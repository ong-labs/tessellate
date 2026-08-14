"""tools/cim_physics.py: real NoC hop-count + CIM device-physics scaling.

These are pure-function unit tests (no LangGraph, no tool wrapping) directly
against the topology-distance formulas and the mapper/profiler analytical
models, isolating physics-correctness from the @wrap_tool_call plumbing
covered elsewhere.
"""

from schemas.config import NoCTopology
from schemas.tools import LayerBitConfig
from tools.cim_physics import crossbar_passes, hop_distance, simulate_cim_profile, simulate_noc_mapping


def _layer(weight_bits=6, activation_bits=6, pruning=0.0, name="l0"):
    return LayerBitConfig(
        layer_name=name, weight_bits=weight_bits, activation_bits=activation_bits, column_pruning_ratio=pruning
    )


# --- hop_distance: real topology graph-distance formulas --------------------


def test_hop_distance_same_tile_is_zero():
    assert hop_distance(NoCTopology.MESH, 3, 3, 16) == 0


def test_hop_distance_mesh_is_manhattan():
    # 4x4 grid: tile0=(0,0), tile1=(0,1) -> 1 hop; tile5=(1,1) -> 2 hops
    assert hop_distance(NoCTopology.MESH, 0, 1, 16) == 1
    assert hop_distance(NoCTopology.MESH, 0, 5, 16) == 2


def test_hop_distance_torus_wraps_around():
    # 4x4 grid: tile0=(0,0), tile3=(0,3) direct dx=3, but torus wraps to 1
    assert hop_distance(NoCTopology.TORUS, 0, 3, 16) == 1
    assert hop_distance(NoCTopology.MESH, 0, 3, 16) == 3  # mesh: no wraparound


def test_hop_distance_ring_takes_shorter_arc():
    assert hop_distance(NoCTopology.RING, 0, 15, 16) == 1
    assert hop_distance(NoCTopology.RING, 0, 8, 16) == 8


def test_hop_distance_bus_is_always_one_hop():
    assert hop_distance(NoCTopology.CROSSBAR_BUS, 0, 15, 16) == 1
    assert hop_distance(NoCTopology.CROSSBAR_BUS, 2, 9, 16) == 1


# --- simulate_noc_mapping: pipelined transfer over real topology distance ---


def test_single_layer_needs_no_noc_transfer(good_hw_config):
    result = simulate_noc_mapping(good_hw_config, [_layer()])

    assert result["tiles_needed"] == 1
    assert result["noc_latency_ms"] == 0.0  # nothing to route to


def test_more_layers_increase_noc_latency(good_hw_config):
    one_layer = simulate_noc_mapping(good_hw_config, [_layer()])
    three_layers = simulate_noc_mapping(good_hw_config, [_layer(), _layer(), _layer()])

    assert three_layers["tiles_needed"] == 3
    assert three_layers["noc_latency_ms"] > one_layer["noc_latency_ms"]


def test_tile_utilization_scales_with_layer_count_and_num_tiles(good_hw_config):
    result = simulate_noc_mapping(good_hw_config, [_layer(), _layer()])
    assert result["tile_utilization"] == round(2 / good_hw_config.num_tiles, 4)


# --- simulate_cim_profile: real ADC/DAC/crossbar energy scaling laws -------


def test_higher_adc_bits_increase_energy_exponentially(good_hw_config):
    low = simulate_cim_profile(good_hw_config, [_layer()])

    hotter_hw = good_hw_config.model_copy(update={"adc_bits": good_hw_config.adc_bits + 2})
    high = simulate_cim_profile(hotter_hw, [_layer()])

    assert high["energy_pj"] > low["energy_pj"] * 2  # exponential, not linear, growth


def test_higher_weight_bits_increase_energy_linearly_via_crossbar_term(good_hw_config):
    low = simulate_cim_profile(good_hw_config, [_layer(weight_bits=2)])
    high = simulate_cim_profile(good_hw_config, [_layer(weight_bits=8)])

    assert high["energy_pj"] > low["energy_pj"]


def test_more_layers_increase_both_energy_and_latency(good_hw_config):
    one_layer = simulate_cim_profile(good_hw_config, [_layer()])
    two_layers = simulate_cim_profile(good_hw_config, [_layer(), _layer()])

    # compute_energy_pj (ADC+DAC+crossbar) scales with layer count; buffer_energy_pj
    # is a one-time buffer-fill cost (simulate_cim_profile's own docstring) and must
    # NOT double just because the run has more layers.
    assert two_layers["compute_energy_pj"] == round(one_layer["compute_energy_pj"] * 2, 6)
    assert two_layers["buffer_energy_pj"] == one_layer["buffer_energy_pj"]
    assert two_layers["energy_pj"] == round(two_layers["compute_energy_pj"] + two_layers["buffer_energy_pj"], 6)
    assert two_layers["latency_ns"] == round(one_layer["latency_ns"] * 2, 6)


def test_buffer_energy_is_zero_when_sram_buffer_kb_unset(good_hw_config):
    hw = good_hw_config.model_copy(update={"sram_buffer_kb": None})
    result = simulate_cim_profile(hw, [_layer()])

    assert result["buffer_energy_pj"] == 0.0
    assert result["energy_pj"] == result["compute_energy_pj"]


def test_buffer_energy_scales_linearly_with_sram_buffer_kb(good_hw_config):
    small = good_hw_config.model_copy(update={"sram_buffer_kb": 32.0})
    large = good_hw_config.model_copy(update={"sram_buffer_kb": 64.0})

    small_result = simulate_cim_profile(small, [_layer()])
    large_result = simulate_cim_profile(large, [_layer()])

    assert large_result["buffer_energy_pj"] == round(small_result["buffer_energy_pj"] * 2, 6)
    # compute_energy_pj must be unaffected by sram_buffer_kb -- it's a
    # crossbar/ADC/DAC-only quantity (tools/calibration.py calibrates
    # against it specifically because it excludes buffer energy).
    assert small_result["compute_energy_pj"] == large_result["compute_energy_pj"]


# --- crossbar_passes: real weight-matrix -> array mapping -------------------


def test_dense_layer_is_tiled_across_arrays_by_its_real_shape(good_hw_config):
    """A 4608x512 matrix (resnet18's layer4 3x3x512x512 conv) on a 128x128
    array needs ceil(4608/128) * ceil(512/128) = 36*4 = 144 passes, not 1."""
    assert crossbar_passes(4608, 512, 1, good_hw_config) == 144


def test_a_matrix_smaller_than_the_array_still_costs_one_pass(good_hw_config):
    assert crossbar_passes(27, 32, 1, good_hw_config) == 1


def test_depthwise_groups_are_packed_not_charged_one_array_each(good_hw_config):
    """A 3x3 depthwise conv over 960 channels is block-diagonal: 9 rows and
    1 column per group, so 128//9 = 14 groups pack into one array ->
    ceil(960/14) = 69 passes. Charging one array per group (7x the arrays,
    960) is the naive mapping this deliberately does not assume -- see
    crossbar_passes' docstring."""
    assert crossbar_passes(9, 960, 960, good_hw_config) == 69


def test_a_group_larger_than_the_array_falls_back_to_per_group_tiling(good_hw_config):
    """rows > crossbar_rows can't be packed at all, so each group is tiled
    on its own rather than silently reported as a single cheap pass."""
    assert crossbar_passes(256, 8, 4, good_hw_config) == 2 * 1 * 4


def test_column_pruning_reduces_crossbar_passes(good_hw_config):
    """Structured column pruning physically removes output channels, so it
    must shrink the mapped matrix -- without this, pruning was a pure
    accuracy cost the multi-objective search had no reason to ever pick."""
    unpruned = crossbar_passes(4608, 512, 1, good_hw_config, pruning_ratio=0.0)
    pruned = crossbar_passes(4608, 512, 1, good_hw_config, pruning_ratio=0.5)

    assert pruned < unpruned
    assert pruned == 36 * 2  # 512 cols -> 256 -> 2 column-tiles instead of 4


# --- shape-aware profiling --------------------------------------------------


_RESNET_LAYER4_SHAPES = {"layer4.1": [(4608, 512, 1), (4608, 512, 1)]}


def test_profile_without_shapes_keeps_the_one_array_per_entry_fallback(good_hw_config):
    """tools/calibration.py's synthetic probe layer really is a single full
    array, so the shape-free call must keep costing exactly one pass."""
    one_entry = simulate_cim_profile(good_hw_config, [_layer(name="layer4.1")])
    two_entries = simulate_cim_profile(good_hw_config, [_layer(name="a"), _layer(name="b")])

    assert two_entries["compute_energy_pj"] == round(one_entry["compute_energy_pj"] * 2, 6)


def test_real_shapes_charge_a_stage_all_of_its_layers(good_hw_config):
    """One stage holds several real layers (resnet18's layer4.1 is two
    3x3x512x512 convs = 288 passes); charging it one array understated the
    stage by that factor."""
    blind = simulate_cim_profile(good_hw_config, [_layer(name="layer4.1")])
    real = simulate_cim_profile(good_hw_config, [_layer(name="layer4.1")], stage_weight_shapes=_RESNET_LAYER4_SHAPES)

    assert real["compute_energy_pj"] == round(blind["compute_energy_pj"] * 288, 6)


def test_an_unknown_stage_name_falls_back_instead_of_guessing(good_hw_config):
    """A stage with no resolvable shape must cost the documented fallback
    of one pass, never an invented shape."""
    blind = simulate_cim_profile(good_hw_config, [_layer(name="not_a_real_stage")])
    real = simulate_cim_profile(
        good_hw_config, [_layer(name="not_a_real_stage")], stage_weight_shapes=_RESNET_LAYER4_SHAPES
    )

    assert real["compute_energy_pj"] == blind["compute_energy_pj"]


def test_pruning_lowers_profiled_energy_once_shapes_are_known(good_hw_config):
    unpruned = simulate_cim_profile(
        good_hw_config, [_layer(name="layer4.1", pruning=0.0)], stage_weight_shapes=_RESNET_LAYER4_SHAPES
    )
    pruned = simulate_cim_profile(
        good_hw_config, [_layer(name="layer4.1", pruning=0.5)], stage_weight_shapes=_RESNET_LAYER4_SHAPES
    )

    assert pruned["compute_energy_pj"] < unpruned["compute_energy_pj"]


def test_noc_tiles_needed_counts_real_arrays_not_layer_entries(good_hw_config):
    mapping = simulate_noc_mapping(
        good_hw_config, [_layer(name="layer4.1")], stage_weight_shapes=_RESNET_LAYER4_SHAPES
    )

    assert mapping["tiles_needed"] == 288
    # 288 tiles on a 16-tile chip is a real overflow, and tile_utilization
    # is capped at 1.0 -- it must report saturation, not a comfortable 0.06.
    assert mapping["tile_utilization"] == 1.0


def test_noc_transfer_latency_stays_per_layer_boundary(good_hw_config):
    """Intra-layer tiles merge partial sums locally; only layer-to-layer
    activations cross the NoC, so a multi-tile layer must not inflate
    transfer latency (see simulate_noc_mapping's docstring)."""
    blind = simulate_noc_mapping(good_hw_config, [_layer(name="layer4.1")])
    real = simulate_noc_mapping(good_hw_config, [_layer(name="layer4.1")], stage_weight_shapes=_RESNET_LAYER4_SHAPES)

    assert real["noc_latency_ms"] == blind["noc_latency_ms"]
