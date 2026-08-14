"""tools/cim_physics.py: real NoC hop-count + CIM device-physics scaling.

These are pure-function unit tests (no LangGraph, no tool wrapping) directly
against the topology-distance formulas and the mapper/profiler analytical
models, isolating physics-correctness from the @wrap_tool_call plumbing
covered elsewhere.
"""

from schemas.config import NoCTopology
from schemas.tools import LayerBitConfig
from tools.cim_physics import hop_distance, simulate_cim_profile, simulate_noc_mapping


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
