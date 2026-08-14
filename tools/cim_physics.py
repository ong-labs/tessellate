"""Physically-grounded analytical models for @mapper / @profiler.

Replaces the earlier placeholder arithmetic in tools/simulators.py with real
device-physics and NoC-architecture scaling relationships:

- NoC routing: real topology hop-count graph distances (mesh Manhattan
  distance, torus wraparound, ring shortest-arc, bus single-hop) and
  bandwidth-limited serialization time -- no arbitrary latency constant
  beyond a standard NoC flit header size.
- CIM profiling (NeuroSim/ISAAC-literature scaling laws): ADC energy scales
  exponentially with resolution (flash-ADC dominated), DAC energy scales
  linearly with resolution, crossbar read/MAC energy scales with array size
  and weight bit-width (bit-serial CIM), and per-layer compute latency is
  bounded by ADC/DAC conversion time -- exploiting CIM's defining property
  that one crossbar performs a full row x col MAC in parallel per cycle.
  These three terms are `simulate_cim_profile`'s `compute_energy_pj` --
  the crossbar-macro-only quantity tools/calibration.py's literature
  references measure and calibrate against.
- SRAM buffer staging: a separate, one-time (not per-layer) cost for
  filling `hw.sram_buffer_kb` worth of on-chip buffer, scaling linearly
  with buffer size -- `simulate_cim_profile`'s `buffer_energy_pj`, deliberately
  *not* part of `compute_energy_pj`/calibration (see that function's
  docstring for why) but included in the total `energy_pj`.

`CIM_TECH_PARAMS` are representative order-of-magnitude reference values,
not silicon-measured numbers for any specific process node -- this is
exactly why `AutoCIMState.calibration_factors` exists (Research_Plan.md 3):
these are a *fast approximation*, meant to be scaled by a per-hw_spec_id
factor learned against precise simulation over time (applied in
tools/simulators.py's profiler_tool, to `compute_energy_pj` only), not
treated as ground truth on their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from schemas.config import HWConfig, NoCTopology
from schemas.tools import LayerBitConfig


@dataclass(frozen=True)
class CIMTechParams:
    """Reference (uncalibrated) technology constants.

    Order-of-magnitude placeholders, not values fit or validated against any
    specific real chip -- there is no sensitivity analysis here showing
    downstream Pareto rankings are robust to a different-but-plausible set
    of these numbers. `calibration_factors`/`calibration_provenance`
    (tools/calibration.py) is this project's actual honesty mechanism for
    that gap: any `hw_spec_id` without a real published-reference match
    stays visibly "uncalibrated" (tools/dashboard.py) rather than silently
    trusting these constants' absolute scale."""

    adc_ref_energy_fj: float = 0.2  # energy of a 1-bit-equivalent flash-ADC conversion step
    dac_ref_energy_fj: float = 0.05  # energy per DAC bit driven
    crossbar_ref_energy_fj: float = 0.01  # energy per (cell * weight-bit) MAC
    adc_cycle_time_ns: float = 0.1  # time per SAR-ADC bit-cycle
    dac_settle_time_ns: float = 0.05  # DAC settling time per bit
    router_flit_header_bits: int = 32  # standard NoC flit header overhead per hop
    sram_buffer_ref_energy_fj_per_bit: float = 0.002  # energy per bit staged into the on-chip SRAM buffer, once per profile


CIM_TECH_PARAMS = CIMTechParams()


# Bump whenever this module's energy/latency numbers change *meaning* -- not
# for a refactor, but whenever a value computed by the old code would be
# wrong to compare against one computed by the new code.
#
# `store.py` keys its energy-derived long-term memory (calibration factors,
# stored Pareto candidates) on this, so bumping it makes every pre-existing
# entry invisible instead of silently mixing incompatible scales into a new
# run. That is not hypothetical: version 1 charged one crossbar array per
# layer_configs entry, version 2 charges the real per-layer pass count, so a
# v1 candidate's energy_pj is ~70x smaller than a v2 candidate's for the
# identical design. A real session loaded five such v1 candidates from a
# previous run and plotted them in the same table as v2 numbers.
#
# 1: one array per layer_configs entry (pre-2026-08-14)
# 2: real per-layer crossbar pass counts, pruning reduces mapped columns
ENERGY_MODEL_VERSION = 2


# ---------------------------------------------------------------------------
# NoC topology: real hop-count graph distance formulas
# ---------------------------------------------------------------------------


def _tile_coordinates(index: int, grid_dim: int) -> Tuple[int, int]:
    return index // grid_dim, index % grid_dim


def hop_distance(topology: NoCTopology, i: int, j: int, ring_or_grid_size: int) -> int:
    """Real graph hop-count between tile `i` and tile `j` under `topology`,
    where `ring_or_grid_size` is the physical tile count the NoC is built
    for (`HWConfig.num_tiles`), independent of how many tiles a given
    workload actually uses."""
    if i == j:
        return 0
    if topology == NoCTopology.CROSSBAR_BUS:
        return 1  # shared bus: every node is one hop from every other
    if topology == NoCTopology.RING:
        direct = abs(i - j)
        return min(direct, ring_or_grid_size - direct)

    grid_dim = max(1, math.ceil(math.sqrt(ring_or_grid_size)))
    xi, yi = _tile_coordinates(i, grid_dim)
    xj, yj = _tile_coordinates(j, grid_dim)
    dx, dy = abs(xi - xj), abs(yi - yj)
    if topology == NoCTopology.TORUS:
        dx = min(dx, grid_dim - dx)
        dy = min(dy, grid_dim - dy)
    return dx + dy  # MESH, and TORUS after wraparound folding


# ---------------------------------------------------------------------------
# @mapper: multi-tile placement + pipelined NoC transfer latency
# ---------------------------------------------------------------------------


def simulate_noc_mapping(
    hw: HWConfig,
    layer_configs: List[LayerBitConfig],
    params: CIMTechParams = CIM_TECH_PARAMS,
    stage_weight_shapes: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
) -> Dict[str, Any]:
    """Layers are placed in `layer_configs` order and pipelined tile-to-tile
    -- the standard layer-pipelined CIM dataflow. Real per-hop topology
    distance (`hop_distance`) and bandwidth-limited transfer time;
    `noc_link_bandwidth_gbps` (Gb/s) converts to bits/ns directly
    (1 Gb/s == 1 bit/ns), so no unit-conversion constant is needed beyond
    the documented flit header size.

    `stage_weight_shapes` (`tools.qat.get_stage_weight_shapes`) gives each
    entry its real tile count (`crossbar_passes`); without it this falls
    back to the old one-tile-per-entry assumption, which silently claimed
    any layer fits one array -- resnet18's `layer4.1` alone really needs
    288 tiles on a 128x128 array, so `tile_utilization` was reported as
    comfortably under 1.0 for workloads that in fact overflow the chip
    many times over.

    Note the two counts are deliberately different things: `tiles_needed`
    is every array activation the workload requires, while the transfer
    loop below stays *per layer boundary*. Tiles within one layer hold
    slices of the same weight matrix and merge partial sums locally; it is
    only the layer-to-layer activation hand-off that crosses the NoC in
    this pipelined dataflow, so inflating the transfer count by intra-layer
    tiles would model traffic that doesn't exist.
    """
    tiles_needed = sum(_passes_for_layer(hw, lc, stage_weight_shapes) for lc in layer_configs)
    bandwidth_bits_per_ns = hw.noc_link_bandwidth_gbps

    total_latency_ns = 0.0
    for idx in range(1, len(layer_configs)):
        hops = hop_distance(hw.noc_topology, idx - 1, idx, hw.num_tiles)
        payload_bits = hw.crossbar_cols * layer_configs[idx].activation_bits
        serialization_ns = payload_bits / bandwidth_bits_per_ns
        header_ns = hops * params.router_flit_header_bits / bandwidth_bits_per_ns
        total_latency_ns += serialization_ns + header_ns

    buffer_bits = (hw.sram_buffer_kb or 0.0) * 8 * 1024
    buffer_delay_ns = buffer_bits / bandwidth_bits_per_ns if buffer_bits else 0.0

    return {
        "noc_latency_ms": round(total_latency_ns / 1e6, 6),
        "buffer_delay_ms": round(buffer_delay_ns / 1e6, 6),
        "tile_utilization": round(min(1.0, tiles_needed / hw.num_tiles), 4),
        "tiles_needed": tiles_needed,
    }


# ---------------------------------------------------------------------------
# Real weight-matrix -> crossbar pass mapping
# ---------------------------------------------------------------------------


def crossbar_passes(rows: int, cols: int, groups: int, hw: HWConfig, pruning_ratio: float = 0.0) -> int:
    """How many `hw.crossbar_rows x hw.crossbar_cols` array activations a
    weight matrix of shape `(rows, cols)` with `groups` groups really costs.

    `rows`/`cols`/`groups` come from `tools.qat.get_stage_weight_shapes`
    (`rows` is the *per-group* input length). Three cases:

    - **Dense / pointwise (`groups == 1`)**: the ordinary tiled GEMM
      mapping -- `ceil(rows/R) * ceil(cols/C)` arrays. This alone is the
      dominant correction: a stage is a group of layers, and a single
      3x3x512x512 conv already needs 144 passes on a 128x128 array, so
      charging one array per stage understated resnet18 by ~70x.
    - **Depthwise (`groups == in_channels`, and any grouped conv)**: the
      matrix is block-diagonal -- group `i` only connects its own `rows`
      inputs to its own `cols/groups` outputs. Packing those small blocks
      down the diagonal of one array fits
      `min(R//rows, C//cols_per_group)` groups per activation, so a
      960-group 3x3 depthwise layer costs ~69 passes, not 960.
    - **A group too large to pack** (a block wider/taller than the array
      itself) falls back to tiling each group independently.

    The packing assumption is load-bearing and deliberately documented
    rather than hidden: measured on mobilenet_v2, packed vs. one-array-per-
    group differ by ~10x in total passes (746 vs. 7363). Packing is what
    real CIM accelerators do with depthwise layers, so it is the default
    here -- but it is an assumption about the *mapper*, not a device
    constant, which is precisely why this project reports depthwise-heavy
    numbers as approximations (see `CIMTechParams`' own docstring on the
    calibration escape hatch).

    `pruning_ratio` is `LayerBitConfig.column_pruning_ratio`: structured
    column pruning physically removes output channels, so it shrinks `cols`
    before tiling. Without this, pruning was a pure accuracy cost with no
    energy reward anywhere in the model, and the multi-objective search had
    no reason to ever prune."""
    effective_cols = max(1, int(cols * (1.0 - min(max(pruning_ratio, 0.0), 1.0))))
    array_rows, array_cols = hw.crossbar_rows, hw.crossbar_cols

    if groups <= 1:
        return math.ceil(rows / array_rows) * math.ceil(effective_cols / array_cols)

    cols_per_group = max(1, effective_cols // groups)
    groups_per_array = min(array_rows // rows, array_cols // cols_per_group) if rows <= array_rows else 0
    if groups_per_array >= 1:
        return math.ceil(groups / groups_per_array)
    # A single group already exceeds the array -- tile each one separately.
    return math.ceil(rows / array_rows) * math.ceil(cols_per_group / array_cols) * groups


def snap_pruning_ratio(
    ratio: float,
    shapes: Optional[List[Tuple[int, int, int]]],
    hw: HWConfig,
    max_ratio: float,
    step: float = 0.01,
) -> float:
    """Rounds `ratio` *down* to the largest pruning ratio that actually
    removes a crossbar array from this stage, or 0.0 if none does.

    This is a dominance argument, not a heuristic. `crossbar_passes` only
    drops when pruning crosses a tile boundary (a 512-column layer on a
    128-column array needs 4 arrays until pruning reaches 0.25), so every
    ratio strictly between two boundaries costs real accuracy and buys
    exactly zero energy -- measured: pruning 0.0/0.1/0.2 gave byte-identical
    energy at every weight_bits while accuracy fell up to 0.12. Snapping
    down therefore never raises energy (same pass count by construction) and
    never lowers accuracy (less weight removed), so the snapped config
    weakly dominates the one the search asked for.

    A stage whose layers already fit in one column-tile (resnet18's early
    64/128-column stages) can never save an array by pruning, so this
    returns 0.0 for it -- pruning there is pure loss no matter the ratio.

    Boundaries are found by scanning `step`-spaced ratios rather than solved
    in closed form: `crossbar_passes` mixes ceil() with the grouped/packed
    depthwise cases, and a scan is both obviously correct and cheap (a few
    dozen integer evaluations per stage)."""
    ratio = min(max(ratio, 0.0), max_ratio)
    if not shapes or ratio <= 0.0:
        return 0.0

    def passes_at(p: float) -> int:
        return sum(crossbar_passes(rows, cols, groups, hw, p) for rows, cols, groups in shapes)

    baseline = passes_at(0.0)
    best = 0.0
    seen = baseline
    candidate = step
    while candidate <= ratio + 1e-9:
        current = passes_at(candidate)
        if current < seen:
            seen = current
            best = round(candidate, 4)
        candidate = round(candidate + step, 4)
    return best


def _passes_for_layer(
    hw: HWConfig,
    lc: LayerBitConfig,
    stage_weight_shapes: Optional[Dict[str, List[Tuple[int, int, int]]]],
) -> int:
    """Total array activations one `layer_configs` entry costs. Falls back
    to 1 (the old shape-blind behavior) when no shape is known for this
    entry -- a synthetic probe layer (`tools.calibration`) or a model whose
    stage names don't resolve, rather than guessing a shape."""
    if not stage_weight_shapes:
        return 1
    shapes = stage_weight_shapes.get(lc.layer_name)
    if not shapes:
        return 1
    return sum(crossbar_passes(rows, cols, groups, hw, lc.column_pruning_ratio) for rows, cols, groups in shapes)


# ---------------------------------------------------------------------------
# @profiler: CIM device-physics energy/latency model
# ---------------------------------------------------------------------------


def simulate_cim_profile(
    hw: HWConfig,
    layer_configs: List[LayerBitConfig],
    params: CIMTechParams = CIM_TECH_PARAMS,
    stage_weight_shapes: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
) -> Dict[str, Any]:
    """See module docstring for the scaling relationships. `compute_energy_pj`
    sums per-layer ADC + DAC + crossbar-MAC energy across all `layer_configs`
    entries -- this is the quantity tools/calibration.py's KNOWN_REFERENCES
    calibrates against, since every published pJ/MAC figure there measures
    only the crossbar macro itself, never a host SRAM buffer. `buffer_energy_pj`
    is a separate, one-time (not per-layer) cost for staging `hw.sram_buffer_kb`
    worth of weights into the on-chip SRAM buffer -- the same "filled once per
    profile" granularity `simulate_noc_mapping`'s own `buffer_delay_ms` already
    assumes for this same field, just for energy instead of latency. Zero if
    `hw.sram_buffer_kb` is unset. `energy_pj` is their sum -- the one number
    @evaluator's Pareto ranking and target-checking actually use.
    `latency_ns` sums per-layer ADC/DAC conversion time, since a CIM
    crossbar's row x col MAC itself completes in parallel per cycle (the
    array-size term does not appear in the latency model, only in energy).

    `stage_weight_shapes` (`tools.qat.get_stage_weight_shapes`, keyed by
    `LayerBitConfig.layer_name`) is what makes each entry cost its *real*
    number of array activations (`crossbar_passes`) instead of exactly one.
    Omitting it keeps the old one-full-array-per-entry behavior, which is
    correct only for a caller whose entry really is a single full array --
    `tools.calibration.compute_calibration_factor`'s synthetic probe layer
    is exactly that case, and deliberately passes nothing here. Every real
    profiling caller (`tools.simulators.profiler_tool`) passes real shapes;
    without them, resnet18's energy came out ~70x low and mobilenet_v2's
    ~37x low, i.e. wrong by an architecture-dependent factor that silently
    biased cross-architecture comparisons."""
    total_compute_energy_fj = 0.0
    total_latency_ns = 0.0

    for lc in layer_configs:
        passes = _passes_for_layer(hw, lc, stage_weight_shapes)
        adc_energy_fj = params.adc_ref_energy_fj * (2**hw.adc_bits) * hw.crossbar_cols
        dac_energy_fj = params.dac_ref_energy_fj * hw.dac_bits * hw.crossbar_rows
        crossbar_energy_fj = params.crossbar_ref_energy_fj * hw.crossbar_rows * hw.crossbar_cols * lc.weight_bits
        total_compute_energy_fj += passes * (adc_energy_fj + dac_energy_fj + crossbar_energy_fj)
        total_latency_ns += passes * (hw.adc_bits * params.adc_cycle_time_ns + hw.dac_bits * params.dac_settle_time_ns)

    buffer_bits = (hw.sram_buffer_kb or 0.0) * 8 * 1024
    buffer_energy_fj = params.sram_buffer_ref_energy_fj_per_bit * buffer_bits

    compute_energy_pj = round(total_compute_energy_fj / 1000.0, 6)
    buffer_energy_pj = round(buffer_energy_fj / 1000.0, 6)

    return {
        "compute_energy_pj": compute_energy_pj,
        "buffer_energy_pj": buffer_energy_pj,
        "energy_pj": round(compute_energy_pj + buffer_energy_pj, 6),
        "latency_ns": round(total_latency_ns, 6),
    }
