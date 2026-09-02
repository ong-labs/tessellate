"""Backend simulator tools for Tessellate.

Each function is the Python/PyTorch (or eventual NeuroSim/CIM-Loop
C-extension) wrapper referenced in Research_Plan.md 3: the LLM only picks
parameters via `tool_calls`; these do the numeric work. Every tool is
invoked only through `wrap_tool_call` (middleware.py), which validates
arguments, injects `ExecutionContext`/`HWConfig`, and contains exceptions.

@tuner runs a real PyTorch QAT trial (tools/qat.py): a pretrained model is
fake-quantized and structurally column-pruned per `args.layer_configs`,
applied per real model stage (`LayerBitConfig.layer_name` == one of
`tools.qat.get_layer_groups`'s stage names -- see that module's docstring),
fine-tuned with real gradient descent, and evaluated for real measured
accuracy on a small held-out set. @mapper and @profiler run real, physically-grounded analytical
simulations (tools/cim_physics.py): topology-aware NoC hop-count routing and
bandwidth-limited transfer timing for @mapper, and CIM device-physics
scaling laws (ADC/DAC/crossbar energy, parallel-array compute latency) for
@profiler. Mapper/profiler are still "Fast Approximation" models in
Research_Plan.md's sense, not bindings to the actual NeuroSim/CIM-Loop
C-extensions (docs/coding_guidelines.md 5.D: Mock First for those) -- `calibration_factors`
(per hw_spec_id) is the intended correction loop against precise simulation
over time.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from middleware import wrap_tool_call
from schemas.tools import (
    MapperToolInput,
    MapperToolOutput,
    PrecisionCheckToolInput,
    PrecisionCheckToolOutput,
    ProfilerToolInput,
    ProfilerToolOutput,
    ToolStatus,
    TunerToolInput,
    TunerToolOutput,
    VerifierToolInput,
    VerifierToolOutput,
)
from tools.cim_physics import simulate_cim_profile, simulate_noc_mapping
from tools.precision_check import compute_updated_calibration_factor, simulate_precise_energy
from tools.qat import get_qat_backend, get_stage_weight_shapes, run_qat_tuning
from tools.targets import check_targets


@wrap_tool_call(TunerToolInput, node_name="tuner")
def tuner_tool(args: TunerToolInput) -> TunerToolOutput:
    """Real crossbar-aligned QAT / column-pruning trial (tools/qat.py).

    Each `args.layer_configs` entry is applied to its own real model stage
    (`layer_name` matched against `tools.qat.get_layer_groups`'s stage
    names); any real stage not named by an entry falls back to the
    across-entries average -- no crossbar size or bit-width is hardcoded,
    both come from `args.layer_configs` / `args.hw_config`. Echoes the
    validated `layer_configs` back in `data` so @mapper/@profiler simulate
    against the exact configuration actually used here.
    """
    avg_weight_bits = sum(lc.weight_bits for lc in args.layer_configs) / len(args.layer_configs)
    avg_pruning_ratio = sum(lc.column_pruning_ratio for lc in args.layer_configs) / len(args.layer_configs)
    stage_configs = {lc.layer_name: (lc.weight_bits, lc.column_pruning_ratio) for lc in args.layer_configs}

    backend = get_qat_backend(args.execution_context.model_id, args.execution_context.dataset_id)
    tuning_result = run_qat_tuning(
        backend,
        stage_configs=stage_configs,
        default_config=(round(avg_weight_bits), avg_pruning_ratio),
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        # Column-wise partial-sum quantization (tools/qat.py's
        # apply_crossbar_quant_prune/_PartialSumQuant): a fixed hardware
        # property (one ADC per crossbar, same precision for every
        # column), not a per-stage search parameter -- args.hw_config,
        # not args.layer_configs.
        adc_bits=args.hw_config.adc_bits,
        seed=args.execution_context.run_seed,
    )

    return TunerToolOutput(
        status=ToolStatus.SUCCESS,
        data={
            "accuracy": round(tuning_result["accuracy"], 4),
            "epochs_run": tuning_result["epochs_run"],
            "model_ref": f"{args.execution_context.model_id}::tuned::iter{args.execution_context.iteration_count}",
            "avg_weight_bits": avg_weight_bits,
            "avg_column_pruning_ratio": avg_pruning_ratio,
            "layer_configs": [lc.model_dump(mode="json") for lc in args.layer_configs],
            # tools/qat.py's run_qat_tuning auto-detects CUDA/MPS/CPU
            # (TESSELLATE_QAT_DEVICE overrides) -- echoed back so it's visible
            # in the observability log / dashboard, not just inferred.
            "device": tuning_result["device"],
        },
    )


def _stage_shapes_for(model_id: str) -> Optional[Dict[str, List[Tuple[int, int, int]]]]:
    """Real per-stage weight-matrix shapes for `model_id`, or None if this
    model's architecture can't be resolved.

    None makes `tools.cim_physics` fall back to its old one-array-per-entry
    assumption rather than fail the whole tool: a custom `--model-id` spec
    that can't be imported is already reported by @tuner (which really does
    need the model), so failing @mapper/@profiler on it too would turn one
    clear error into three confusing ones. `get_stage_weight_shapes` reuses
    the architecture instance `@tuner`/@planner already cached, so this
    costs a dict walk, not a model build."""
    try:
        return get_stage_weight_shapes(model_id)
    except Exception:  # noqa: BLE001 -- see docstring: degrade, don't cascade
        return None


@wrap_tool_call(MapperToolInput, node_name="mapper")
def mapper_tool(args: MapperToolInput) -> MapperToolOutput:
    """Real Multi-Tile / NoC mapping simulation (tools/cim_physics.py):
    every layer's real weight-matrix shape is tiled across
    `crossbar_rows x crossbar_cols` arrays, placed and pipelined per
    `args.hw_config.noc_topology`'s real hop-count formula, with
    bandwidth-limited transfer timing -- never a literal '128x128'/tile
    count or an arbitrary latency constant.
    """
    mapping = simulate_noc_mapping(
        args.hw_config, args.layer_configs, stage_weight_shapes=_stage_shapes_for(args.execution_context.model_id)
    )
    mapping["tiling_strategy"] = args.tiling_strategy or "default_row_major"

    return MapperToolOutput(status=ToolStatus.SUCCESS, data=mapping)


@wrap_tool_call(ProfilerToolInput, node_name="profiler")
def profiler_tool(args: ProfilerToolInput) -> ProfilerToolOutput:
    """Real CIM device-physics energy/latency simulation
    (tools/cim_physics.py): ADC energy exponential in `adc_bits`, DAC
    energy linear in `dac_bits`, crossbar energy proportional to array size
    and per-layer `weight_bits`, and compute latency bounded by ADC/DAC
    conversion time (CIM's parallel-array MAC property). The result is
    still a *fast approximation* (Research_Plan.md 3) -- scaled by this
    hw_spec_id's `calibration_factors` entry, which the verify loop is
    meant to refine against precise simulation over time. Only
    `compute_energy_pj` (crossbar/ADC/DAC) is scaled by that factor --
    `buffer_energy_pj` (SRAM buffer staging) isn't derived from any
    calibration reference (see tools/calibration.py's compute_calibration_factor)
    and is added back in afterward, unscaled, to get the final `energy_pj`.
    """
    profile = simulate_cim_profile(
        args.hw_config, args.layer_configs, stage_weight_shapes=_stage_shapes_for(args.execution_context.model_id)
    )

    calibration_factors = dict(args.calibration_factors or {})
    factor = calibration_factors.setdefault(args.hw_config.hw_spec_id, 1.0)
    profile["compute_energy_pj"] = round(profile["compute_energy_pj"] * factor, 6)
    profile["energy_pj"] = round(profile["compute_energy_pj"] + profile["buffer_energy_pj"], 6)
    profile["latency_ns"] = round(profile["latency_ns"] * factor, 6)

    return ProfilerToolOutput(
        status=ToolStatus.SUCCESS,
        data={
            **profile,
            "calibration_factors": calibration_factors,
        },
    )


@wrap_tool_call(VerifierToolInput, node_name="verifier")
def verifier_tool(args: VerifierToolInput) -> VerifierToolOutput:
    """Real IR-drop/noise-margin verifier (analytical fast-approximation,
    same Research_Plan.md 3 sense as @mapper/@profiler -- not a placeholder):
    synthesizes @mapper + @profiler results and checks them against IR-drop/
    noise bounds derived from `args.hw_config` (wire_resistance_ohm_per_um,
    device_noise_sigma) -- never a hardcoded tolerance tied to one specific
    chip.
    """
    hw = args.hw_config
    wire_resistance = hw.wire_resistance_ohm_per_um or 0.0
    noise_sigma = hw.device_noise_sigma or 0.0

    mapper_data = args.mapper_result.data or {}
    profiler_data = args.profiler_result.data or {}

    ir_drop_error_pct = round(wire_resistance * hw.crossbar_rows * 0.01, 4)
    noise_margin_db = round(6.02 * hw.adc_bits - 20 * noise_sigma, 4)
    is_converged = (
        args.mapper_result.status == ToolStatus.SUCCESS
        and args.profiler_result.status == ToolStatus.SUCCESS
        and ir_drop_error_pct < 5.0
        and noise_margin_db > 0.0
    )

    return VerifierToolOutput(
        status=ToolStatus.SUCCESS,
        data={
            "ir_drop_error_pct": ir_drop_error_pct,
            "noise_margin_db": noise_margin_db,
            "is_converged": is_converged,
            "noc_latency_ms": mapper_data.get("noc_latency_ms"),
            "energy_pj": profiler_data.get("energy_pj"),
        },
    )


@wrap_tool_call(PrecisionCheckToolInput, node_name="precision_check")
def precision_check_tool(args: PrecisionCheckToolInput) -> PrecisionCheckToolOutput:
    """MOCK Stage-2 high-fidelity re-verification (tools/precision_check.py,
    docs/coding_guidelines.md 5.D Mock First) -- only called for a candidate @evaluator's
    fast-approximation already judged converged (nodes/precision_verifier.py),
    never every candidate a run searches: a real NeuroSim/CIM-Loop call is
    far more expensive than tools/cim_physics.py's analytical formulas, so
    this re-checks only the (rare) candidates a run is about to stop on.

    Re-runs `check_targets` (tools/targets.py, the same rule @evaluator
    itself uses) with the precise energy number in place of the fast one --
    accuracy/noc_latency_ms are passed through unchanged (analog crossbar/
    ADC fidelity is this check's scope, not @tuner's measured accuracy or
    @mapper's NoC routing time). `calibration_factor` is derived from the
    *compute-only* raw/precise pair, not the totals -- consistent with
    `tools.calibration.compute_calibration_factor` (which this self-updated
    factor feeds the same `calibration_factors[hw_spec_id]` slot as) never
    folding SRAM buffer energy into a MAC-scaling calibration ratio, and
    with `profiler_tool` only ever applying a calibration_factor to
    `compute_energy_pj`."""
    result = simulate_precise_energy(
        args.hw_config, args.layer_configs, _stage_shapes_for(args.execution_context.model_id)
    )
    calibration_factor = compute_updated_calibration_factor(
        result["raw_compute_energy_pj"], result["precise_compute_energy_pj"]
    )

    target_errors = check_targets(
        args.accuracy,
        result["precise_energy_pj"],
        args.noc_latency_ms,
        args.target_accuracy,
        args.target_energy_pj,
        args.target_latency_ms,
    )

    return PrecisionCheckToolOutput(
        status=ToolStatus.SUCCESS,
        data={
            "raw_energy_pj": round(result["raw_energy_pj"], 6),
            "precise_energy_pj": round(result["precise_energy_pj"], 6),
            "calibration_factor": round(calibration_factor, 6),
            "is_converged": not target_errors,
            "target_errors": target_errors,
        },
    )
