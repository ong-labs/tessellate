"""Compares Tessellate's searched result against a fixed, unoptimized
"naive" baseline configuration -- the search-free reference point a
hand-tuned human design would plausibly pick without spending any
optimization effort (uniform weight_bits/activation_bits, zero column
pruning, every stage).

Why this exists, distinct from tools/run_ablation.py: that module compares
this project's own search strategies against *each other*
(multi_objective/single_objective/random) -- never against a config nobody
searched for at all. Without this, "the agent found a good candidate"
has no reference point to call "good" against.

Both sides are evaluated across the same N seeds
(tools.repeat_search's machinery), not once each -- QAT training has its
own stochasticity (dropout, batch shuffling) even for a *fixed*
layer_configs (tools.qat.run_qat_tuning's own seed), so a single-run
"baseline vs search" comparison couldn't tell a genuine search advantage
from one lucky/unlucky QAT run on either side.

The baseline candidate is evaluated through the exact same tuner -> mapper
-> profiler -> verifier path tools/batch_warmup.py's warm-up candidates
use (`_evaluate_one_candidate`) -- same physics model, same calibration,
same QAT budget as the searched side, so the only thing that differs is
whether a layer_configs point was searched for at all.

Usage:
    python -m tools.baseline_compare --model-id resnet18 \\
        --hw-config examples/hw_configs/default_cim_v1_128x128.json \\
        --seeds 5 --target-accuracy 0.6

Note the same honesty caveat tools/run_ablation.py's own report carries:
this is "do baseline and search land in visibly different places", backed
by tools/repeat_search.py's Student's t confidence interval on each side --
not a p-value for "search beats baseline" as a formal hypothesis test.

A second, optional baseline (`--grid`): a coarse grid over uniform
(weight_bits, column_pruning_ratio) pairs -- still applied identically
across every stage, unlike the real search's per-stage-independent points
-- evaluated the same way. Distinct from tools/run_ablation.py's `random`
strategy (which samples this same uniform space randomly): a grid
systematically covers it instead. Each grid point is evaluated at a
*single* seed, not replicated across the N used elsewhere -- an explicit,
documented simplification (see `run_grid_search_baseline`'s docstring for
why) to keep the real-QAT-training cost of a whole grid bounded, not a
claim that a grid point's own single-run noise has been averaged out the
way the naive-baseline/search columns' have.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import main
from nodes.evaluator import build_candidate_entry, validate_metrics
from nodes.planner import real_stage_names
from schemas.config import HWConfig
from schemas.tools import LayerBitConfig
from tools.batch_warmup import _evaluate_one_candidate
from tools.calibration import bootstrap_approximate_calibration_factors, bootstrap_calibration_factors
from tools.repeat_search import run_repeated_search, summarize

_DEFAULT_BASELINE_WEIGHT_BITS = 8
_DEFAULT_BASELINE_ACTIVATION_BITS = 8
_DEFAULT_BASELINE_PRUNING_RATIO = 0.0


def naive_baseline_layer_configs(
    model_id: str,
    weight_bits: int = _DEFAULT_BASELINE_WEIGHT_BITS,
    activation_bits: int = _DEFAULT_BASELINE_ACTIVATION_BITS,
    column_pruning_ratio: float = _DEFAULT_BASELINE_PRUNING_RATIO,
) -> List[LayerBitConfig]:
    """One `LayerBitConfig` per real stage (`nodes.planner.real_stage_names`
    -- never a hardcoded stage list, docs/coding_guidelines.md C), all set to the same
    fixed values: the "didn't search at all, picked an unremarkable
    default" reference point every stage of this model would plausibly get
    from a human who spent zero optimization effort."""
    return [
        LayerBitConfig(
            layer_name=name,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            column_pruning_ratio=column_pruning_ratio,
        )
        for name in real_stage_names(model_id)
    ]


def run_naive_baseline_replicate(
    model_id: str,
    hw_config: HWConfig,
    run_seed: int,
    *,
    dataset_id: Optional[str] = None,
    weight_bits: int = _DEFAULT_BASELINE_WEIGHT_BITS,
    activation_bits: int = _DEFAULT_BASELINE_ACTIVATION_BITS,
    column_pruning_ratio: float = _DEFAULT_BASELINE_PRUNING_RATIO,
    allow_approximate_calibration: bool = True,
) -> Dict[str, Any]:
    """One QAT trial of the fixed naive `layer_configs` at `run_seed` --
    varies only QAT's own training stochasticity (there's no search here
    to vary anything else). Returns the same shape
    `tools.repeat_search.run_one_replicate` returns, so
    `tools.repeat_search.summarize` works on it unmodified: `iteration_count`
    is always 1 (one fixed evaluation, not a multi-iteration search), and
    `is_converged` means what it means for every other candidate this
    project evaluates directly (`tools/batch_warmup.py`'s own convention):
    @verifier's physical IR-drop/noise check, not a --target-* gate (this
    baseline was never searched, so there's no search decision to gate)."""
    layer_configs = naive_baseline_layer_configs(model_id, weight_bits, activation_bits, column_pruning_ratio)
    calibration_factors = (
        bootstrap_approximate_calibration_factors(hw_config)
        if allow_approximate_calibration
        else bootstrap_calibration_factors(hw_config)
    )
    metrics = _evaluate_one_candidate(model_id, hw_config, layer_configs, calibration_factors, dataset_id, run_seed)

    validation_errors = validate_metrics(metrics)
    verifier_data = (metrics.get("verifier") or {}).get("data") or {}
    is_converged = not validation_errors and bool(verifier_data.get("is_converged", False))
    candidate = build_candidate_entry([], metrics, 1, is_converged, has_validation_errors=bool(validation_errors))

    return {
        "run_seed": run_seed,
        "is_converged": is_converged,
        "iteration_count": 1,
        "accuracy": candidate.get("accuracy") if candidate else None,
        "energy_pj": candidate.get("energy_pj") if candidate else None,
        "noc_latency_ms": candidate.get("noc_latency_ms") if candidate else None,
    }


def run_repeated_naive_baseline(
    model_id: str, hw_config: HWConfig, n_seeds: int, base_seed: int = 0, **kwargs: Any
) -> List[Dict[str, Any]]:
    """`n_seeds` naive-baseline replicates at `base_seed`, `base_seed + 1`,
    ... -- the same seed range convention `tools.repeat_search.
    run_repeated_search` uses, so `compare_search_vs_baseline` can hand both
    sides the identical seeds."""
    return [run_naive_baseline_replicate(model_id, hw_config, base_seed + i, **kwargs) for i in range(n_seeds)]


_DEFAULT_GRID_WEIGHT_BITS = (4, 6, 8)
# 0.25/0.5 included deliberately: on resnet18 over a 128-column array the
# 0.0/0.1/0.2 sweep this used to default to never crosses a single crossbar
# tile boundary, so `crossbar_passes` (tools/cim_physics.py) returns the
# identical pass count for all three and the grid's energy column comes out
# equal to the last decimal while accuracy drops -- pruning appears to be
# pure loss because the grid never sampled a ratio that actually removes an
# array. Costs a wider sweep (3x5 points, not 3x3).
_DEFAULT_GRID_PRUNING_RATIOS = (0.0, 0.1, 0.2, 0.25, 0.5)


def grid_search_points(
    weight_bits_grid: List[int] = _DEFAULT_GRID_WEIGHT_BITS,
    pruning_ratio_grid: List[float] = _DEFAULT_GRID_PRUNING_RATIOS,
) -> List[Tuple[int, float]]:
    """Every (weight_bits, column_pruning_ratio) pair in the cartesian
    product -- a systematic, non-adaptive alternative to this project's
    real per-stage NSGA-II search (which explores each stage's own
    independent point) and distinct from tools.run_ablation.py's `random`
    strategy (which samples this same uniform space randomly, not by
    grid)."""
    return [(w, p) for w in weight_bits_grid for p in pruning_ratio_grid]


def run_grid_search_baseline(
    model_id: str,
    hw_config: HWConfig,
    run_seed: int = 0,
    *,
    dataset_id: Optional[str] = None,
    weight_bits_grid: List[int] = _DEFAULT_GRID_WEIGHT_BITS,
    pruning_ratio_grid: List[float] = _DEFAULT_GRID_PRUNING_RATIOS,
    allow_approximate_calibration: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluates every grid point (uniform weight_bits/activation_bits/
    column_pruning_ratio across all stages, same as `run_naive_baseline_
    replicate` -- this just sweeps its (weight_bits, column_pruning_ratio)
    args over the grid) at a *single* `run_seed`, not replicated across
    multiple seeds -- a real cost trade-off, stated plainly: replicating
    every grid point across the same N seeds the baseline/search columns
    use would multiply real-QAT-training cost by the grid size, which for
    a "coarse" grid meant to stay cheap defeats its own purpose. Each
    result dict also carries its own `weight_bits`/`column_pruning_ratio`
    so `best_grid_point` (and any caller) can report *which* point won,
    not just its metrics."""
    results = []
    for weight_bits, pruning_ratio in grid_search_points(weight_bits_grid, pruning_ratio_grid):
        result = run_naive_baseline_replicate(
            model_id,
            hw_config,
            run_seed,
            dataset_id=dataset_id,
            weight_bits=weight_bits,
            activation_bits=weight_bits,
            column_pruning_ratio=pruning_ratio,
            allow_approximate_calibration=allow_approximate_calibration,
        )
        result["weight_bits"] = weight_bits
        result["column_pruning_ratio"] = pruning_ratio
        results.append(result)
    return results


def best_grid_point(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Best converged grid point by accuracy (falling back to every point
    if none converged) -- the same "prefer converged, highest accuracy"
    rule `tools.repeat_search._best_candidate` uses, since a one-shot grid
    sweep has no Pareto-rank/candidate_history machinery of its own to
    lean on here."""
    converged = [r for r in results if r["is_converged"]]
    candidates = converged or results
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["accuracy"] or 0.0, -(r["energy_pj"] or float("inf"))))


def compare_search_vs_baseline(
    model_id: str,
    hw_config: HWConfig,
    n_seeds: int,
    base_seed: int = 0,
    *,
    dataset_id: Optional[str] = None,
    search_strategy: str = "multi_objective",
    target_accuracy: Optional[float] = None,
    target_energy_pj: Optional[float] = None,
    target_latency_ms: Optional[float] = None,
    allow_approximate_calibration: bool = True,
    auto_hitl_max_rounds: int = 5,
    search_budget_iterations: Optional[int] = None,
    weight_bits: int = _DEFAULT_BASELINE_WEIGHT_BITS,
    activation_bits: int = _DEFAULT_BASELINE_ACTIVATION_BITS,
    column_pruning_ratio: float = _DEFAULT_BASELINE_PRUNING_RATIO,
    include_grid: bool = False,
    grid_weight_bits: List[int] = _DEFAULT_GRID_WEIGHT_BITS,
    grid_pruning_ratios: List[float] = _DEFAULT_GRID_PRUNING_RATIOS,
) -> Dict[str, Dict[str, Any]]:
    """Runs the naive baseline and the real search across the identical seed
    range (`base_seed`..`base_seed + n_seeds - 1`) -- any difference in
    outcome is attributable to search vs. no-search, not to a different
    random draw on one side. Returns `{"baseline": {...}, "search": {...}}`,
    each `{"results": [...], "summary": {...}}` (`tools.repeat_search.summarize`'s
    own shape) -- deliberately the same two-key structure
    `tools.run_ablation.run_ablation` uses per strategy, so the two report
    formats stay visually consistent for anyone reading both.

    `include_grid=True` adds a third `"grid"` key: `{"results": [...],
    "best": {...} or None}` from `run_grid_search_baseline`/`best_grid_point`
    at `run_seed=base_seed` -- a single seed, not the full `n_seeds` range
    (see `run_grid_search_baseline`'s docstring for the cost trade-off), so
    it has no `summary`/confidence interval the way the other two arms do."""
    baseline_results = run_repeated_naive_baseline(
        model_id,
        hw_config,
        n_seeds,
        base_seed=base_seed,
        dataset_id=dataset_id,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        column_pruning_ratio=column_pruning_ratio,
        allow_approximate_calibration=allow_approximate_calibration,
    )
    search_results = run_repeated_search(
        model_id,
        hw_config,
        n_seeds,
        base_seed=base_seed,
        target_accuracy=target_accuracy,
        target_energy_pj=target_energy_pj,
        target_latency_ms=target_latency_ms,
        dataset_id=dataset_id,
        search_strategy=search_strategy,
        allow_approximate_calibration=allow_approximate_calibration,
        auto_hitl_max_rounds=auto_hitl_max_rounds,
        search_budget_iterations=search_budget_iterations,
    )
    comparison = {
        "baseline": {"results": baseline_results, "summary": summarize(baseline_results)},
        "search": {"results": search_results, "summary": summarize(search_results)},
    }
    if include_grid:
        grid_results = run_grid_search_baseline(
            model_id,
            hw_config,
            run_seed=base_seed,
            dataset_id=dataset_id,
            weight_bits_grid=grid_weight_bits,
            pruning_ratio_grid=grid_pruning_ratios,
            allow_approximate_calibration=allow_approximate_calibration,
        )
        comparison["grid"] = {"results": grid_results, "best": best_grid_point(grid_results)}
    return comparison


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4g}"


def _metric_str(stats: Optional[Dict[str, Any]]) -> str:
    if stats is None:
        return "-"
    if stats["ci_low"] is None:
        return f"{stats['mean']:.3g} (n={stats['n']}, no CI)"
    return f"{stats['mean']:.3g} [{stats['ci_low']:.3g}, {stats['ci_high']:.3g}]"


def print_comparison_report(comparison: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{'=' * 88}\nBaseline (unsearched) vs. search comparison\n{'=' * 88}")
    header = f"{'':>10} {'converged':>10} {'accuracy (mean, 95% CI)':>30} {'energy_pj (mean, 95% CI)':>32}"
    print(header)
    for label in ("baseline", "search"):
        summary = comparison[label]["summary"]
        n = summary["n_replicates"]
        converged_str = f"{summary['n_converged']}/{n}"
        acc_str = _metric_str(summary.get("accuracy"))
        energy_str = _metric_str(summary.get("energy_pj"))
        print(f"{label:>10} {converged_str:>10} {acc_str:>30} {energy_str:>32}")

    grid = comparison.get("grid")
    if grid is not None:
        best = grid["best"]
        n_grid = len(grid["results"])
        n_grid_converged = sum(1 for r in grid["results"] if r["is_converged"])
        if best is None:
            print(f"{'grid':>10} {f'{n_grid_converged}/{n_grid}':>10} {'-':>30} {'-':>32}")
        else:
            best_str = (
                f"weight_bits={best['weight_bits']} pruning={best['column_pruning_ratio']}"
            )
            print(
                f"{'grid':>10} {f'{n_grid_converged}/{n_grid}':>10} "
                f"{_fmt(best['accuracy']):>30} {_fmt(best['energy_pj']):>32}"
            )
            print(f"{'':>10} best grid point: {best_str}")

    print(
        "\nNote: 'baseline' never searched anything -- fixed weight_bits/activation_bits/"
        "column_pruning_ratio, every stage. 'search' is this project's normal multi-iteration "
        "search. Read this as 'do the two visibly land in different places, and by how much', "
        "backed by each side's own Student's t 95% CI (tools/repeat_search.py) -- not a formal "
        "significance test for 'search beats baseline' as a hypothesis."
        + (
            "\n'grid' (if shown) is a single-seed sweep over uniform (weight_bits, "
            "column_pruning_ratio) pairs, not replicated across seeds like the other two rows -- "
            "its one number is a coarse point estimate, not a distribution."
            if grid is not None
            else ""
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Tessellate's searched result against a fixed, unsearched baseline "
        "config, both evaluated across the same N seeds."
    )
    parser.add_argument("--model-id", default="resnet18")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument(
        "--search-strategy",
        default="multi_objective",
        help="Same meaning as main.py's --search-strategy for the 'search' side.",
    )
    parser.add_argument("--hw-config", required=True, metavar="PATH")
    parser.add_argument("--seeds", type=int, default=5, metavar="N")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--target-energy-pj", type=float, default=None)
    parser.add_argument("--target-latency-ms", type=float, default=None)
    parser.add_argument("--auto-hitl-max-rounds", type=int, default=5)
    parser.add_argument(
        "--search-budget-iterations",
        type=int,
        default=None,
        metavar="N",
        help="Same meaning as main.py's flag, applied to the 'search' arm: keep "
        "searching for N iterations and report the best target-meeting candidate, "
        "rather than stopping at the first one. Set this to make the search arm "
        "comparable against --grid, which always evaluates every point and reports "
        "its best.",
    )
    parser.add_argument("--exact-calibration-only", action="store_true")
    parser.add_argument(
        "--baseline-weight-bits", type=int, default=_DEFAULT_BASELINE_WEIGHT_BITS, metavar="BITS"
    )
    parser.add_argument(
        "--baseline-activation-bits", type=int, default=_DEFAULT_BASELINE_ACTIVATION_BITS, metavar="BITS"
    )
    parser.add_argument(
        "--baseline-pruning-ratio", type=float, default=_DEFAULT_BASELINE_PRUNING_RATIO, metavar="RATIO"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also run a coarse single-seed grid sweep over uniform (weight_bits, "
        "column_pruning_ratio) pairs (see module docstring) and report its best point.",
    )
    parser.add_argument(
        "--grid-weight-bits",
        type=int,
        nargs="+",
        default=list(_DEFAULT_GRID_WEIGHT_BITS),
        metavar="BITS",
        help="Weight-bit values to sweep with --grid.",
    )
    parser.add_argument(
        "--grid-pruning-ratios",
        type=float,
        nargs="+",
        default=list(_DEFAULT_GRID_PRUNING_RATIOS),
        metavar="RATIO",
        help="Column-pruning-ratio values to sweep with --grid.",
    )
    parser.add_argument("--out", default=None, metavar="PATH", help="Write full comparison as JSON here.")
    return parser.parse_args()


def main_cli() -> None:
    args = _parse_args()
    hw_config = main.load_hw_config(args.hw_config)
    comparison = compare_search_vs_baseline(
        args.model_id,
        hw_config,
        args.seeds,
        base_seed=args.base_seed,
        dataset_id=args.dataset_id,
        search_strategy=args.search_strategy,
        target_accuracy=args.target_accuracy,
        target_energy_pj=args.target_energy_pj,
        target_latency_ms=args.target_latency_ms,
        allow_approximate_calibration=not args.exact_calibration_only,
        auto_hitl_max_rounds=args.auto_hitl_max_rounds,
        search_budget_iterations=args.search_budget_iterations,
        weight_bits=args.baseline_weight_bits,
        activation_bits=args.baseline_activation_bits,
        column_pruning_ratio=args.baseline_pruning_ratio,
        include_grid=args.grid,
        grid_weight_bits=args.grid_weight_bits,
        grid_pruning_ratios=args.grid_pruning_ratios,
    )
    print_comparison_report(comparison)
    if args.out:
        Path(args.out).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\n[baseline_compare] wrote {args.out}")


if __name__ == "__main__":
    main_cli()
