"""Multi-seed repeat runner: N independent replicates of the same
(model_id, hw_config, targets) search, reporting the distribution of the
converged (or best-attempted) candidate's accuracy/energy_pj/noc_latency_ms
across replicates.

Why this exists: a single session's Pareto front can't distinguish "the
search reliably finds a good candidate" from "got lucky once" -- and before
`state.py`'s `run_seed` existed, every session against the same
(model_id, hw_spec_id) explored the *exact same* warm-up/NSGA-II points
anyway (nodes/planner.py's `search_seed_for` was a pure function of
(model_id, hw_spec_id)), so "run it again" wasn't even a real second
sample. Each replicate here gets its own `run_seed` (varies which
candidates the search proposes -- see search_seed_for's docstring -- and
QAT training's own stochasticity via `tools.qat.run_qat_tuning`'s `seed`),
its own throwaway checkpoint DB, and its own throwaway long-term-learning
store (main.py's `DEFAULT_LONG_TERM_STORE_DB`, temporarily overridden) --
replicates must never learn from each other's `candidate_history`, or
they'd stop being independent samples.

Usage:
    python -m tools.repeat_search --model-id resnet18 \\
        --hw-config examples/hw_configs/default_cim_v1_128x128.json \\
        --seeds 5 --target-accuracy 0.6

Reports mean/stdev/min/max plus a Student's t confidence interval across
replicates -- `scipy.stats.t` (already a transitive dependency via pymoo,
no new package needed) rather than a normal-distribution z-interval,
since t accounts for the extra uncertainty of estimating variance from a
handful of samples instead of assuming it's known. Still an approximation
worth stating plainly: it assumes each replicate's metric is roughly
normally distributed around the true mean, which this few samples (a
"handful of replicates", each a real multi-iteration QAT search) can't
actually verify -- the interval is genuinely computed, not decorative, but
wide/unstable at these N is expected and honest, not a bug.
"""

import argparse
import json
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langgraph.checkpoint.sqlite import SqliteSaver
from scipy import stats as scipy_stats

import main
from graph import build_graph
from nodes.planner import SEARCH_STRATEGIES
from schemas.config import HWConfig
from tools.search import best_known_candidate


def run_one_replicate(
    model_id: str,
    hw_config: HWConfig,
    run_seed: int,
    *,
    target_accuracy: Optional[float] = None,
    target_energy_pj: Optional[float] = None,
    target_latency_ms: Optional[float] = None,
    dataset_id: Optional[str] = None,
    search_strategy: str = "multi_objective",
    search_budget_iterations: Optional[int] = None,
    allow_approximate_calibration: bool = True,
    auto_hitl: bool = True,
    auto_hitl_max_rounds: int = 5,
) -> Dict[str, Any]:
    """Runs one full session (via `main.run_session`, unmodified -- this
    reuses the exact same code path a normal `python main.py` invocation
    takes, not a parallel re-implementation) with a throwaway checkpoint DB
    and long-term store, then reads the final state back for its result.
    `auto_hitl` defaults to True here (unlike `main.run_session`'s own
    default) -- N replicates run unattended, so a HITL prompt blocking on
    stdin would hang the whole batch."""
    print(f"\n### repeat_search: run_seed={run_seed} ###")
    tmp_dir = Path(tempfile.mkdtemp(prefix="tessellate_repeat_"))
    thread_id = f"repeat-seed-{run_seed}"
    original_long_term_db = main.DEFAULT_LONG_TERM_STORE_DB
    main.DEFAULT_LONG_TERM_STORE_DB = tmp_dir / "long_term_store.sqlite"
    try:
        with SqliteSaver.from_conn_string(str(tmp_dir / "checkpoint.sqlite")) as checkpointer:
            main.run_session(
                model_id,
                hw_config,
                thread_id,
                checkpointer,
                target_accuracy=target_accuracy,
                target_energy_pj=target_energy_pj,
                target_latency_ms=target_latency_ms,
                allow_approximate_calibration=allow_approximate_calibration,
                auto_hitl=auto_hitl,
                auto_hitl_max_rounds=auto_hitl_max_rounds,
                dataset_id=dataset_id,
                run_seed=run_seed,
                search_strategy=search_strategy,
                search_budget_iterations=search_budget_iterations,
            )
            state = build_graph(checkpointer=checkpointer).get_state({"configurable": {"thread_id": thread_id}}).values
    finally:
        main.DEFAULT_LONG_TERM_STORE_DB = original_long_term_db
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # low_confidence_conclusion (state.py): hitl_node's conclude-with-best
    # path also sets is_converged=True (so evaluator_router still routes to
    # END), but candidate_history[-1] there is just whatever the last,
    # likely-failing search iteration produced -- not a target-meeting
    # answer. real_converged is what best_known_candidate's
    # session_is_converged and this replicate's own reported "is_converged"
    # must both use instead of the raw state field, per both docstrings.
    low_confidence = bool(state.get("low_confidence_conclusion", False))
    real_converged = bool(state.get("is_converged", False)) and not low_confidence
    best = best_known_candidate(state.get("candidate_history", []), real_converged)
    return {
        "run_seed": run_seed,
        "search_strategy": search_strategy,
        "is_converged": real_converged,
        "low_confidence_conclusion": low_confidence,
        "iteration_count": state.get("iteration_count", 0),
        "accuracy": best.get("accuracy") if best else None,
        "energy_pj": best.get("energy_pj") if best else None,
        "noc_latency_ms": best.get("noc_latency_ms") if best else None,
    }


def run_repeated_search(
    model_id: str, hw_config: HWConfig, n_seeds: int, base_seed: int = 0, **kwargs: Any
) -> List[Dict[str, Any]]:
    """`n_seeds` replicates, `run_seed` = `base_seed`, `base_seed + 1`, ...,
    `base_seed + n_seeds - 1` -- sequential, not threaded/multiprocessed:
    each replicate is itself a real multi-iteration QAT search already
    reusing whatever GPU is available, so concurrent replicates would just
    contend for the same device (tools/batch_warmup.py's module docstring
    describes the same concurrent-CUDA hazard)."""
    return [run_one_replicate(model_id, hw_config, base_seed + i, **kwargs) for i in range(n_seeds)]


def _confidence_interval(values: List[float], confidence_level: float) -> Optional[Tuple[float, float]]:
    """Student's t two-sided CI for the mean, `(low, high)` -- `None` below
    2 samples (no variance estimate to build an interval from at all).
    `stats.t.ppf` needs `df = n - 1` degrees of freedom; `margin = t_crit *
    stdev / sqrt(n)` is the standard CI half-width formula."""
    n = len(values)
    if n < 2:
        return None
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0:
        return (mean, mean)  # every replicate agreed exactly -- a real, zero-width interval, not an error
    t_crit = scipy_stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
    margin = t_crit * stdev / (n**0.5)
    return (mean - margin, mean + margin)


def summarize(results: List[Dict[str, Any]], confidence_level: float = 0.95) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_replicates": len(results),
        "n_converged": sum(1 for r in results if r["is_converged"]),
        # Replicates hitl_node's conclude-with-best path ended, not a real
        # target-meeting convergence -- reported separately so it's never
        # silently folded into (or mistaken for) n_converged.
        "n_low_confidence": sum(1 for r in results if r.get("low_confidence_conclusion")),
        "confidence_level": confidence_level,
    }
    for metric in ("accuracy", "energy_pj", "noc_latency_ms", "iteration_count"):
        values = [r[metric] for r in results if r[metric] is not None]
        if not values:
            summary[metric] = None
            continue
        ci = _confidence_interval(values, confidence_level)
        summary[metric] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) >= 2 else None,
            "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None,
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return summary


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4g}"


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print(f"\n=== Multi-seed repeat search: {summary['n_converged']}/{summary['n_replicates']} replicates converged ===")
    if summary.get("n_low_confidence"):
        print(
            f"    ({summary['n_low_confidence']}/{summary['n_replicates']} of the rest gave up and auto-concluded "
            "with their best-known candidate -- see the 'low_conf' column, not counted as converged above)"
        )
    print(f"{'seed':>5} {'converged':>10} {'low_conf':>9} {'iters':>6} {'accuracy':>10} {'energy_pj':>12} {'latency_ms':>12}")
    for r in results:
        print(
            f"{r['run_seed']:>5} {str(r['is_converged']):>10} {str(r.get('low_confidence_conclusion', False)):>9} "
            f"{r['iteration_count']:>6} {_fmt(r['accuracy']):>10} {_fmt(r['energy_pj']):>12} {_fmt(r['noc_latency_ms']):>12}"
        )
    print()
    for metric in ("accuracy", "energy_pj", "noc_latency_ms"):
        stats = summary.get(metric)
        if stats is None:
            print(f"{metric}: no converged/attempted candidate had this value across any replicate")
            continue
        stdev_str = f"{stats['stdev']:.4g}" if stats["stdev"] is not None else "n/a (need >=2 samples)"
        ci_str = (
            f"[{stats['ci_low']:.4g}, {stats['ci_high']:.4g}]"
            if stats["ci_low"] is not None
            else "n/a (need >=2 samples)"
        )
        print(
            f"{metric}: mean={stats['mean']:.4g} stdev={stdev_str} "
            f"{summary['confidence_level']:.0%} CI={ci_str} "
            f"min={stats['min']:.4g} max={stats['max']:.4g} (n={stats['n']})"
        )
    print(
        f"\nNote: {summary['confidence_level']:.0%} CI is a Student's t-interval for the mean "
        "(scipy.stats.t) -- a real computed interval, not decorative, but it assumes each "
        "replicate's metric is roughly normally distributed around the true mean, which this "
        "few independent replicates can't actually verify. Expect it to be wide (sometimes "
        "uselessly so) at small N -- that width is itself honest information about how much "
        "this run count can support, not a formatting default to ignore."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same Tessellate search N times with independent seeds and report variance."
    )
    parser.add_argument("--model-id", default="resnet18", help="Same meaning as main.py's --model-id.")
    parser.add_argument("--dataset-id", default=None, help="Same meaning as main.py's --dataset-id.")
    parser.add_argument(
        "--search-strategy",
        default="multi_objective",
        choices=SEARCH_STRATEGIES,
        help="Same meaning as main.py's --search-strategy -- normally left at the default here too; "
        "tools/run_ablation.py is what actually compares strategies against each other.",
    )
    parser.add_argument("--hw-config", required=True, metavar="PATH", help="Same meaning as main.py's --hw-config.")
    parser.add_argument("--seeds", type=int, default=5, metavar="N", help="Number of independent replicates to run.")
    parser.add_argument("--base-seed", type=int, default=0, help="First run_seed; replicates use base_seed..base_seed+N-1.")
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--target-energy-pj", type=float, default=None)
    parser.add_argument("--target-latency-ms", type=float, default=None)
    parser.add_argument(
        "--auto-hitl-max-rounds",
        type=int,
        default=5,
        help="Per-replicate cap on auto-resolved HITL rounds (main.py's --auto-hitl-max-rounds).",
    )
    parser.add_argument("--exact-calibration-only", action="store_true")
    parser.add_argument("--out", default=None, metavar="PATH", help="Write raw per-seed results + summary as JSON here.")
    return parser.parse_args()


def main_cli() -> None:
    args = _parse_args()
    hw_config = main.load_hw_config(args.hw_config)
    results = run_repeated_search(
        args.model_id,
        hw_config,
        args.seeds,
        base_seed=args.base_seed,
        target_accuracy=args.target_accuracy,
        target_energy_pj=args.target_energy_pj,
        target_latency_ms=args.target_latency_ms,
        dataset_id=args.dataset_id,
        search_strategy=args.search_strategy,
        allow_approximate_calibration=not args.exact_calibration_only,
        auto_hitl_max_rounds=args.auto_hitl_max_rounds,
    )
    summary = summarize(results)
    print_report(results, summary)
    if args.out:
        Path(args.out).write_text(json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8")
        print(f"\n[repeat_search] wrote {args.out}")


if __name__ == "__main__":
    main_cli()
