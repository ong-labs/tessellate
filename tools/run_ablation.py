"""Ablation comparison: runs the identical search (same model_id, hw_config,
targets, seeds) under each of `nodes.planner.SEARCH_STRATEGIES` --
`multi_objective` (this project's real NSGA-II acquisition), `single_objective`
(the accuracy-only IDW-UCB acquisition this project actually shipped before
the NSGA-II rewrite), and `random` (no surrogate at all) -- and reports a
real comparison table across them, using `tools.repeat_search`'s N-seed
replicate machinery so each strategy's numbers are themselves a distribution,
not one lucky run each.

This turns a claim ("multi-objective search is better") this project's own
commit history already found evidence for once (tools/search.py's module
docstring: a real run's single-objective surrogate-phase candidates never
beat the best warm-up candidate's accuracy) into a repeatable, on-demand
measurement instead of a one-off historical anecdote.

Usage:
    python -m tools.run_ablation --model-id resnet18 \\
        --hw-config examples/hw_configs/default_cim_v1_128x128.json \\
        --seeds 5 --target-accuracy 0.6

Each strategy uses the *same* `seeds` (base_seed..base_seed+N-1) -- so any
difference in outcome is attributable to the acquisition rule, not to a
different random draw per strategy.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import main
from nodes.planner import SEARCH_STRATEGIES
from schemas.config import HWConfig
from tools.repeat_search import run_repeated_search, summarize


def run_ablation(
    model_id: str,
    hw_config: HWConfig,
    n_seeds: int,
    base_seed: int = 0,
    strategies: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Returns `{strategy: {"results": [...], "summary": {...}}}`, one entry
    per strategy in `strategies` (default: all of `SEARCH_STRATEGIES`), each
    built from `tools.repeat_search.run_repeated_search`'s same N-replicate
    machinery."""
    out: Dict[str, Dict[str, Any]] = {}
    for strategy in strategies or SEARCH_STRATEGIES:
        print(f"\n{'=' * 70}\nAblation: search_strategy={strategy!r}\n{'=' * 70}")
        results = run_repeated_search(
            model_id, hw_config, n_seeds, base_seed=base_seed, search_strategy=strategy, **kwargs
        )
        out[strategy] = {"results": results, "summary": summarize(results)}
    return out


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4g}"


def print_ablation_report(ablation: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{'=' * 78}\nAblation comparison ({len(ablation)} strategies)\n{'=' * 78}")
    header = f"{'strategy':>18} {'converged':>10} {'accuracy (mean+/-sd)':>22} {'energy_pj (mean+/-sd)':>24} {'iters (mean)':>14}"
    print(header)
    for strategy, data in ablation.items():
        summary = data["summary"]
        n = summary["n_replicates"]
        acc = summary.get("accuracy")
        energy = summary.get("energy_pj")
        iters = summary.get("iteration_count")
        acc_str = "-" if acc is None else f"{acc['mean']:.3g}+/-{_fmt(acc['stdev'])}"
        energy_str = "-" if energy is None else f"{energy['mean']:.4g}+/-{_fmt(energy['stdev'])}"
        iters_str = "-" if iters is None else f"{iters['mean']:.1f}"
        converged_str = f"{summary['n_converged']}/{n}"
        print(f"{strategy:>18} {converged_str:>10} {acc_str:>22} {energy_str:>24} {iters_str:>14}")
    print(
        "\nNote: 'converged' means the replicate's search actually met every configured "
        "--target-*; accuracy/energy_pj are the mean/stdev of each replicate's best (or "
        "converged) candidate, not a formal significance test -- read this as 'do the "
        "strategies land in visibly different places', not as a p-value."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare search_strategy options (multi_objective/single_objective/random) "
        "under the same model_id/hw_config/targets/seeds."
    )
    parser.add_argument("--model-id", default="resnet18")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--hw-config", required=True, metavar="PATH")
    parser.add_argument("--seeds", type=int, default=5, metavar="N")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(SEARCH_STRATEGIES),
        choices=SEARCH_STRATEGIES,
        help="Subset of strategies to compare (default: all of them).",
    )
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--target-energy-pj", type=float, default=None)
    parser.add_argument("--target-latency-ms", type=float, default=None)
    parser.add_argument("--auto-hitl-max-rounds", type=int, default=5)
    parser.add_argument("--exact-calibration-only", action="store_true")
    parser.add_argument("--out", default=None, metavar="PATH", help="Write full per-strategy results as JSON here.")
    return parser.parse_args()


def main_cli() -> None:
    args = _parse_args()
    hw_config = main.load_hw_config(args.hw_config)
    ablation = run_ablation(
        args.model_id,
        hw_config,
        args.seeds,
        base_seed=args.base_seed,
        strategies=args.strategies,
        target_accuracy=args.target_accuracy,
        target_energy_pj=args.target_energy_pj,
        target_latency_ms=args.target_latency_ms,
        dataset_id=args.dataset_id,
        allow_approximate_calibration=not args.exact_calibration_only,
        auto_hitl_max_rounds=args.auto_hitl_max_rounds,
    )
    print_ablation_report(ablation)
    if args.out:
        Path(args.out).write_text(json.dumps(ablation, indent=2), encoding="utf-8")
        print(f"\n[run_ablation] wrote {args.out}")


if __name__ == "__main__":
    main_cli()
