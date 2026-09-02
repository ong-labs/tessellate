"""Compares Tessellate's own search acquisition (tools/search.py's
NSGA-II over an IDW surrogate, the "multi_objective" strategy) against
Optuna (a well-established, widely-used general-purpose hyperparameter
optimization library) running the identical real QAT + physics evaluation
pipeline this project already has -- isolates whether this project's
hand-built acquisition strategy earns its keep over an off-the-shelf
multi-objective optimizer on the exact same problem, or whether a generic
tool would do just as well.

Why this exists (roadmap Step 1.5, toward "믿고 자동화해도 되는가"): before
committing to the much larger integration of an external DSE tool
(DNN+NeuroSim's own search framework -- Step 2, see the deferred Tier 3
comparison), this checks a cheaper question first, reusing this project's
own real physics/QAT stack directly so no new config-translation or
simulator-integration cost is paid just to run it.

Real architectural difference worth stating plainly: this project's own
"multi_objective" strategy is two-phase -- real LHS warm-up, then NSGA-II
searching over a *surrogate*'s (cheap IDW-regression) predictions, only
occasionally spending a real QAT trial. Optuna here has no such surrogate
phase: `optuna.samplers.NSGAIISampler` proposes a new point every trial,
and every trial is *always* a real QAT+physics evaluation -- so this
compares "sample-efficient surrogate-guided search" against "direct
real-evaluation genetic search", not two identical algorithms on different
libraries. That is the actual question worth answering: does the
surrogate phase pay for its own complexity in fewer real QAT trials, or
would a well-known optimizer running directly on real trials do as well or
better?

Both sides stop at the *first* candidate meeting every configured
--target-* (mirrors nodes/evaluator.py's own is_converged rule exactly),
so `iteration_count`/trial-count-to-converge is a fair, direct comparison
of real-QAT-trial cost -- neither side is asked to keep improving after
its first satisficing candidate (see README.md's "왜 per-stage 탐색이
그리드 최고점을 놓쳤는가" finding for why holding that constant here too,
rather than optimizing each side to its own different stopping rule, is
the fair comparison).

Usage:
    python -m tools.optuna_compare --model-id resnet18 \\
        --hw-config examples/hw_configs/default_cim_v1_128x128.json \\
        --seeds 3 --target-accuracy 0.7
"""

import argparse
from typing import Any, Dict, List, Optional

import optuna
from optuna.samplers import NSGAIISampler

import main
from nodes.evaluator import build_candidate_entry, validate_metrics
from nodes.planner import real_stage_names, search_bounds
from schemas.config import HWConfig
from schemas.tools import LayerBitConfig
from tools.batch_warmup import _evaluate_one_candidate
from tools.calibration import bootstrap_approximate_calibration_factors, bootstrap_calibration_factors
from tools.repeat_search import run_repeated_search, summarize
from tools.run_ablation import print_ablation_report
from tools.search import best_known_candidate
from tools.targets import check_targets

# This project prints its own per-trial progress lines; Optuna's default
# logger would otherwise interleave a second, redundant progress stream.
optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEFAULT_OPTUNA_MAX_TRIALS = 40


def run_optuna_replicate(
    model_id: str,
    hw_config: HWConfig,
    run_seed: int,
    *,
    target_accuracy: Optional[float] = None,
    target_energy_pj: Optional[float] = None,
    target_latency_ms: Optional[float] = None,
    dataset_id: Optional[str] = None,
    allow_approximate_calibration: bool = True,
    max_trials: int = _DEFAULT_OPTUNA_MAX_TRIALS,
) -> Dict[str, Any]:
    """One Optuna study: up to `max_trials` real QAT+physics evaluations
    (a hard cap standing in for the real-cost budget `auto_hitl_max_rounds`
    bounds on the multi_objective side -- the default comfortably exceeds
    the ~24 iterations this project's own search needed to converge or
    give up at its default hw_config/target combination, see README.md,
    so Optuna is never starved of budget relative to the comparison).
    Stops at the first candidate meeting every configured target (the same
    rule nodes/evaluator.py's `is_converged` uses). Returns the same shape
    `tools.repeat_search.run_one_replicate` does, so
    `tools.repeat_search.summarize`/`print_report` work on it unmodified."""
    stage_names = real_stage_names(model_id)
    bit_bounds, pruning_bounds = search_bounds(hw_config.hw_spec_id)
    calibration_factors = (
        bootstrap_approximate_calibration_factors(hw_config)
        if allow_approximate_calibration
        else bootstrap_calibration_factors(hw_config)
    )

    history: List[Dict[str, Any]] = []
    converged: Dict[str, Any] = {"candidate": None, "trial_number": None}

    def objective(trial: "optuna.Trial") -> tuple:
        # Same per-stage (weight_bits, column_pruning_ratio) construction
        # nodes/planner.py's points_to_stage_configs uses -- activation_bits
        # mirrors weight_bits, never a separately-searched dimension, so
        # Optuna explores the identical search space this project's own
        # search does, not a differently-shaped one.
        layer_configs = []
        for name in stage_names:
            bits = trial.suggest_int(f"{name}__weight_bits", int(round(bit_bounds[0])), int(round(bit_bounds[1])))
            pruning = trial.suggest_float(f"{name}__pruning", pruning_bounds[0], pruning_bounds[1])
            layer_configs.append(
                LayerBitConfig(layer_name=name, weight_bits=bits, activation_bits=bits, column_pruning_ratio=round(pruning, 4))
            )

        metrics = _evaluate_one_candidate(model_id, hw_config, layer_configs, calibration_factors, dataset_id, run_seed)
        validation_errors = validate_metrics(metrics)
        tuner_data = (metrics.get("tuner") or {}).get("data") or {}
        verifier_data = (metrics.get("verifier") or {}).get("data") or {}
        hw_verified = not validation_errors and bool(verifier_data.get("is_converged", False))

        target_errors: List[str] = []
        if not validation_errors:
            target_errors = check_targets(
                tuner_data.get("accuracy"),
                verifier_data.get("energy_pj"),
                verifier_data.get("noc_latency_ms"),
                target_accuracy,
                target_energy_pj,
                target_latency_ms,
            )
        is_converged = hw_verified and not target_errors

        candidate = build_candidate_entry(history, metrics, trial.number + 1, is_converged, has_validation_errors=bool(validation_errors))
        print(
            f"[optuna trial {trial.number + 1}] "
            + (f"accuracy={candidate['accuracy']:.4g} energy_pj={candidate['energy_pj']:.6g}" if candidate else "invalid/incomplete")
        )
        if candidate is None:
            # Schema-invalid or an upstream tool failed this trial -- tell
            # Optuna to discard it rather than feed it a fabricated
            # objective value, the same "incomplete data is never silently
            # fine" rule build_candidate_entry's other callers already
            # follow (nodes/evaluator.py, tools/batch_warmup.py).
            raise optuna.TrialPruned()

        history.append(candidate)
        if is_converged and converged["candidate"] is None:
            converged["candidate"] = candidate
            converged["trial_number"] = trial.number + 1

        return candidate["accuracy"], candidate["energy_pj"], candidate["noc_latency_ms"]

    def _stop_once_converged(study: "optuna.Study", _trial: "optuna.trial.FrozenTrial") -> None:
        if converged["candidate"] is not None:
            study.stop()

    study = optuna.create_study(directions=["maximize", "minimize", "minimize"], sampler=NSGAIISampler(seed=run_seed))
    study.optimize(objective, n_trials=max_trials, callbacks=[_stop_once_converged])

    best = converged["candidate"] or best_known_candidate(history, session_is_converged=False)
    return {
        "run_seed": run_seed,
        "is_converged": converged["candidate"] is not None,
        "iteration_count": converged["trial_number"] or len(history),
        "accuracy": best.get("accuracy") if best else None,
        "energy_pj": best.get("energy_pj") if best else None,
        "noc_latency_ms": best.get("noc_latency_ms") if best else None,
    }


def run_repeated_optuna(
    model_id: str, hw_config: HWConfig, n_seeds: int, base_seed: int = 0, **kwargs: Any
) -> List[Dict[str, Any]]:
    """Same `base_seed`..`base_seed + n_seeds - 1` convention
    `tools.repeat_search.run_repeated_search` uses, so both sides of the
    comparison see identical seeds."""
    return [run_optuna_replicate(model_id, hw_config, base_seed + i, **kwargs) for i in range(n_seeds)]


def compare_multi_objective_vs_optuna(
    model_id: str,
    hw_config: HWConfig,
    n_seeds: int,
    base_seed: int = 0,
    *,
    dataset_id: Optional[str] = None,
    target_accuracy: Optional[float] = None,
    target_energy_pj: Optional[float] = None,
    target_latency_ms: Optional[float] = None,
    allow_approximate_calibration: bool = True,
    auto_hitl_max_rounds: int = 5,
    optuna_max_trials: int = _DEFAULT_OPTUNA_MAX_TRIALS,
) -> Dict[str, Dict[str, Any]]:
    """Runs this project's own `multi_objective` search (via
    `tools.repeat_search.run_repeated_search`, the real graph/HITL path
    unmodified) and Optuna (`run_repeated_optuna`) across the identical
    seed range, both gated by the identical `--target-*` stopping rule.
    Returns the same `{label: {"results": [...], "summary": {...}}}` shape
    `tools.run_ablation.run_ablation` does, so its `print_ablation_report`
    renders both without a separate formatter."""
    mo_results = run_repeated_search(
        model_id,
        hw_config,
        n_seeds,
        base_seed=base_seed,
        target_accuracy=target_accuracy,
        target_energy_pj=target_energy_pj,
        target_latency_ms=target_latency_ms,
        dataset_id=dataset_id,
        search_strategy="multi_objective",
        allow_approximate_calibration=allow_approximate_calibration,
        auto_hitl_max_rounds=auto_hitl_max_rounds,
    )
    optuna_results = run_repeated_optuna(
        model_id,
        hw_config,
        n_seeds,
        base_seed=base_seed,
        target_accuracy=target_accuracy,
        target_energy_pj=target_energy_pj,
        target_latency_ms=target_latency_ms,
        dataset_id=dataset_id,
        allow_approximate_calibration=allow_approximate_calibration,
        max_trials=optuna_max_trials,
    )
    return {
        "multi_objective (this project)": {"results": mo_results, "summary": summarize(mo_results)},
        "optuna (NSGA-II, direct)": {"results": optuna_results, "summary": summarize(optuna_results)},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare this project's own multi_objective search against Optuna's NSGA-II sampler "
        "running the identical real QAT+physics evaluation pipeline."
    )
    parser.add_argument("--model-id", default="resnet18")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--hw-config", required=True, metavar="PATH")
    parser.add_argument("--seeds", type=int, default=5, metavar="N")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--target-energy-pj", type=float, default=None)
    parser.add_argument("--target-latency-ms", type=float, default=None)
    parser.add_argument("--auto-hitl-max-rounds", type=int, default=5)
    parser.add_argument("--optuna-max-trials", type=int, default=_DEFAULT_OPTUNA_MAX_TRIALS)
    parser.add_argument("--exact-calibration-only", action="store_true")
    parser.add_argument("--out", default=None, metavar="PATH", help="Write full comparison as JSON here.")
    return parser.parse_args()


def main_cli() -> None:
    import json
    from pathlib import Path

    args = _parse_args()
    hw_config = main.load_hw_config(args.hw_config)
    comparison = compare_multi_objective_vs_optuna(
        args.model_id,
        hw_config,
        args.seeds,
        base_seed=args.base_seed,
        dataset_id=args.dataset_id,
        target_accuracy=args.target_accuracy,
        target_energy_pj=args.target_energy_pj,
        target_latency_ms=args.target_latency_ms,
        allow_approximate_calibration=not args.exact_calibration_only,
        auto_hitl_max_rounds=args.auto_hitl_max_rounds,
        optuna_max_trials=args.optuna_max_trials,
    )
    print_ablation_report(comparison)
    if args.out:
        Path(args.out).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\n[optuna_compare] wrote {args.out}")


if __name__ == "__main__":
    main_cli()
