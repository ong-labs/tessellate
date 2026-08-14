"""tools/optuna_compare.py: multi_objective (this project's own search) vs
Optuna's NSGA-II sampler, both running the real (stubbed-backend, see
tests/conftest.py's autouse stub_tuner_qat_backend) tuner/mapper/profiler/
verifier pipeline -- same no-network/no-real-training contract every other
test in this suite relies on."""

from typing import Any, Dict, List

from tools.optuna_compare import (
    compare_multi_objective_vs_optuna,
    run_optuna_replicate,
    run_repeated_optuna,
)


def test_run_optuna_replicate_returns_the_expected_shape(registered_hw_config):
    result = run_optuna_replicate("resnet18", registered_hw_config, run_seed=0, max_trials=3)
    assert set(result.keys()) == {"run_seed", "is_converged", "iteration_count", "accuracy", "energy_pj", "noc_latency_ms"}
    assert result["run_seed"] == 0


def test_run_optuna_replicate_stops_at_the_first_converged_trial(registered_hw_config):
    # No --target-* configured -- check_targets gates nothing, so the very
    # first physically-valid (verifier-passing) trial already satisfies
    # is_converged, the same "no configured targets means this dimension
    # isn't gated" rule nodes/evaluator.py's own is_converged follows. A
    # generous max_trials makes this a real assertion (it stopped early),
    # not just "it happened to run out of budget at 1".
    result = run_optuna_replicate("resnet18", registered_hw_config, run_seed=0, max_trials=20)
    assert result["is_converged"] is True
    assert result["iteration_count"] == 1


def test_run_optuna_replicate_uses_the_full_budget_when_target_is_unreachable(registered_hw_config):
    result = run_optuna_replicate(
        "resnet18", registered_hw_config, run_seed=0, target_accuracy=999.0, max_trials=3
    )
    assert result["is_converged"] is False
    assert result["iteration_count"] == 3
    # Still reports the best attempt across the whole (non-converged)
    # history, never nothing -- same tools.search.best_known_candidate
    # fallback tools.repeat_search.run_one_replicate itself relies on.
    assert result["accuracy"] is not None


def test_run_repeated_optuna_uses_base_seed_offset(registered_hw_config, monkeypatch):
    import tools.optuna_compare as optuna_compare_module

    seen_seeds: List[int] = []
    monkeypatch.setattr(
        optuna_compare_module,
        "run_optuna_replicate",
        lambda model_id, hw_config, run_seed, **kwargs: seen_seeds.append(run_seed)
        or {"run_seed": run_seed, "is_converged": False, "iteration_count": 0, "accuracy": None, "energy_pj": None, "noc_latency_ms": None},
    )
    run_repeated_optuna("resnet18", registered_hw_config, n_seeds=3, base_seed=10)
    assert seen_seeds == [10, 11, 12]


def test_compare_multi_objective_vs_optuna_returns_ablation_shaped_dict(registered_hw_config, monkeypatch):
    import tools.optuna_compare as optuna_compare_module

    canned: Dict[str, Any] = {
        "run_seed": 0,
        "is_converged": True,
        "iteration_count": 1,
        "accuracy": 0.5,
        "energy_pj": 10.0,
        "noc_latency_ms": 0.001,
    }
    monkeypatch.setattr(optuna_compare_module, "run_repeated_search", lambda *a, **k: [dict(canned)])
    monkeypatch.setattr(optuna_compare_module, "run_repeated_optuna", lambda *a, **k: [dict(canned)])

    comparison = compare_multi_objective_vs_optuna("resnet18", registered_hw_config, n_seeds=1)

    assert set(comparison.keys()) == {"multi_objective (this project)", "optuna (NSGA-II, direct)"}
    for entry in comparison.values():
        assert set(entry.keys()) == {"results", "summary"}
        assert entry["summary"]["n_replicates"] == 1
