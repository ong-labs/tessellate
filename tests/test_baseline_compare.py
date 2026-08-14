"""tools/baseline_compare.py: fixed unsearched baseline vs. this project's
real search, both across the same N seeds.

`run_naive_baseline_replicate`'s test is a real (if tiny) end-to-end
evaluation -- it exercises the same stub_tuner_qat_backend/stub_planner_llm
autouse fixtures every other full-graph test in this suite relies on
(tests/conftest.py), so it's still no-network/no-real-training, same as
tests/test_repeat_search.py's own run_one_replicate test.
"""

from typing import Any, Dict, List

import tools.baseline_compare as baseline_compare_module
from nodes.planner import real_stage_names
from tools.baseline_compare import (
    best_grid_point,
    compare_search_vs_baseline,
    grid_search_points,
    naive_baseline_layer_configs,
    run_grid_search_baseline,
    run_naive_baseline_replicate,
    run_repeated_naive_baseline,
)


def test_naive_baseline_layer_configs_covers_every_real_stage():
    configs = naive_baseline_layer_configs("resnet18")
    assert [c.layer_name for c in configs] == real_stage_names("resnet18")


def test_naive_baseline_layer_configs_uses_fixed_uniform_values():
    configs = naive_baseline_layer_configs("resnet18", weight_bits=4, activation_bits=6, column_pruning_ratio=0.2)
    assert all(c.weight_bits == 4 for c in configs)
    assert all(c.activation_bits == 6 for c in configs)
    assert all(c.column_pruning_ratio == 0.2 for c in configs)


def test_naive_baseline_layer_configs_defaults_to_no_pruning_8_bit():
    configs = naive_baseline_layer_configs("resnet18")
    assert all(c.weight_bits == 8 for c in configs)
    assert all(c.activation_bits == 8 for c in configs)
    assert all(c.column_pruning_ratio == 0.0 for c in configs)


def test_run_naive_baseline_replicate_returns_repeat_search_shaped_result(registered_hw_config):
    result = run_naive_baseline_replicate("resnet18", registered_hw_config, run_seed=0)
    assert set(result.keys()) == {"run_seed", "is_converged", "iteration_count", "accuracy", "energy_pj", "noc_latency_ms"}
    assert result["run_seed"] == 0
    assert result["iteration_count"] == 1


def test_run_naive_baseline_replicate_produces_real_metrics_for_a_converging_hw_config(registered_hw_config):
    result = run_naive_baseline_replicate("resnet18", registered_hw_config, run_seed=0)
    assert result["is_converged"] is True
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["energy_pj"] > 0
    assert result["noc_latency_ms"] >= 0


def test_run_repeated_naive_baseline_uses_base_seed_offset(registered_hw_config, monkeypatch):
    seen_seeds: List[int] = []

    def _fake_replicate(model_id, hw_config, run_seed, **kwargs):
        seen_seeds.append(run_seed)
        return {"run_seed": run_seed, "is_converged": True, "iteration_count": 1, "accuracy": 0.5, "energy_pj": 10.0, "noc_latency_ms": 0.001}

    monkeypatch.setattr(baseline_compare_module, "run_naive_baseline_replicate", _fake_replicate)
    run_repeated_naive_baseline("resnet18", registered_hw_config, n_seeds=3, base_seed=10)
    assert seen_seeds == [10, 11, 12]


def _fake_run_repeated_search(model_id, hw_config, n_seeds, base_seed=0, **kwargs):
    return [
        {
            "run_seed": base_seed + i,
            "is_converged": True,
            "iteration_count": 4,
            "accuracy": 0.6,
            "energy_pj": 100.0,
            "noc_latency_ms": 0.001,
        }
        for i in range(n_seeds)
    ]


def _fake_run_repeated_naive_baseline(model_id, hw_config, n_seeds, base_seed=0, **kwargs):
    return [
        {
            "run_seed": base_seed + i,
            "is_converged": True,
            "iteration_count": 1,
            "accuracy": 0.4,
            "energy_pj": 150.0,
            "noc_latency_ms": 0.002,
        }
        for i in range(n_seeds)
    ]


def test_compare_search_vs_baseline_uses_the_same_seed_range_for_both_sides(registered_hw_config, monkeypatch):
    seen_seed_args: List[Dict[str, Any]] = []

    def _spy_search(model_id, hw_config, n_seeds, base_seed=0, **kwargs):
        seen_seed_args.append({"side": "search", "n_seeds": n_seeds, "base_seed": base_seed})
        return _fake_run_repeated_search(model_id, hw_config, n_seeds, base_seed, **kwargs)

    def _spy_baseline(model_id, hw_config, n_seeds, base_seed=0, **kwargs):
        seen_seed_args.append({"side": "baseline", "n_seeds": n_seeds, "base_seed": base_seed})
        return _fake_run_repeated_naive_baseline(model_id, hw_config, n_seeds, base_seed, **kwargs)

    monkeypatch.setattr(baseline_compare_module, "run_repeated_search", _spy_search)
    monkeypatch.setattr(baseline_compare_module, "run_repeated_naive_baseline", _spy_baseline)

    compare_search_vs_baseline("resnet18", registered_hw_config, n_seeds=3, base_seed=7)

    assert {(a["n_seeds"], a["base_seed"]) for a in seen_seed_args} == {(3, 7)}
    assert {a["side"] for a in seen_seed_args} == {"search", "baseline"}


def test_compare_search_vs_baseline_returns_both_results_and_summaries(registered_hw_config, monkeypatch):
    monkeypatch.setattr(baseline_compare_module, "run_repeated_search", _fake_run_repeated_search)
    monkeypatch.setattr(baseline_compare_module, "run_repeated_naive_baseline", _fake_run_repeated_naive_baseline)

    comparison = compare_search_vs_baseline("resnet18", registered_hw_config, n_seeds=2)

    assert set(comparison.keys()) == {"baseline", "search"}
    for side in ("baseline", "search"):
        assert "results" in comparison[side]
        assert "summary" in comparison[side]
        assert comparison[side]["summary"]["n_replicates"] == 2


# --- Grid search baseline (tier 2) -----------------------------------------


def test_grid_search_points_is_the_cartesian_product():
    points = grid_search_points(weight_bits_grid=[4, 8], pruning_ratio_grid=[0.0, 0.1])
    assert set(points) == {(4, 0.0), (4, 0.1), (8, 0.0), (8, 0.1)}
    assert len(points) == 4


def test_run_grid_search_baseline_evaluates_every_grid_point_once(registered_hw_config, monkeypatch):
    seen_points: List[Any] = []

    def _fake_replicate(model_id, hw_config, run_seed, *, weight_bits, activation_bits, column_pruning_ratio, **kwargs):
        seen_points.append((weight_bits, column_pruning_ratio))
        return {"run_seed": run_seed, "is_converged": True, "iteration_count": 1, "accuracy": 0.5, "energy_pj": 10.0, "noc_latency_ms": 0.001}

    monkeypatch.setattr(baseline_compare_module, "run_naive_baseline_replicate", _fake_replicate)
    results = run_grid_search_baseline(
        "resnet18", registered_hw_config, run_seed=3, weight_bits_grid=[4, 8], pruning_ratio_grid=[0.0, 0.2]
    )

    assert set(seen_points) == {(4, 0.0), (4, 0.2), (8, 0.0), (8, 0.2)}
    assert len(results) == 4
    assert all(r["weight_bits"] in (4, 8) and r["column_pruning_ratio"] in (0.0, 0.2) for r in results)


def test_run_grid_search_baseline_uses_a_single_seed_for_every_point(registered_hw_config, monkeypatch):
    seen_seeds: List[int] = []

    def _fake_replicate(model_id, hw_config, run_seed, **kwargs):
        seen_seeds.append(run_seed)
        return {"run_seed": run_seed, "is_converged": True, "iteration_count": 1, "accuracy": 0.5, "energy_pj": 10.0, "noc_latency_ms": 0.001}

    monkeypatch.setattr(baseline_compare_module, "run_naive_baseline_replicate", _fake_replicate)
    run_grid_search_baseline("resnet18", registered_hw_config, run_seed=9, weight_bits_grid=[4, 8], pruning_ratio_grid=[0.0, 0.1, 0.2])

    assert seen_seeds == [9] * 6  # every one of the 2x3 grid points, same seed


def test_best_grid_point_prefers_converged_then_highest_accuracy():
    results = [
        {"weight_bits": 4, "column_pruning_ratio": 0.0, "is_converged": False, "accuracy": 0.9, "energy_pj": 10.0},
        {"weight_bits": 6, "column_pruning_ratio": 0.1, "is_converged": True, "accuracy": 0.5, "energy_pj": 20.0},
        {"weight_bits": 8, "column_pruning_ratio": 0.2, "is_converged": True, "accuracy": 0.7, "energy_pj": 15.0},
    ]
    best = best_grid_point(results)
    assert best["weight_bits"] == 8 and best["column_pruning_ratio"] == 0.2


def test_best_grid_point_falls_back_to_all_results_when_none_converged():
    results = [
        {"weight_bits": 4, "column_pruning_ratio": 0.0, "is_converged": False, "accuracy": 0.3, "energy_pj": 10.0},
        {"weight_bits": 6, "column_pruning_ratio": 0.1, "is_converged": False, "accuracy": 0.6, "energy_pj": 20.0},
    ]
    best = best_grid_point(results)
    assert best["weight_bits"] == 6


def test_best_grid_point_is_none_for_empty_results():
    assert best_grid_point([]) is None


def test_run_grid_search_baseline_produces_real_metrics(registered_hw_config):
    # Real (stub-QAT-backed) evaluation, small grid to keep this test fast.
    results = run_grid_search_baseline(
        "resnet18", registered_hw_config, run_seed=0, weight_bits_grid=[8], pruning_ratio_grid=[0.0]
    )
    assert len(results) == 1
    assert results[0]["accuracy"] is not None
    assert results[0]["weight_bits"] == 8
    assert results[0]["column_pruning_ratio"] == 0.0


def test_compare_search_vs_baseline_includes_grid_when_requested(registered_hw_config, monkeypatch):
    monkeypatch.setattr(baseline_compare_module, "run_repeated_search", _fake_run_repeated_search)
    monkeypatch.setattr(baseline_compare_module, "run_repeated_naive_baseline", _fake_run_repeated_naive_baseline)

    def _fake_grid(model_id, hw_config, run_seed=0, **kwargs):
        return [{"run_seed": run_seed, "is_converged": True, "accuracy": 0.5, "energy_pj": 10.0, "noc_latency_ms": 0.001, "weight_bits": 8, "column_pruning_ratio": 0.0}]

    monkeypatch.setattr(baseline_compare_module, "run_grid_search_baseline", _fake_grid)

    comparison = compare_search_vs_baseline("resnet18", registered_hw_config, n_seeds=2, include_grid=True)
    assert "grid" in comparison
    assert comparison["grid"]["best"]["weight_bits"] == 8


def test_compare_search_vs_baseline_omits_grid_by_default(registered_hw_config, monkeypatch):
    monkeypatch.setattr(baseline_compare_module, "run_repeated_search", _fake_run_repeated_search)
    monkeypatch.setattr(baseline_compare_module, "run_repeated_naive_baseline", _fake_run_repeated_naive_baseline)

    comparison = compare_search_vs_baseline("resnet18", registered_hw_config, n_seeds=2)
    assert "grid" not in comparison
