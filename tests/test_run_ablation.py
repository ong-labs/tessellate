"""tools/run_ablation.py: strategy comparison built on tools/repeat_search.py."""

from typing import Any, Dict, List

import tools.run_ablation as run_ablation_module
from nodes.planner import SEARCH_STRATEGIES
from tools.run_ablation import run_ablation


def _fake_run_repeated_search(model_id, hw_config, n_seeds, base_seed=0, search_strategy="multi_objective", **kwargs):
    return [
        {
            "run_seed": base_seed + i,
            "search_strategy": search_strategy,
            "is_converged": True,
            "iteration_count": 4,
            "accuracy": 0.5,
            "energy_pj": 100.0,
            "noc_latency_ms": 0.001,
        }
        for i in range(n_seeds)
    ]


def test_run_ablation_covers_every_strategy_by_default(monkeypatch, registered_hw_config):
    monkeypatch.setattr(run_ablation_module, "run_repeated_search", _fake_run_repeated_search)
    ablation = run_ablation("resnet18", registered_hw_config, n_seeds=2)
    assert set(ablation.keys()) == set(SEARCH_STRATEGIES)
    for strategy, data in ablation.items():
        assert data["summary"]["n_replicates"] == 2
        assert all(r["search_strategy"] == strategy for r in data["results"])


def test_run_ablation_respects_strategies_subset(monkeypatch, registered_hw_config):
    monkeypatch.setattr(run_ablation_module, "run_repeated_search", _fake_run_repeated_search)
    ablation = run_ablation("resnet18", registered_hw_config, n_seeds=1, strategies=["random"])
    assert set(ablation.keys()) == {"random"}


def test_run_ablation_uses_same_seeds_across_strategies(monkeypatch, registered_hw_config):
    seen_seed_args: List[Dict[str, Any]] = []

    def _spy(model_id, hw_config, n_seeds, base_seed=0, search_strategy="multi_objective", **kwargs):
        seen_seed_args.append({"strategy": search_strategy, "n_seeds": n_seeds, "base_seed": base_seed})
        return _fake_run_repeated_search(model_id, hw_config, n_seeds, base_seed, search_strategy, **kwargs)

    monkeypatch.setattr(run_ablation_module, "run_repeated_search", _spy)
    run_ablation("resnet18", registered_hw_config, n_seeds=3, base_seed=5)

    # Every strategy must see the identical (n_seeds, base_seed) -- any
    # difference in outcome across strategies must be attributable to the
    # acquisition rule, not to a different random draw per strategy.
    assert {(a["n_seeds"], a["base_seed"]) for a in seen_seed_args} == {(3, 5)}
    assert {a["strategy"] for a in seen_seed_args} == set(SEARCH_STRATEGIES)
