"""tools/repeat_search.py: multi-seed replicate runner.

`run_one_replicate`'s isolation test is a real (if tiny) end-to-end
session -- it exercises the same stub_tuner_qat_backend/stub_planner_llm
autouse fixtures every other full-graph test in this suite relies on
(tests/conftest.py), so it's still no-network/no-real-training, same as
the rest of this suite.
"""

import math
from typing import Any, Dict, List

import tools.repeat_search as repeat_search_module
from tools.repeat_search import _confidence_interval, run_one_replicate, summarize
from tools.search import best_known_candidate


def _candidate(iteration: int, accuracy: float, is_converged: bool = False) -> Dict[str, Any]:
    return {
        "iteration": iteration,
        "accuracy": accuracy,
        "energy_pj": 100.0,
        "noc_latency_ms": 0.001,
        "is_converged": is_converged,
        "pareto_rank": 1,
    }


def test_best_candidate_returns_the_last_entry_when_the_session_converged():
    # Only the session-level flag decides this -- nothing gets appended to
    # candidate_history after the entry that made the session converge (the
    # graph stops immediately), so the last entry is the true accepted
    # answer regardless of which earlier entries also happen to carry their
    # own is_converged=True (see the warm-up-poisoning test below for why
    # per-entry flags alone are not trustworthy).
    history = [_candidate(1, 0.9, is_converged=True), _candidate(2, 0.3), _candidate(3, 0.5, is_converged=True)]
    assert best_known_candidate(history, session_is_converged=True) is history[2]


def test_best_candidate_falls_back_to_highest_accuracy_when_session_did_not_converge():
    history = [_candidate(1, 0.4), _candidate(2, 0.9), _candidate(3, 0.1)]
    assert best_known_candidate(history, session_is_converged=False) is history[1]


def test_best_candidate_ignores_per_entry_is_converged_flags_when_session_did_not_converge():
    # Regression test for a real bug (found via an actual end-to-end
    # reproduction, not a hypothetical): tools/batch_warmup.py's warm-up
    # candidates set their own is_converged from @verifier's hw-only
    # physical check alone -- never gated by target_accuracy/target_energy_pj/
    # target_latency_ms at all. A session whose warm-up batch ran (always
    # producing some is_converged=True entries) but then never met a real
    # target for any later search iteration still has those warm-up
    # is_converged=True entries sitting in candidate_history despite the
    # *session* never truly converging. The old logic ("any entry with
    # is_converged=True -> return the last one") took that branch anyway and
    # returned an arbitrary warm-up candidate instead of the true best
    # attempt across the whole history.
    history = [
        _candidate(1, 0.5, is_converged=True),  # warm-up candidate: hw-converged, never target-gated
        _candidate(2, 0.25, is_converged=True),  # warm-up candidate
        _candidate(3, 0.375, is_converged=True),  # warm-up candidate -- old buggy code returned this one
        _candidate(4, 0.9, is_converged=False),  # real search iteration: missed the target, but genuinely best
    ]
    assert best_known_candidate(history, session_is_converged=False) is history[3]


def test_best_candidate_none_for_empty_history():
    assert best_known_candidate([], session_is_converged=True) is None
    assert best_known_candidate([], session_is_converged=False) is None


def test_summarize_computes_mean_stdev_min_max():
    results: List[Dict[str, Any]] = [
        {"run_seed": 0, "is_converged": True, "iteration_count": 5, "accuracy": 0.6, "energy_pj": 100.0, "noc_latency_ms": 0.001},
        {"run_seed": 1, "is_converged": True, "iteration_count": 7, "accuracy": 0.8, "energy_pj": 120.0, "noc_latency_ms": 0.002},
        {"run_seed": 2, "is_converged": False, "iteration_count": 10, "accuracy": None, "energy_pj": None, "noc_latency_ms": None},
    ]
    summary = summarize(results)

    assert summary["n_replicates"] == 3
    assert summary["n_converged"] == 2
    assert summary["accuracy"]["n"] == 2  # the non-converged replicate with accuracy=None is excluded
    assert summary["accuracy"]["mean"] == 0.7
    assert summary["accuracy"]["min"] == 0.6
    assert summary["accuracy"]["max"] == 0.8
    assert summary["accuracy"]["stdev"] is not None


def test_summarize_stdev_is_none_for_a_single_sample():
    results = [{"run_seed": 0, "is_converged": True, "iteration_count": 4, "accuracy": 0.5, "energy_pj": 1.0, "noc_latency_ms": 0.001}]
    summary = summarize(results)
    assert summary["accuracy"]["stdev"] is None


def test_summarize_includes_confidence_interval_bracketing_the_mean():
    results: List[Dict[str, Any]] = [
        {"run_seed": 0, "is_converged": True, "iteration_count": 5, "accuracy": 0.6, "energy_pj": 100.0, "noc_latency_ms": 0.001},
        {"run_seed": 1, "is_converged": True, "iteration_count": 7, "accuracy": 0.8, "energy_pj": 120.0, "noc_latency_ms": 0.002},
        {"run_seed": 2, "is_converged": True, "iteration_count": 6, "accuracy": 0.7, "energy_pj": 110.0, "noc_latency_ms": 0.0015},
    ]
    summary = summarize(results)
    stats = summary["accuracy"]
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]


def test_summarize_confidence_interval_is_none_for_a_single_sample():
    results = [{"run_seed": 0, "is_converged": True, "iteration_count": 4, "accuracy": 0.5, "energy_pj": 1.0, "noc_latency_ms": 0.001}]
    summary = summarize(results)
    assert summary["accuracy"]["ci_low"] is None
    assert summary["accuracy"]["ci_high"] is None


def test_summarize_default_confidence_level_is_95_percent():
    results: List[Dict[str, Any]] = [
        {"run_seed": 0, "is_converged": True, "iteration_count": 5, "accuracy": 0.6, "energy_pj": 100.0, "noc_latency_ms": 0.001},
        {"run_seed": 1, "is_converged": True, "iteration_count": 7, "accuracy": 0.8, "energy_pj": 120.0, "noc_latency_ms": 0.002},
    ]
    summary = summarize(results)
    assert summary["confidence_level"] == 0.95


def test_confidence_interval_is_none_below_two_samples():
    assert _confidence_interval([1.0], 0.95) is None


def test_confidence_interval_is_zero_width_when_every_value_agrees():
    # Every replicate landed on the exact same number -- a real, zero-width
    # interval (deterministic search + deterministic QAT given the same
    # seed range would produce this), not an error case.
    assert _confidence_interval([2.0, 2.0, 2.0], 0.95) == (2.0, 2.0)


def test_confidence_interval_widens_with_higher_confidence_level():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_95 = _confidence_interval(values, 0.95)
    ci_99 = _confidence_interval(values, 0.99)

    width_95 = ci_95[1] - ci_95[0]
    width_99 = ci_99[1] - ci_99[0]
    assert width_99 > width_95  # a higher confidence level demands a wider interval


def test_confidence_interval_matches_scipy_t_formula_directly():
    from scipy import stats as scipy_stats

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = len(values)
    mean = sum(values) / n
    stdev = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    expected_margin = t_crit * stdev / (n**0.5)

    ci_low, ci_high = _confidence_interval(values, 0.95)
    assert math.isclose(ci_low, mean - expected_margin, rel_tol=1e-9)
    assert math.isclose(ci_high, mean + expected_margin, rel_tol=1e-9)


def test_summarize_counts_low_confidence_conclusions_separately_from_converged():
    results: List[Dict[str, Any]] = [
        {"run_seed": 0, "is_converged": True, "low_confidence_conclusion": False, "iteration_count": 5, "accuracy": 0.9, "energy_pj": 100.0, "noc_latency_ms": 0.001},
        {"run_seed": 1, "is_converged": False, "low_confidence_conclusion": True, "iteration_count": 24, "accuracy": 0.68, "energy_pj": 90.0, "noc_latency_ms": 0.001},
    ]
    summary = summarize(results)
    assert summary["n_converged"] == 1
    assert summary["n_low_confidence"] == 1


def test_summarize_metric_is_none_when_no_replicate_has_a_value():
    results = [{"run_seed": 0, "is_converged": False, "iteration_count": 3, "accuracy": None, "energy_pj": None, "noc_latency_ms": None}]
    summary = summarize(results)
    assert summary["accuracy"] is None


def test_run_one_replicate_restores_default_long_term_store_db(registered_hw_config):
    # run_one_replicate temporarily overrides main.DEFAULT_LONG_TERM_STORE_DB
    # (module-level global) so this replicate's learnings don't leak into --
    # or get polluted by -- the real repo's store; it must always restore
    # the original value afterward, success or failure, or every later
    # main.run_session() call in the same process (e.g. a second replicate)
    # would silently keep using this replicate's throwaway path.
    import main as main_module

    original = main_module.DEFAULT_LONG_TERM_STORE_DB
    run_one_replicate("resnet18", registered_hw_config, run_seed=0, auto_hitl_max_rounds=1)
    assert main_module.DEFAULT_LONG_TERM_STORE_DB == original


def test_two_replicates_do_not_share_long_term_learnings(registered_hw_config, monkeypatch):
    # The whole point of a throwaway long-term store per replicate: if
    # replicate 0 converges and its Pareto front leaked into replicate 1's
    # store, replicate 1 would load it ("이미 학습된 후보...LHS warm-up 배치를
    # 건너뜁니다") and skip its own warm-up -- no longer an independent
    # sample. Assert on the printed session-start line instead of internal
    # state: it's the same signal a researcher watching the console sees.
    printed: List[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    run_one_replicate("resnet18", registered_hw_config, run_seed=0, auto_hitl_max_rounds=1)
    run_one_replicate("resnet18", registered_hw_config, run_seed=1, auto_hitl_max_rounds=1)

    assert not any("이미 학습된 후보" in line for line in printed)


def test_run_repeated_search_uses_base_seed_offset(registered_hw_config, monkeypatch):
    seen_seeds: List[int] = []
    monkeypatch.setattr(
        repeat_search_module,
        "run_one_replicate",
        lambda model_id, hw_config, run_seed, **kwargs: seen_seeds.append(run_seed)
        or {"run_seed": run_seed, "is_converged": False, "iteration_count": 0, "accuracy": None, "energy_pj": None, "noc_latency_ms": None},
    )
    repeat_search_module.run_repeated_search("resnet18", registered_hw_config, n_seeds=3, base_seed=10)
    assert seen_seeds == [10, 11, 12]
