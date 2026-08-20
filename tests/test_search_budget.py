"""`search_budget_iterations` (state.py): the run's stopping rule.

Unset (the default) is satisficing -- @evaluator stops at the first
candidate meeting every configured target. Set, the search keeps going for
that many iterations and then concludes with the best target-meeting
candidate it saw.

The rule exists because satisficing made this project's own search-vs-grid
comparison unanswerable: `tools/baseline_compare.py`'s grid evaluates every
point and reports its maximum, while the search stopped at its first hit,
so the two arms were never selecting by the same rule. These tests pin the
satisficing path as unchanged (every ordinary run must behave exactly as
before) and cover what the budget path adds.
"""

import pytest

from graph import evaluator_router
from nodes.evaluator import MAX_RETRY_LIMIT, evaluator_node
from tests.test_evaluator_candidate_history import _metrics


@pytest.fixture
def budget_state(state_factory):
    """A candidate that meets a configured 0.7 accuracy target, at iteration
    1 of a run with nothing evaluated yet -- overridden per test."""

    def _make(**overrides):
        defaults = {
            "metrics_store": _metrics(accuracy=0.8),
            "iteration_count": 1,
            "retry_count": 0,
            "candidate_history": [],
            "target_accuracy": 0.7,
        }
        defaults.update(overrides)
        return state_factory(**defaults)

    return _make


def _entry(iteration, accuracy, is_converged):
    return {
        "iteration": iteration,
        "accuracy": accuracy,
        "energy_pj": 100.0,
        "noc_latency_ms": 0.001,
        "avg_weight_bits": 6,
        "avg_column_pruning_ratio": 0.1,
        "layer_configs": [{"layer_name": "stage0", "weight_bits": 6, "activation_bits": 6, "column_pruning_ratio": 0.1}],
        "is_converged": is_converged,
        "pareto_rank": 1,
    }


# --- satisficing (default) stays exactly as it was ---------------------------


def test_without_a_budget_the_first_target_meeting_candidate_still_stops_the_run(budget_state):
    result = evaluator_node(budget_state())
    assert result["is_converged"] is True
    assert "final_candidate" not in result  # nothing to select: the answer is candidate_history[-1]


def test_without_a_budget_a_missed_target_still_takes_the_failure_path(budget_state):
    result = evaluator_node(budget_state(metrics_store=_metrics(accuracy=0.1)))
    assert result["is_converged"] is False
    assert result["retry_count"] == 1
    assert result["failure_history"][0]["reason"].startswith("accuracy ")


# --- the candidate entry's own flag ------------------------------------------


def test_a_candidate_meeting_targets_is_flagged_even_when_the_budget_keeps_searching(budget_state):
    # The entry's is_converged records whether *that candidate* met every
    # target, not whether the run stopped. Conflating them would file every
    # pre-budget hit as a failure and make it unselectable at the end.
    result = evaluator_node(budget_state(search_budget_iterations=5))
    assert result["is_converged"] is False  # the run continues
    assert result["candidate_history"][0]["is_converged"] is True  # the candidate did meet targets


def test_a_candidate_missing_targets_is_not_flagged_under_a_budget(budget_state):
    result = evaluator_node(budget_state(search_budget_iterations=5, metrics_store=_metrics(accuracy=0.1)))
    assert result["candidate_history"][0]["is_converged"] is False


# --- continuing on an unexhausted budget -------------------------------------


def test_meeting_a_target_mid_budget_resets_the_retry_counter_instead_of_stalling(budget_state):
    # Without this, a search that keeps succeeding would climb to
    # MAX_RETRY_LIMIT and fire a HITL interrupt purely for succeeding.
    result = evaluator_node(budget_state(search_budget_iterations=5, retry_count=MAX_RETRY_LIMIT - 1))
    assert result["retry_count"] == 0
    assert result["needs_hitl"] is False
    assert "failure_history" not in result


def test_missing_a_target_mid_budget_still_counts_as_a_failure(budget_state):
    result = evaluator_node(
        budget_state(search_budget_iterations=5, metrics_store=_metrics(accuracy=0.1), retry_count=1)
    )
    assert result["retry_count"] == 2
    assert result["failure_history"][0]["reason"].startswith("accuracy ")


# --- budget exhaustion -------------------------------------------------------


def test_an_exhausted_budget_concludes_with_the_best_target_meeting_candidate(budget_state):
    history = [_entry(1, 0.95, True), _entry(2, 0.72, True), _entry(3, 0.2, False)]
    result = evaluator_node(
        budget_state(search_budget_iterations=4, iteration_count=4, candidate_history=history, metrics_store=_metrics(accuracy=0.75))
    )

    assert result["is_converged"] is True
    assert result["low_confidence_conclusion"] is False
    # 0.95 from iteration 1, not this iteration's 0.75 and not the highest
    # accuracy overall if that one had missed its target.
    assert result["final_candidate"]["iteration"] == 1
    assert result["final_candidate"]["accuracy"] == 0.95


def test_an_exhausted_budget_that_never_met_a_target_concludes_with_low_confidence(budget_state):
    history = [_entry(1, 0.2, False), _entry(2, 0.3, False)]
    result = evaluator_node(
        budget_state(search_budget_iterations=3, iteration_count=3, candidate_history=history, metrics_store=_metrics(accuracy=0.25))
    )

    assert result["is_converged"] is True  # the run stops
    assert result["low_confidence_conclusion"] is True  # but reports honestly
    assert result["final_candidate"]["accuracy"] == 0.3  # best attempt, not a target-meeting answer


def test_a_budget_overshoot_still_concludes(budget_state):
    # iteration_count > budget, not just ==: resuming a session whose budget
    # was lowered, or any off-by-one in the loop, must still terminate.
    result = evaluator_node(
        budget_state(search_budget_iterations=2, iteration_count=7, candidate_history=[_entry(1, 0.9, True)])
    )
    assert result["is_converged"] is True


def test_the_final_candidate_can_be_this_very_iteration(budget_state):
    result = evaluator_node(
        budget_state(search_budget_iterations=2, iteration_count=2, candidate_history=[_entry(1, 0.71, True)], metrics_store=_metrics(accuracy=0.99))
    )
    assert result["final_candidate"]["accuracy"] == 0.99


# --- routing -----------------------------------------------------------------


def test_a_low_confidence_budget_conclusion_skips_the_precise_re_check():
    # Same reasoning as hitl_node's conclude-with-best path: no candidate
    # ever met its targets, so there is no converged verdict for
    # @precision_verifier to confirm and no reason to spend a real
    # NeuroSim/CACTI call re-verifying a known miss.
    assert (
        evaluator_router({"is_converged": True, "low_confidence_conclusion": True})
        == "Concluded (low confidence)"
    )


def test_a_real_budget_conclusion_still_routes_through_the_precise_re_check():
    assert evaluator_router({"is_converged": True, "low_confidence_conclusion": False}) == "Converged (Done)"


def test_satisficing_convergence_is_unaffected_by_the_new_routing_branch():
    assert evaluator_router({"is_converged": True}) == "Converged (Done)"
