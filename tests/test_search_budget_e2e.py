"""A `search_budget_iterations` run must actually reach its budget.

Satisficing never runs the graph loop for long: @evaluator either converges
within a few iterations or stalls into a HITL interrupt at MAX_RETRY_LIMIT,
and each interrupt ends the invocation, so LangGraph's per-invocation
recursion budget resets every time. A budget run deliberately removes both
of those brakes -- a candidate that keeps meeting its targets neither
converges nor accumulates retries -- so the loop runs continuously
for as many iterations as the budget asks for. Verified up to a budget of
40 (~200 supersteps) when this was written; the assertion below runs a
cheaper 12 as a standing guardrail, since a limit hit would surface as a
GraphRecursionError hours into a real multi-seed comparison run.
"""

from graph import build_graph


def test_a_budget_run_reaches_its_budget_without_exhausting_the_recursion_limit(
    state_factory, registered_hw_config
):
    budget = 12  # ~5 supersteps per iteration -- well past LangGraph's default 25
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-budget-e2e"}}
    init_state = state_factory(
        hw_spec_id=registered_hw_config.hw_spec_id,
        target_accuracy=0.0,  # met by every candidate: nothing stops the loop early
        search_budget_iterations=budget,
    )

    result = compiled.invoke(init_state, config=config)

    assert result["iteration_count"] >= budget
    assert result["is_converged"] is True
    assert result["low_confidence_conclusion"] is False
    assert result["final_candidate"] is not None
