"""[checklist item 1] hitl_human_approval must be the ONLY pause mechanism.

`workflow.compile()` in graph.py must never register static
interrupt_before/interrupt_after nodes; pausing happens exclusively via the
dynamic `interrupt()` call inside hitl_node, and resuming happens exclusively
via `Command(resume=...)` -- never `graph.update_state()` (docs/coding_guidelines.md 5.B).
"""

import inspect

from langgraph.types import Command

import graph as graph_module
from graph import build_graph
from nodes.evaluator import MAX_RETRY_LIMIT


def test_compile_call_passes_no_interrupt_before_or_after():
    """Source-level guardrail directly on the workflow.compile(...) call."""
    source = inspect.getsource(graph_module.build_graph)
    compile_call = source[source.index("workflow.compile("):]
    assert "interrupt_before" not in compile_call
    assert "interrupt_after" not in compile_call


def test_compiled_graph_has_no_static_interrupts_registered():
    """Runtime guardrail: the compiled graph itself was never configured
    with static interrupt_before/interrupt_after node lists."""
    compiled = build_graph()
    assert compiled.interrupt_before_nodes == []
    assert compiled.interrupt_after_nodes == []


def test_dynamic_interrupt_pauses_at_hitl_and_resumes_with_command(state_factory, registered_bad_hw_config):
    """Full graph integration test: a never-converging HWConfig must drive
    the graph to pause at hitl_human_approval via dynamic interrupt() after
    MAX_RETRY_LIMIT failed attempts, and Command(resume=...) must safely
    continue execution with the sanitization contract (checklist item 3)
    applied on the way back through @planner.
    """
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-hitl-thread"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    result = compiled.invoke(init_state, config=config)

    assert "__interrupt__" in result, "graph must pause via dynamic interrupt(), not run to completion"
    assert result["needs_hitl"] is True
    assert result["retry_count"] == MAX_RETRY_LIMIT
    assert result["is_converged"] is False

    interrupt_obj = result["__interrupt__"][0]
    assert "reason" in interrupt_obj.value
    assert interrupt_obj.value["retry_count"] == MAX_RETRY_LIMIT
    # main.py's suggest_override_bounds() needs hw_spec_id + targets to
    # size its suggestion against the actual miss and this hw_spec_id's
    # real bit ceiling -- must reach the interrupt payload, not just live
    # in state and never get handed to whoever resumes the session.
    assert interrupt_obj.value["hw_spec_id"] == registered_bad_hw_config.hw_spec_id
    assert "target_accuracy" in interrupt_obj.value

    # Resume and inspect the very next @planner update -- this is where the
    # sanitization contract (checklist item 3) must fire, right after
    # hitl_node's interrupt() return value is consumed. The bad HW spec
    # never converges, so a plain .invoke() here would keep looping through
    # another full retry cycle and land back on a *second* interrupt rather
    # than returning right after @planner -- so we inspect the update
    # stream instead of only the final return value.
    resumed_chunks = list(
        compiled.stream(
            Command(resume={"new_bounds": {"note": "researcher intervened"}}),
            config=config,
            stream_mode="updates",
        )
    )
    first_planner_update = next(chunk["planner"] for chunk in resumed_chunks if "planner" in chunk)

    assert first_planner_update["human_overrides"] == {}
    assert first_planner_update["needs_hitl"] is False
    assert first_planner_update["retry_count"] == 0
    assert first_planner_update["iteration_count"] == MAX_RETRY_LIMIT + 1

    # Still-bad HW correctly drives the graph through another full retry
    # cycle and back to a second dynamic interrupt, rather than looping
    # forever or silently giving up (regression coverage for the
    # unconditional-retry_count-reset bug fixed in nodes/planner.py).
    assert any("__interrupt__" in chunk for chunk in resumed_chunks)


def test_conclude_with_best_resume_routes_straight_to_end(state_factory, registered_bad_hw_config):
    """main.py's auto-hitl giving up after its round cap (or prompt_for_override's
    choice [3]) resumes with {"conclude_with_best": True} -- hitl_node must
    set is_converged/low_confidence_conclusion and graph.py's hitl_router
    must route straight to END, never back to planner (which would just
    retry against the same never-converging HW spec forever)."""
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-conclude-thread"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    first = compiled.invoke(init_state, config=config)
    assert "__interrupt__" in first  # same first pause as the always-fails case above

    result = compiled.invoke(Command(resume={"conclude_with_best": True}), config=config)

    assert "__interrupt__" not in result, "conclude_with_best must reach END, not loop back through planner"
    assert result["is_converged"] is True
    assert result["low_confidence_conclusion"] is True
    assert result["needs_hitl"] is False


def test_good_hw_config_converges_without_ever_interrupting(state_factory, registered_hw_config):
    """Control case: a converging HWConfig must reach END without ever
    pausing, proving interrupts are conditional (needs_hitl-driven), not
    unconditional/static.
    """
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-converge-thread"}}
    init_state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)

    result = compiled.invoke(init_state, config=config)

    assert "__interrupt__" not in result
    assert result["is_converged"] is True


def test_resuming_with_empty_bounds_restores_the_full_retry_budget(state_factory, registered_bad_hw_config):
    """Regression (nodes/hitl.py): `main.py`'s override prompt explicitly
    offers "빈칸=변경 없이 재시도", which resumes with `{"new_bounds": {}}`.
    @planner's sanitization only fires when there *is* an override to
    consume, so `retry_count` stayed at MAX_RETRY_LIMIT and @evaluator
    re-fired HITL on every single following iteration -- the researcher was
    prompted again after one iteration instead of after another full retry
    budget. hitl_node now resets the counter on both resume shapes.
    """
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-empty-bounds-thread"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    first = compiled.invoke(init_state, config=config)
    assert first["retry_count"] == MAX_RETRY_LIMIT

    resumed_chunks = list(
        compiled.stream(Command(resume={"new_bounds": {}}), config=config, stream_mode="updates")
    )

    evaluator_updates = [chunk["evaluator"] for chunk in resumed_chunks if "evaluator" in chunk]
    # The very next evaluation starts a fresh budget (retry 1 of
    # MAX_RETRY_LIMIT), instead of continuing at MAX_RETRY_LIMIT + 1 and
    # immediately demanding the researcher again.
    assert evaluator_updates[0]["retry_count"] == 1
    assert evaluator_updates[0]["needs_hitl"] is False
    # And the still-bad HW does eventually pause again -- after a full
    # budget, not after one iteration.
    assert len(evaluator_updates) == MAX_RETRY_LIMIT
    assert evaluator_updates[-1]["needs_hitl"] is True
