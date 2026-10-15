"""hitl_human_approval node.

Uses LangGraph's v0.2+ dynamic `interrupt()` standard (docs/coding_guidelines.md B):
no `interrupt_before` in `compile()`, and no `graph.update_state()` call
here that would duplicate the state update `interrupt()`'s return value
already provides on resume (`Command(resume=...)`).

Two resume shapes: `{"new_bounds": {...}}` (routes back to @planner with
narrowed bounds, the original/only behavior) and `{"conclude_with_best":
True}` (routes straight to `END` instead -- for `main.py`'s auto-hitl
giving up after its round cap, rather than leaving the session paused
forever with no real path to a final state). `graph.py`'s `hitl_router`
reads `low_confidence_conclusion` to pick the edge.

Both resume paths reset `retry_count`, not just the one carrying new
bounds. `nodes/planner.py`'s own sanitization only fires when there *is* an
override to consume (deliberately -- resetting it on every planner call
would erase @evaluator's counter and loop forever), so resuming with empty
`new_bounds` -- which `main.py`'s `_prompt_manual_override` explicitly
offers as "빈칸=변경 없이 재시도" -- used to leave `retry_count` at
MAX_RETRY_LIMIT: @evaluator then re-fired HITL on *every* following
iteration instead of granting another full retry budget (reproduced: first
pause at iteration 3, then a pause at 4, 5, 6, ...). Resetting here is safe
in a way resetting in @planner is not: this node only runs when HITL
actually fired, so the counter restarts once per researcher decision, never
once per iteration.

The conclude path deliberately does not *select* a candidate itself: it
only records `low_confidence_conclusion=True`, and each reporting consumer
(`main.py`'s `print_final_state`, `tools/dashboard.py`) applies
`tools.search.best_known_candidate` to the existing `candidate_history`.
Keeping selection out of this node means there is exactly one selection
rule to keep correct, and no stale copy of a "chosen" candidate on state
that a later iteration could contradict.
"""

from typing import Any, Dict

from langgraph.types import interrupt

from state import TessellateState


def hitl_node(state: TessellateState) -> Dict[str, Any]:
    human_input = interrupt(
        {
            "reason": "Convergence stalled -- researcher review requested",
            "iteration_count": state.get("iteration_count", 0),
            "retry_count": state.get("retry_count", 0),
            "failure_history": state.get("failure_history", []),
            # Non-null only when this exact iteration's @planner LLM call
            # flagged something (state.py's planner_flagged_anomaly) -- this
            # may be why HITL fired now instead of at MAX_RETRY_LIMIT.
            "anomaly_note": state.get("planner_flagged_anomaly"),
            # Every bound a researcher has already approved this run
            # (state.py, nodes/planner.py merges each newly-approved
            # human_overrides into this) -- main.py's suggest_override_bounds()
            # needs this so a *later* HITL round's suggestion narrows from
            # what's already active, not from the hw's raw ceiling every
            # time (which could suggest something looser than an override
            # already approved, silently undoing it).
            "search_bound_overrides": state.get("search_bound_overrides", {}),
            "latest_metrics": state.get("metrics_store", {}),
            # hw_spec_id + targets: main.py's suggest_override_bounds() needs
            # these to size its suggestion against how far off target the
            # candidate actually is, capped at this hw_spec's real ADC/DAC
            # bit ceiling -- neither was in this payload before, so that
            # suggestion was a fixed step regardless of how big the miss was.
            "hw_spec_id": state.get("hw_spec_id"),
            "target_accuracy": state.get("target_accuracy"),
            "target_energy_pj": state.get("target_energy_pj"),
            "target_latency_ms": state.get("target_latency_ms"),
        }
    )

    if isinstance(human_input, dict) and human_input.get("conclude_with_best"):
        # Skips @precision_verifier deliberately -- this candidate was never
        # judged converged by @evaluator's real target check in the first
        # place, so there's nothing for the (expensive) precise re-check to
        # confirm; it would only cost a real NeuroSim/CACTI call to
        # re-verify a candidate everyone already knows didn't meet target.
        return {
            "human_overrides": {},
            "needs_hitl": False,
            "is_converged": True,
            "low_confidence_conclusion": True,
        }

    new_overrides = human_input.get("new_bounds", {}) if isinstance(human_input, dict) else {}

    return {
        "human_overrides": new_overrides,
        "needs_hitl": False,
        # Reset regardless of whether `new_overrides` is empty -- see the
        # module docstring: @planner's own sanitization can't cover the
        # resume-unchanged case without looping forever.
        "retry_count": 0,
    }
