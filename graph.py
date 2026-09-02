"""Tessellate LangGraph assembly (Research_Plan.md 2절).

START -> planner -> tuner -> (mapper, profiler fan-out) -> verifier -> evaluator
                                                                            |
                            +----------------- Re-plan with History -------+
                            |                                              |
                       hitl_human_approval <--- HITL Interrupt ------------+
                            |    (or: Concluded (low conf) -> END)         |
                            +-> planner                                   +--> Converged (Done) -> precision_verifier
                                                                                                            |
                                                    +-------------- Re-plan with History -------------------+
                                                    |                                                       |
                                               hitl_human_approval <--- HITL Interrupt ---------------------+
                                                    |    (or: Concluded (low conf) -> END)                  |
                                                    +-> planner                                             +--> Converged (Done) -> END

@precision_verifier (mock Stage-2 high-fidelity re-check, docs/coding_guidelines.md 5.D Mock
First) sits between @evaluator's fast-approximation convergence call and a
real `END` -- @evaluator's own "Converged (Done)" routes here first instead
of straight to `END`, and this node re-routes through the exact same three
outcomes @evaluator_router already defines (reused directly, not
duplicated) once its own precision check confirms (or overturns) that
verdict. Only a candidate the fast approximation already thinks is done
ever reaches this node -- never every candidate a search tries.

HITL pausing uses only the dynamic `interrupt()` call inside `hitl_node`
(checklist item 1 / docs/coding_guidelines.md 5.B) -- `workflow.compile()` below must never
be given `interrupt_before`/`interrupt_after`. `hitl_human_approval`'s own
outgoing edge is conditional (`hitl_router`): a normal resume (new search
bounds) still goes to `planner`, but `main.py`'s auto-hitl giving up after
its round cap (`conclude_with_best`) routes straight to `END` instead,
skipping `precision_verifier` (the candidate was never judged converged in
the first place, so there's nothing for the precise re-check to confirm).
"""

from typing import Literal, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nodes import (
    evaluator_node,
    evaluator_router,
    hitl_node,
    mapper_node,
    planner_node,
    precision_verifier_node,
    profiler_node,
    tuner_node,
    verifier_node,
)
from state import TessellateState


def hitl_router(state: TessellateState) -> Literal["Concluded (low confidence)", "Re-plan"]:
    """hitl_node's `conclude_with_best` resume (main.py's auto-hitl giving
    up after its round cap) sets `low_confidence_conclusion` -- routes
    straight to `END` instead of the normal `planner` retry, skipping
    @precision_verifier the same way @evaluator_router already does for a
    candidate that was never judged converged in the first place."""
    return "Concluded (low confidence)" if state.get("low_confidence_conclusion") else "Re-plan"


def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    workflow = StateGraph(TessellateState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("tuner", tuner_node)
    workflow.add_node("mapper", mapper_node)
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("precision_verifier", precision_verifier_node)
    workflow.add_node("hitl_human_approval", hitl_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "tuner")

    # Fan-out: mapper/profiler run in parallel, then fan back into verifier.
    workflow.add_edge("tuner", "mapper")
    workflow.add_edge("tuner", "profiler")
    workflow.add_edge("mapper", "verifier")
    workflow.add_edge("profiler", "verifier")

    workflow.add_edge("verifier", "evaluator")
    workflow.add_conditional_edges(
        "hitl_human_approval",
        hitl_router,
        {
            "Concluded (low confidence)": END,
            "Re-plan": "planner",
        },
    )

    workflow.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            # Was END directly before @precision_verifier existed -- the
            # fast-approximation's own "converged" verdict is no longer
            # final by itself, only a trigger for the (rare, expensive)
            # precise re-check.
            "Converged (Done)": "precision_verifier",
            # Only a `search_budget_iterations` run reaches this from
            # @evaluator -- see evaluator_router. Same destination and same
            # reasoning as hitl_router's identically-named outcome.
            "Concluded (low confidence)": END,
            "HITL Interrupt": "hitl_human_approval",
            "Re-plan with History": "planner",
        },
    )
    workflow.add_conditional_edges(
        "precision_verifier",
        evaluator_router,  # same outcomes, same state fields -- see module docstring
        {
            "Converged (Done)": END,
            "Concluded (low confidence)": END,
            "HITL Interrupt": "hitl_human_approval",
            "Re-plan with History": "planner",
        },
    )

    # Checklist item 1 (strict guardrail): dynamic interrupt() inside
    # hitl_node is the only pause mechanism -- compile() must not declare
    # interrupt_before/interrupt_after.
    #
    # Defaults to MemorySaver (in-process only -- a session dies with the
    # process, exactly the limitation main.py's run_session works around
    # for real CLI sessions) so every existing caller (tests, the module-
    # level `graph` singleton below) keeps today's fully-isolated-per-call
    # behavior with zero changes. main.py passes a real persistent
    # checkpointer (SqliteSaver) for actual CLI sessions.
    return workflow.compile(checkpointer=checkpointer or MemorySaver())


graph = build_graph()
