"""@evaluator node + router.

Research_Plan.md 2절: "Pydantic Schema 인터페이스 검증, State Store 이력 누적,
수렴도 평가 및 자율 피드백 라우팅". This node validates every tool output
produced this iteration against its `schemas/tools.py` model, accumulates
`failure_history` on non-convergence, and decides `is_converged`/
`needs_hitl`/`retry_count`; `evaluator_router` reads those flags to route
the corresponding conditional edge (Converged / HITL / Re-plan).

`is_converged` is the AND of two independent checks, both of which must
pass: @verifier's own IR-drop/noise physical-feasibility check (unchanged,
still computed in tools/simulators.py's verifier_tool), and -- new --
whether this candidate actually meets every *configured*
`target_accuracy`/`target_energy_pj`/`target_latency_ms` (state.py). A
hardware config can be physically viable yet still produce a candidate
nowhere near good enough to deploy (e.g. real accuracy 0.20 against a 0.7
target); before this, that scenario was reported as "Converged (Done)"
purely because IR-drop/noise looked fine, which let a run stop having
never actually met its real research goal. Missing target fields (`None`,
the default) skip that dimension's check entirely -- a run that never
configures targets behaves exactly as before.

This is also where `candidate_history` (state.py) gets its one entry per
fully-evaluated iteration: @evaluator is the first node with the complete
multi-objective picture (accuracy from @tuner, energy/latency from
@verifier's synthesis of @mapper+@profiler), so it's the only place a real
Pareto rank (tools/search.py) can be computed -- @tuner's own QAT trial has
no basis to rank itself against candidates it never saw. @planner reads
`candidate_history` back to drive its warm-up/surrogate-guided search.

MAX_RETRY_LIMIT is a control-flow constant (loop-termination policy), not a
hardware parameter, so it is defined here rather than sourced from
`HWConfig` -- docs/coding_guidelines.md C only prohibits hardcoding hardware specs.
"""

from typing import Any, Dict, List, Literal, Optional

from nodes.common import run_id_for
from observability import log_event
from schemas.tools import (
    MapperToolOutput,
    ProfilerToolOutput,
    ToolStatus,
    TunerToolOutput,
    VerifierToolOutput,
)
from state import TessellateState
from tools.search import best_known_candidate, compute_pareto_rank
from tools.targets import check_targets  # noqa: F401 -- re-exported; see tools/targets.py's docstring

MAX_RETRY_LIMIT = 3

_OUTPUT_SCHEMAS = (
    ("tuner", TunerToolOutput),
    ("mapper", MapperToolOutput),
    ("profiler", ProfilerToolOutput),
    ("verifier", VerifierToolOutput),
)


def validate_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Validation errors (empty if all four tools' outputs are present and
    schema-valid with `ToolStatus.SUCCESS`) -- pulled out of `evaluator_node`
    so `tools/batch_warmup.py`'s parallel warm-up runner validates each
    candidate's raw tool results with the exact same rule the sequential
    graph loop uses, rather than a second, potentially-diverging copy of
    this logic."""
    validation_errors = []
    for node_name, schema in _OUTPUT_SCHEMAS:
        raw = metrics.get(node_name)
        if raw is None:
            validation_errors.append(f"{node_name}: missing from metrics_store")
            continue
        try:
            parsed = schema(**raw)
        except Exception as exc:  # noqa: BLE001 -- record and keep validating the rest
            validation_errors.append(f"{node_name}: {exc}")
            continue
        if parsed.status != ToolStatus.SUCCESS:
            validation_errors.append(f"{node_name}: status={parsed.status}, error={parsed.error}")
    return validation_errors


def build_candidate_entry(
    history: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    iteration_count: int,
    is_converged: bool,
    has_validation_errors: bool,
) -> Optional[Dict[str, Any]]:
    """None when this iteration's multi-objective result is incomplete (a
    sub-tool failed/is missing, or is otherwise schema-invalid) -- an
    incomplete candidate has no trustworthy accuracy/energy/latency to rank
    or to feed the surrogate, so it must not be recorded into
    `candidate_history`. This is deliberately keyed off `has_validation_errors`
    rather than only "are these fields non-None": @verifier always returns
    ToolStatus.SUCCESS itself and echoes whatever @mapper/@profiler gave it,
    so a failed upstream tool doesn't reliably leave verifier's own fields
    None -- `validate_metrics`'s result is the authoritative signal that
    this iteration's numbers are trustworthy.

    Takes `history` (the accumulated `candidate_history` to rank against)
    directly rather than the full `TessellateState` -- `tools/batch_warmup.py`
    builds entries for candidates that were never part of any graph state,
    just a growing in-memory list, and this keeps that reuse a plain data
    dependency instead of requiring a fake state dict."""
    if has_validation_errors:
        return None

    tuner_data = (metrics.get("tuner") or {}).get("data") or {}
    verifier_data = (metrics.get("verifier") or {}).get("data") or {}

    candidate = {
        "iteration": iteration_count,
        "avg_weight_bits": tuner_data.get("avg_weight_bits"),
        "avg_column_pruning_ratio": tuner_data.get("avg_column_pruning_ratio"),
        # Full per-stage (weight_bits, column_pruning_ratio) record, the
        # exact config @tuner actually used this iteration -- @planner's
        # per-stage surrogate (tools.search.predict_accuracy_stages) needs
        # this, not just the across-stage averages above, to place this
        # candidate in the real 2*n_stages-dimensional search space.
        "layer_configs": tuner_data.get("layer_configs"),
        "accuracy": tuner_data.get("accuracy"),
        "energy_pj": verifier_data.get("energy_pj"),
        "noc_latency_ms": verifier_data.get("noc_latency_ms"),
        "is_converged": is_converged,
    }
    if any(candidate[key] is None for key in ("avg_weight_bits", "avg_column_pruning_ratio", "accuracy", "energy_pj", "noc_latency_ms")):
        return None

    candidate["pareto_rank"] = compute_pareto_rank(candidate, history)
    return candidate


def evaluator_node(state: TessellateState) -> Dict[str, Any]:
    metrics = state.get("metrics_store", {})
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)

    validation_errors = validate_metrics(metrics)

    tuner_data = (metrics.get("tuner") or {}).get("data") or {}
    verifier_data = (metrics.get("verifier") or {}).get("data") or {}
    hw_verified = not validation_errors and bool(verifier_data.get("is_converged", False))

    # Only meaningful once the schema/HW-feasibility checks above already
    # passed -- an incomplete/invalid iteration has no trustworthy
    # accuracy/energy/latency to judge against a target in the first place.
    target_errors: List[str] = []
    if not validation_errors:
        target_errors = check_targets(
            tuner_data.get("accuracy"),
            verifier_data.get("energy_pj"),
            verifier_data.get("noc_latency_ms"),
            state.get("target_accuracy"),
            state.get("target_energy_pj"),
            state.get("target_latency_ms"),
        )

    meets_targets = hw_verified and not target_errors

    # `meets_targets`, not the run-level stop decision below: a candidate
    # entry's own `is_converged` records whether *that candidate* met every
    # configured target, which is what best_known_candidate selects on.
    # Identical to the old argument under satisficing (where the two
    # coincide); under a budget they must not be conflated, or every
    # candidate found before the budget ran out would be recorded as a
    # failure and become unselectable.
    candidate_entry = build_candidate_entry(
        state.get("candidate_history", []), metrics, iteration_count, meets_targets, has_validation_errors=bool(validation_errors)
    )
    candidate_history: List[Dict[str, Any]] = [candidate_entry] if candidate_entry is not None else []

    budget = state.get("search_budget_iterations")
    budget_exhausted = budget is not None and iteration_count >= budget

    if meets_targets and budget is None:
        log_event(
            run_id_for(state),
            node="evaluator",
            event="candidate_evaluated",
            iteration=iteration_count,
            is_converged=True,
            needs_hitl=False,
            accuracy=(candidate_entry or {}).get("accuracy"),
            energy_pj=(candidate_entry or {}).get("energy_pj"),
            noc_latency_ms=(candidate_entry or {}).get("noc_latency_ms"),
            pareto_rank=(candidate_entry or {}).get("pareto_rank"),
        )
        return {"is_converged": True, "needs_hitl": False, "candidate_history": candidate_history}

    if budget is not None:
        history = list(state.get("candidate_history", [])) + candidate_history
        if budget_exhausted:
            # `low_confidence_conclusion` (state.py) already means exactly
            # "the run stopped without any candidate meeting its targets,
            # and what follows is only the best of what was tried" -- every
            # reporting consumer distinguishes it already, so a budget that
            # ran out empty-handed reuses it rather than adding a second
            # flag meaning the same thing.
            any_met = any(c.get("is_converged") for c in history)
            final_candidate = best_known_candidate(history, any_met)
            log_event(
                run_id_for(state),
                node="evaluator",
                event="search_budget_exhausted",
                iteration=iteration_count,
                is_converged=any_met,
                budget=budget,
                candidates_meeting_targets=sum(1 for c in history if c.get("is_converged")),
                accuracy=(final_candidate or {}).get("accuracy"),
                energy_pj=(final_candidate or {}).get("energy_pj"),
                selected_iteration=(final_candidate or {}).get("iteration"),
            )
            return {
                "is_converged": True,
                "needs_hitl": False,
                "low_confidence_conclusion": not any_met,
                "final_candidate": final_candidate,
                "candidate_history": candidate_history,
            }
        if meets_targets:
            # Spending the rest of the budget on a target already met is
            # progress, not a stall: leaving `retry_count` to climb here
            # would fire a HITL interrupt at MAX_RETRY_LIMIT purely because
            # the search kept succeeding, and record each hit in
            # `failure_history` as though it had missed.
            log_event(
                run_id_for(state),
                node="evaluator",
                event="candidate_evaluated",
                iteration=iteration_count,
                is_converged=False,
                needs_hitl=False,
                met_targets_continuing_on_budget=True,
                budget=budget,
                accuracy=(candidate_entry or {}).get("accuracy"),
                energy_pj=(candidate_entry or {}).get("energy_pj"),
                noc_latency_ms=(candidate_entry or {}).get("noc_latency_ms"),
                pareto_rank=(candidate_entry or {}).get("pareto_rank"),
            )
            return {
                "is_converged": False,
                "needs_hitl": False,
                "retry_count": 0,
                "candidate_history": candidate_history,
            }

    new_retry_count = retry_count + 1
    # planner_flagged_anomaly (state.py): @planner's LLM sets this fresh
    # every iteration (nodes/planner.py), so it's always this iteration's
    # own value, never a stale one from an earlier call -- an anomaly
    # pulls in a researcher immediately instead of waiting for
    # MAX_RETRY_LIMIT routine misses to accumulate.
    anomaly_note = state.get("planner_flagged_anomaly")
    needs_hitl = new_retry_count >= MAX_RETRY_LIMIT or bool(anomaly_note)
    if validation_errors:
        reason = "; ".join(validation_errors)
    elif not hw_verified:
        reason = "verifier reported not converged"
    else:
        reason = "; ".join(target_errors)
    failure_entry = {
        "iteration": iteration_count,
        "retry_count": new_retry_count,
        "reason": reason,
    }
    log_event(
        run_id_for(state),
        node="evaluator",
        event="candidate_evaluated",
        iteration=iteration_count,
        is_converged=False,
        needs_hitl=needs_hitl,
        retry_count=new_retry_count,
        reason=failure_entry["reason"],
        anomaly_note=anomaly_note,
        accuracy=(candidate_entry or {}).get("accuracy"),
        energy_pj=(candidate_entry or {}).get("energy_pj"),
        noc_latency_ms=(candidate_entry or {}).get("noc_latency_ms"),
        pareto_rank=(candidate_entry or {}).get("pareto_rank"),
    )

    return {
        "is_converged": False,
        "needs_hitl": needs_hitl,
        "retry_count": new_retry_count,
        "failure_history": [failure_entry],
        "candidate_history": candidate_history,
    }


def evaluator_router(
    state: TessellateState,
) -> Literal["Converged (Done)", "Concluded (low confidence)", "HITL Interrupt", "Re-plan with History"]:
    if state.get("is_converged", False):
        # A `search_budget_iterations` run whose budget ran out with nothing
        # ever meeting its targets stops here the same way hitl_node's
        # conclude-with-best path does, and for the same reason: there is
        # no converged verdict for @precision_verifier to confirm, so
        # routing there would spend a real NeuroSim/CACTI call re-verifying
        # a candidate already known to have missed. Unreachable for a
        # satisficing run, where `is_converged` only ever means a real
        # target-check pass.
        if state.get("low_confidence_conclusion", False):
            return "Concluded (low confidence)"
        return "Converged (Done)"
    if state.get("needs_hitl", False):
        return "HITL Interrupt"
    return "Re-plan with History"
