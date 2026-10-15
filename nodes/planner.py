"""@planner node.

Research_Plan.md 2/3 attribute surrogate-model construction (BO/NSGA-II
normalization, LHS sampling) and history-based search-bound updates to
@planner. `tools/search.py` implements the real version of that: real
n-dimensional Latin Hypercube Sampling for the first `warmup_count(n_stages)`
iterations, then an IDW-surrogate feeding a genuinely multi-objective
(accuracy/energy_pj/noc_latency_ms) real NSGA-II (`pymoo`) acquisition once
real `candidate_history` (state.py, populated by @evaluator) exists -- one
independent (weight_bits, column_pruning_ratio)
pair *per real model stage* (`tools.qat.get_layer_groups`), not a single
flat point fanned out via a fixed rule: the search space is genuinely
`2 * len(stage_names)`-dimensional (20 for resnet18's 10 stages, up to 40
for mobilenet_v2's 20). `points_to_stage_configs` only rounds/clamps each
stage's searched point into a `LayerBitConfig` -- it introduces no bias of
its own; whatever differences exist between stages come from the search
itself finding them independently, not from a hardcoded edge-stage rule.

The LLM does NOT set `layer_configs` -- `stage_configs` (the search's own
output) is what actually reaches @tuner, unconditionally, whether or not
the LLM call even succeeds. This is a deliberate departure from
Research_Plan.md 3's original "LLM은 오직 파라미터 판단 및 의사결정
(tool_calls)만 수행합니다": a real production run (this session) caught a
local tool-calling model reasoning backwards about an accuracy shortfall
(reducing weight_bits "to improve accuracy" -- the opposite of what
actually helps), silently producing a worse candidate than the search's
own suggestion. The LLM's forced tool call (`PlannerLayerDecision`,
schemas/planner.py) is narrowed to `rationale`/`rationale_ko` (explaining
the already-decided candidate, never free-form text parsed for numbers --
what Research_Plan.md 1 calls "수치 환각 (Hallucination)") and
`anomaly_note` (an early-warning flag a researcher should see now, not
just after `MAX_RETRY_LIMIT` routine misses -- see `state.py`'s
`planner_flagged_anomaly` and `nodes/evaluator.py`). Hardware bounds
(`HWConfig.adc_bits`/`dac_bits`) still bound `stage_configs` via
`search_bounds()` below, same as always -- now the *only* enforcement
point, since there is no longer an LLM-proposed value to separately clamp.

This module also owns the state-sanitization contract every other node
depends on: when `human_overrides` was just produced by `hitl_node`,
consume it and reset `needs_hitl`/`retry_count` in the same step so it
doesn't re-trigger the same HITL interrupt forever (checklist item 3 /
docs/coding_guidelines.md A). Only the transient `human_overrides`/`needs_hitl`/
`retry_count` signal is reset this way -- the bounds themselves are merged
into `search_bound_overrides` (state.py) in the same step, and every
`search_bounds()` call below reads that persisted, merged dict, not
just-this-iteration's `human_overrides`, so an approved override keeps
constraining every later candidate for the rest of the run.

Sanitization only fires when there actually is an override to consume.
`retry_count` must otherwise survive ordinary "Re-plan with History" loops
untouched -- it is `@evaluator`'s counter towards `MAX_RETRY_LIMIT`
(nodes/evaluator.py); resetting it unconditionally on every planner call
would erase that counter every time evaluator increments it, so
`needs_hitl` could never become True and the graph would loop forever
(reproduced via GraphRecursionError while testing graph.py).
"""

import zlib
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from llm import (
    LLMCallFailed,
    LLMCallRecord,
    check_budget,
    get_fallback_chat_model,
    get_planner_chat_model,
    invoke_with_retry_and_fallback,
)
from middleware import get_hw_config
from nodes.common import run_id_for
from observability import log_event
from schemas.planner import PlannerLayerDecision
from schemas.tools import LayerBitConfig
from state import TessellateState, merge_dicts
from tools.cim_physics import snap_pruning_ratio
from tools.qat import get_layer_groups, get_stage_weight_shapes
from tools.search import (
    Bounds,
    StagePoint,
    propose_stage_candidate_random,
    propose_stage_candidate_single_objective,
    propose_stage_candidates_via_surrogate,
    warmup_stage_candidate,
)

SEARCH_STRATEGIES = ("multi_objective", "single_objective", "random")

_FALLBACK_STAGE_NAME = "model_default"

# Forcing tool_choice by name requires the exact name the schema converts
# to; deriving it from the class avoids the two ever drifting apart.
_TOOL_NAME = PlannerLayerDecision.__name__

# Search-space policy constants (not hardware specs -- docs/coding_guidelines.md C only
# prohibits hardcoding hardware numbers, and the upper bit-width bound below
# is still derived from HWConfig, never a literal).
#
# Warm-up count scales with the real per-stage dimensionality
# (2 * n_stages): a flat 3 samples is nowhere near enough to seed a
# 12-40-dimensional search space (curse of dimensionality), but scaling
# unboundedly (e.g. the textbook ~10x-dimensions BO heuristic) would mean
# hundreds of real QAT trials at 37s-2min each before the surrogate phase
# even starts. `warmup_count` is the deliberate middle ground:
# `_WARMUP_SAMPLES_PER_STAGE` warm-up samples per stage (not just one --
# measured via tools/baseline_compare.py that a flat 1x ratio left
# resnet18's surrogate (10 stages -> exactly 10 real warm-up points over a
# 20-dim space) unreliable at holding a specific --target-accuracy: only
# 1/3 seeds converged within the auto-hitl retry budget), capped at
# `_MAX_WARMUP_CANDIDATES` so a 20-stage model (mobilenet_v2) still bounds
# warm-up wall-clock instead of scaling with model size -- raised from 12
# alongside the per-stage multiplier so it doesn't immediately re-cap
# resnet18's now-larger warm-up count back down to the old value.
_WARMUP_CANDIDATES = 3
_WARMUP_SAMPLES_PER_STAGE = 1.5
_MAX_WARMUP_CANDIDATES = 20
_WEIGHT_BITS_MIN = 2
_PRUNING_RATIO_MAX = 0.5


def warmup_count(n_stages: int) -> int:
    return min(_MAX_WARMUP_CANDIDATES, max(_WARMUP_CANDIDATES, round(n_stages * _WARMUP_SAMPLES_PER_STAGE)))


def search_seed_for(model_id: str, hw_spec_id: str, run_seed: int = 0) -> int:
    """Deterministic per-`(model_id, hw_spec_id, run_seed)` seed for LHS
    warm-up sampling + NSGA-II acquisition -- crc32, not Python's builtin
    `hash()`, since str hashing is randomized per-process
    (`PYTHONHASHSEED`) unless disabled, which would make "same warm-up
    index -> same point" false across runs/processes. Public (not just
    inlined in `_propose_stage_points`) so `tools/batch_warmup.py`'s
    parallel warm-up runner derives the exact same seed the sequential
    path would have used for this `(model_id, hw_spec_id, run_seed)` --
    computing all warm-up points up front must reproduce, not diverge
    from, what `_propose_stage_points` would have produced one at a time.

    `run_seed` (state.py's `run_seed`, `TessellateState`) defaults to 0,
    identical to this function's behavior before the parameter existed --
    without it, *every* session on the same (model_id, hw_spec_id) always
    explored the exact same warm-up points and NSGA-II trajectory, so
    running "the same search" more than once could never produce an
    independent sample: `tools/repeat_search.py`'s whole premise (N
    independent replicates for a variance/CI estimate a single run can't
    give) needs `run_seed` to actually vary the search, not just the seed
    number's cosmetic value."""
    return zlib.crc32(f"{model_id}::{hw_spec_id}::{run_seed}".encode())


_SYSTEM_PROMPT = (
    "You are the @planner agent in Tessellate, a CIM-chip HW-SW "
    "co-design optimizer. You are given the target model's real stages, a "
    "per_stage_suggested_candidate -- one (weight_bits, column_pruning_ratio) "
    "pair per stage, already finalized by a statistical search (Latin "
    "Hypercube warm-up sampling or a multi-objective NSGA-II acquisition "
    "over prior measured candidates) -- plus the target hardware's bounds "
    "and recent failure history. You do NOT choose or adjust weight_bits/ "
    f"column_pruning_ratio yourself. Call `{_TOOL_NAME}` -- a tool call "
    "only, never plain text -- with exactly these fields:\n"
    "1. rationale: briefly explain why per_stage_suggested_candidate is a "
    "reasonable next step, referencing the stated hw bounds and "
    "recent_failure_history. Do not propose different numbers -- describe "
    "the candidate you were given.\n"
    "2. rationale_ko: a Korean translation of the same rationale -- same "
    "content, not a summary or a different justification. Write it the way a "
    "Korean computer-architecture lab writes a seminar note: formal ~입니다/"
    "~합니다 register, and established Korean terms rather than phonetic "
    "transliteration (surrogate model -> 서로게이트 모델, candidate -> 후보, "
    "pruning -> 프루닝, quantization -> 양자화, crossbar -> 크로스바). Design-space "
    "search is 탐색, never 검색 (검색 means looking something up). Leave "
    "identifiers, field names, units and the search-phase tag exactly as "
    "given, in English (weight_bits, column_pruning_ratio, hw_spec_id, pJ, "
    "'LHS warm-up candidate 1/15') -- never translate or transliterate those; "
    "'warm-up' in particular is a phase name, not the adjective 따뜻한.\n"
    "3. anomaly_note: leave this null unless recent_failure_history or "
    "per_stage_suggested_candidate shows something genuinely concerning "
    "enough that a researcher should look at it right now, not just after "
    "the usual retry budget runs out -- e.g. the exact same candidate "
    "failing repeatedly with no change, or a metric that looks physically "
    "implausible. Do not set it for routine, expected-looking failures; "
    "that is what the normal retry loop already handles."
)


def _clamp_bounds(lo: float, hi: float, override_lo: Any, override_hi: Any) -> Bounds:
    """Narrows (lo, hi) using override_lo/override_hi if they are numeric --
    never widens past the hw-derived bound, and if a bad pair would invert
    the range (lo > hi), ignores the override entirely and falls back to
    the original (lo, hi) rather than producing an unusable search range."""
    new_lo = max(lo, override_lo) if isinstance(override_lo, (int, float)) else lo
    new_hi = min(hi, override_hi) if isinstance(override_hi, (int, float)) else hi
    if new_lo > new_hi:
        return lo, hi
    return new_lo, new_hi


def search_bounds(hw_spec_id: str, overrides: Optional[Dict[str, Any]] = None) -> Tuple[Bounds, Bounds]:
    """`overrides` (from `human_overrides`/HITL `new_bounds`) is applied here
    as a hard clamp on the actual search range -- not just advisory text in
    the LLM prompt (`_build_user_prompt`'s `researcher_overrides` line still
    shows it too, but that alone never guaranteed compliance, especially
    from smaller/local tool-calling models). Recognized keys: weight_bits_min,
    weight_bits_max, pruning_ratio_min, pruning_ratio_max -- all optional,
    and each can only narrow its dimension's hw-derived bound, never exceed
    the physical adc_bits/dac_bits-derived ceiling.

    Why `dac_bits` bounds *weight* bits: `points_to_stage_configs` sets
    `activation_bits == weight_bits` (one searched precision per stage, not
    two independent axes), and activation precision is what a DAC physically
    drives, so the single searched value has to respect the tighter of the
    two converter resolutions. The consequence is worth knowing when reading
    a run: on a `dac_bits=4` spec (examples/hw_configs/default_cim_v1_128x128.json)
    this caps the whole search at 4-bit weights regardless of how many
    levels the crossbar cell could actually hold. Splitting weight and
    activation precision into separate search dimensions -- so `adc_bits`
    bounds one and `dac_bits` the other -- would remove that coupling, at
    the cost of doubling the search space to `4 * n_stages`."""
    try:
        hw = get_hw_config(hw_spec_id)
        bits_max = min(8, hw.adc_bits, hw.dac_bits)
    except Exception:
        bits_max = 8
    bits_max = max(bits_max, _WEIGHT_BITS_MIN + 1)

    bit_bounds = (_WEIGHT_BITS_MIN, bits_max)
    pruning_bounds = (0.0, _PRUNING_RATIO_MAX)
    if overrides:
        bit_bounds = _clamp_bounds(*bit_bounds, overrides.get("weight_bits_min"), overrides.get("weight_bits_max"))
        pruning_bounds = _clamp_bounds(*pruning_bounds, overrides.get("pruning_ratio_min"), overrides.get("pruning_ratio_max"))
    return bit_bounds, pruning_bounds


def real_stage_names(model_id: str) -> List[str]:
    """Real per-stage layer names for `model_id` (tools/qat.py), in true
    front-to-back architectural order -- `group_quantizable_layers` builds
    its dict while iterating `model.named_modules()`, which visits stages in
    definition/forward order, and dict insertion order is preserved; do NOT
    sort alphabetically here, or the per-stage search points from
    `_propose_stage_points` would land on the wrong stage names when zipped
    against them. Falls back to a single synthetic stage name for an
    unknown `model_id` so this module still works before @tuner's model
    registry covers it -- @tuner then applies that one entry's config
    uniformly (tools/qat.py's `default_config` fallback)."""
    try:
        return list(get_layer_groups(model_id).keys())
    except Exception:
        return [_FALLBACK_STAGE_NAME]


def _propose_stage_points(
    state: TessellateState, stage_names: List[str], overrides: Optional[Dict[str, Any]] = None
) -> Tuple[StagePoint, str]:
    """Returns one (weight_bits, column_pruning_ratio) point per real stage
    (in `stage_names` order) and a short tag describing which search phase
    produced it, for the LLM prompt/messages."""
    bit_bounds, pruning_bounds = search_bounds(state["hw_spec_id"], overrides)
    history = state.get("candidate_history", [])
    n_warmup = warmup_count(len(stage_names))
    seed = search_seed_for(state["model_id"], state["hw_spec_id"], state.get("run_seed", 0))

    if len(history) < n_warmup:
        point = warmup_stage_candidate(len(history), n_warmup, len(stage_names), bit_bounds, pruning_bounds, seed)
        return point, f"LHS warm-up candidate {len(history) + 1}/{n_warmup}"

    # Warm-up is always real LHS regardless of strategy -- ablation
    # (tools/run_ablation.py) only varies the *acquisition* phase after
    # warm-up, so every strategy starts every replicate from the same kind
    # of cold-start history, isolating the comparison to the one thing
    # actually being ablated.
    strategy = state.get("search_strategy") or "multi_objective"
    # iteration-varying seed offset: propose_stage_candidate_random has no
    # history to diversify against (unlike the surrogate strategies, whose
    # predictions already shift every call as history grows), so reusing
    # one fixed seed across the whole surrogate phase would repeat the
    # exact same "random" point on every iteration.
    iteration_seed = seed + len(history)
    if strategy == "random":
        point = propose_stage_candidate_random(stage_names, bit_bounds, pruning_bounds, iteration_seed)
        return point, "random-search acquisition (ablation baseline, no surrogate)"
    if strategy == "single_objective":
        point = propose_stage_candidate_single_objective(history, stage_names, bit_bounds, pruning_bounds, seed)
        return point, f"single-objective surrogate acquisition over {len(history)} prior candidates (ablation baseline)"
    point = propose_stage_candidates_via_surrogate(history, stage_names, bit_bounds, pruning_bounds, seed)
    return point, f"surrogate-model acquisition over {len(history)} prior candidates"


def points_to_stage_configs(
    stage_points: StagePoint,
    stage_names: List[str],
    hw: Optional[Any] = None,
    stage_weight_shapes: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
) -> List[LayerBitConfig]:
    """One `LayerBitConfig` per real stage: each stage's own independently
    searched (weight_bits, column_pruning_ratio) point, rounded/clamped to
    physically realizable values -- no per-stage bias is applied here, since
    the search itself (`_propose_stage_points`) already explored each stage
    independently.

    `weight_bits` is rounded to an integer for the obvious reason: there is
    no 4.7-bit crossbar cell. Passing `hw` *and* `stage_weight_shapes`
    (`tools.qat.get_stage_weight_shapes`) applies the same kind of rounding
    to the pruning axis via `tools.cim_physics.snap_pruning_ratio` -- energy
    only drops when pruning removes a whole crossbar array, so a ratio
    between two tile boundaries is not a realizable design point either, it
    is strictly the same hardware with more weights thrown away. Snapping
    down weakly dominates (same passes, less accuracy lost); see that
    function's docstring for the measured evidence. Omit both to keep the
    raw clamp, which is what callers without a resolved model/hardware
    (tests, `tools/optuna_compare.py`'s independent search space) still
    use."""
    configs = []
    for name, (bits, pruning) in zip(stage_names, stage_points):
        stage_bits = max(_WEIGHT_BITS_MIN, round(bits))
        stage_pruning = round(min(max(pruning, 0.0), _PRUNING_RATIO_MAX), 4)
        if hw is not None and stage_weight_shapes is not None:
            stage_pruning = snap_pruning_ratio(
                stage_pruning, stage_weight_shapes.get(name), hw, _PRUNING_RATIO_MAX
            )
        configs.append(
            LayerBitConfig(
                layer_name=name, weight_bits=stage_bits, activation_bits=stage_bits, column_pruning_ratio=stage_pruning
            )
        )
    return configs


def _snapping_context(model_id: str, hw_spec_id: str) -> Tuple[Optional[Any], Optional[Dict[str, List[Tuple[int, int, int]]]]]:
    """`(hw, stage_weight_shapes)` for `points_to_stage_configs`' pruning
    snap, or `(None, None)` if either can't be resolved.

    Degrading to `(None, None)` keeps the raw clamp rather than failing the
    planner over a model/hardware lookup: an unregistered `hw_spec_id` or an
    unimportable custom `--model-id` already surfaces as its own error
    elsewhere (`search_bounds` and @tuner respectively). `tools/batch_warmup.py`
    calls this same helper so the parallel warm-up path snaps identically to
    the sequential one -- the two must agree candidate-for-candidate."""
    try:
        return get_hw_config(hw_spec_id), get_stage_weight_shapes(model_id)
    except Exception:  # noqa: BLE001 -- see docstring: degrade to the raw clamp
        return None, None


def _describe_hw_bounds(hw_spec_id: str) -> str:
    try:
        hw = get_hw_config(hw_spec_id)
    except Exception:
        return f"hw_spec_id={hw_spec_id!r} (HWConfig not registered yet; assume a conservative 8-bit bound)"
    return (
        f"hw_spec_id={hw.hw_spec_id!r}, crossbar={hw.crossbar_rows}x{hw.crossbar_cols}, "
        f"num_tiles={hw.num_tiles}, adc_bits={hw.adc_bits}, dac_bits={hw.dac_bits}"
    )


def _build_user_prompt(
    state: TessellateState,
    overrides: Dict[str, Any],
    search_tag: str,
    stage_configs: List[LayerBitConfig],
) -> str:
    verifier_data = (state.get("metrics_store", {}).get("verifier") or {}).get("data") or {}
    suggested = ", ".join(
        f"{lc.layer_name}(weight_bits={lc.weight_bits}, column_pruning_ratio={lc.column_pruning_ratio})"
        for lc in stage_configs
    )
    lines = [
        f"model_id: {state['model_id']}",
        f"hardware: {_describe_hw_bounds(state['hw_spec_id'])}",
        f"real_model_stages: {[lc.layer_name for lc in stage_configs]}",
        # Each stage below was searched independently ({search_tag}) -- not
        # one flat point fanned out by a fixed rule. This is the final
        # config; you are explaining it, not proposing an alternative.
        f"per_stage_suggested_candidate ({search_tag}): {suggested}",
        f"iteration_count: {state.get('iteration_count', 0)}",
        f"recent_failure_history: {state.get('failure_history', [])[-3:]}",
        f"latest_verifier_metrics: {verifier_data}",
    ]
    if overrides:
        lines.append(
            f"researcher_overrides (already applied as hard bounds on per_stage_suggested_candidate above; "
            f"stay within them): {overrides}"
        )
    return "\n".join(lines)


def _extract_decision(response: Any) -> PlannerLayerDecision:
    for call in getattr(response, "tool_calls", None) or []:
        if call.get("name") == _TOOL_NAME:
            return PlannerLayerDecision(**call["args"])
    raise RuntimeError(f"planner LLM response had no {_TOOL_NAME!r} tool call: {response!r}")


def _explain_decision(
    state: TessellateState,
    overrides: Dict[str, Any],
    search_tag: str,
    stage_configs: List[LayerBitConfig],
    iteration_count: int,
) -> Tuple[str, Optional[str], Optional[str], "LLMCallRecord"]:
    """Real LLM call, retried on transient errors via
    `llm.invoke_with_retry_and_fallback` (rate limits/timeouts/5xx -- see
    llm.py) -- if the primary provider stays down through every retry, this
    automatically fails over to `llm.get_fallback_chat_model()` (the local
    default) and retries there before giving up, instead of immediately
    falling back to the template rationale. Then forced through
    tool-calling. Raises `LLMCallFailed` -- carrying the real attempts/
    tokens/cost `LLMCallRecord` even for this failure, so a billable call
    that came back malformed still gets logged accurately -- on any failure
    neither the retry nor the provider-fallback layer absorbed (including a
    well-formed response with no usable tool call); the caller must catch
    that and fall back to its own template rationale. `get_planner_chat_model()`
    itself raising (e.g. missing API key/model env var) is not caught here --
    that's a config problem retrying can't fix, and has no LLM call to build
    a record from; the caller handles it as a plain exception. This overall
    containment mirrors `middleware.wrap_tool_call`'s containment of
    backend-tool exceptions.

    Returns `(rationale, rationale_ko, anomaly_note, usage_record)` --
    notably no `layer_configs`: `stage_configs` (the search's own output)
    is what the caller actually uses, unconditionally (module docstring)."""
    chat_model = get_planner_chat_model().bind_tools([PlannerLayerDecision], tool_choice=_TOOL_NAME)
    fallback_model = get_fallback_chat_model()
    if fallback_model is not None:
        fallback_model = fallback_model.bind_tools([PlannerLayerDecision], tool_choice=_TOOL_NAME)
    response, usage_record = invoke_with_retry_and_fallback(
        chat_model,
        fallback_model,
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(state, overrides, search_tag, stage_configs)),
        ],
        node="planner",
        iteration=iteration_count,
    )
    try:
        decision = _extract_decision(response)
    except Exception as exc:  # a real (billable) call came back malformed -- not a transient error to retry
        usage_record.succeeded = False
        usage_record.error = f"{type(exc).__name__}: {exc}"
        raise LLMCallFailed(usage_record) from exc
    return decision.rationale, decision.rationale_ko, decision.anomaly_note, usage_record


def planner_node(state: TessellateState) -> Dict[str, Any]:
    iteration_count = state.get("iteration_count", 0) + 1
    new_overrides = state.get("human_overrides", {})  # read once, before clearing
    # Merge, not replace: bounds a researcher already approved in an earlier
    # HITL round must keep applying alongside (or be narrowed further by) a
    # newly-approved one, not be dropped the moment a different key changes.
    overrides = merge_dicts(state.get("search_bound_overrides", {}), new_overrides)

    update: Dict[str, Any] = {"iteration_count": iteration_count}
    messages: List[Dict[str, Any]] = []

    if new_overrides:
        # Sanitize immediately after consuming (checklist item 3): prevents
        # the HITL flag / the retry counter that triggered it from leaking
        # into the next iteration. The approved bounds themselves are NOT
        # transient -- they persist in search_bound_overrides (merged above)
        # for the rest of the run, not just this one candidate.
        update["human_overrides"] = {}
        update["search_bound_overrides"] = new_overrides
        update["needs_hitl"] = False
        update["retry_count"] = 0
        messages.append(
            {
                "role": "system",
                "content": f"[planner] applied human_overrides at iteration {iteration_count}: {new_overrides}",
            }
        )

    stage_names = real_stage_names(state["model_id"])
    stage_points, search_tag = _propose_stage_points(state, stage_names, overrides)
    stage_configs = points_to_stage_configs(
        stage_points, stage_names, *_snapping_context(state["model_id"], state["hw_spec_id"])
    )

    # stage_configs is what actually reaches @tuner on every path below,
    # whether or not the LLM call even runs -- the LLM only ever supplies
    # rationale/rationale_ko/anomaly_note now (module docstring: it no
    # longer proposes layer_configs itself). No separate post-hoc bounds
    # clamp is needed either: stage_configs already came from
    # search_bounds(state["hw_spec_id"], overrides) above, so it's
    # override-aware and hw-ceiling-safe by construction.
    anomaly_note: Optional[str] = None
    budget_status = check_budget(state.get("llm_usage", []))
    if budget_status.exceeded:
        # Kill-switch: skip the LLM call entirely (never even constructs a
        # chat model) rather than retry/fallback after spending more --
        # retry/backoff (llm.invoke_with_retry) only contains failures
        # *within* one call, it has no concept of a cumulative per-run cap.
        used_llm = False
        usage_record = LLMCallRecord(
            node="planner", iteration=iteration_count, attempts=0, succeeded=False, latency_ms=0.0, error=budget_status.reason
        )
        rationale = f"LLM call skipped: {budget_status.reason}"
        rationale_ko = None  # fixed English template, not an LLM output to translate
        messages.append({"role": "system", "content": f"[planner] {rationale}; using the {search_tag} search point directly"})
    else:
        used_llm = True
        try:
            rationale, rationale_ko, anomaly_note, usage_record = _explain_decision(
                state, overrides, search_tag, stage_configs, iteration_count
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[planner] iteration {iteration_count} ({search_tag}): {rationale}",
                }
            )
        except LLMCallFailed as exc:  # a real LLM call happened (possibly after retries) but didn't pan out
            used_llm = False
            usage_record = exc.record
            rationale = f"LLM proposal failed after {usage_record.attempts} attempt(s) ({usage_record.error})"
            rationale_ko = None
            messages.append({"role": "system", "content": f"[planner] {rationale}; using the {search_tag} search point directly"})
        except Exception as exc:  # no LLM call was ever made (e.g. get_planner_chat_model() config error)
            used_llm = False
            usage_record = LLMCallRecord(
                node="planner", iteration=iteration_count, attempts=0, succeeded=False, latency_ms=0.0, error=f"{type(exc).__name__}: {exc}"
            )
            rationale = f"LLM proposal failed ({type(exc).__name__}: {exc})"
            rationale_ko = None
            messages.append({"role": "system", "content": f"[planner] {rationale}; using the {search_tag} search point directly"})

    layer_config_dicts = [lc.model_dump(mode="json") for lc in stage_configs]
    update["planned_layer_configs"] = layer_config_dicts
    update["planner_flagged_anomaly"] = anomaly_note
    update["llm_usage"] = [usage_record.to_dict()]
    decision_entry = {
        "iteration": iteration_count,
        "search_tag": search_tag,
        "used_llm": used_llm,
        "rationale": rationale,
        "rationale_ko": rationale_ko,
        "anomaly_note": anomaly_note,
        "layer_configs": layer_config_dicts,
    }
    update["planner_decisions"] = [decision_entry]
    if anomaly_note:
        messages.append({"role": "system", "content": f"[planner] anomaly flagged at iteration {iteration_count}: {anomaly_note}"})
    log_event(
        run_id_for(state),
        node="planner",
        event="candidate_proposed",
        iteration=iteration_count,
        search_tag=search_tag,
        used_llm=used_llm,
        rationale=rationale,
        anomaly_note=anomaly_note,
        n_stages=len(stage_names),
    )
    update["messages"] = messages

    return update
