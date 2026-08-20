"""Real multi-candidate search for @planner (Research_Plan.md 2/3's "BO/NSGA-II
정규화 및 LHS 최소 샘플링 기반 대리 모델" and "정규화 다중 목적 탐색").

Previously @planner asked its LLM for a candidate with no statistical
grounding, and @tuner reported a hardcoded `pareto_rank: 1` that no single
QAT trial can actually justify (Pareto rank is inherently relative to other
evaluated candidates). This module is the missing piece underneath the LLM
call:

- `warmup_candidate`/`propose_via_surrogate`: the original 2D (single
  weight_bits, column_pruning_ratio point for the whole model) search --
  real Latin Hypercube Sampling for the first few iterations, then a
  lightweight real surrogate (inverse-distance-weighted regression +
  upper-confidence-bound acquisition) once there's history. Kept as-is for
  callers/tests that only need one flat point.
- `warmup_stage_candidate`/`propose_stage_candidates_via_surrogate`: the
  real per-stage generalization. A candidate is one (weight_bits,
  column_pruning_ratio) pair *per real model stage*
  (`tools.qat.get_layer_groups`), so the search space is
  `2 * len(stage_names)`-dimensional (20 for resnet18's 10 stages, up to 40
  for mobilenet_v2's 20) instead of a fixed 2D point that
  `nodes/planner.py` used to fan out to every stage via one hardcoded
  edge-stage-bias rule. `warmup_stage_candidate` still uses real LHS (now
  n-dimensional -- each of the `2 * n_stages` dimensions independently
  stratified, same construction as the 2D case).

  `propose_stage_candidates_via_surrogate` is genuinely multi-objective
  (accuracy, energy_pj, noc_latency_ms all three, via `pymoo`'s real
  NSGA-II) as of this revision -- it used to only ever model accuracy
  (`predict_accuracy_stages`) and leave energy/latency out of the search
  entirely, with `compute_pareto_rank` below only ever ranking candidates
  *after* they were already evaluated for real. That meant the search
  itself was single-objective in practice despite the module's NSGA-II
  namesake -- confirmed in production (a real run's surrogate-phase
  candidates never beat the best warm-up candidate's accuracy, because
  nothing was steering search toward the accuracy/energy Pareto front, just
  toward accuracy alone). `predict_energy_stages`/`predict_latency_stages`
  (the same IDW regression as `predict_accuracy_stages`, just fit against
  `energy_pj`/`noc_latency_ms` history instead) now feed a 3-objective
  `pymoo.core.problem.Problem`; `NSGA2` explores over the *surrogate's*
  predictions (still cheap IDW math, run for many generations) to find a
  well-spread Pareto front, and one point is picked (uniformly at random,
  seeded) from that front as the next *real* candidate to actually
  evaluate -- NSGA-II itself never touches a real QAT trial, only ever
  what the IDW surrogates predict one would produce.

  This is a deliberate exception to this project's usual minimal-
  dependency stance (`pymoo` pulls in `scipy`/`matplotlib`/etc. --
  genuinely heavy): decided explicitly in favor of a real, validated
  multi-objective algorithm over extending the hand-rolled single-
  objective UCB scan to three objectives itself.
- `compute_pareto_rank`: real non-dominated-sorting-style rank (NSGA-II's
  core ranking idea) against the actual accumulated `candidate_history`,
  computed once real multi-objective results (accuracy, energy, latency)
  exist for a candidate -- this ranks already-*evaluated* real candidates
  for reporting (tools/dashboard.py's Pareto chart), a different job from
  `propose_stage_candidates_via_surrogate`'s NSGA-II search over the
  surrogate's predictions of *not-yet-evaluated* candidates above.

- `propose_stage_candidate_random`/`propose_stage_candidate_single_objective`:
  ablation baselines (`tools/run_ablation.py`, `state.py`'s `search_strategy`)
  -- "no search at all" and "the prior single-objective acquisition this
  project actually shipped", respectively, so the current multi-objective
  NSGA-II acquisition can be compared against concrete alternatives, not
  just asserted to be better.

@planner still makes the final call via forced tool-calling
(nodes/planner.py) -- this module produces the statistically-grounded
suggestion the LLM prompt is anchored to, not a replacement for it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

Bounds = Tuple[float, float]


# ---------------------------------------------------------------------------
# Warm-up: real Latin Hypercube Sampling
# ---------------------------------------------------------------------------


def _latin_hypercube_1d(n_samples: int, bounds: Bounds, seed: int) -> List[float]:
    """One real LHS dimension: `n_samples` equal-width strata, one uniform
    sample per stratum, then shuffled -- every stratum is used exactly once,
    which is what distinguishes LHS from plain uniform random sampling."""
    low, high = bounds
    rng = random.Random(seed)
    width = (high - low) / n_samples
    points = [rng.uniform(low + i * width, low + (i + 1) * width) for i in range(n_samples)]
    rng.shuffle(points)
    return points


def latin_hypercube_candidates(
    n_samples: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> List[Tuple[float, float]]:
    """`n_samples` (weight_bits, column_pruning_ratio) points via 2D LHS:
    each dimension independently stratified via `_latin_hypercube_1d`, then
    paired by index -- the standard LHS construction."""
    bits = _latin_hypercube_1d(n_samples, bit_bounds, seed)
    pruning = _latin_hypercube_1d(n_samples, pruning_bounds, seed + 1)
    return list(zip(bits, pruning))


def warmup_candidate(
    index: int, n_warmup: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> Tuple[float, float]:
    """The `index`-th warm-up candidate. Deterministic in `(index, seed)`, so
    it never needs its own cursor in `AutoCIMState` -- calling this again
    with the same `index` (e.g. a HITL-resumed retry of the same iteration)
    reproduces the same point rather than silently sampling a new one."""
    candidates = latin_hypercube_candidates(n_warmup, bit_bounds, pruning_bounds, seed)
    return candidates[index % n_warmup]


# ---------------------------------------------------------------------------
# Per-stage (n-dimensional) warm-up: real n-D Latin Hypercube Sampling
# ---------------------------------------------------------------------------

StagePoint = List[Tuple[float, float]]  # one (weight_bits, column_pruning_ratio) pair per real stage


def latin_hypercube_stage_candidates(
    n_samples: int, n_stages: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> List[StagePoint]:
    """`n_samples` candidates, each a list of `n_stages` independent
    (weight_bits, column_pruning_ratio) pairs -- real n-D LHS: each of the
    `2 * n_stages` dimensions (two per stage) is stratified independently
    via `_latin_hypercube_1d` with its own seed offset, exactly like
    `latin_hypercube_candidates`'s 2 dimensions, just `n_stages` times as
    many of them."""
    dims = [
        _latin_hypercube_1d(n_samples, bit_bounds if d % 2 == 0 else pruning_bounds, seed + d)
        for d in range(2 * n_stages)
    ]
    rows = zip(*dims)
    return [[(row[2 * s], row[2 * s + 1]) for s in range(n_stages)] for row in rows]


def warmup_stage_candidate(
    index: int, n_warmup: int, n_stages: int, bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> StagePoint:
    """The `index`-th per-stage warm-up candidate -- the n-D analogue of
    `warmup_candidate`, same determinism-in-`(index, seed)` guarantee."""
    candidates = latin_hypercube_stage_candidates(n_warmup, n_stages, bit_bounds, pruning_bounds, seed)
    return candidates[index % n_warmup]


# ---------------------------------------------------------------------------
# Surrogate-guided phase: IDW regression + UCB acquisition
# ---------------------------------------------------------------------------


def _normalize(value: float, bounds: Bounds) -> float:
    low, high = bounds
    return (value - low) / (high - low) if high > low else 0.0


def predict_accuracy(
    bits: float, pruning: float, history: Sequence[Dict[str, Any]], bit_bounds: Bounds, pruning_bounds: Bounds
) -> Tuple[float, float]:
    """Inverse-distance-weighted prediction of accuracy at `(bits, pruning)`
    from `history` entries (each needs `avg_weight_bits`,
    `avg_column_pruning_ratio`, `accuracy`), plus an uncertainty proxy
    (lower when close, well-sampled history exists nearby; ~1.0 with no
    history at all) used as the exploration term in `propose_via_surrogate`.
    """
    target = (_normalize(bits, bit_bounds), _normalize(pruning, pruning_bounds))
    weights: List[float] = []
    values: List[float] = []
    for entry in history:
        point = (
            _normalize(entry["avg_weight_bits"], bit_bounds),
            _normalize(entry["avg_column_pruning_ratio"], pruning_bounds),
        )
        distance = math.hypot(target[0] - point[0], target[1] - point[1])
        if distance < 1e-6:
            return entry["accuracy"], 0.0
        weights.append(1.0 / (distance**2))
        values.append(entry["accuracy"])

    if not weights:
        return 0.5, 1.0

    total_weight = sum(weights)
    predicted = sum(w * v for w, v in zip(weights, values)) / total_weight
    uncertainty = 1.0 / (1.0 + total_weight)
    return predicted, uncertainty


def propose_via_surrogate(
    history: Sequence[Dict[str, Any]],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
    grid_size: int = 9,
    exploration_weight: float = 0.3,
) -> Tuple[float, float]:
    """Scans a `grid_size` x `grid_size` grid over the valid search space and
    returns the point maximizing `predicted_accuracy + exploration_weight *
    uncertainty` -- a standard upper-confidence-bound acquisition rule over
    the IDW surrogate in `predict_accuracy`."""
    bit_lo, bit_hi = bit_bounds
    pruning_lo, pruning_hi = pruning_bounds

    best_score = -math.inf
    best_point = (bit_lo, pruning_lo)
    for i in range(grid_size):
        bits = bit_lo + (bit_hi - bit_lo) * i / (grid_size - 1)
        for j in range(grid_size):
            pruning = pruning_lo + (pruning_hi - pruning_lo) * j / (grid_size - 1)
            predicted, uncertainty = predict_accuracy(bits, pruning, history, bit_bounds, pruning_bounds)
            score = predicted + exploration_weight * uncertainty
            if score > best_score:
                best_score = score
                best_point = (bits, pruning)
    return best_point


# ---------------------------------------------------------------------------
# Per-stage (n-dimensional) surrogate-guided phase: IDW regression over the
# full per-stage vector + random-search UCB acquisition
# ---------------------------------------------------------------------------


def _stage_vector_from_layer_configs(
    layer_configs: Any, stage_names: Sequence[str]
) -> "StagePoint | None":
    """Recovers the (weight_bits, column_pruning_ratio) pair per stage from
    a `candidate_history` entry's stored `layer_configs` (the exact
    validated per-stage config @tuner used that iteration -- echoed through
    `TunerToolOutput.data`, see tools/simulators.py), in `stage_names`
    order. Returns None if this entry doesn't cover exactly `stage_names`
    (a different model_id's history, or an entry recorded before this field
    existed) -- such entries are real data, just not comparable point-for-
    point to the current search space, so `predict_accuracy_stages` skips
    them rather than guessing a partial vector."""
    if not layer_configs:
        return None
    by_name = {entry["layer_name"]: entry for entry in layer_configs}
    if set(by_name) != set(stage_names):
        return None
    return [(by_name[name]["weight_bits"], by_name[name]["column_pruning_ratio"]) for name in stage_names]


def _normalize_stage_point(point: StagePoint, bit_bounds: Bounds, pruning_bounds: Bounds) -> Tuple[float, ...]:
    flat: List[float] = []
    for bits, pruning in point:
        flat.append(_normalize(bits, bit_bounds))
        flat.append(_normalize(pruning, pruning_bounds))
    return tuple(flat)


def _predict_stage_value(
    point: StagePoint,
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
    value_key: str,
    neutral_default: float,
) -> Tuple[float, float]:
    """The shared inverse-distance-weighted regression behind
    `predict_accuracy_stages`/`predict_energy_stages`/`predict_latency_stages`
    -- same math, just fit against a different `candidate_history` field
    (`value_key`). A history entry missing `value_key` (e.g. a failed
    candidate that never reached @profiler) is skipped for *this*
    objective's fit only -- it can still count for the other two objectives
    via the other calls, same as a stage-name mismatch is skipped only for
    the entries that don't cover `stage_names`. `neutral_default` differs
    per objective since only accuracy has a natural bounded range: 0.5 is a
    reasonable "no idea" prior on [0, 1], but energy_pj/noc_latency_ms have
    no such range, so 0.0 with maximal uncertainty is used instead --
    harmless in practice, since a real run always has warm-up history by
    the time this is called (nodes/planner.py only reaches the surrogate
    phase once `len(history) >= n_warmup`)."""
    target = _normalize_stage_point(point, bit_bounds, pruning_bounds)
    weights: List[float] = []
    values: List[float] = []
    for entry in history:
        if entry.get(value_key) is None:
            continue
        vector = _stage_vector_from_layer_configs(entry.get("layer_configs"), stage_names)
        if vector is None:
            continue
        entry_point = _normalize_stage_point(vector, bit_bounds, pruning_bounds)
        distance = math.sqrt(sum((t - p) ** 2 for t, p in zip(target, entry_point)))
        if distance < 1e-6:
            return entry[value_key], 0.0
        weights.append(1.0 / (distance**2))
        values.append(entry[value_key])

    if not weights:
        return neutral_default, 1.0

    total_weight = sum(weights)
    predicted = sum(w * v for w, v in zip(weights, values)) / total_weight
    uncertainty = 1.0 / (1.0 + total_weight)
    return predicted, uncertainty


def predict_accuracy_stages(
    point: StagePoint,
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
) -> Tuple[float, float]:
    """The n-dimensional analogue of `predict_accuracy`: same
    inverse-distance-weighted regression, just over the full
    `2 * len(stage_names)`-dimensional normalized point (built from each
    history entry's stored per-stage `layer_configs` via
    `_stage_vector_from_layer_configs`) instead of one flat 2D point."""
    return _predict_stage_value(point, history, stage_names, bit_bounds, pruning_bounds, "accuracy", 0.5)


def predict_energy_stages(
    point: StagePoint,
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
) -> Tuple[float, float]:
    """`predict_accuracy_stages`'s sibling for `energy_pj` -- lower is
    better (this project's minimization convention), unlike accuracy."""
    return _predict_stage_value(point, history, stage_names, bit_bounds, pruning_bounds, "energy_pj", 0.0)


def predict_latency_stages(
    point: StagePoint,
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
) -> Tuple[float, float]:
    """`predict_accuracy_stages`'s sibling for `noc_latency_ms` -- lower is
    better, same as `predict_energy_stages`."""
    return _predict_stage_value(point, history, stage_names, bit_bounds, pruning_bounds, "noc_latency_ms", 0.0)


class _SurrogateMultiObjectiveProblem(Problem):
    """Wraps the three IDW surrogates above as a real 3-objective `pymoo`
    problem: accuracy, energy_pj, noc_latency_ms, all three at once --
    `NSGA2` explores over these surrogate *predictions* (cheap IDW math),
    never a real QAT trial, to find where the next real trial should go.
    Objectives are UCB/LCB-adjusted the same way the old single-objective
    scan was: accuracy is negated (pymoo always minimizes, so maximizing
    accuracy is minimizing its negation) and its exploration bonus
    subtracted from that negation (i.e. added to the accuracy estimate);
    energy_pj/noc_latency_ms are already a minimization objective, so their
    exploration bonus is subtracted directly -- an optimistic (low) bound
    on both, encouraging trying under-sampled regions same as accuracy's
    optimistic (high) bound does."""

    def __init__(
        self,
        history: Sequence[Dict[str, Any]],
        stage_names: Sequence[str],
        bit_bounds: Bounds,
        pruning_bounds: Bounds,
        exploration_weight: float,
    ) -> None:
        self._history = history
        self._stage_names = stage_names
        self._bit_bounds = bit_bounds
        self._pruning_bounds = pruning_bounds
        self._exploration_weight = exploration_weight
        n_dims = 2 * len(stage_names)
        xl = np.array([bit_bounds[0] if d % 2 == 0 else pruning_bounds[0] for d in range(n_dims)])
        xu = np.array([bit_bounds[1] if d % 2 == 0 else pruning_bounds[1] for d in range(n_dims)])
        super().__init__(n_var=n_dims, n_obj=3, n_constr=0, xl=xl, xu=xu)

    def row_to_point(self, row: "np.ndarray") -> StagePoint:
        return [(float(row[2 * s]), float(row[2 * s + 1])) for s in range(len(self._stage_names))]

    def _evaluate(self, x: "np.ndarray", out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        f = np.zeros((x.shape[0], 3))
        for i, row in enumerate(x):
            point = self.row_to_point(row)
            accuracy, accuracy_unc = predict_accuracy_stages(
                point, self._history, self._stage_names, self._bit_bounds, self._pruning_bounds
            )
            energy, energy_unc = predict_energy_stages(
                point, self._history, self._stage_names, self._bit_bounds, self._pruning_bounds
            )
            latency, latency_unc = predict_latency_stages(
                point, self._history, self._stage_names, self._bit_bounds, self._pruning_bounds
            )
            f[i, 0] = -(accuracy + self._exploration_weight * accuracy_unc)
            f[i, 1] = energy - self._exploration_weight * energy_unc
            f[i, 2] = latency - self._exploration_weight * latency_unc
        out["F"] = f


def propose_stage_candidate_random(
    stage_names: Sequence[str], bit_bounds: Bounds, pruning_bounds: Bounds, seed: int
) -> StagePoint:
    """Random-search ablation baseline (`tools/run_ablation.py`): ignores
    `history` entirely, uniform-random within bounds every call -- no
    surrogate, no acquisition rule, just "try something valid." The floor
    a real acquisition strategy (single- or multi-objective surrogate-
    guided) needs to actually beat for "the search helps" to be a
    demonstrated claim rather than an assumed one."""
    rng = random.Random(seed)
    return [(rng.uniform(*bit_bounds), rng.uniform(*pruning_bounds)) for _ in stage_names]


def propose_stage_candidate_single_objective(
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
    seed: int,
    n_candidates: int = 200,
    exploration_weight: float = 0.3,
) -> StagePoint:
    """Single-objective ablation baseline (`tools/run_ablation.py`): the
    acquisition rule this project actually shipped before the multi-
    objective NSGA-II rewrite (this module's own docstring) --
    `predict_accuracy_stages`'s IDW surrogate, accuracy only, energy/latency
    left entirely out of the search itself (only ever used *after* the
    fact by `compute_pareto_rank`). That prior version was confirmed in
    production to plateau at the best warm-up candidate's accuracy, with
    nothing steering search toward the accuracy/energy/latency Pareto
    front -- this function reproduces that same single-objective behavior
    on demand, deliberately, so the current multi-objective version can be
    compared against it rather than only against its own history.

    UCB over `n_candidates` random points, not a grid: `propose_via_surrogate`'s
    2D version can afford a `grid_size x grid_size` grid, but a grid is
    infeasible once `2 * n_stages` climbs past a handful of dimensions
    (resnet18 alone is already 20-dimensional). Real random sampling +
    UCB selection is the same acquisition *rule* as the multi-objective
    version's NSGA-II step, just without a population-based search over it."""
    rng = random.Random(seed)
    best_score = -math.inf
    best_point: StagePoint = [(bit_bounds[0], pruning_bounds[0]) for _ in stage_names]
    for _ in range(n_candidates):
        point = [(rng.uniform(*bit_bounds), rng.uniform(*pruning_bounds)) for _ in stage_names]
        predicted, uncertainty = predict_accuracy_stages(point, history, stage_names, bit_bounds, pruning_bounds)
        score = predicted + exploration_weight * uncertainty
        if score > best_score:
            best_score = score
            best_point = point
    return best_point


def propose_stage_candidates_via_surrogate(
    history: Sequence[Dict[str, Any]],
    stage_names: Sequence[str],
    bit_bounds: Bounds,
    pruning_bounds: Bounds,
    seed: int,
    n_candidates: int = 200,
    exploration_weight: float = 0.3,
    n_generations: int = 30,
) -> StagePoint:
    """Genuinely multi-objective acquisition: runs real NSGA-II (`pymoo`)
    against `_SurrogateMultiObjectiveProblem` -- `n_candidates` is now the
    GA's population size per generation (kept under its original name since
    callers/tests already pass it; it plays the same "how many points does
    this acquisition step consider" role as the old random-sample count),
    evolved for `n_generations` generations. This produces a well-spread
    Pareto-optimal *set* over the surrogate's predictions, not a single
    point -- one point is picked from that set uniformly at random (seeded,
    so still deterministic in `seed` like every other sampler in this
    module) as the next real candidate to evaluate. Falls back to a single
    uniform-random point in bounds only if NSGA-II returns no solutions at
    all (defensive; shouldn't happen for this unconstrained problem)."""
    problem = _SurrogateMultiObjectiveProblem(history, stage_names, bit_bounds, pruning_bounds, exploration_weight)
    algorithm = NSGA2(pop_size=n_candidates)
    result = minimize(problem, algorithm, ("n_gen", n_generations), seed=seed, verbose=False)

    rng = random.Random(seed)
    pareto_x = result.X
    if pareto_x is None or len(pareto_x) == 0:
        return [(rng.uniform(*bit_bounds), rng.uniform(*pruning_bounds)) for _ in stage_names]
    chosen_row = pareto_x[rng.randrange(len(pareto_x))]
    return problem.row_to_point(chosen_row)


# ---------------------------------------------------------------------------
# Real Pareto rank (NSGA-II-style non-dominated sorting), computed online
# ---------------------------------------------------------------------------


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """`a` dominates `b`: at least as good on every objective (higher
    accuracy, lower energy, lower latency) and strictly better on at least
    one."""
    not_worse = a["accuracy"] >= b["accuracy"] and a["energy_pj"] <= b["energy_pj"] and a["noc_latency_ms"] <= b["noc_latency_ms"]
    strictly_better = a["accuracy"] > b["accuracy"] or a["energy_pj"] < b["energy_pj"] or a["noc_latency_ms"] < b["noc_latency_ms"]
    return not_worse and strictly_better


def compute_pareto_rank(candidate: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> int:
    """1 = non-dominated (Pareto-optimal) against every entry in `history`;
    otherwise `1 + max(dominator's own pareto_rank)` -- an incremental
    approximation of full non-dominated sorting appropriate for the
    sequential (one-candidate-at-a-time) search loop this project runs,
    rather than batch-sorting a whole generation at once as textbook
    NSGA-II does."""
    dominating = [entry for entry in history if _dominates(entry, candidate)]
    if not dominating:
        return 1
    return 1 + max(entry.get("pareto_rank", 1) for entry in dominating)


def best_known_candidate(
    candidate_history: Sequence[Dict[str, Any]], session_is_converged: bool = False
) -> Optional[Dict[str, Any]]:
    """The candidate a caller should treat as this session's answer.

    `session_is_converged=True` means some entry actually met every
    configured target. Under the default satisficing stopping rule
    (`search_budget_iterations` unset, state.py) that entry is
    `candidate_history[-1]`: the graph stops immediately after the first
    hit, so nothing is appended afterward and the last entry is the answer
    by construction. Under a budget the search deliberately keeps going
    past its first hit, so the last entry is merely the last one *tried* --
    the answer is the best of every entry that met the targets, which is
    what the `met` branch below selects. That branch reduces to
    `candidate_history[-1]` in the satisficing case (only one entry is ever
    flagged), so both rules share one implementation rather than diverging.

    "Best" here is highest accuracy among target-meeting candidates -- the
    same rule `tools/baseline_compare.py`'s `best_grid_point` applies to the
    grid. Matching it deliberately: comparing a search that stopped early
    against a grid that reported its maximum was the asymmetry that made
    the two columns unanswerable in the first place.

    Note this trusts each entry's own `is_converged` field to mean "met
    every configured target". That is true of @evaluator's entries and --
    since `tools/batch_warmup.py` was made to apply `check_targets` too --
    of warm-up entries as well. Before that, warm-up flagged candidates on
    @verifier's physical check alone, so an entry could carry
    `is_converged=True` having never come near the accuracy target; a
    "best among flagged" rule would have silently selected one.

    `False` (including the
    `low_confidence_conclusion` case -- see state.py's field docstring)
    means no entry is known to have met the targets, so this returns the
    highest-accuracy attempt across the *entire* history instead -- a
    session that never converged, or gave up via hitl_node's
    conclude-with-best path, still has a real "best it found" worth
    reporting, not just whatever its last (likely failing) iteration was.

    Callers must pass `session_is_converged` as `state["is_converged"] and
    not state.get("low_confidence_conclusion")`, never `state["is_converged"]`
    alone -- hitl_node's low-confidence path also sets `is_converged=True`
    (so @evaluator_router still routes to `Converged (Done)` / `END`), but
    its `candidate_history[-1]` is whatever the last, likely-failing search
    iteration happened to produce, not the selected answer.

    Originally `tools/repeat_search.py`'s private `_best_candidate` (moved
    here so every consumer that has to answer "which candidate was this
    session's answer" selects the same way instead of duplicating or
    silently diverging: `main.py`'s `print_final_state`,
    `tools/repeat_search.py`, and `tools/optuna_compare.py`). Note that
    `nodes/hitl.py` is deliberately *not* one of them -- its
    conclude-with-best path only records the `low_confidence_conclusion`
    flag on state, leaving the actual selection to these reporting
    consumers, so no candidate is ever picked twice by two different
    rules."""
    if session_is_converged and candidate_history:
        met = [c for c in candidate_history if c.get("is_converged")]
        # Falls back to the last entry when nothing is flagged: a caller
        # asserting convergence over a history that records none is a
        # contradiction this function must not resolve by inventing a
        # winner, and the last entry is what it returned before.
        return max(met, key=lambda c: c.get("accuracy") or 0.0) if met else candidate_history[-1]
    if candidate_history:
        return max(candidate_history, key=lambda c: c.get("accuracy") or 0.0)
    return None
