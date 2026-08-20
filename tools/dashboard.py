"""Static HTML dashboard for one AutoCIM-Agent run (state.py's `AutoCIMState`).

Real multi-researcher use needs more than `state["messages"]`'s free text
to build trust in / debug a run: which candidate was tried each iteration
and *why* (`planner_decisions`, appended by `nodes/planner.py`), what it
actually measured (`candidate_history`, appended by `nodes/evaluator.py`),
and how many LLM calls it took, including retries/failures (`llm_usage`).
This module joins those three already-structured `AutoCIMState` fields -- the authoritative
numeric record -- into one self-contained HTML report.

`observability.py`'s JSON-Lines log is a separate, narrative/audit channel
(when/in-what-order things happened) -- this module reads `AutoCIMState`
directly, not that log, since state is the source of truth for the actual
numbers.

No charting dependency: the Pareto-front-movement chart is hand-rolled
inline SVG (a handful of <circle>/<polyline> elements), consistent with
this project's "no new heavy dependency" stance elsewhere (tools/search.py).
"""

from __future__ import annotations

import html
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from schemas.config import HWConfig
from state import AutoCIMState


def _iteration_rows(state: AutoCIMState) -> List[Dict[str, Any]]:
    """One row per evaluated candidate, joining `candidate_history`
    (what happened) against `planner_decisions` (why it was tried) by
    `iteration` -- the two lists are appended by different nodes
    (nodes/evaluator.py, nodes/planner.py) so neither alone has both."""
    decisions = {d["iteration"]: d for d in state.get("planner_decisions", [])}
    rows = []
    for candidate in state.get("candidate_history", []):
        iteration = candidate.get("iteration")
        decision = decisions.get(iteration, {})
        rows.append(
            {
                **candidate,
                "search_tag": decision.get("search_tag"),
                "rationale": decision.get("rationale"),
                "rationale_ko": decision.get("rationale_ko"),
                "used_llm": decision.get("used_llm"),
            }
        )
    return rows


_WARMUP_TAG_RE = re.compile(r"^LHS warm-up candidate (\d+)/(\d+)$")
_SURROGATE_TAG_RE = re.compile(r"^surrogate-model acquisition over (\d+) prior candidates$")


def _bi(en: str, ko: str) -> str:
    """Wraps static UI prose (never LLM/data-derived text -- rationale,
    search_tag, model_id/hw_spec_id stay in their original language,
    see render_dashboard_html) in both languages; the JS toggle in
    render_dashboard_html's <script> flips which one has the `hidden`
    attribute. Both are always in the DOM so switching language doesn't
    reflow/lengthen the page -- only the previously-hidden span un-hides
    in place of the one that just hid."""
    return f'<span data-lang="en">{en}</span><span data-lang="ko" hidden>{ko}</span>'


def _search_phase_summary_html(rows: List[Dict[str, Any]]) -> str:
    """Explains *why* the per-iteration table below can show the same
    "LHS warm-up candidate" search phase many times in a row: warm-up is a
    fixed-size, uniform space-filling sample (`nodes/planner.py`'s
    `warmup_count`) whose total is baked into its own tag text ("X/N"),
    parsed back out here rather than re-derived from model/hw config (this
    module intentionally has no dependency on that, see module docstring)
    -- so a reader watching a run doesn't have to infer "when does this
    stop" from a wall of repeated rows alone. Deliberately does not invent
    a total for the optimization phase that follows warm-up: it really has
    none (runs until a convergence target is hit or the run is stopped)."""
    ordered = sorted(rows, key=lambda r: r["iteration"]) if rows else []
    if not ordered:
        return f'<p class="phase-note">{_bi("No candidates evaluated yet.", "아직 평가된 후보가 없습니다.")}</p>'

    latest = ordered[-1]
    tag = str(latest.get("search_tag") or "")
    warmup_match = _WARMUP_TAG_RE.match(tag)
    surrogate_match = _SURROGATE_TAG_RE.match(tag)

    if warmup_match:
        done, total = int(warmup_match.group(1)), int(warmup_match.group(2))
        pct = min(100, round(done / total * 100)) if total else 0
        title_en = f"Warm-up phase &mdash; {done}/{total} candidates ({pct}%)"
        title_ko = f"워밍업(warm-up) 단계 &mdash; {done}/{total} 완료 ({pct}%)"
        note_en = (
            "Warm-up uniformly samples the search space and always runs exactly "
            f"<strong>{total}</strong> times for this model/hardware, regardless of how it's going &mdash; "
            "that count is fixed up front, not a sign of trouble. Once it finishes, the optimizer switches "
            "to surrogate-model-guided search, which has <strong>no fixed total</strong>: it keeps proposing "
            "candidates until a convergence target is met or the run is stopped."
        )
        note_ko = (
            "워밍업은 탐색 공간을 균일하게 샘플링하는 단계로, 진행 상황과 무관하게 이 모델/하드웨어 기준 "
            f"항상 정확히 <strong>{total}</strong>번 실행됩니다 &mdash; 미리 정해진 횟수일 뿐, 문제가 있다는 "
            "신호가 아닙니다. 이 단계가 끝나면 서로게이트 모델 기반 탐색으로 전환되며, 그 단계는 "
            "<strong>정해진 총 횟수가 없습니다</strong>: 수렴 목표를 만족하거나 실행이 중단될 때까지 후보 "
            "제안을 계속합니다."
        )
        return (
            '<div class="phase-box phase-warmup">'
            f'<div class="phase-title">{_bi(title_en, title_ko)}</div>'
            f'<div class="phase-bar"><div class="phase-bar-fill" style="width:{pct}%"></div></div>'
            f'<p class="phase-note">{_bi(note_en, note_ko)}</p>'
            "</div>"
        )
    if surrogate_match:
        n_prior = int(surrogate_match.group(1))
        title_en = f"Optimization phase &mdash; surrogate-model-guided search ({n_prior} prior candidate(s) so far)"
        title_ko = f"최적화 단계 &mdash; 서로게이트 모델 기반 탐색 (지금까지 이전 후보 {n_prior}개 반영)"
        note_en = (
            "Warm-up is complete. From here there is no fixed iteration count: the optimizer keeps proposing "
            "candidates until a convergence target is met (if one was set) or the run is stopped."
        )
        note_ko = (
            "워밍업이 완료되었습니다. 이제부터는 정해진 반복 횟수가 없습니다: 수렴 목표가 설정되어 있다면 "
            "이를 만족하거나, 실행이 중단될 때까지 최적화 탐색을 계속합니다."
        )
        return (
            '<div class="phase-box phase-search">'
            f'<div class="phase-title">{_bi(title_en, title_ko)}</div>'
            f'<p class="phase-note">{_bi(note_en, note_ko)}</p>'
            "</div>"
        )
    return ""


def _is_warmup_row(row: Dict[str, Any]) -> bool:
    return bool(_WARMUP_TAG_RE.match(str(row.get("search_tag") or "")))


def _pareto_front_2d(plottable: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The non-dominated set on the two *plotted* axes alone (lower
    energy_pj better, higher accuracy better), energy-ascending.

    Deliberately not `pareto_rank`: that field is computed over three
    objectives (accuracy, energy, latency), so on a real 18-candidate
    session 13 candidates carried rank 1 -- colouring by it marked almost
    everything, which is exactly the "front" a reader cannot see. Restricted
    to the two axes actually on screen, the same session yields 6 points,
    and they form a staircase a viewer can read off directly. The table
    below still reports the real 3-objective `pareto_rank`; the legend says
    so, since the two genuinely disagree and hiding that would be worse
    than explaining it."""
    front: List[Dict[str, Any]] = []
    best_accuracy: Optional[float] = None
    # -accuracy breaks energy ties in favour of the more accurate candidate,
    # so the tied-but-worse one is correctly dropped by the strict >.
    for row in sorted(plottable, key=lambda r: (r["energy_pj"], -r["accuracy"])):
        if best_accuracy is None or row["accuracy"] > best_accuracy:
            best_accuracy = row["accuracy"]
            front.append(row)
    return front


def _axis_ticks(low: float, high: float, n: int = 4) -> List[float]:
    return [low] if high == low else [low + (high - low) * i / (n - 1) for i in range(n)]


def _extend_for_target(low: float, high: float, target: Optional[float]) -> Tuple[float, float]:
    """Widens an axis domain so a target reference line lands inside the
    plot -- but only when the target is at most one full data span outside
    it. A target far away from everything measured (e.g. an energy budget
    100x below any candidate) would otherwise squash every real point into
    a single pixel column; in that case the domain is left alone and the
    line is simply not drawn (`_svg_pareto_chart` range-checks before
    emitting it), which is the honest rendering of "nothing on this chart
    is anywhere near the target"."""
    if target is None:
        return low, high
    span = high - low
    slack = span if span > 0 else (abs(low) * 0.1 or 1.0)
    if low - slack <= target <= high + slack:
        return min(low, target), max(high, target)
    return low, high


def _svg_pareto_chart(
    rows: List[Dict[str, Any]],
    targets: Optional[Dict[str, Optional[float]]] = None,
    width: int = 720,
    height: int = 430,
    pad: int = 58,
) -> str:
    """energy_pj (x) vs accuracy (y, inverted so higher accuracy trends
    upward), one point per iteration; points on the two-axis Pareto front
    (`_pareto_front_2d`) are highlighted and joined by a staircase.

    The staircase is the front's actual meaning made visible: each riser is
    the extra energy a candidate has to pay to reach the next accuracy
    level. Scattered highlighted dots stated the same data but left the
    reader to trace the frontier by eye.

    `targets` ({"accuracy": ..., "energy_pj": ...}, either key optional)
    draws the run's configured convergence targets as dashed reference
    lines, so "did any candidate actually clear the bar" is answerable from
    the chart rather than only from the table.

    **The trajectory line covers the search phase only.** Warm-up points are
    Latin Hypercube samples -- deliberately spread across the space with no
    meaningful order between them -- so joining them by iteration index drew
    noise, not a trajectory: on a real 18-candidate session (15 warm-up) the
    polyline ran 3401px across a 466px plot diagonal, crossing itself
    roughly seven times. Warm-up is now drawn as unconnected hollow marks,
    which also surfaces something the old chart hid: which candidates were
    random exploration versus surrogate-guided search.

    Only search-phase points and the final candidate carry a printed
    iteration number. Labelling all 18 put 8 pairs of labels within 24px of
    each other (closest 7.3px) on the same real session. Warm-up points are
    deliberately left unlabelled for the same reason their line was dropped:
    an LHS sample's index carries no information, so printing it is noise --
    and front membership, the thing that *is* worth seeing, is already
    carried by colour. (Labelling every Pareto-rank-1 point instead was
    tried and rejected: 13 of these 18 candidates are rank 1, since the rank
    is computed over three objectives, so it barely filtered anything.)
    Every point still reports its full detail through the `<title>` tooltip.

    Axis label text is anchored in the margin outside the
    [pad, width-pad] x [pad, height-pad] plot rectangle (the y-axis label
    rotated, centered on the left margin) rather than at a fixed (x, y)
    inside it -- the old top-left placement sat at exactly (pad, pad),
    which is also where the min-energy/max-accuracy point can land, so the
    label text and a data point could overlap."""
    plottable = [r for r in rows if r.get("accuracy") is not None and r.get("energy_pj") is not None]
    if not plottable:
        return "<p>No evaluated candidates yet.</p>"

    accuracies = [r["accuracy"] for r in plottable]
    energies = [r["energy_pj"] for r in plottable]
    targets = targets or {}
    target_accuracy, target_energy = targets.get("accuracy"), targets.get("energy_pj")
    acc_min, acc_max = _extend_for_target(min(accuracies), max(accuracies), target_accuracy)
    e_min, e_max = _extend_for_target(min(energies), max(energies), target_energy)

    def x(energy: float) -> float:  # energy_pj increases left -> right
        if e_max == e_min:
            return width / 2
        return pad + (energy - e_min) / (e_max - e_min) * (width - 2 * pad)

    def y(accuracy: float) -> float:  # higher accuracy is better -> plotted higher
        if acc_max == acc_min:
            return height / 2
        return height - pad - (accuracy - acc_min) / (acc_max - acc_min) * (height - 2 * pad)

    ordered = sorted(plottable, key=lambda r: r["iteration"])
    searched = [r for r in ordered if not _is_warmup_row(r)]
    last_iteration = ordered[-1]["iteration"]
    front_iterations = {r["iteration"] for r in _pareto_front_2d(plottable)}

    # Plot frame + gridded tick values: without them the axes carried no
    # magnitude at all (a reader could not tell 1.6M pJ from 16 pJ), and with
    # only the two end values there was nothing to read an interior point
    # against -- the gap between two candidates was eyeballed, not measured.
    axis = [
        f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" '
        f'fill="none" stroke="#2b3a4a" stroke-width="1"/>'
    ]
    energy_ticks = _axis_ticks(e_min, e_max)
    for i, tick in enumerate(energy_ticks):
        tx = x(tick)
        # The end ticks are anchored inward: centred, the leftmost 7-digit
        # energy value ran under the y-axis tick column (15px apart on a
        # real session) and the rightmost overflowed the plot's right edge.
        anchor = "start" if i == 0 else "end" if i == len(energy_ticks) - 1 else "middle"
        axis.append(
            f'<line x1="{tx:.1f}" y1="{pad}" x2="{tx:.1f}" y2="{height - pad}" stroke="#1b2532" stroke-width="1"/>'
            f'<text x="{tx:.1f}" y="{height - pad + 16}" font-size="10" fill="#93a4b8" '
            f'text-anchor="{anchor}">{tick:,.0f}</text>'
        )
    for tick in _axis_ticks(acc_min, acc_max):
        ty = y(tick)
        axis.append(
            f'<line x1="{pad}" y1="{ty:.1f}" x2="{width - pad}" y2="{ty:.1f}" stroke="#1b2532" stroke-width="1"/>'
            f'<text x="{pad - 8}" y="{ty + 3:.1f}" font-size="10" fill="#93a4b8" '
            f'text-anchor="end">{tick:.3f}</text>'
        )

    # Configured convergence targets as dashed reference lines. Range-checked
    # because _extend_for_target deliberately refuses to widen the domain for
    # a target that is nowhere near the measured data.
    if target_accuracy is not None and acc_min <= target_accuracy <= acc_max:
        ty = y(target_accuracy)
        axis.append(
            f'<line x1="{pad}" y1="{ty:.1f}" x2="{width - pad}" y2="{ty:.1f}" stroke="#e0b354" '
            f'stroke-width="1.5" stroke-dasharray="6 4"/>'
            f'<text x="{width - pad - 4}" y="{ty - 6:.1f}" font-size="10" fill="#e0b354" '
            f'text-anchor="end">target accuracy {target_accuracy:.3f}</text>'
        )
    if target_energy is not None and e_min <= target_energy <= e_max:
        tx = x(target_energy)
        axis.append(
            f'<line x1="{tx:.1f}" y1="{pad}" x2="{tx:.1f}" y2="{height - pad}" stroke="#e0b354" '
            f'stroke-width="1.5" stroke-dasharray="6 4"/>'
            f'<text x="{tx + 5:.1f}" y="{pad + 12}" font-size="10" fill="#e0b354" '
            f'text-anchor="start">target energy {target_energy:,.0f}</text>'
        )

    # The front itself, as a staircase: each riser is the extra energy the
    # next accuracy level costs.
    front_line = ""
    front_points = _pareto_front_2d(plottable)
    if len(front_points) > 1:
        steps: List[str] = []
        previous_y: Optional[float] = None
        for r in front_points:
            px, py = x(r["energy_pj"]), y(r["accuracy"])
            if previous_y is not None:
                steps.append(f"{px:.1f},{previous_y:.1f}")  # tread: pay energy at the old accuracy
            steps.append(f"{px:.1f},{py:.1f}")  # riser: the accuracy that energy buys
            previous_y = py
        front_line = (
            f'<polyline points="{" ".join(steps)}" fill="none" stroke="#5b9dff" stroke-width="2" opacity="0.85"/>'
        )

    trajectory = ""
    if len(searched) > 1:
        path = " ".join(f"{x(r['energy_pj']):.1f},{y(r['accuracy']):.1f}" for r in searched)
        trajectory = (
            f'<polyline points="{path}" fill="none" stroke="#4a6b8a" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )

    marks = []
    # Two candidates can land on the same point -- iterations 16 and 17 of a
    # real session shared an energy figure exactly and differed by 0.004
    # accuracy, putting their two labels 3.4px apart and rendering both
    # unreadable. Each label is pushed straight down until it clears every
    # label already placed.
    placed_labels: List[Tuple[float, float]] = []
    for r in ordered:
        cx, cy = x(r["energy_pj"]), y(r["accuracy"])
        on_front = r["iteration"] in front_iterations
        warmup = _is_warmup_row(r)
        tooltip = (
            f"iter {r['iteration']} ({'warm-up' if warmup else 'search'}): "
            f"energy={r['energy_pj']:,.2f}pJ, accuracy={r['accuracy']:.3f}, "
            f"3-objective rank={r.get('pareto_rank')}"
        )
        color = "#5b9dff" if on_front else "#5c6b7e"
        if on_front:
            # Front membership is the one distinction this chart exists to
            # show, and it used to be carried by hue alone -- #5b9dff against
            # #5c6b7e is only 2.00:1, so a projector washing out the blues, a
            # greyscale handout, or a red-green colour vision deficiency
            # erases the front entirely. This concentric ring restates it as
            # a shape difference; the colour now reinforces rather than
            # carries it. Drawn before the marker so it sits behind.
            marks.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{6.5 if warmup else 8.5:.1f}" fill="none" '
                f'stroke="{color}" stroke-width="1.2" opacity="0.55"/>'
            )
        if warmup:
            shape = (
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="none" '
                f'stroke="{color}" stroke-width="1.5" opacity="{1.0 if on_front else 0.7}">'
            )
        else:
            shape = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" opacity="0.95">'
        marks.append(f"{shape}<title>{html.escape(tooltip)}</title></circle>")
        if not warmup or r["iteration"] == last_iteration:
            # Clear the marker's real outer edge, which the front ring above
            # widens to 8.5px -- a fixed +10 left the digits sitting on the
            # ring stroke rather than beside it.
            lx, ly = cx + (12.0 if on_front else 10.0), cy + 4
            while any(abs(lx - px) < 20 and abs(ly - py) < 11 for px, py in placed_labels):
                ly += 11
            placed_labels.append((lx, ly))
            # paint-order=stroke draws a panel-coloured halo *behind* the
            # glyphs: displaced labels land on the trajectory/staircase they
            # were displaced away from, and a line running through a two-digit
            # number at 10px makes it unreadable.
            marks.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10" fill="#c8d3e0" '
                f'stroke="#111821" stroke-width="3" paint-order="stroke">{r["iteration"]}</text>'
            )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" '
        # Capped well above the viewBox width and centred: pinned to `width`
        # the chart sat in the left half of a 1340px-wide panel with the rest
        # blank, and every tick/label rendered at its minimum legible size.
        f'style="max-width:{width * 1.35:.0f}px;display:block;margin:0 auto" role="img" '
        f'aria-label="energy vs accuracy per iteration; filled marks are surrogate-guided search '
        f'candidates joined in order, hollow marks are unordered LHS warm-up samples, '
        f'blue marks and the staircase are the energy-accuracy Pareto front, '
        f'dashed amber lines are this run\'s configured targets">'
        + "".join(axis)
        + trajectory
        + front_line
        + "".join(marks)
        + f'<text x="{width / 2:.1f}" y="{height - 8}" font-size="11" fill="#93a4b8" '
        f'text-anchor="middle">energy_pj &#8594; (lower=better)</text>'
        + f'<text x="14" y="{height / 2:.1f}" font-size="11" fill="#93a4b8" text-anchor="middle" '
        f'transform="rotate(-90 14 {height / 2:.1f})">accuracy &#8593;</text>'
        + "</svg>"
    )


def _pareto_legend_html(has_targets: bool) -> str:
    """Separate from `_svg_pareto_chart` since these are plain HTML
    swatches, not SVG -- simpler to keep them (and their bilingual labels,
    `_bi`) out of the hand-rolled SVG string entirely.

    Ends with an explicit note that the chart's front and the table's
    `pareto_rank` are computed over different objective sets, because on a
    real session they visibly disagree (see `_pareto_front_2d`) and a reader
    comparing the two would otherwise conclude one of them is wrong."""
    target_item = (
        '<span class="legend-item"><span class="legend-swatch" '
        'style="background:transparent;border-top:2px dashed #e0b354;border-radius:0;height:0"></span>'
        f'{_bi("Dashed amber: configured target", "노란 점선: 이번 실행에 설정된 목표값")}</span>'
        if has_targets
        else ""
    )
    return (
        '<div class="legend">'
        # The ring is named before the colour: it is the channel that
        # survives a washed-out projector or a greyscale print, so a reader
        # who cannot separate the two blues still has a stated cue.
        '<span class="legend-item"><span class="legend-swatch legend-ring" '
        'style="background:#5b9dff"></span>'
        f'{_bi("Ringed: on the energy-accuracy Pareto front", "테두리 있는 점: 에너지-정확도 파레토 프론트 위의 후보")}</span>'
        '<span class="legend-item"><span class="legend-swatch" style="background:#5c6b7e"></span>'
        f'{_bi("Dominated on these two axes", "이 두 축 기준으로 피지배인 후보")}</span>'
        '<span class="legend-item"><span class="legend-swatch" '
        'style="background:transparent;border:1.5px solid #5c6b7e"></span>'
        f'{_bi("Hollow: LHS warm-up (unordered samples)", "빈 원: LHS 워밍업 (표본 간 순서 없음)")}</span>'
        '<span class="legend-item"><span class="legend-swatch" '
        'style="background:#5b9dff;border-radius:0;height:2px"></span>'
        f'{_bi("Staircase: the front itself", "계단선: 파레토 프론트")}</span>'
        '<span class="legend-item"><span class="legend-swatch" '
        'style="background:transparent;border-top:2px dashed #4a6b8a;border-radius:0;height:0"></span>'
        f'{_bi("Dashed: surrogate-guided search order", "점선: 서로게이트 기반 탐색 순서")}</span>'
        f"{target_item}"
        "</div>"
        f'<p class="summary">{_bi("The front drawn here is over the two plotted axes only. The pareto_rank column "
                                 "in the table below is computed over three objectives (accuracy, energy, latency), "
                                 "so a candidate can be rank 1 there and still sit below this staircase.",
                                 "이 그래프의 프론트는 화면에 표시된 두 축만을 기준으로 산출한 것입니다. 아래 표의 "
                                 "pareto_rank 열은 정확도·에너지·지연 시간 세 가지 목적을 모두 반영하므로, 표에서 "
                                 "랭크 1인 후보가 이 계단선 아래에 위치할 수 있습니다.")}</p>'
    )


def _format_or_dash(value: Any, fmt: str = "{:.4f}") -> str:
    return fmt.format(value) if isinstance(value, (int, float)) else "-"


def _format_significant(value: Any, digits: int = 3) -> str:
    """Fixed-decimal formatting silently zeroes out whole columns: a real
    session's noc_latency_ms values span 0.000342-0.000432 ms, every one of
    which `{:.3f}` prints as "0.000" -- reading as "the latency model
    contributes nothing" rather than "these differ by 26%". Scaling the
    decimals to the magnitude keeps `digits` significant figures whatever
    the units turn out to be."""
    if not isinstance(value, (int, float)):
        return "-"
    if value == 0:
        return "0"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(value))))
    return f"{value:,.{decimals}f}"


def _row_html(r: Dict[str, Any]) -> str:
    """Rationale is rendered in full (no char-count truncation) -- a
    hard `[:80]` slice cut mid-word/mid-sentence, which read as broken
    rather than intentionally short. `.rationale-cell`'s CSS wraps it
    into a fixed-width column instead, so the complete text is always
    visible without stretching the row or clipping anything.

    rationale_ko (schemas/planner.py's `PlannerLayerDecision.rationale_ko`,
    the @planner LLM's own Korean translation of the same justification --
    NOT machine-translated here) follows the same EN/KO toggle as the rest
    of the page's static UI text (`_bi`). Older recorded decisions have no
    rationale_ko (field didn't exist yet) -- those fall back to showing the
    English rationale in both language modes rather than an empty cell."""
    rationale = html.escape(str(r.get("rationale") or ""))
    rationale_ko = r.get("rationale_ko")
    rationale_html = _bi(rationale, html.escape(rationale_ko) if rationale_ko else rationale)
    return (
        "<tr>"
        f"<td>{r['iteration']}</td>"
        f"<td>{html.escape(str(r.get('search_tag') or '-'))}</td>"
        f"<td>{_format_or_dash(r.get('accuracy'))}</td>"
        f"<td>{_format_or_dash(r.get('energy_pj'), '{:.2f}')}</td>"
        f"<td>{_format_significant(r.get('noc_latency_ms'))}</td>"
        f"<td>{r.get('pareto_rank', '-')}</td>"
        f"<td>{_bi('yes', '예') if r.get('is_converged') else _bi('no', '아니요')}</td>"
        f'<td class="rationale-cell">{rationale_html}</td>'
        "</tr>"
    )


def _warmup_group_html(group: List[Dict[str, Any]]) -> str:
    """Collapses a contiguous run of LHS warm-up rows into one
    <details>-backed summary row (native disclosure, no JS) instead of
    `warmup_count(n_stages)` individual ones -- at the real max
    (nodes/planner.py's `_MAX_WARMUP_CANDIDATES`) those would
    otherwise dominate the table before the surrogate-guided search even
    starts, and (unlike a real planner decision) their rationale never
    varies with anything a reader would compare row-to-row (tools/
    batch_warmup.py skips the LLM entirely). Expands in place -- collapsed
    by default so the page isn't taller than it needs to be, matching this
    dashboard's EN/KO toggle's same "don't lengthen the page" approach."""
    first, last = group[0]["iteration"], group[-1]["iteration"]
    summary_en = f"Warm-up batch (iterations {first}-{last}, {len(group)} candidates, no LLM confirmation)"
    summary_ko = f"워밍업 배치 (반복 {first}-{last}, {len(group)}개 후보, LLM 개입 없음)"
    inner_rows = "\n".join(_row_html(r) for r in group)
    return (
        '<tr class="warmup-group-row">'
        '<td colspan="8" class="warmup-group-cell">'
        "<details class=\"warmup-details\">"
        f"<summary>{_bi(summary_en, summary_ko)}</summary>"
        '<table class="warmup-inner-table"><tbody>'
        f"{inner_rows}"
        "</tbody></table>"
        "</details>"
        "</td>"
        "</tr>"
    )


def _table_rows_html(rows: List[Dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["iteration"])
    blocks: List[str] = []
    i = 0
    while i < len(ordered):
        if _is_warmup_row(ordered[i]):
            j = i + 1
            while j < len(ordered) and _is_warmup_row(ordered[j]):
                j += 1
            group = ordered[i:j]
            # A lone warm-up row isn't worth collapsing behind a click --
            # only a real multi-row batch is.
            blocks.append(_warmup_group_html(group) if len(group) >= 2 else _row_html(group[0]))
            i = j
        else:
            blocks.append(_row_html(ordered[i]))
            i += 1
    return "\n".join(blocks)


def _calibration_section_html(state: AutoCIMState) -> str:
    """Whether this run's energy/latency numbers are grounded in a real
    published reference (`tools.calibration`) or are still the uncorrected
    analytical fast-approximation (`tools.cim_physics`) -- the honest stand-
    in for "precise simulator vs fast-approximation error margin" this
    project can actually make, given @mapper/@profiler don't bind a real
    NeuroSim/CIM-Loop instance (docs/coding_guidelines.md 5.D Mock First). Uncalibrated is
    not an error state -- it's the expected case for any hw_spec_id that
    doesn't exactly match a known reference -- but it belongs on the
    dashboard, not buried in a code comment, since it directly bears on how
    much to trust the absolute accuracy/energy/latency numbers below."""
    hw_spec_id = state.get("hw_spec_id")
    factor = (state.get("calibration_factors") or {}).get(hw_spec_id)
    provenance = (state.get("calibration_provenance") or {}).get(hw_spec_id)
    hw_spec_id_esc = html.escape(str(hw_spec_id))

    if factor is None or provenance is None:
        warn_en = (
            f"&#9888; hw_spec_id={hw_spec_id_esc} is <strong>uncalibrated</strong> -- no published "
            "literature reference exactly matches this hardware configuration (tools/calibration.py). "
            "energy_pj/latency_ns figures below are analytical order-of-magnitude estimates "
            "(tools/cim_physics.py), not validated against precise simulation or silicon. "
            "Safe for comparing candidates <em>within this run</em>; do not treat the absolute numbers "
            "as a validated benchmark."
        )
        warn_ko = (
            f"&#9888; hw_spec_id={hw_spec_id_esc}는 <strong>캘리브레이션되지 않았습니다</strong> -- 이 하드웨어 "
            "구성과 정확히 일치하는 공개 문헌 참조값이 없습니다 (tools/calibration.py). 아래 energy_pj/"
            "latency_ns 수치는 정밀 시뮬레이션이나 실측 대비 검증되지 않은, 자릿수 수준의 해석적 "
            "추정값(tools/cim_physics.py)입니다. <em>이번 실행 내에서</em> 후보끼리 비교하는 용도로는 "
            "안전하지만, 절대값을 검증된 벤치마크로 취급하지 마십시오."
        )
        return f'<p class="calib-warning">{_bi(warn_en, warn_ko)}</p>'

    if provenance.get("precision_verified"):
        raw_energy = provenance.get("raw_energy_pj")
        precise_energy = provenance.get("precise_energy_pj")
        precision_ok_en = (
            f"&#10003; hw_spec_id={hw_spec_id_esc} is calibrated: analytical energy is scaled by "
            f"<strong>{factor:.4f}x</strong> against this run's own Stage-2 precision re-check "
            f"({raw_energy} &rarr; {precise_energy} pJ, iteration {provenance.get('iteration')}) -- "
            "<em>ADC energy is always a real literature-fit model (hwcomponents_adc); DAC/crossbar-MAC/"
            "SRAM-buffer energy are real device models too (hwcomponents_neurosim/hwcomponents_cacti) "
            "only where this session ran with those Docker-only packages installed, analytical "
            "placeholders otherwise (tools/precision_check.py)</em>."
        )
        precision_ok_ko = (
            f"&#10003; hw_spec_id={hw_spec_id_esc}는 <strong>보정 완료</strong> 상태입니다. 해석적 에너지 "
            f"추정값을 본 실행에서 자체 수행한 2단계 정밀 재검증 결과({raw_energy} &rarr; {precise_energy} pJ, "
            f"iteration {provenance.get('iteration')})에 맞추어 <strong>{factor:.4f}배</strong>로 보정하였습니다. "
            "<em>ADC 에너지는 모든 환경에서 문헌 기반 실측 모델(hwcomponents_adc)을 사용합니다. 반면 "
            "DAC/크로스바-MAC/SRAM 버퍼 에너지는 Docker 전용 패키지"
            "(hwcomponents_neurosim/hwcomponents_cacti)가 설치된 환경에서 실행한 경우에 한하여 실제 소자 "
            "모델이 적용되며, 그 외의 환경에서는 해석적 자리 표시자로 대체됩니다"
            "(tools/precision_check.py).</em>"
        )
        return f'<p class="calib-ok">{_bi(precision_ok_en, precision_ok_ko)}</p>'

    source = html.escape(str(provenance.get("source") or ""))
    note = html.escape(str(provenance.get("note") or ""))
    ref_value = provenance.get("reference_energy_pj_per_mac")
    ok_en = (
        f"&#10003; hw_spec_id={hw_spec_id_esc} is calibrated: analytical energy is scaled by "
        f"<strong>{factor:.4f}x</strong> against a real published reference ({ref_value} pJ/MAC) -- "
        f"<em>{source}</em>."
    )
    ok_ko = (
        f"&#10003; hw_spec_id={hw_spec_id_esc}는 캘리브레이션되었습니다: 해석적 에너지 추정값을 실제 공개된 "
        f"참조값({ref_value} pJ/MAC) 대비 <strong>{factor:.4f}배</strong>로 보정했습니다 -- <em>{source}</em>."
    )
    caveat_html = ""
    if note:
        caveat_html = "<br>" + _bi(
            f"<strong>Stated uncertainty:</strong> {note}", f"<strong>명시된 불확실성:</strong> {note}"
        )
    return f'<p class="calib-ok">{_bi(ok_en, ok_ko)}{caveat_html}</p>'


def _spec_cell(label_en: str, label_ko: str, value: str) -> str:
    return f'<div class="spec-k">{_bi(label_en, label_ko)}</div><div class="spec-v">{value}</div>'


_UNSET = None  # sentinel-free: HWConfig's optional fields are genuinely None when unstated


def _or_unset(value: Any, suffix: str = "") -> str:
    """Optional HWConfig fields (wire_resistance, device_noise_sigma,
    sram_buffer_kb, tech_node_nm, device_type) are legitimately unset for a
    spec that doesn't model them -- rendering those as a blank cell would
    read as a bug, and rendering a made-up default would be worse."""
    if value is _UNSET:
        return f'<span class="spec-unset">{_bi("not specified", "미지정")}</span>'
    # Only the value is escaped -- `suffix` is this module's own literal
    # markup (unit entities like &#8486;/&micro;m), not data.
    return f"{html.escape(f'{value:g}' if isinstance(value, float) else str(value))}{suffix}"


def _run_config_section_html(state: AutoCIMState, hw_config: Optional[HWConfig]) -> str:
    """The exact workload and hardware this run was executed against.

    Previously the report named only `model_id`/`hw_spec_id`, which is an
    opaque label: a reader could not tell whether "cim_v1_128x128" meant an
    8-bit or a 4-bit ADC, how many tiles were available, or what device
    non-idealities were modelled -- all of which change every energy and
    accuracy number below. `hw_config` is passed in rather than read from
    state because `AutoCIMState` carries the id, not the resolved spec
    (main.py's `write_dashboard` resolves it through middleware's registry);
    when it can't be resolved the section degrades to the ids alone and says
    so, rather than inventing values (docs/coding_guidelines.md 5.C)."""
    dataset = state.get("dataset_id")
    workload = [
        _spec_cell("Model", "대상 모델", html.escape(str(state.get("model_id")))),
        _spec_cell(
            "Dataset",
            "데이터셋",
            html.escape(str(dataset)) if dataset else _bi("cifar10 (model default)", "cifar10 (모델 기본값)"),
        ),
        _spec_cell("Search strategy", "탐색 전략", html.escape(str(state.get("search_strategy")))),
        _spec_cell("Run seed", "난수 시드", html.escape(str(state.get("run_seed")))),
    ]

    if hw_config is None:
        hardware = [
            _spec_cell("Hardware spec id", "하드웨어 사양 ID", html.escape(str(state.get("hw_spec_id")))),
        ]
        note = (
            f'<p class="summary">{_bi("The full hardware specification could not be resolved for this report "
                                      "(it is registered at run time, not stored in the run state), so only its "
                                      "identifier is shown.", "이 리포트에서는 하드웨어 사양 전체를 확인할 수 "
                                      "없습니다 (사양은 실행 시점에 등록되며 실행 상태에 저장되지 않습니다). "
                                      "따라서 식별자만 표시합니다.")}</p>'
        )
    else:
        hardware = [
            _spec_cell("Hardware spec id", "하드웨어 사양 ID", html.escape(hw_config.hw_spec_id)),
            _spec_cell(
                "Crossbar array",
                "크로스바 배열",
                f"{hw_config.crossbar_rows} &times; {hw_config.crossbar_cols}",
            ),
            _spec_cell("Tiles on chip", "온칩 타일 수", str(hw_config.num_tiles)),
            _spec_cell("ADC / DAC resolution", "ADC / DAC 해상도", f"{hw_config.adc_bits} bit / {hw_config.dac_bits} bit"),
            _spec_cell(
                "NoC",
                "네트워크 온 칩",
                f"{html.escape(hw_config.noc_topology.value)}, {hw_config.noc_link_bandwidth_gbps:g} Gbps",
            ),
            _spec_cell("Wire resistance", "배선 저항", _or_unset(hw_config.wire_resistance_ohm_per_um, " &#8486;/&micro;m")),
            _spec_cell("Device noise &sigma;", "소자 잡음 &sigma;", _or_unset(hw_config.device_noise_sigma)),
            _spec_cell("SRAM buffer per tile", "타일당 SRAM 버퍼", _or_unset(hw_config.sram_buffer_kb, " KB")),
            _spec_cell("Technology node", "공정 노드", _or_unset(hw_config.tech_node_nm, " nm")),
            _spec_cell(
                "Cell device technology",
                "셀 소자 기술",
                _or_unset(hw_config.device_type.value if hw_config.device_type else None),
            ),
        ]
        note = ""

    return (
        '<div class="spec-panels">'
        f'<div class="panel"><div class="panel-title">{_bi("Workload", "대상 워크로드")}</div>'
        f'<div class="spec-grid">{"".join(workload)}</div></div>'
        f'<div class="panel"><div class="panel-title">{_bi("Hardware specification", "하드웨어 사양")}</div>'
        f'<div class="spec-grid">{"".join(hardware)}</div></div>'
        "</div>"
        f"{note}"
    )


def _targets_section_html(state: AutoCIMState) -> str:
    """What "converged" was actually being measured against.

    `is_converged` is the only outcome the header reports, but it is a
    conjunction: a candidate must clear *every configured* target
    (target_accuracy/target_energy_pj/target_latency_ms, state.py) **and**
    pass @verifier's IR-drop/noise feasibility check. A reader shown only
    the verdict cannot tell which of those bound, nor whether an unset
    target means "met easily" or "never checked" -- so each dimension is
    listed with its target, this run's best measured value, and whether the
    target gated the run at all.

    The best value per dimension is a per-column best across every
    candidate, so the three cells generally come from three *different*
    candidates -- stated in the note, because reading them as one achievable
    design point would overstate the result. The candidate that actually met
    everything simultaneously is called out separately."""
    history = [c for c in state.get("candidate_history", []) if isinstance(c, dict)]

    def best(field: str, better) -> Optional[float]:
        values = [c[field] for c in history if isinstance(c.get(field), (int, float))]
        return better(values) if values else None

    # Each dimension carries its own formatter rather than a shared format
    # string: latency needs `_format_significant` (its real values are
    # ~0.0004 ms, which any fixed-decimal format prints as a flat "0.000").
    dimensions = [
        (
            "accuracy",
            "정확도",
            state.get("target_accuracy"),
            best("accuracy", max),
            "{:.4f}".format,
            lambda b, t: b >= t,
        ),
        (
            "energy_pj",
            "에너지 (pJ)",
            state.get("target_energy_pj"),
            best("energy_pj", min),
            "{:,.2f}".format,
            lambda b, t: b <= t,
        ),
        (
            "latency_ms",
            "지연 시간 (ms)",
            state.get("target_latency_ms"),
            best("noc_latency_ms", min),
            _format_significant,
            lambda b, t: b <= t,
        ),
    ]

    rows = []
    for label_en, label_ko, target, achieved, fmt, meets in dimensions:
        if target is None:
            target_cell = f'<span class="spec-unset">{_bi("not set (not gated)", "설정 안 함 (수렴 판정 미적용)")}</span>'
            verdict = '<span class="spec-unset">&mdash;</span>'
        else:
            target_cell = fmt(target)
            if achieved is None:
                verdict = f'<span class="spec-unset">{_bi("no data yet", "측정값 없음")}</span>'
            elif meets(achieved, target):
                verdict = f'<span class="verdict-ok">&#10003; {_bi("met", "달성")}</span>'
            else:
                verdict = f'<span class="verdict-miss">&#10007; {_bi("not met", "미달성")}</span>'
        achieved_cell = fmt(achieved) if achieved is not None else "-"
        rows.append(
            f"<tr><td>{_bi(label_en, label_ko)}</td><td>{target_cell}</td>"
            f"<td>{achieved_cell}</td><td>{verdict}</td></tr>"
        )

    converged = next((c for c in reversed(history) if c.get("is_converged")), None)
    if converged is None:
        converged_line = _bi(
            "No single candidate has met every configured target together yet.",
            "설정된 목표를 동시에 모두 충족한 후보는 아직 없습니다.",
        )
    else:
        detail = (
            f"iteration {converged.get('iteration')} &mdash; "
            f"accuracy {_format_or_dash(converged.get('accuracy'))}, "
            f"energy {_format_or_dash(converged.get('energy_pj'), '{:,.2f}')} pJ, "
            f"latency {_format_significant(converged.get('noc_latency_ms'))} ms"
        )
        converged_line = _bi(
            f"Met every configured target together at {detail}.",
            f"설정된 목표를 동시에 모두 충족한 후보: {detail}.",
        )

    return (
        '<div class="table-scroll"><table class="targets-table"><thead><tr>'
        f'<th>{_bi("objective", "목적 지표")}</th><th>{_bi("target", "목표값")}</th>'
        f'<th>{_bi("best this run", "이번 실행 최고값")}</th><th>{_bi("verdict", "달성 여부")}</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        f'<p class="summary">{converged_line}</p>'
        f'<p class="summary">{_bi("Each best-this-run figure is the best value over all candidates for that "
                                 "objective alone, so the three can come from three different candidates. "
                                 "Convergence additionally requires @verifier&#39;s IR-drop and device-noise "
                                 "feasibility check to pass, which no target column can show.",
                                 "‘이번 실행 최고값’은 각 목적 지표별로 개별 산출한 최고값이므로, 세 값이 서로 다른 "
                                 "후보에서 나온 것일 수 있습니다. 또한 수렴 판정에는 @verifier의 IR 강하·소자 잡음 "
                                 "타당성 검사 통과가 추가로 요구되며, 이는 목표값 표에 나타나지 않습니다.")}</p>'
    )


def _llm_usage_summary(state: AutoCIMState) -> Dict[str, Any]:
    usage = state.get("llm_usage", [])
    return {
        "n_calls": len(usage),
        "total_attempts": sum(u.get("attempts", 0) for u in usage),
        "n_failed": sum(1 for u in usage if not u.get("succeeded")),
    }


def render_dashboard_html(state: AutoCIMState, hw_config: Optional[HWConfig] = None) -> str:
    """Pure function of `state` (plus the optional resolved `hw_config`, which
    `AutoCIMState` only names by id) -- no I/O, no LangGraph/checkpointer
    dependency, so it's directly unit-testable against a synthetic state
    (state_factory in tests) exactly like the rest of this project's
    Mock-First-tested tools. `hw_config=None` renders every other section
    unchanged and degrades the hardware panel to the id alone."""
    rows = _iteration_rows(state)
    usage = _llm_usage_summary(state)
    chart_targets = {"accuracy": state.get("target_accuracy"), "energy_pj": state.get("target_energy_pj")}
    has_targets = any(v is not None for v in chart_targets.values())

    title = f"{state.get('model_id')} / {state.get('hw_spec_id')}"
    low_confidence = bool(state.get("low_confidence_conclusion"))
    # low_confidence_conclusion (hitl_node's conclude-with-best path, see
    # state.py's field docstring) also sets is_converged=True so the graph
    # can still route to END -- the badge below must not read that as a
    # real target-meeting convergence.
    converged = bool(state.get("is_converged")) and not low_confidence
    llm_summary_en = (
        f'LLM calls: {usage["n_calls"]} node-invocation(s), {usage["total_attempts"]} total attempt(s), '
        f'{usage["n_failed"]} failed after retries'
    )
    llm_summary_ko = (
        f'LLM 호출: {usage["n_calls"]}회(노드 호출 기준), 총 시도 {usage["total_attempts"]}회, '
        f'재시도 후 실패 {usage["n_failed"]}건'
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AutoCIM-Agent report: {html.escape(title)}</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #0b0f14;
    --bg-panel: #111821;
    --bg-panel-alt: #161f2b;
    --border: #253242;
    --text: #e6edf3;
    --text-muted: #93a4b8;
    --blue: #5b9dff;
    --amber-bg: #241d0d;
    --amber-border: #4d3a14;
    --amber-text: #ecd9a8;
    --green-bg: #0f2019;
    --green-border: #1e4432;
    --green-text: #bfe9d3;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    margin: 0;
    padding: 2.5rem clamp(1rem, 4vw, 3rem) 4rem;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }}
  h1 {{ font-size: 1.35rem; font-weight: 600; margin: 0; }}
  h2 {{
    font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
    color: var(--text-muted); border-bottom: 1px solid var(--border); padding-bottom: .5rem;
    margin: 2.4rem 0 1rem;
  }}
  .header-row {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }}
  .lang-toggle {{ display: flex; gap: 4px; }}
  .lang-btn {{
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 12px; background: var(--bg-panel); border: 1px solid var(--border);
    color: var(--text-muted); padding: 4px 10px; border-radius: 5px; cursor: pointer;
  }}
  .lang-btn:hover {{ color: var(--text); }}
  .lang-btn.active {{ color: var(--blue); border-color: var(--blue); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 14px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-muted); }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  /* Mirrors the chart's own front marker: a detached concentric ring, drawn
     with box-shadow so the swatch keeps its 10px footprint and the legend
     row does not reflow. */
  .legend-ring {{ box-shadow: 0 0 0 1.2px var(--bg-panel), 0 0 0 2.6px currentColor; color: #5b9dff; }}
  .meta-bar {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .9rem 0 0; }}
  .chip {{
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 12.5px; background: var(--bg-panel); border: 1px solid var(--border);
    color: var(--text-muted); padding: 4px 10px; border-radius: 5px;
  }}
  .chip strong {{ color: var(--text); font-weight: 600; }}
  .chip.status-ok strong {{ color: var(--green-text); }}
  .chip.status-pending strong {{ color: var(--amber-text); }}
  table {{
    border-collapse: collapse; width: 100%; margin-top: .5rem;
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden;
  }}
  th, td {{ border-bottom: 1px solid var(--border); padding: 8px 12px; text-align: left; font-size: 13px; vertical-align: top; }}
  th {{
    background: var(--bg-panel-alt); color: var(--text-muted); font-weight: 600;
    text-transform: uppercase; font-size: 11px; letter-spacing: .05em;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--bg-panel-alt); }}
  td {{ font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; }}
  td.rationale-cell {{
    font-family: inherit; white-space: normal; word-break: break-word;
    max-width: 380px; color: var(--text);
  }}
  tr.warmup-group-row:hover {{ background: none; }}
  td.warmup-group-cell {{ padding: 0; }}
  .warmup-details {{ padding: 10px 12px; }}
  .warmup-details > summary {{
    cursor: pointer; list-style: none; font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 13px; color: var(--text-muted);
  }}
  .warmup-details > summary::-webkit-details-marker {{ display: none; }}
  .warmup-details > summary::before {{ content: "\\25B6  "; color: var(--blue); }}
  .warmup-details[open] > summary::before {{ content: "\\25BC  "; }}
  .warmup-details[open] > summary {{ color: var(--text); margin-bottom: 8px; }}
  table.warmup-inner-table {{ margin: 0; }}
  /* An 8-column monospace table has a hard min-content width (~735px);
     below that it used to push the whole page sideways, which is what a
     reader sees as "the right edge is cut off" when zooming in. Scrolling
     it inside its own box keeps the rest of the page in place. */
  .table-scroll {{ overflow-x: auto; }}
  .panel {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; }}
  .panel-title {{
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em;
    color: var(--text-muted); margin-bottom: 10px;
  }}
  .spec-panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }}
  .spec-grid {{ display: grid; grid-template-columns: auto 1fr; gap: 6px 18px; align-items: baseline; }}
  .spec-k {{ font-size: 12.5px; color: var(--text-muted); }}
  .spec-v {{
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 13px; color: var(--text); text-align: right;
  }}
  /* No italic: Korean fonts ship no true italic, so the browser synthesises
     a slanted face that renders smeared and noticeably harder to read than
     the upright text around it. Colour alone carries "unset" here. */
  .spec-unset {{ color: #6c7a8c; font-family: inherit; }}
  table.targets-table td:first-child {{ font-family: inherit; }}
  .verdict-ok {{ color: var(--green-text); font-family: inherit; }}
  .verdict-miss {{ color: var(--amber-text); font-family: inherit; }}
  .calib-warning {{ background: var(--amber-bg); border: 1px solid var(--amber-border); color: var(--amber-text); padding: 12px 16px; border-radius: 8px; }}
  .calib-ok {{ background: var(--green-bg); border: 1px solid var(--green-border); color: var(--green-text); padding: 12px 16px; border-radius: 8px; }}
  .phase-box {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; }}
  .phase-title {{ font-weight: 600; color: var(--text); margin-bottom: 6px; }}
  .phase-warmup .phase-title {{ color: var(--blue); }}
  .phase-bar {{ height: 6px; background: var(--bg-panel-alt); border-radius: 3px; overflow: hidden; margin: 8px 0 10px; }}
  .phase-bar-fill {{ height: 100%; background: var(--blue); }}
  .phase-note {{ color: var(--text-muted); font-size: 13px; margin: 0; }}
  .summary {{ color: var(--text-muted); font-size: 13.5px; margin: .6rem 0 0; }}
</style>
</head>
<body>
<div class="header-row">
  <h1>AutoCIM-Agent report</h1>
  <div class="lang-toggle" role="group" aria-label="Language / 언어">
    <button type="button" class="lang-btn active" data-setlang="en" onclick="setDashboardLang('en')">EN</button>
    <button type="button" class="lang-btn" data-setlang="ko" onclick="setDashboardLang('ko')">한국어</button>
  </div>
</div>
<div class="meta-bar">
  <span class="chip">model_id <strong>{html.escape(str(state.get("model_id")))}</strong></span>
  <span class="chip">hw_spec_id <strong>{html.escape(str(state.get("hw_spec_id")))}</strong></span>
  <span class="chip">iteration_count <strong>{state.get("iteration_count")}</strong></span>
  <span class="chip {"status-ok" if converged else "status-pending"}">is_converged <strong>{converged}</strong></span>
  {f'<span class="chip status-pending">{_bi("low-confidence conclusion", "낮은 신뢰도 종결")} <strong>{_bi("review before trusting", "신뢰 전 검토 필요")}</strong></span>' if low_confidence else ""}
</div>
<p class="summary">{_bi(llm_summary_en, llm_summary_ko)}</p>
{f'<p class="calib-warning">{_bi("This session gave up without meeting every configured target -- the reported candidate is just the best one the search tried, not a real convergence.", "이 세션은 설정된 목표를 모두 충족하지 못한 채 종료되었습니다 -- 보고된 후보는 탐색이 시도한 것 중 최선일 뿐, 실제 수렴이 아닙니다.")}</p>' if low_confidence else ""}

<h2>{_bi("Run configuration", "실행 조건")}</h2>
{_run_config_section_html(state, hw_config)}

<h2>{_bi("Targets vs. best measured", "목표값 대비 실측 최고값")}</h2>
{_targets_section_html(state)}

<h2>{_bi("Search progress", "탐색 진행 상황")}</h2>
{_search_phase_summary_html(rows)}

<h2>{_bi("Calibration / precision confidence", "캘리브레이션 / 정밀도 신뢰도")}</h2>
{_calibration_section_html(state)}

<h2>{_bi("Pareto front (energy vs. accuracy)", "파레토 프론트 (에너지 대비 정확도)")}</h2>
<div class="panel">
{_svg_pareto_chart(rows, chart_targets)}
{_pareto_legend_html(has_targets)}
</div>

<h2>{_bi("Per-iteration candidates", "반복별 후보 목록")}</h2>
<div class="table-scroll">
<table>
<thead><tr>
<th>{_bi("iter", "반복")}</th>
<th>{_bi("search phase", "탐색 단계")}</th>
<th>{_bi("accuracy", "정확도")}</th>
<th>energy_pj</th>
<th>latency_ms</th>
<th>pareto_rank</th>
<th>{_bi("converged", "수렴 여부")}</th>
<th>{_bi("rationale", "판단 근거")}</th>
</tr></thead>
<tbody>
{_table_rows_html(rows)}
</tbody>
</table>
</div>
<script>
function setDashboardLang(lang) {{
  document.querySelectorAll('[data-lang]').forEach(function (el) {{
    el.hidden = el.getAttribute('data-lang') !== lang;
  }});
  document.querySelectorAll('.lang-btn').forEach(function (btn) {{
    btn.classList.toggle('active', btn.dataset.setlang === lang);
  }});
}}
</script>
</body>
</html>"""
