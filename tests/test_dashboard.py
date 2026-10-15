"""tools/dashboard.py: joins candidate_history (what happened) against
planner_decisions (why it was tried) and llm_usage (real LLM cost) into a
self-contained HTML report -- a pure function of TessellateState, no I/O, so
tested purely against synthetic state (state_factory), no real graph run."""

import re

from graph import build_graph
from schemas.config import HWConfig, NoCTopology
from tools.dashboard import (
    _pareto_front_2d,
    _pareto_legend_html,
    _svg_pareto_chart,
    render_dashboard_html,
)

_PLANNER_DECISIONS = [
    {"iteration": 1, "search_tag": "LHS warm-up candidate 1/3", "used_llm": True, "rationale": "exploring the space"},
    {"iteration": 2, "search_tag": "surrogate-model acquisition over 1 prior candidates", "used_llm": False, "rationale": "LLM proposal failed"},
]

_CANDIDATE_HISTORY = [
    {
        "iteration": 1,
        "avg_weight_bits": 6,
        "avg_column_pruning_ratio": 0.1,
        "accuracy": 0.7,
        "energy_pj": 8.0,
        "noc_latency_ms": 1.5,
        "is_converged": False,
        "pareto_rank": 1,
    },
    {
        "iteration": 2,
        "avg_weight_bits": 8,
        "avg_column_pruning_ratio": 0.0,
        "accuracy": 0.9,
        "energy_pj": 4.0,
        "noc_latency_ms": 1.0,
        "is_converged": True,
        "pareto_rank": 1,
    },
]


def test_render_dashboard_html_with_no_candidates_yet(state_factory):
    state = state_factory()

    output = render_dashboard_html(state)

    assert "No evaluated candidates yet." in output
    assert "0 node-invocation(s)" in output


def test_render_dashboard_html_joins_candidate_history_and_planner_decisions_by_iteration(state_factory):
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=_PLANNER_DECISIONS)

    output = render_dashboard_html(state)

    assert "LHS warm-up candidate 1/3" in output
    assert "surrogate-model acquisition over 1 prior candidates" in output
    assert "exploring the space" in output
    assert "0.7000" in output  # iteration 1 accuracy
    assert "0.9000" in output  # iteration 2 accuracy


def test_render_dashboard_html_includes_pareto_chart_points_per_iteration(state_factory):
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=_PLANNER_DECISIONS)

    output = render_dashboard_html(state)

    assert "<svg" in output
    # Markers are the circles carrying a <title> tooltip and so closed with
    # </circle>; the Pareto-front ring is a separate self-closing <circle>
    # with no tooltip, so counting "<circle" would count it as a candidate.
    assert output.count("</circle>") == 2  # one marker per evaluated candidate


def test_render_dashboard_html_reports_llm_call_counts(state_factory):
    """No cost figure is shown -- this project targets local LLMs, where
    there's no per-call billing to estimate (TESSELLATE_LLM_COST_PER_1K_* is
    for the rare paid-API case and isn't surfaced on the dashboard)."""
    usage = [
        {"node": "planner", "iteration": 1, "attempts": 1, "succeeded": True, "estimated_cost_usd": 0.02},
        {"node": "planner", "iteration": 2, "attempts": 2, "succeeded": False, "estimated_cost_usd": None},
    ]
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=_PLANNER_DECISIONS, llm_usage=usage)

    output = render_dashboard_html(state)

    assert "2 node-invocation(s)" in output
    assert "1 failed after retries" in output
    assert "$0.0200" not in output
    assert "estimated cost" not in output.lower()


def test_render_dashboard_html_collapses_a_multi_row_warmup_batch(state_factory):
    """3+ consecutive "LHS warm-up candidate" rows (tools/batch_warmup.py's
    parallel path seeds one such planner_decisions entry per candidate --
    see nodes/planner.py/tools/batch_warmup.py) must render as one
    collapsed <details> summary row instead of N individual <tr>s, so a
    real run's up to 12 warm-up candidates don't dominate the table before
    the surrogate-guided search even starts. The individual candidates
    must still be present in the HTML (inside the collapsed block), just
    not as top-level rows."""
    warmup_decisions = [
        {
            "iteration": i,
            "search_tag": f"LHS warm-up candidate {i}/3",
            "used_llm": False,
            "rationale": f"warmup rationale {i}",
        }
        for i in (1, 2, 3)
    ]
    warmup_candidates = [
        {
            "iteration": i,
            "avg_weight_bits": 4,
            "avg_column_pruning_ratio": 0.1,
            "accuracy": 0.1 * i,
            "energy_pj": 5.0,
            "noc_latency_ms": 1.0,
            "is_converged": False,
            "pareto_rank": 1,
        }
        for i in (1, 2, 3)
    ]
    state = state_factory(candidate_history=warmup_candidates, planner_decisions=warmup_decisions)

    output = render_dashboard_html(state)

    assert 'class="warmup-details"' in output
    assert "Warm-up batch (iterations 1-3, 3 candidates" in output
    for i in (1, 2, 3):
        assert f"warmup rationale {i}" in output
    # Collapsed, not rendered as three separate top-level rows.
    assert output.count('<tr class="warmup-group-row">') == 1


def test_render_dashboard_html_handles_a_candidate_with_no_matching_planner_decision(state_factory):
    """A candidate_history entry with no corresponding planner_decisions
    entry (e.g. state from before this feature existed) must not crash the
    join -- it should just render without a search phase/rationale."""
    state = state_factory(candidate_history=_CANDIDATE_HISTORY, planner_decisions=[])

    output = render_dashboard_html(state)

    assert "0.7000" in output
    assert "0.9000" in output


def test_calibration_section_warns_when_hw_spec_id_is_uncalibrated(state_factory):
    state = state_factory(calibration_factors={}, calibration_provenance={})

    output = render_dashboard_html(state)

    assert "uncalibrated" in output
    assert '<p class="calib-warning">' in output
    assert '<p class="calib-ok">' not in output


def test_calibration_section_shows_factor_and_citation_when_calibrated(state_factory):
    state = state_factory(
        hw_spec_id="calibrated_hw",
        calibration_factors={"calibrated_hw": 1.2345},
        calibration_provenance={
            "calibrated_hw": {
                "reference_energy_pj_per_mac": 0.047,
                "source": "Peng et al., NeuroSim V1.5",
                "note": "assumes 1 MAC == 1 op",
            }
        },
    )

    output = render_dashboard_html(state)

    assert '<p class="calib-ok">' in output
    assert "1.2345x" in output
    assert "Peng et al., NeuroSim V1.5" in output
    assert "assumes 1 MAC == 1 op" in output
    assert '<p class="calib-warning">' not in output


def test_calibration_section_shows_precision_verified_status_distinctly_from_a_literature_match(state_factory):
    """nodes/precision_verifier.py's own calibration_provenance update
    (`"precision_verified": True`) must render as calibrated, but must never
    be confused with a real published-reference match (previous test) --
    it's a Mock First (docs/coding_guidelines.md D) placeholder re-check, not a citation."""
    state = state_factory(
        hw_spec_id="precision_checked_hw",
        calibration_factors={"precision_checked_hw": 1.05},
        calibration_provenance={
            "precision_checked_hw": {
                "precision_verified": True,
                "raw_energy_pj": 10.0,
                "precise_energy_pj": 10.5,
                "iteration": 4,
            }
        },
    )

    output = render_dashboard_html(state)

    assert '<p class="calib-ok">' in output
    assert "1.0500x" in output
    assert "precision" in output.lower()
    assert '<p class="calib-warning">' not in output


def test_calibration_section_ignores_factors_for_other_hw_spec_ids(state_factory):
    """A calibration_factors/calibration_provenance entry seeded for a
    *different* hw_spec_id (e.g. left over from a prior session on the same
    process) must not be mistaken for this run's own calibration status."""
    state = state_factory(
        hw_spec_id="this_run_hw",
        calibration_factors={"some_other_hw": 2.0},
        calibration_provenance={"some_other_hw": {"reference_energy_pj_per_mac": 0.1, "source": "x", "note": None}},
    )

    output = render_dashboard_html(state)

    assert "uncalibrated" in output


def test_dashboard_renders_from_a_real_stubbed_graph_run(state_factory, registered_bad_hw_config):
    """End-to-end: a real (stubbed) multi-iteration graph run's resulting
    state -- not hand-built synthetic state -- must render cleanly, proving
    planner_decisions/candidate_history/llm_usage actually line up the way
    the rest of this file assumes."""
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-dashboard-e2e-thread"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    result = compiled.invoke(init_state, config=config)

    output = render_dashboard_html(result)

    assert "<svg" in output
    assert len(result["candidate_history"]) == output.count("</circle>")
    # every candidate_history iteration has a matching planner_decisions
    # entry, so its search_tag/rationale must show up in the rendered table
    for decision in result["planner_decisions"]:
        assert decision["search_tag"] in output


# --- Pareto chart: warm-up vs search-phase rendering ------------------------


def _chart_rows():
    """4 LHS warm-up samples + 2 surrogate-guided search candidates."""
    rows = []
    for i in range(1, 5):
        rows.append(
            {
                "iteration": i,
                "accuracy": 0.60 + i * 0.02,
                "energy_pj": 1_600_000 + i * 50_000,
                "pareto_rank": 1,
                "search_tag": f"LHS warm-up candidate {i}/4",
            }
        )
    for i in (5, 6):
        rows.append(
            {
                "iteration": i,
                "accuracy": 0.80 + i * 0.01,
                "energy_pj": 1_700_000 + i * 10_000,
                "pareto_rank": 1,
                "search_tag": "surrogate-model acquisition over 4 prior candidates",
            }
        )
    return rows


def test_pareto_chart_connects_only_search_phase_points():
    """LHS warm-up samples have no meaningful order between them, so joining
    them by iteration index draws noise: on a real 18-candidate session the
    polyline ran 3401px across a 466px plot diagonal. Only the surrogate-
    guided points -- which genuinely are a trajectory -- get the line."""
    svg = _svg_pareto_chart(_chart_rows())

    points = re.search(r'<polyline points="([^"]+)"', svg).group(1).split()
    assert len(points) == 2  # the two search candidates, not all six


def test_pareto_chart_omits_the_trajectory_entirely_during_warm_up():
    """A run still in warm-up has no trajectory to draw yet. The Pareto
    staircase (a different polyline, drawn in the front colour) is unrelated
    and must still appear -- warm-up samples have a front from candidate
    one."""
    warmup_only = [r for r in _chart_rows() if "warm-up" in r["search_tag"]]

    svg = _svg_pareto_chart(warmup_only)

    assert "#4a6b8a" not in svg  # the search-order trajectory
    assert '<polyline points="' in svg and "#5b9dff" in svg  # the front staircase


def test_pareto_chart_draws_warm_up_hollow_and_search_filled():
    svg = _svg_pareto_chart(_chart_rows())

    assert svg.count('r="3.5" fill="none"') == 4  # hollow warm-up marks
    assert len(re.findall(r'r="5" fill="#', svg)) == 2  # filled search marks


def test_pareto_chart_carries_iteration_ids_instead_of_printing_labels():
    """Printed iteration numbers were dropped: at 150 candidates the ones
    that mattered clustered in one corner, and the displacement needed to
    keep them apart left each number sitting away from the point it named
    (leader lines back to the marker only added more ink). Every marker now
    carries its iteration as `data-iter` instead, and the per-iteration
    table highlights the matching point on click -- on demand, one at a
    time, instead of all of them all the time."""
    svg = _svg_pareto_chart(_chart_rows())

    assert not re.findall(r'fill="#c8d3e0"[^>]*>(\d+)</text>', svg)
    tagged = sorted(int(n) for n in re.findall(r'<circle data-iter="(\d+)"', svg))
    assert tagged == [r["iteration"] for r in sorted(_chart_rows(), key=lambda r: r["iteration"])]
    assert 'id="pareto-highlight"' in svg  # the ring the table moves onto a point


def test_pareto_chart_prints_axis_end_values():
    """Without them the axes carried no magnitude at all -- a reader could
    not tell 1.6M pJ from 16 pJ."""
    svg = _svg_pareto_chart(_chart_rows())

    assert "1,650,000" in svg and "1,800,000" in svg  # energy min/max
    assert "0.620" in svg and "0.860" in svg  # accuracy min/max


def test_pareto_chart_scales_to_its_container():
    """Fixed pixel width overflowed narrow viewports; the viewBox keeps the
    aspect ratio while the element itself scales down."""
    svg = _svg_pareto_chart(_chart_rows())

    assert 'width="100%"' in svg
    assert "viewBox=" in svg


# --- Pareto chart: two-axis front, staircase, target reference lines --------


def test_pareto_chart_front_is_over_the_two_plotted_axes_not_pareto_rank():
    """Every row in _chart_rows() carries pareto_rank 1 (that rank is over
    three objectives, and on a real session 13 of 18 candidates held it), so
    colouring by it would mark all six points and show no front at all. The
    chart's front is recomputed over energy/accuracy alone."""
    front = _pareto_front_2d(_chart_rows())

    assert [r["iteration"] for r in front] == [1, 2, 5, 6]
    # iteration 3 ties iteration 5 on energy but is far less accurate, and
    # iteration 4 costs the most while being beaten on accuracy by both.
    assert {3, 4}.isdisjoint({r["iteration"] for r in front})


def test_pareto_chart_draws_the_front_as_a_staircase():
    """Tread then riser per step: the extra energy is paid first, then the
    accuracy it buys -- 4 front points give 7 vertices, not 4."""
    svg = _svg_pareto_chart(_chart_rows())

    staircase = re.findall(r'<polyline points="([^"]+)" fill="none" stroke="#5b9dff"', svg)
    assert len(staircase) == 1
    assert len(staircase[0].split()) == 7


def test_pareto_chart_draws_configured_targets_as_reference_lines():
    svg = _svg_pareto_chart(_chart_rows(), {"accuracy": 0.75, "energy_pj": 1_700_000})

    assert "target accuracy 0.750" in svg
    assert "target energy 1,700,000" in svg
    # Counts lines specifically: the click-to-highlight ring is a <circle>
    # in the same amber, and it is not a target reference line.
    assert len(re.findall(r'<line[^>]*stroke="#e0b354"', svg)) == 2


def test_pareto_chart_extends_its_domain_so_a_nearby_target_stays_visible():
    """A target just outside the measured range still has to be drawable --
    otherwise "no candidate reached the bar" is invisible, which is the one
    case a reader most needs to see."""
    svg = _svg_pareto_chart(_chart_rows(), {"accuracy": 0.95})

    assert "target accuracy 0.950" in svg
    # The domain now carries 6% headroom past the extreme data point, so the
    # target no longer lands exactly on a tick -- what matters is that the
    # axis reaches past it, which is what makes the line drawable at all.
    ticks = [float(t) for t in re.findall(r'text-anchor="end">([0-9.]+)</text>', svg)]
    assert max(ticks) >= 0.95


def test_pareto_chart_omits_a_target_that_is_nowhere_near_the_data():
    """Widening the domain to reach an energy budget 100x below anything
    measured would squash every real candidate into one pixel column. The
    line is dropped instead; the targets table still reports the miss."""
    svg = _svg_pareto_chart(_chart_rows(), {"energy_pj": 10_000})

    assert "target energy" not in svg
    assert "1,650,000" in svg  # the data's own domain is untouched


# --- Run configuration and target sections ---------------------------------


def test_run_configuration_section_reports_the_real_hardware_numbers(state_factory, good_hw_config):
    """hw_spec_id alone is an opaque label -- a reader cannot tell an 8-bit
    ADC from a 4-bit one, and every energy figure in the report depends on
    it."""
    output = render_dashboard_html(state_factory(run_seed=3), good_hw_config)

    assert "128 &times; 128" in output
    assert "8 bit / 4 bit" in output  # adc_bits / dac_bits
    assert "mesh, 10 Gbps" in output
    assert "0.001 &#8486;/&micro;m" in output  # wire_resistance_ohm_per_um
    assert "64 KB" in output  # sram_buffer_kb
    assert ">3<" in output  # run_seed, not the default 0


def test_run_configuration_marks_unstated_hardware_fields_rather_than_inventing_them(state_factory):
    """good_hw_config states no tech_node_nm/device_type -- a blank cell
    would read as a bug and a made-up default would be worse (docs/coding_guidelines.md
    5.C)."""
    hw = HWConfig(
        hw_spec_id="spec_without_optionals",
        crossbar_rows=64,
        crossbar_cols=64,
        num_tiles=4,
        adc_bits=6,
        dac_bits=2,
        noc_topology=NoCTopology.RING,
        noc_link_bandwidth_gbps=5.0,
    )

    output = render_dashboard_html(state_factory(), hw)

    assert output.count("미지정") == 5  # wire resistance, noise sigma, SRAM, tech node, device type


def test_run_configuration_degrades_to_the_id_when_the_spec_cannot_be_resolved(state_factory):
    """main.py's write_dashboard passes None when middleware's registry has
    no entry -- that must cost the hardware panel, not the whole report."""
    output = render_dashboard_html(state_factory(hw_spec_id="unregistered_spec"))

    assert "unregistered_spec" in output
    assert "실행 조건" in output
    assert "크로스바 배열" not in output


def test_targets_section_reports_target_best_and_verdict_per_objective(state_factory):
    state = state_factory(
        target_accuracy=0.8,
        target_energy_pj=5.0,
        candidate_history=_CANDIDATE_HISTORY,
    )

    output = render_dashboard_html(state)

    assert "0.8000" in output  # the accuracy target
    assert "0.9000" in output  # best accuracy measured (iteration 2)
    assert "달성" in output
    assert "설정 안 함 (수렴 판정 미적용)" in output  # target_latency_ms was never set


def test_targets_section_flags_an_unmet_target(state_factory):
    state = state_factory(target_accuracy=0.95, candidate_history=_CANDIDATE_HISTORY)

    output = render_dashboard_html(state)

    assert "미달성" in output


def test_targets_section_names_the_candidate_that_met_everything_at_once(state_factory):
    """The per-objective bests generally come from different candidates, so
    reading them as one design point would overstate the result -- the
    candidate that actually converged is called out separately."""
    state = state_factory(target_accuracy=0.8, candidate_history=_CANDIDATE_HISTORY)

    output = render_dashboard_html(state)

    assert "설정된 목표를 동시에 모두 충족한 후보: iteration 2" in output


def test_targets_section_says_so_when_nothing_has_converged(state_factory):
    history = [{**c, "is_converged": False} for c in _CANDIDATE_HISTORY]

    output = render_dashboard_html(state_factory(target_accuracy=0.99, candidate_history=history))

    assert "설정된 목표를 동시에 모두 충족한 후보는 아직 없습니다." in output


def test_latency_keeps_significant_digits_instead_of_collapsing_to_zero(state_factory):
    """A real session's noc_latency_ms values span 0.000342-0.000432 ms.
    Under a fixed {:.3f} every one printed as "0.000", reading as "latency
    is zero" rather than "these differ by 26%"."""
    history = [
        {**_CANDIDATE_HISTORY[0], "noc_latency_ms": 0.000342},
        {**_CANDIDATE_HISTORY[1], "noc_latency_ms": 0.000432},
    ]

    output = render_dashboard_html(state_factory(target_accuracy=0.8, candidate_history=history))

    assert "0.000342" in output
    assert "0.000432" in output
    assert ">0.000<" not in output


def test_pareto_front_membership_is_not_carried_by_colour_alone():
    """Front membership is the distinction this chart exists to show, and
    #5b9dff against #5c6b7e is only 2.00:1 -- a washed-out projector, a
    greyscale handout, or a red-green colour vision deficiency would erase
    it. Every front point therefore also carries a concentric ring, a shape
    cue that survives all three.
    """
    rows = [
        {"iteration": 1, "energy_pj": 100.0, "accuracy": 0.90, "pareto_rank": 1},  # front
        {"iteration": 2, "energy_pj": 120.0, "accuracy": 0.50, "pareto_rank": 2},  # dominated
    ]

    svg = _svg_pareto_chart(rows)

    # Rings are self-closing and tooltip-less; markers are closed with
    # </circle>. The click-to-highlight ring is self-closing too, so it is
    # excluded by id -- it marks whatever the reader selected, not front
    # membership.
    self_closing = svg.count("<circle") - svg.count("</circle>")
    rings = self_closing - svg.count('id="pareto-highlight"')
    assert rings == 1  # exactly the one front point, not both, not neither


def test_the_legend_names_the_ring_before_the_colour():
    """A cue nobody is told about is not a cue. The legend has to state the
    ring, not just show a blue dot and call it the front."""
    legend = _pareto_legend_html(has_targets=False)

    assert "legend-ring" in legend
    assert "Ringed" in legend
    assert "테두리" in legend
