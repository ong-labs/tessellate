"""@planner's real per-stage search space (nodes/planner.py + tools/search.py).

Every other planner test (tests/test_planner_llm.py) runs against
FAKE_LAYER_GROUPS' 2 stages (tests/conftest.py's autouse
`stub_planner_layer_groups`), which is too small to actually exercise the
"per-real-stage independent search" behavior this module adds -- with only
2 stages, `warmup_count` never rises above its floor of 3, and a bug that
silently fanned one flat point out to every stage would be invisible. These
tests monkeypatch `get_layer_groups` again, locally, to a larger stage set.
"""

import nodes.planner as planner_module

FAKE_STAGES = {
    "stage_a": ["stage_a"],
    "stage_b": ["stage_b"],
    "stage_c": ["stage_c"],
    "stage_d": ["stage_d"],
    "stage_e": ["stage_e"],
}


def test_warmup_count_scales_with_real_stage_count(monkeypatch, state_factory, registered_hw_config):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)

    stage_names = planner_module.real_stage_names(state["model_id"])
    assert len(stage_names) == 5
    # round(5 * _WARMUP_SAMPLES_PER_STAGE=1.5) = 8, between floor(3) and cap(20)
    assert planner_module.warmup_count(len(stage_names)) == 8


def test_warmup_count_is_capped_for_a_high_stage_count():
    assert planner_module.warmup_count(20) == planner_module._MAX_WARMUP_CANDIDATES


def test_warmup_count_samples_more_than_once_per_stage():
    # Regression guard for the exact reliability gap tools/baseline_compare.py
    # measured: a flat 1x-per-stage ratio left resnet18 (10 stages) with
    # only 10 real warm-up points over its 20-dim search space, and only
    # 1/3 seeds held a --target-accuracy=0.7 floor within the auto-hitl
    # retry budget. warmup_count must now exceed n_stages itself (not just
    # equal it) whenever the multiplied-and-capped result allows.
    assert planner_module.warmup_count(10) > 10
    assert planner_module.warmup_count(10) == round(10 * planner_module._WARMUP_SAMPLES_PER_STAGE)


def test_warmup_count_floor_is_unchanged_for_very_few_stages():
    # 2 stages * 1.5 = 3.0, exactly the floor -- still bottoms out at
    # _WARMUP_CANDIDATES, same as before this multiplier existed.
    assert planner_module.warmup_count(2) == planner_module._WARMUP_CANDIDATES


def test_propose_stage_points_returns_one_independent_point_per_stage(
    monkeypatch, state_factory, registered_hw_config
):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    stage_names = planner_module.real_stage_names(state["model_id"])

    stage_points, search_tag = planner_module._propose_stage_points(state, stage_names)

    assert len(stage_points) == len(stage_names)
    assert "LHS warm-up" in search_tag
    # Real per-stage independence: not every stage collapsed to one shared
    # point, which the old fixed first/last-stage-bias expansion would do.
    assert len(set(stage_points)) > 1


def test_planner_node_produces_one_layer_config_per_real_stage_with_llm_fallback(
    monkeypatch, state_factory, registered_hw_config
):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))

    def _raise():
        raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(planner_module, "get_planner_chat_model", _raise)

    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    update = planner_module.planner_node(state)

    assert {lc["layer_name"] for lc in update["planned_layer_configs"]} == set(FAKE_STAGES)
    assert len(update["planned_layer_configs"]) == 5


def test_surrogate_phase_uses_candidate_historys_layer_configs(monkeypatch, state_factory, registered_hw_config):
    """Once warm-up is exhausted, the surrogate must actually be able to
    read prior candidates back via candidate_history's `layer_configs`
    field (nodes/evaluator.py) -- not just the old across-stage averages --
    to place them in the real per-stage search space."""
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)
    stage_names = planner_module.real_stage_names(state["model_id"])
    n_warmup = planner_module.warmup_count(len(stage_names))

    good_layer_configs = [
        {"layer_name": name, "weight_bits": 8, "activation_bits": 8, "column_pruning_ratio": 0.0}
        for name in stage_names
    ]
    history = [
        {"layer_configs": good_layer_configs, "accuracy": 0.95}
        for _ in range(n_warmup)  # enough entries to exit the warm-up phase
    ]
    state = {**state, "candidate_history": history}

    stage_points, search_tag = planner_module._propose_stage_points(state, stage_names)

    assert "surrogate-model acquisition" in search_tag
    assert len(stage_points) == len(stage_names)


# --- search_seed_for's run_seed (multi-seed replicate support) --------------
#
# Before run_seed existed, this was a pure function of (model_id,
# hw_spec_id) -- every session against the same target always explored the
# exact same warm-up/NSGA-II points, so running "the same search" twice
# could never produce an independent sample. tools/repeat_search.py's whole
# premise (N replicates -> a real variance estimate) depends on run_seed
# actually changing the seed, not just being accepted and ignored.


def test_search_seed_for_is_deterministic_given_same_run_seed():
    a = planner_module.search_seed_for("resnet18", "cim_v1", run_seed=7)
    b = planner_module.search_seed_for("resnet18", "cim_v1", run_seed=7)
    assert a == b


def test_search_seed_for_varies_with_run_seed():
    seeds = {planner_module.search_seed_for("resnet18", "cim_v1", run_seed=s) for s in range(5)}
    assert len(seeds) == 5  # 5 distinct run_seeds -> 5 distinct search seeds


def test_search_seed_for_default_run_seed_is_zero():
    assert planner_module.search_seed_for("resnet18", "cim_v1") == planner_module.search_seed_for(
        "resnet18", "cim_v1", run_seed=0
    )


# --- search_strategy branching (ablation: tools/run_ablation.py) ------------


def _post_warmup_state(state_factory, registered_hw_config, stage_names, **overrides):
    n_warmup = planner_module.warmup_count(len(stage_names))
    good_layer_configs = [
        {"layer_name": name, "weight_bits": 8, "activation_bits": 8, "column_pruning_ratio": 0.0}
        for name in stage_names
    ]
    history = [{"layer_configs": good_layer_configs, "accuracy": 0.95} for _ in range(n_warmup)]
    state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id, candidate_history=history)
    state.update(overrides)
    return state


def test_default_search_strategy_is_multi_objective(monkeypatch, state_factory, registered_hw_config):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    stage_names = list(FAKE_STAGES.keys())
    state = _post_warmup_state(state_factory, registered_hw_config, stage_names)

    _, search_tag = planner_module._propose_stage_points(state, stage_names)
    assert "surrogate-model acquisition" in search_tag
    assert "ablation baseline" not in search_tag


def test_random_search_strategy_produces_random_tag_and_no_surrogate(monkeypatch, state_factory, registered_hw_config):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    stage_names = list(FAKE_STAGES.keys())
    state = _post_warmup_state(state_factory, registered_hw_config, stage_names, search_strategy="random")

    stage_points, search_tag = planner_module._propose_stage_points(state, stage_names)
    assert "random-search acquisition" in search_tag
    assert len(stage_points) == len(stage_names)


def test_single_objective_search_strategy_produces_expected_tag(monkeypatch, state_factory, registered_hw_config):
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    stage_names = list(FAKE_STAGES.keys())
    state = _post_warmup_state(state_factory, registered_hw_config, stage_names, search_strategy="single_objective")

    stage_points, search_tag = planner_module._propose_stage_points(state, stage_names)
    assert "single-objective surrogate acquisition" in search_tag
    assert len(stage_points) == len(stage_names)


def test_random_search_strategy_varies_point_across_iterations(monkeypatch, state_factory, registered_hw_config):
    # propose_stage_candidate_random has no history to diversify against,
    # so nodes/planner.py must offset its seed by len(history) itself --
    # otherwise every post-warmup iteration under the random strategy
    # would propose the exact same point forever.
    monkeypatch.setattr(planner_module, "get_layer_groups", lambda model_id: dict(FAKE_STAGES))
    stage_names = list(FAKE_STAGES.keys())
    state = _post_warmup_state(state_factory, registered_hw_config, stage_names, search_strategy="random")

    first_points, _ = planner_module._propose_stage_points(state, stage_names)
    state["candidate_history"] = state["candidate_history"] + [state["candidate_history"][-1]]
    second_points, _ = planner_module._propose_stage_points(state, stage_names)
    assert first_points != second_points


# --- pruning snap wiring ----------------------------------------------------


def test_points_to_stage_configs_without_context_keeps_the_raw_clamp():
    """Callers with no resolved model/hardware (tests, optuna_compare's own
    search space) must keep the pre-snap behavior exactly."""
    configs = planner_module.points_to_stage_configs([(4.4, 0.2)], ["layer4.1"])

    assert configs[0].column_pruning_ratio == 0.2
    assert configs[0].weight_bits == 4


def test_points_to_stage_configs_snaps_pruning_when_given_real_context(registered_hw_config):
    """With real shapes+hardware, a pruning ratio that removes no crossbar
    array is dropped to one that does (tools/cim_physics.py's
    snap_pruning_ratio) -- weight_bits rounding is unaffected."""
    from middleware import get_hw_config

    hw = get_hw_config(registered_hw_config.hw_spec_id)
    shapes = {"wide": [(4608, 512, 1)], "narrow": [(147, 64, 1)]}

    configs = planner_module.points_to_stage_configs(
        [(4.4, 0.2), (6.6, 0.3)], ["wide", "narrow"], hw, shapes
    )

    assert configs[0].column_pruning_ratio == 0.0  # 0.2 removes no array on a 512-col layer
    assert configs[1].column_pruning_ratio == 0.0  # 64-col layer can never save one
    assert [c.weight_bits for c in configs] == [4, 7]


def test_snapping_context_degrades_to_none_for_an_unresolvable_model():
    """An unimportable custom --model-id must fall back to the raw clamp,
    not fail the planner -- @tuner already reports that error."""
    assert planner_module._snapping_context("no.such.module:build", "whatever") == (None, None)
