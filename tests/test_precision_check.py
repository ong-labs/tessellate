"""tools/precision_check.py's Stage-2 verification -- the ADC energy term
is a real model (hwcomponents_adc, arXiv:2404.06553) in every environment;
DAC/crossbar-MAC (hwcomponents_neurosim) and SRAM buffer-staging energy
(hwcomponents_cacti) are real models only where those Docker-only packages
are installed, and fall back to the analytical placeholder
(tools/cim_physics.py) otherwise (see that module's docstring for why).
Since whether hwcomponents_neurosim/hwcomponents_cacti are actually
installed depends on which environment runs this suite (absent on this
project's own Windows dev machine and in CI, present inside the Docker
image), fallback-path tests explicitly monkeypatch `_NEUROSIM_AVAILABLE`/
`_real_sram_buffer_energy_fj` to their "unavailable" state rather than
relying on them already being that value -- this keeps every test's
outcome deterministic regardless of where it runs, and isolates each term
from the others even though this project's real Dockerfile always installs
NeuroSim and CACTI together. Available-path tests instead monkeypatch the
real-model functions directly, so they verify simulate_precise_energy's
delta arithmetic without needing the real compiled binaries; they cannot
exercise `_nearest_neurosim_tech_node_m`/`_clamped_cacti_tech_node_m`/the
real `RowDrivers`/`MemoryCell`/`SRAM` calls themselves -- that only happens
inside the Docker image."""

import subprocess
from unittest.mock import MagicMock

from schemas.config import CIMDeviceType, HWConfig, NoCTopology
from schemas.tools import LayerBitConfig
from tools import precision_check
from tools.cim_physics import CIM_TECH_PARAMS, simulate_cim_profile
from tools.precision_check import (
    _DEFAULT_TECH_NODE_NM,
    _call_docker_precision_bridge,
    _neurosim_cell_config_filename,
    _real_adc_conversion_energy_fj,
    _real_crossbar_mac_energy_fj_per_pass,
    _real_dac_conversion_energy_fj,
    _real_sram_buffer_energy_fj,
    compute_updated_calibration_factor,
    simulate_precise_energy,
)

_LAYER_CONFIGS = [LayerBitConfig(layer_name="conv", weight_bits=6, activation_bits=6, column_pruning_ratio=0.1)]


def _hw(**overrides) -> HWConfig:
    defaults = dict(
        hw_spec_id="test_hw",
        crossbar_rows=128,
        crossbar_cols=128,
        num_tiles=16,
        adc_bits=8,
        dac_bits=4,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.001,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    defaults.update(overrides)
    return HWConfig(**defaults)


def test_real_adc_energy_scales_with_bit_width():
    # hwcomponents_adc's literature-fit model: more resolution costs more
    # energy per conversion -- the same physical direction the analytical
    # placeholder's 2**adc_bits term assumes, now from a real ADC survey.
    low = _real_adc_conversion_energy_fj(_hw(adc_bits=4))
    high = _real_adc_conversion_energy_fj(_hw(adc_bits=8))
    assert high > low > 0.0


def test_real_adc_energy_uses_default_tech_node_when_unset():
    unset = _real_adc_conversion_energy_fj(_hw(tech_node_nm=None))
    explicit_default = _real_adc_conversion_energy_fj(_hw(tech_node_nm=_DEFAULT_TECH_NODE_NM))
    assert unset == explicit_default


def test_neurosim_cell_config_filename_maps_each_device_type_to_a_distinct_file():
    # Pure filename lookup -- independent of _NEUROSIM_AVAILABLE, so this
    # runs identically whether or not hwcomponents_neurosim is installed.
    filenames = {
        device_type: _neurosim_cell_config_filename(device_type)
        for device_type in (CIMDeviceType.RRAM, CIMDeviceType.SRAM, CIMDeviceType.PCM, CIMDeviceType.STTRAM)
    }
    assert len(set(filenames.values())) == len(filenames)  # each device technology gets its own real .cell file
    for filename in filenames.values():
        assert filename.endswith(".cell")
        assert filename != "placeholder.cell"


def test_neurosim_cell_config_filename_falls_back_to_placeholder_when_device_type_unset():
    assert _neurosim_cell_config_filename(None) == "placeholder.cell"


def _stub_neurosim_component(monkeypatch, attr_name, cache=None):
    # hwcomponents_neurosim's own Crossbar cache is keyed on rows/cols/
    # tech_node/timing/etc. but NOT cell_config (confirmed by inspecting
    # its cache-key construction, and by direct measurement inside the
    # Docker image: two hw_specs sharing everything but device_type
    # returned identical "real" energy until the cache was cleared between
    # them). _real_dac_conversion_energy_fj/_real_crossbar_mac_energy_fj_per_pass
    # must clear hwcomponents_neurosim's shared CACHE before every call, or a
    # different device_type silently reuses a stale simulation from
    # whichever cell ran first. This doesn't need the real package: it
    # stubs RowDrivers/MemoryCell/ColDrivers and spies on the cache's
    # .clear().
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", True)
    monkeypatch.setattr(precision_check, "_NEUROSIM_TECH_NODES_M", [32e-9], raising=False)
    monkeypatch.setattr(precision_check, "_NEUROSIM_SCRIPT_DIR", "/fake/neurosim", raising=False)
    if cache is None:
        cache = MagicMock()
    monkeypatch.setattr(precision_check, "_NEUROSIM_CACHE", cache, raising=False)
    dummy_component = MagicMock()
    dummy_component.read.return_value.energy = 1e-13
    monkeypatch.setattr(precision_check, attr_name, MagicMock(return_value=dummy_component), raising=False)
    return cache


def test_real_dac_energy_clears_the_neurosim_cache_before_each_call(monkeypatch):
    cache = _stub_neurosim_component(monkeypatch, "RowDrivers")
    _real_dac_conversion_energy_fj(_hw())
    cache.clear.assert_called_once()


def test_real_crossbar_mac_energy_clears_the_neurosim_cache_before_each_call(monkeypatch):
    cache = _stub_neurosim_component(monkeypatch, "MemoryCell")
    _stub_neurosim_component(monkeypatch, "ColDrivers", cache=cache)
    _real_crossbar_mac_energy_fj_per_pass(_hw())
    cache.clear.assert_called_once()


def test_real_crossbar_mac_energy_sums_memcell_and_precharge_energy(monkeypatch):
    # MemCellType: SRAM's real .cell file has no resistance/read-voltage
    # data (only cell geometry), so hwcomponents_neurosim's MemoryCell alone
    # always returns exactly 0.0 for it -- not a genuine near-zero
    # measurement, an architecturally-inapplicable one (see
    # _real_crossbar_mac_energy_fj_per_pass's docstring). ColDrivers with
    # adc_resolution=0 supplies the missing column-precharge contribution
    # that NeuroSim's own compiled core still counts for SRAM even when the
    # bitline-discharge term itself is zero.
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", True)
    monkeypatch.setattr(precision_check, "_NEUROSIM_TECH_NODES_M", [32e-9], raising=False)
    monkeypatch.setattr(precision_check, "_NEUROSIM_SCRIPT_DIR", "/fake/neurosim", raising=False)
    monkeypatch.setattr(precision_check, "_NEUROSIM_CACHE", MagicMock(), raising=False)

    memcell = MagicMock()
    memcell.read.return_value.energy = 0.0  # SRAM's real, architecturally-zero value
    col_drivers = MagicMock()
    col_drivers.read.return_value.energy = 1.7375e-14  # 17.375 fJ, a real ColDrivers(adc_resolution=0) measurement

    monkeypatch.setattr(precision_check, "MemoryCell", MagicMock(return_value=memcell), raising=False)
    monkeypatch.setattr(precision_check, "ColDrivers", MagicMock(return_value=col_drivers), raising=False)

    result = _real_crossbar_mac_energy_fj_per_pass(_hw(device_type=CIMDeviceType.SRAM))
    assert result == 17.375


def _force_no_real_buffer(monkeypatch):
    # Isolates ADC/DAC/crossbar-only tests from the SRAM buffer term --
    # forced regardless of environment, since this project's real Dockerfile
    # always installs hwcomponents_cacti alongside hwcomponents_neurosim
    # (both live at once inside the Docker image).
    monkeypatch.setattr(precision_check, "_real_sram_buffer_energy_fj", lambda hw: None)


def test_simulate_precise_energy_replaces_only_the_adc_term(monkeypatch):
    # Forced False regardless of environment: isolates the ADC-only claim
    # even when run inside the Docker image, where hwcomponents_neurosim
    # is actually installed and would otherwise also swap DAC/crossbar-MAC.
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    _force_no_real_buffer(monkeypatch)
    hw = _hw(adc_bits=8)
    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    assert result["raw_energy_pj"] == raw_profile["energy_pj"]

    n_conversions = hw.crossbar_cols * len(_LAYER_CONFIGS)
    analytical_adc_pj = CIM_TECH_PARAMS.adc_ref_energy_fj * (2**hw.adc_bits) * n_conversions / 1000.0
    real_adc_pj = _real_adc_conversion_energy_fj(hw) * n_conversions / 1000.0
    expected = round(raw_profile["energy_pj"] - analytical_adc_pj + real_adc_pj, 6)
    assert result["precise_energy_pj"] == expected


def test_real_dac_and_crossbar_energy_are_none_without_neurosim(monkeypatch):
    # Forced False regardless of environment (see module docstring) --
    # both real-model functions must report "nothing to swap in" rather
    # than crash whenever hwcomponents_neurosim isn't available.
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    assert _real_dac_conversion_energy_fj(_hw()) is None
    assert _real_crossbar_mac_energy_fj_per_pass(_hw()) is None


def test_simulate_precise_energy_dac_and_crossbar_terms_untouched_without_neurosim(monkeypatch):
    # Isolate the claim in the module docstring for the "no
    # hwcomponents_neurosim" case (forced False regardless of environment):
    # only the ADC term should differ between the analytical and "precise"
    # numbers here -- if the DAC/crossbar-MAC terms also silently changed
    # without the real package available, the fallback contract would be
    # broken.
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    _force_no_real_buffer(monkeypatch)
    hw_low_adc = _hw(adc_bits=4)  # >= _MIN_REAL_ADC_BITS, real model applies to both
    hw_high_adc = _hw(adc_bits=16)
    # Difference in raw (analytical) energy between these two hw specs is
    # entirely attributable to the ADC term (crossbar/DAC don't depend on
    # adc_bits at all in tools.cim_physics.simulate_cim_profile).
    raw_low = simulate_cim_profile(hw_low_adc, _LAYER_CONFIGS)["energy_pj"]
    raw_high = simulate_cim_profile(hw_high_adc, _LAYER_CONFIGS)["energy_pj"]
    precise_low = simulate_precise_energy(hw_low_adc, _LAYER_CONFIGS)["precise_energy_pj"]
    precise_high = simulate_precise_energy(hw_high_adc, _LAYER_CONFIGS)["precise_energy_pj"]
    # Both raw and precise must move in the same direction (more ADC bits
    # -> more energy) even though the magnitude of the ADC contribution
    # itself differs between the analytical and real models.
    assert raw_high > raw_low
    assert precise_high > precise_low


def test_simulate_precise_energy_falls_back_to_analytical_below_min_real_adc_bits(monkeypatch):
    # hwcomponents_adc.ADC asserts n_bits >= 4 -- a hw_spec with fewer ADC
    # bits than that (HWConfig.adc_bits itself allows down to 1) must not
    # crash simulate_precise_energy or fabricate a number the real model
    # was never fit to produce; it should just keep the analytical value.
    # _NEUROSIM_AVAILABLE forced False so DAC/crossbar-MAC don't also swap
    # in for real (which would happen inside the Docker image) and mask
    # what this test is actually isolating: the ADC-bits fallback alone.
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    _force_no_real_buffer(monkeypatch)
    hw = _hw(adc_bits=2)
    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)
    assert result["precise_energy_pj"] == raw_profile["energy_pj"]


def test_real_adc_conversion_energy_is_none_below_min_bits():
    assert _real_adc_conversion_energy_fj(_hw(adc_bits=3)) is None


def test_simulate_precise_energy_replaces_dac_term_when_neurosim_available(monkeypatch):
    # adc_bits=2 disables the real ADC term (below _MIN_REAL_ADC_BITS); the
    # crossbar-MAC function is also explicitly forced to None so the DAC
    # swap is the only source of any raw/precise difference regardless of
    # whether this suite happens to run where hwcomponents_neurosim is
    # actually installed (the Docker image) or not.
    hw = _hw(adc_bits=2, dac_bits=4)
    monkeypatch.setattr(precision_check, "_real_dac_conversion_energy_fj", lambda hw: 500.0)
    monkeypatch.setattr(precision_check, "_real_crossbar_mac_energy_fj_per_pass", lambda hw: None)
    _force_no_real_buffer(monkeypatch)

    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    n_layers = len(_LAYER_CONFIGS)
    analytical_dac_fj = CIM_TECH_PARAMS.dac_ref_energy_fj * hw.dac_bits * hw.crossbar_rows * n_layers
    real_dac_fj = 500.0 * n_layers
    expected = round(raw_profile["energy_pj"] + (real_dac_fj - analytical_dac_fj) / 1000.0, 6)
    assert result["precise_energy_pj"] == expected


def test_simulate_precise_energy_replaces_crossbar_term_when_neurosim_available(monkeypatch):
    # adc_bits=2 disables the real ADC term; the DAC function is also
    # explicitly forced to None so the crossbar swap is isolated regardless
    # of whether hwcomponents_neurosim is actually installed here.
    hw = _hw(adc_bits=2)
    monkeypatch.setattr(precision_check, "_real_crossbar_mac_energy_fj_per_pass", lambda hw: 20.0)
    monkeypatch.setattr(precision_check, "_real_dac_conversion_energy_fj", lambda hw: None)
    _force_no_real_buffer(monkeypatch)

    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    analytical_crossbar_fj = sum(
        CIM_TECH_PARAMS.crossbar_ref_energy_fj * hw.crossbar_rows * hw.crossbar_cols * lc.weight_bits
        for lc in _LAYER_CONFIGS
    )
    real_crossbar_fj = sum(20.0 * lc.weight_bits for lc in _LAYER_CONFIGS)
    expected = round(raw_profile["energy_pj"] + (real_crossbar_fj - analytical_crossbar_fj) / 1000.0, 6)
    assert result["precise_energy_pj"] == expected


def test_real_sram_buffer_energy_is_none_without_cacti(monkeypatch):
    # Forced unavailable regardless of environment (see module docstring).
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", False)
    assert _real_sram_buffer_energy_fj(_hw(sram_buffer_kb=64.0)) is None


def test_real_sram_buffer_energy_is_none_when_sram_buffer_kb_unset():
    # Nothing to model -- HWConfig.sram_buffer_kb's own gt=0 constraint
    # means this is the "not every hardware spec states one" case, distinct
    # from _CACTI_AVAILABLE being False.
    assert _real_sram_buffer_energy_fj(_hw(sram_buffer_kb=None)) is None


def test_simulate_precise_energy_replaces_buffer_term_when_cacti_available(monkeypatch):
    # adc_bits=2 disables the real ADC term; DAC/crossbar are also
    # explicitly forced to None so the buffer swap is isolated regardless
    # of whether hwcomponents_neurosim/hwcomponents_cacti are actually
    # installed here.
    hw = _hw(adc_bits=2, sram_buffer_kb=64.0)
    monkeypatch.setattr(precision_check, "_real_dac_conversion_energy_fj", lambda hw: None)
    monkeypatch.setattr(precision_check, "_real_crossbar_mac_energy_fj_per_pass", lambda hw: None)
    monkeypatch.setattr(precision_check, "_real_sram_buffer_energy_fj", lambda hw: 50_000.0)

    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    analytical_buffer_fj = CIM_TECH_PARAMS.sram_buffer_ref_energy_fj_per_bit * (hw.sram_buffer_kb * 8 * 1024)
    expected = round(raw_profile["energy_pj"] + (50_000.0 - analytical_buffer_fj) / 1000.0, 6)
    assert result["precise_energy_pj"] == expected
    # compute_energy_pj is untouched -- only the buffer term was replaced.
    assert result["precise_compute_energy_pj"] == result["raw_compute_energy_pj"]


def test_simulate_precise_energy_exposes_compute_only_pair_for_calibration(monkeypatch):
    # raw_compute_energy_pj/precise_compute_energy_pj must exclude
    # buffer_energy_pj -- this is what compute_updated_calibration_factor
    # (tools/simulators.py's precision_check_tool) is evaluated against,
    # not the totals, mirroring tools.calibration.compute_calibration_factor.
    # Forced False/None regardless of environment (adc_bits=2 alone isn't
    # enough inside the Docker image, where hwcomponents_neurosim/_cacti
    # are actually installed and would otherwise replace DAC/crossbar/buffer
    # for real, defeating this test's "nothing replaced" premise).
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    _force_no_real_buffer(monkeypatch)
    hw = _hw(adc_bits=2, sram_buffer_kb=64.0)  # ADC below _MIN_REAL_ADC_BITS
    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    assert result["raw_compute_energy_pj"] == raw_profile["compute_energy_pj"]
    assert result["precise_compute_energy_pj"] == raw_profile["compute_energy_pj"]
    assert result["raw_energy_pj"] == raw_profile["energy_pj"]
    assert result["precise_energy_pj"] == raw_profile["energy_pj"]


def test_compute_updated_calibration_factor_is_the_precise_over_raw_ratio():
    assert compute_updated_calibration_factor(100.0, 120.0) == 1.2


def test_compute_updated_calibration_factor_guards_against_zero_or_negative_raw_energy():
    assert compute_updated_calibration_factor(0.0, 120.0) == 1.0
    assert compute_updated_calibration_factor(-5.0, 120.0) == 1.0


def test_docker_bridge_not_attempted_by_default(monkeypatch):
    # AUTOCIM_PRECISION_BACKEND unset -> "local" default (same local-first/
    # opt-in-cloud pattern as AUTOCIM_PLANNER_MODEL): subprocess.run must
    # never be called, regardless of whether hwcomponents_neurosim/_cacti
    # are importable in this process.
    monkeypatch.delenv("AUTOCIM_PRECISION_BACKEND", raising=False)
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", False)
    run_spy = MagicMock()
    monkeypatch.setattr(subprocess, "run", run_spy)

    simulate_precise_energy(_hw(adc_bits=2), _LAYER_CONFIGS)  # adc_bits=2: skips the always-on ADC term too

    run_spy.assert_not_called()


def test_docker_bridge_skipped_when_local_real_packages_already_available(monkeypatch):
    # Already running where hwcomponents_neurosim/_cacti are both real (the
    # Docker image itself) -- must not shell out to `docker run` from
    # inside a container (no docker-in-docker).
    monkeypatch.setenv("AUTOCIM_PRECISION_BACKEND", "docker")
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", True)
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", True)
    run_spy = MagicMock()
    monkeypatch.setattr(subprocess, "run", run_spy)
    _force_no_real_buffer(monkeypatch)
    monkeypatch.setattr(precision_check, "_real_dac_conversion_energy_fj", lambda hw: None)
    monkeypatch.setattr(precision_check, "_real_crossbar_mac_energy_fj_per_pass", lambda hw: None)

    simulate_precise_energy(_hw(adc_bits=2), _LAYER_CONFIGS)

    run_spy.assert_not_called()


def test_docker_bridge_result_used_verbatim_when_it_succeeds(monkeypatch):
    monkeypatch.setenv("AUTOCIM_PRECISION_BACKEND", "docker")
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", False)
    bridged_result = {
        "raw_energy_pj": 1.0,
        "precise_energy_pj": 999.0,
        "raw_compute_energy_pj": 1.0,
        "precise_compute_energy_pj": 999.0,
    }
    monkeypatch.setattr(precision_check, "_call_docker_precision_bridge", lambda hw, lcs: bridged_result)

    result = simulate_precise_energy(_hw(), _LAYER_CONFIGS)

    assert result == bridged_result


def test_docker_bridge_falls_back_to_local_when_docker_missing(monkeypatch):
    monkeypatch.setenv("AUTOCIM_PRECISION_BACKEND", "docker")
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", False)
    _force_no_real_buffer(monkeypatch)
    hw = _hw(adc_bits=2)  # disables the ADC term too, isolating "nothing real got swapped in"

    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError("docker not found")))
    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    assert result["precise_energy_pj"] == raw_profile["energy_pj"]


def test_docker_bridge_falls_back_when_daemon_not_running(monkeypatch):
    monkeypatch.setenv("AUTOCIM_PRECISION_BACKEND", "docker")
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", False)
    _force_no_real_buffer(monkeypatch)
    hw = _hw(adc_bits=2)

    failed_proc = MagicMock(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=failed_proc))
    raw_profile = simulate_cim_profile(hw, _LAYER_CONFIGS)
    result = simulate_precise_energy(hw, _LAYER_CONFIGS)

    assert result["precise_energy_pj"] == raw_profile["energy_pj"]


def test_docker_bridge_falls_back_on_timeout(monkeypatch):
    monkeypatch.setenv("AUTOCIM_PRECISION_BACKEND", "docker")
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", False)
    monkeypatch.setattr(
        subprocess, "run", MagicMock(side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=120.0))
    )

    assert _call_docker_precision_bridge(_hw(), _LAYER_CONFIGS) is None


def test_docker_bridge_falls_back_on_malformed_output(monkeypatch):
    monkeypatch.setattr(precision_check, "_NEUROSIM_AVAILABLE", False)
    monkeypatch.setattr(precision_check, "_CACTI_AVAILABLE", False)
    ok_proc = MagicMock(returncode=0, stdout="not json")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=ok_proc))

    assert _call_docker_precision_bridge(_hw(), _LAYER_CONFIGS) is None


def test_docker_bridge_uses_configured_image_name(monkeypatch):
    # Forces the one-shot `docker run --rm` fallback (warm-keeping
    # "unavailable") so the image name -- irrelevant to `docker exec`,
    # which addresses the container by name instead -- actually appears in
    # the command this test inspects.
    monkeypatch.setattr(precision_check, "_ensure_docker_container_running", lambda image, container: False)
    monkeypatch.setenv("AUTOCIM_PRECISION_DOCKER_IMAGE", "my-custom-image:tag")
    ok_proc = MagicMock(returncode=0, stdout='{"raw_energy_pj": 1.0}')
    run_spy = MagicMock(return_value=ok_proc)
    monkeypatch.setattr(subprocess, "run", run_spy)

    _call_docker_precision_bridge(_hw(), _LAYER_CONFIGS)

    called_args = run_spy.call_args[0][0]
    assert "my-custom-image:tag" in called_args


def test_docker_bridge_defaults_to_autocim_agent_image(monkeypatch):
    monkeypatch.setattr(precision_check, "_ensure_docker_container_running", lambda image, container: False)
    monkeypatch.delenv("AUTOCIM_PRECISION_DOCKER_IMAGE", raising=False)
    ok_proc = MagicMock(returncode=0, stdout='{"raw_energy_pj": 1.0}')
    run_spy = MagicMock(return_value=ok_proc)
    monkeypatch.setattr(subprocess, "run", run_spy)

    _call_docker_precision_bridge(_hw(), _LAYER_CONFIGS)

    called_args = run_spy.call_args[0][0]
    assert "autocim-agent" in called_args


def test_docker_bridge_uses_docker_exec_when_container_already_warm(monkeypatch):
    monkeypatch.setattr(precision_check, "_ensure_docker_container_running", lambda image, container: True)
    monkeypatch.setenv("AUTOCIM_PRECISION_DOCKER_CONTAINER", "my-warm-container")
    ok_proc = MagicMock(returncode=0, stdout='{"raw_energy_pj": 1.0}')
    run_spy = MagicMock(return_value=ok_proc)
    monkeypatch.setattr(subprocess, "run", run_spy)

    _call_docker_precision_bridge(_hw(), _LAYER_CONFIGS)

    called_args = run_spy.call_args[0][0]
    assert called_args[:3] == ["docker", "exec", "-i"]
    assert "my-warm-container" in called_args
    assert "run" not in called_args  # never the one-shot `docker run --rm` path


def test_ensure_docker_container_running_reuses_an_already_running_container(monkeypatch):
    running_inspect = MagicMock(returncode=0, stdout="true\n")
    run_spy = MagicMock(return_value=running_inspect)
    monkeypatch.setattr(subprocess, "run", run_spy)

    assert precision_check._ensure_docker_container_running("autocim-agent", "my-container") is True
    # Only the inspect call -- never rm/run, since it was already running.
    run_spy.assert_called_once()
    assert run_spy.call_args[0][0][:2] == ["docker", "inspect"]


def test_ensure_docker_container_running_recreates_a_stopped_container(monkeypatch):
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return MagicMock(returncode=0, stdout="false\n")
        return MagicMock(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert precision_check._ensure_docker_container_running("autocim-agent", "my-container") is True
    assert calls[0][:2] == ["docker", "inspect"]
    assert calls[1][:3] == ["docker", "rm", "-f"]
    assert calls[2][:3] == ["docker", "run", "-d"]


def test_ensure_docker_container_running_returns_false_when_docker_missing(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError("docker not found")))
    assert precision_check._ensure_docker_container_running("autocim-agent", "my-container") is False
