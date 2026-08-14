"""tools/calibration.py: seeding @profiler's `calibration_factors` from a
real, cited literature reference (not an invented number), and staying
honestly uncalibrated (factor 1.0) for any HWConfig that doesn't exactly
match a known reference.
"""

from schemas.calibration import CalibrationReference
from schemas.config import CIMDeviceType, HWConfig, NoCTopology
from schemas.tools import LayerBitConfig
from tools.calibration import (
    KNOWN_REFERENCES,
    bootstrap_approximate_calibration_factors,
    bootstrap_approximate_calibration_provenance,
    bootstrap_calibration_factors,
    bootstrap_calibration_provenance,
    compute_approximate_calibration_factor,
    compute_calibration_factor,
    describe_approximate_calibration,
    describe_calibration,
    find_matching_reference,
    find_nearest_reference,
)
from tools.cim_physics import simulate_cim_profile

_REFERENCE = KNOWN_REFERENCES[0]  # the NeuroSim V1.5 128x128/7-bit-ADC point


def _hw(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw",
        crossbar_rows=_REFERENCE.crossbar_rows,
        crossbar_cols=_REFERENCE.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_exact_matching_hw_config():
    factors = bootstrap_calibration_factors(_hw())
    assert list(factors.keys()) == ["calib_test_hw"]
    assert factors["calib_test_hw"] > 0


def test_bootstrap_is_empty_when_adc_bits_differs():
    assert bootstrap_calibration_factors(_hw(adc_bits=_REFERENCE.adc_bits + 1)) == {}


def test_bootstrap_is_empty_when_crossbar_size_differs():
    assert bootstrap_calibration_factors(_hw(crossbar_rows=64)) == {}
    assert bootstrap_calibration_factors(_hw(crossbar_cols=64)) == {}


def test_compute_calibration_factor_equals_reference_over_predicted_per_mac():
    hw = _hw()
    probe = LayerBitConfig(
        layer_name="_calibration_probe",
        weight_bits=_REFERENCE.weight_bits,
        activation_bits=_REFERENCE.weight_bits,
        column_pruning_ratio=0.0,
    )
    # compute_energy_pj (crossbar/ADC/DAC only), not the total energy_pj --
    # published references never measure a host SRAM buffer's energy.
    predicted_energy_pj_per_mac = simulate_cim_profile(hw, [probe])["compute_energy_pj"] / (
        hw.crossbar_rows * hw.crossbar_cols
    )

    factor = compute_calibration_factor(hw, _REFERENCE)

    assert factor == _REFERENCE.reference_energy_pj_per_mac / predicted_energy_pj_per_mac


def test_known_reference_has_a_real_citation():
    # Guards against a future reference entry being added without a source
    # -- this module's entire premise is "grounded in real data, not
    # invented," so an uncited entry would be a regression of that premise.
    for reference in KNOWN_REFERENCES:
        assert reference.source.strip()


# --- Calibration provenance (which reference, and its stated uncertainty) --


def test_find_matching_reference_returns_the_exact_match():
    assert find_matching_reference(_hw()) is _REFERENCE


def test_find_matching_reference_is_none_when_nothing_matches():
    assert find_matching_reference(_hw(adc_bits=_REFERENCE.adc_bits + 1)) is None


def test_describe_calibration_surfaces_source_and_note_for_a_calibrated_hw():
    described = describe_calibration(_hw())
    assert described["source"] == _REFERENCE.source
    assert described["note"] == _REFERENCE.note
    assert described["reference_energy_pj_per_mac"] == _REFERENCE.reference_energy_pj_per_mac


def test_describe_calibration_is_none_for_an_uncalibrated_hw():
    assert describe_calibration(_hw(crossbar_rows=64)) is None


def test_bootstrap_calibration_provenance_matches_bootstrap_calibration_factors_keys():
    """Both bootstrap_* functions must agree on which hw_spec_ids get
    populated -- a factor with no matching provenance (or vice versa) would
    let tools/dashboard.py show a number with no traceable source, or a
    citation for a candidate whose energy wasn't actually corrected by it."""
    hw = _hw()
    assert set(bootstrap_calibration_provenance(hw).keys()) == set(bootstrap_calibration_factors(hw).keys())
    assert bootstrap_calibration_provenance(hw) == {hw.hw_spec_id: describe_calibration(hw)}


def test_bootstrap_calibration_provenance_is_empty_when_uncalibrated():
    assert bootstrap_calibration_provenance(_hw(crossbar_rows=64)) == {}


# --- Second reference (Correll et al. 2025, 256x64/8-bit-ADC/3-bit-weight) --

_REFERENCE_2 = KNOWN_REFERENCES[1]


def _hw2(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_2",
        crossbar_rows=_REFERENCE_2.crossbar_rows,
        crossbar_cols=_REFERENCE_2.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_2.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_second_reference_exact_match():
    factors = bootstrap_calibration_factors(_hw2())
    assert list(factors.keys()) == ["calib_test_hw_2"]
    assert factors["calib_test_hw_2"] > 0


def test_two_references_do_not_cross_match():
    # find_matching_reference must pick the exact reference for each distinct
    # (crossbar_rows, crossbar_cols, adc_bits) triple, never the other one.
    assert find_matching_reference(_hw()) is _REFERENCE
    assert find_matching_reference(_hw2()) is _REFERENCE_2


# --- Third reference (Chen, Chen & Gu 2021, 64x32/5-bit-ADC/4-bit-weight) --

_REFERENCE_3 = KNOWN_REFERENCES[2]


def _hw3(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_3",
        crossbar_rows=_REFERENCE_3.crossbar_rows,
        crossbar_cols=_REFERENCE_3.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_3.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_third_reference_exact_match():
    factors = bootstrap_calibration_factors(_hw3())
    assert list(factors.keys()) == ["calib_test_hw_3"]
    assert factors["calib_test_hw_3"] > 0


def test_three_references_do_not_cross_match():
    assert find_matching_reference(_hw()) is _REFERENCE
    assert find_matching_reference(_hw2()) is _REFERENCE_2
    assert find_matching_reference(_hw3()) is _REFERENCE_3


# --- Fourth & fifth references (Yu et al. 2020, 128x128/1-bit- and
# 5-bit-ADC/1-bit-weight -- same physical macro, two ADC-mode endpoints) --

_REFERENCE_4 = KNOWN_REFERENCES[3]  # adc_bits=1 endpoint
_REFERENCE_5 = KNOWN_REFERENCES[4]  # adc_bits=5 endpoint


def _hw4(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_4",
        crossbar_rows=_REFERENCE_4.crossbar_rows,
        crossbar_cols=_REFERENCE_4.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_4.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def _hw5(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_5",
        crossbar_rows=_REFERENCE_5.crossbar_rows,
        crossbar_cols=_REFERENCE_5.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_5.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_fourth_and_fifth_reference_exact_match():
    factors_4 = bootstrap_calibration_factors(_hw4())
    factors_5 = bootstrap_calibration_factors(_hw5())
    assert list(factors_4.keys()) == ["calib_test_hw_4"]
    assert list(factors_5.keys()) == ["calib_test_hw_5"]
    assert factors_4["calib_test_hw_4"] > 0
    assert factors_5["calib_test_hw_5"] > 0


def test_fourth_and_fifth_references_are_distinguished_by_adc_bits_alone():
    # Same crossbar_rows/crossbar_cols (128x128) as each other and as the
    # first (NeuroSim) reference -- only adc_bits (1 vs. 5 vs. 7) tells them
    # apart. This is the one pair in KNOWN_REFERENCES where crossbar size
    # alone is not enough to disambiguate.
    assert find_matching_reference(_hw4()) is _REFERENCE_4
    assert find_matching_reference(_hw5()) is _REFERENCE_5
    assert find_matching_reference(_hw()) is _REFERENCE


# --- Sixth reference (Garg, Jia, Phadke & Yu 2026 FeFET, 64x64/4-bit-ADC) --
#
# Shares its (64, 64, 4-bit-ADC) key with the ninth reference (Dong/Sinangil
# SRAM, added below) -- the one real collision in KNOWN_REFERENCES,
# resolved by device_type. FeFET itself has no representable device_type
# (schemas.config.CIMDeviceType has no FeFET member -- hwcomponents_neurosim
# ships no FeFET cell), so it can no longer be reached by exact match now
# that Dong coexists at this key: a device_type-unset query is honestly
# ambiguous between them (see test_sixth_reference_key_is_ambiguous_without_
# device_type below), not silently resolved to either.

_REFERENCE_6 = KNOWN_REFERENCES[5]


def _hw6(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_6",
        crossbar_rows=_REFERENCE_6.crossbar_rows,
        crossbar_cols=_REFERENCE_6.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_6.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_sixth_reference_key_is_ambiguous_without_device_type():
    # _hw6() has no device_type set, and its (64, 64, 4-bit-ADC) key is now
    # shared by two references (FeFET, unrepresentable device_type=None,
    # and Dong/Sinangil SRAM, device_type=SRAM) -- find_matching_reference
    # must refuse to guess between them rather than silently picking either.
    assert find_matching_reference(_hw6()) is None
    assert bootstrap_calibration_factors(_hw6()) == {}


def test_six_references_do_not_cross_match():
    assert find_matching_reference(_hw()) is _REFERENCE
    assert find_matching_reference(_hw2()) is _REFERENCE_2
    assert find_matching_reference(_hw3()) is _REFERENCE_3
    assert find_matching_reference(_hw4()) is _REFERENCE_4
    assert find_matching_reference(_hw5()) is _REFERENCE_5
    assert find_matching_reference(_hw6()) is None  # ambiguous -- see test above


# --- Seventh reference (Deaville, Zhang & Verma 2022, 256x512/6-bit-ADC/1-bit-weight, MRAM) --

_REFERENCE_7 = KNOWN_REFERENCES[6]


def _hw7(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_7",
        crossbar_rows=_REFERENCE_7.crossbar_rows,
        crossbar_cols=_REFERENCE_7.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_7.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_seventh_reference_exact_match():
    factors = bootstrap_calibration_factors(_hw7())
    assert list(factors.keys()) == ["calib_test_hw_7"]
    assert factors["calib_test_hw_7"] > 0


def test_seven_references_do_not_cross_match():
    assert find_matching_reference(_hw()) is _REFERENCE
    assert find_matching_reference(_hw2()) is _REFERENCE_2
    assert find_matching_reference(_hw3()) is _REFERENCE_3
    assert find_matching_reference(_hw4()) is _REFERENCE_4
    assert find_matching_reference(_hw5()) is _REFERENCE_5
    assert find_matching_reference(_hw6()) is None  # ambiguous -- see FeFET section above
    assert find_matching_reference(_hw7()) is _REFERENCE_7


# --- Eighth reference (Korea Univ. / arXiv:2211.16008, 256x80/4-bit-ADC/8-bit-weight) --

_REFERENCE_8 = KNOWN_REFERENCES[7]


def _hw8(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_8",
        crossbar_rows=_REFERENCE_8.crossbar_rows,
        crossbar_cols=_REFERENCE_8.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_8.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_eighth_reference_exact_match():
    factors = bootstrap_calibration_factors(_hw8())
    assert list(factors.keys()) == ["calib_test_hw_8"]
    assert factors["calib_test_hw_8"] > 0


def test_eight_references_do_not_cross_match():
    assert find_matching_reference(_hw()) is _REFERENCE
    assert find_matching_reference(_hw2()) is _REFERENCE_2
    assert find_matching_reference(_hw3()) is _REFERENCE_3
    assert find_matching_reference(_hw4()) is _REFERENCE_4
    assert find_matching_reference(_hw5()) is _REFERENCE_5
    assert find_matching_reference(_hw6()) is None  # ambiguous -- see FeFET section above
    assert find_matching_reference(_hw7()) is _REFERENCE_7
    assert find_matching_reference(_hw8()) is _REFERENCE_8


# --- Ninth reference (Dong, Sinangil et al. 2020, 64x64/4-bit-ADC/4-bit-weight,
# SRAM -- restored at the same key as the sixth (FeFET) reference now that
# device_type disambiguates them) --

_REFERENCE_9 = KNOWN_REFERENCES[8]


def _hw9(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_9",
        crossbar_rows=_REFERENCE_9.crossbar_rows,
        crossbar_cols=_REFERENCE_9.crossbar_cols,
        num_tiles=16,
        adc_bits=_REFERENCE_9.adc_bits,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
        device_type=_REFERENCE_9.device_type,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_bootstrap_returns_factor_for_ninth_reference_exact_match():
    factors = bootstrap_calibration_factors(_hw9())
    assert list(factors.keys()) == ["calib_test_hw_9"]
    assert factors["calib_test_hw_9"] > 0


def test_ninth_reference_is_disambiguated_from_sixth_by_device_type():
    # Same (64, 64, 4-bit-ADC) key as _hw6(), but device_type=SRAM picks out
    # the Dong/Sinangil reference specifically instead of returning None.
    assert find_matching_reference(_hw9()) is _REFERENCE_9
    assert find_matching_reference(_hw9()) is not _REFERENCE_6


def test_nine_references_do_not_cross_match():
    assert find_matching_reference(_hw()) is _REFERENCE
    assert find_matching_reference(_hw2()) is _REFERENCE_2
    assert find_matching_reference(_hw3()) is _REFERENCE_3
    assert find_matching_reference(_hw4()) is _REFERENCE_4
    assert find_matching_reference(_hw5()) is _REFERENCE_5
    assert find_matching_reference(_hw6()) is None  # ambiguous -- see FeFET section above
    assert find_matching_reference(_hw7()) is _REFERENCE_7
    assert find_matching_reference(_hw8()) is _REFERENCE_8
    assert find_matching_reference(_hw9()) is _REFERENCE_9


# --- find_matching_reference's device_type disambiguation, general contract
# (synthetic references via monkeypatched KNOWN_REFERENCES -- exercises the
# mechanism directly, independent of which real collisions currently exist
# in KNOWN_REFERENCES) --


def _synthetic_reference(device_type, weight_bits=4) -> CalibrationReference:
    return CalibrationReference(
        crossbar_rows=32,
        crossbar_cols=32,
        adc_bits=6,
        weight_bits=weight_bits,
        reference_energy_pj_per_mac=0.01,
        source="synthetic test fixture, not a real citation",
        device_type=device_type,
    )


def _synthetic_hw(**overrides) -> HWConfig:
    base = dict(
        hw_spec_id="calib_test_hw_synthetic",
        crossbar_rows=32,
        crossbar_cols=32,
        num_tiles=16,
        adc_bits=6,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_matching_reference_picks_the_device_type_matching_candidate_among_three(monkeypatch):
    rram_ref = _synthetic_reference(CIMDeviceType.RRAM)
    sram_ref = _synthetic_reference(CIMDeviceType.SRAM)
    pcm_ref = _synthetic_reference(CIMDeviceType.PCM)
    monkeypatch.setattr("tools.calibration.KNOWN_REFERENCES", [rram_ref, sram_ref, pcm_ref])

    assert find_matching_reference(_synthetic_hw(device_type=CIMDeviceType.RRAM)) is rram_ref
    assert find_matching_reference(_synthetic_hw(device_type=CIMDeviceType.SRAM)) is sram_ref
    assert find_matching_reference(_synthetic_hw(device_type=CIMDeviceType.PCM)) is pcm_ref


def test_matching_reference_is_none_when_queried_device_type_matches_no_tied_candidate(monkeypatch):
    rram_ref = _synthetic_reference(CIMDeviceType.RRAM)
    sram_ref = _synthetic_reference(CIMDeviceType.SRAM)
    monkeypatch.setattr("tools.calibration.KNOWN_REFERENCES", [rram_ref, sram_ref])

    # STTRAM matches neither tied candidate -- must not fall back to either.
    assert find_matching_reference(_synthetic_hw(device_type=CIMDeviceType.STTRAM)) is None


def test_matching_reference_is_none_when_device_type_unset_and_candidates_are_tied(monkeypatch):
    rram_ref = _synthetic_reference(CIMDeviceType.RRAM)
    sram_ref = _synthetic_reference(CIMDeviceType.SRAM)
    monkeypatch.setattr("tools.calibration.KNOWN_REFERENCES", [rram_ref, sram_ref])

    assert find_matching_reference(_synthetic_hw()) is None  # device_type defaults to None


def test_matching_reference_ignores_device_type_when_only_one_candidate_shares_the_key(monkeypatch):
    # A single reference at a key is returned regardless of device_type --
    # matches this function's original behavior for every non-colliding key
    # (most hw_configs never state a device_type at all).
    lone_ref = _synthetic_reference(CIMDeviceType.RRAM)
    monkeypatch.setattr("tools.calibration.KNOWN_REFERENCES", [lone_ref])

    assert find_matching_reference(_synthetic_hw()) is lone_ref
    assert find_matching_reference(_synthetic_hw(device_type=CIMDeviceType.SRAM)) is lone_ref


# --- Approximate (nearest-reference, scaling-law) calibration -------------


def _unmatched_hw(**overrides) -> HWConfig:
    """A HWConfig that exactly matches no KNOWN_REFERENCES entry -- distinct
    from every (crossbar_rows, crossbar_cols, adc_bits) triple currently in
    the list."""
    base = dict(
        hw_spec_id="calib_test_hw_unmatched",
        crossbar_rows=100,
        crossbar_cols=100,
        num_tiles=16,
        adc_bits=6,
        dac_bits=8,
        noc_topology=NoCTopology.MESH,
        noc_link_bandwidth_gbps=10.0,
        wire_resistance_ohm_per_um=0.05,
        device_noise_sigma=0.05,
        sram_buffer_kb=64.0,
    )
    base.update(overrides)
    return HWConfig(**base)


def test_find_nearest_reference_is_exact_zero_distance_for_an_exact_match():
    nearest, distance = find_nearest_reference(_hw())
    assert nearest is _REFERENCE
    assert distance == 0.0


def test_find_nearest_reference_returns_closest_by_distance_for_unmatched_hw():
    from tools.calibration import _reference_distance

    assert find_matching_reference(_unmatched_hw()) is None  # precondition: no exact match

    nearest, distance = find_nearest_reference(_unmatched_hw())

    assert nearest in KNOWN_REFERENCES
    assert distance > 0
    # No other reference should be strictly closer than the one returned.
    for reference in KNOWN_REFERENCES:
        assert distance <= _reference_distance(_unmatched_hw(), reference) + 1e-9


def test_find_nearest_reference_prefers_device_type_match_among_tied_distances():
    # Regression test for a real bug found via an actual end-to-end main.py
    # run (not caught by find_matching_reference's own tests): _hw9() (the
    # restored Dong/Sinangil SRAM reference's own shape, device_type=SRAM)
    # is tied at distance 0.0 with _REFERENCE_6 (FeFET, device_type=None) --
    # since find_nearest_reference is main.py's actual *default* path
    # (allow_approximate_calibration=True), a naive min() would silently
    # return whichever tied reference comes first in KNOWN_REFERENCES'
    # list order (FeFET) regardless of hw.device_type, giving the wrong
    # calibration factor for a hw_config that explicitly states SRAM.
    nearest, distance = find_nearest_reference(_hw9())
    assert nearest is _REFERENCE_9
    assert nearest is not _REFERENCE_6
    assert distance == 0.0

    # And the two calibration paths (exact vs. approximate/default) must
    # now agree, since bootstrap_approximate_calibration_factors's whole
    # premise is "identical to the exact path whenever distance is 0".
    assert bootstrap_approximate_calibration_factors(_hw9()) == bootstrap_calibration_factors(_hw9())


def test_find_nearest_reference_ignores_device_type_when_hw_states_none(monkeypatch):
    # No basis to prefer either tied candidate when hw doesn't state a
    # device_type -- falls back to KNOWN_REFERENCES' list order (matches
    # this function's original, pre-device_type tie-break behavior).
    rram_ref = _synthetic_reference(CIMDeviceType.RRAM)
    sram_ref = _synthetic_reference(CIMDeviceType.SRAM)
    monkeypatch.setattr("tools.calibration.KNOWN_REFERENCES", [rram_ref, sram_ref])

    nearest, distance = find_nearest_reference(_synthetic_hw())  # device_type unset
    assert nearest is rram_ref  # first in list order
    assert distance == 0.0


def test_compute_approximate_calibration_factor_uses_reference_own_config_not_hw_config():
    # The nearest reference to _unmatched_hw() (100x100, 6-bit) among the
    # current KNOWN_REFERENCES is the NeuroSim 128x128/7-bit point (smallest
    # log2-array-size + adc_bits distance). The approximate factor must equal
    # compute_calibration_factor evaluated with the *reference's* own
    # crossbar_rows/cols/adc_bits, not the query hw's.
    hw = _unmatched_hw()
    nearest, _distance = find_nearest_reference(hw)

    factor = compute_approximate_calibration_factor(hw, nearest)

    reference_shaped_hw = hw.model_copy(
        update={"crossbar_rows": nearest.crossbar_rows, "crossbar_cols": nearest.crossbar_cols, "adc_bits": nearest.adc_bits}
    )
    assert factor == compute_calibration_factor(reference_shaped_hw, nearest)


def test_bootstrap_approximate_calibration_factors_returns_a_factor_for_an_unmatched_hw():
    hw = _unmatched_hw()
    assert bootstrap_calibration_factors(hw) == {}  # exact path stays honestly uncalibrated

    approximate_factors = bootstrap_approximate_calibration_factors(hw)

    assert list(approximate_factors.keys()) == ["calib_test_hw_unmatched"]
    assert approximate_factors["calib_test_hw_unmatched"] > 0


def test_bootstrap_approximate_calibration_factors_matches_exact_for_an_exact_match():
    # For a HWConfig that *does* exactly match a reference, the nearest
    # reference is that exact match (distance 0), so the approximate path
    # must agree with the exact path exactly.
    hw = _hw()
    assert bootstrap_approximate_calibration_factors(hw) == bootstrap_calibration_factors(hw)


def test_bootstrap_approximate_calibration_factors_is_empty_when_no_known_references(monkeypatch):
    import tools.calibration as calibration_module

    monkeypatch.setattr(calibration_module, "KNOWN_REFERENCES", [])
    assert find_nearest_reference(_hw()) is None
    assert bootstrap_approximate_calibration_factors(_hw()) == {}


def test_describe_approximate_calibration_flags_approximate_and_names_matched_config():
    hw = _unmatched_hw()
    nearest, distance = find_nearest_reference(hw)

    described = describe_approximate_calibration(hw)

    assert described["approximate"] is True
    assert described["source"] == nearest.source
    assert described["matched_crossbar_rows"] == nearest.crossbar_rows
    assert described["matched_crossbar_cols"] == nearest.crossbar_cols
    assert described["matched_adc_bits"] == nearest.adc_bits
    assert described["distance"] == round(distance, 4)


def test_bootstrap_approximate_calibration_provenance_matches_factors_keys():
    hw = _unmatched_hw()
    assert set(bootstrap_approximate_calibration_provenance(hw).keys()) == set(
        bootstrap_approximate_calibration_factors(hw).keys()
    )


def test_reference_distance_is_zero_only_for_an_exact_array_shape_match():
    """Regression: `_reference_distance` compared only the array's total
    cell count, so a differently-shaped array with the same product came
    out at distance 0.0 -- breaking this function's own stated "zero iff
    exact match" invariant. Found on a real run: `cim_v1_128x128`
    (128x128, 8-bit ADC) reported `approximate ... distance=0.0` against
    the 256x64/8-bit Correll reference, which reads as an exact match while
    converting half as many columns per pass.
    """
    from tools.calibration import _reference_distance

    hw = _hw()  # 128x128, 7-bit ADC -- _REFERENCE's own shape
    same_area_different_shape = CalibrationReference(
        crossbar_rows=hw.crossbar_rows * 2,
        crossbar_cols=hw.crossbar_cols // 2,  # identical rows*cols product
        adc_bits=hw.adc_bits,
        weight_bits=_REFERENCE.weight_bits,
        reference_energy_pj_per_mac=_REFERENCE.reference_energy_pj_per_mac,
        source=_REFERENCE.source,
        note=_REFERENCE.note,
    )

    assert _reference_distance(hw, _REFERENCE) == 0.0
    assert _reference_distance(hw, same_area_different_shape) > 0.0


def test_nearest_reference_prefers_the_same_array_shape_over_equal_cell_count():
    """The end-to-end consequence of the fix, on the hw_config this project
    actually demos with: 128x128/8-bit must resolve to the 128x128/7-bit
    NeuroSim reference (exact shape, one ADC bit away) rather than the
    256x64/8-bit one (exact ADC, 4x aspect-ratio mismatch)."""
    hw = _hw(adc_bits=8)  # no exact match: no 128x128/8-bit reference exists
    assert find_matching_reference(hw) is None

    nearest, distance = find_nearest_reference(hw)

    assert (nearest.crossbar_rows, nearest.crossbar_cols) == (hw.crossbar_rows, hw.crossbar_cols)
    assert distance == 1.0  # exactly the one-bit ADC gap, nothing else
