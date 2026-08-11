"""Invariant tests for the gait scheduler."""

from __future__ import annotations

import numpy as np
import pytest

from quadruped_gait.algorithm.gait import (
    GAIT_LIBRARY,
    GAIT_NAMES,
    GaitParameters,
    contact_schedule,
    exact_duty_factors,
    exact_supported_fraction,
    gait,
    leg_phases,
    sampled_duty_factors,
    stance_count_durations,
    stance_count_extrema,
)
from quadruped_gait.model.geometry import LEG_COUNT, LegId

DUTY_TOLERANCE = 2e-3
# The closed form schedule is arithmetic on floats, so the only error it carries is
# rounding. This is many orders of magnitude below the sampling error it replaces.
EXACT_TOLERANCE = 1e-12


def _midpoint_times(parameters: GaitParameters, cycles: int, per_cycle: int) -> np.ndarray:
    """Return sample times at cell midpoints, which never land on a phase boundary."""
    count = cycles * per_cycle
    return (np.arange(count, dtype=np.float64) + 0.5) * parameters.period / per_cycle


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_every_library_gait_is_well_formed(name: str) -> None:
    parameters = gait(name)
    assert parameters.name == name
    assert 0.0 < parameters.duty_factor <= 1.0
    assert len(parameters.phase_offsets) == LEG_COUNT
    assert parameters.stance_duration + parameters.swing_duration == pytest.approx(
        parameters.period
    )


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_the_schedule_is_periodic_with_the_configured_period(name: str) -> None:
    parameters = gait(name)
    times = _midpoint_times(parameters, cycles=1, per_cycle=257)
    for shift in (1, 2, 5):
        base = contact_schedule(parameters, times)
        shifted = contact_schedule(parameters, times + shift * parameters.period)
        np.testing.assert_array_equal(base, shifted)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_phases_are_periodic_with_the_configured_period(name: str) -> None:
    parameters = gait(name)
    times = _midpoint_times(parameters, cycles=1, per_cycle=97)
    base = leg_phases(parameters, times)
    shifted = leg_phases(parameters, times + 3.0 * parameters.period)
    np.testing.assert_allclose(base, shifted, atol=1e-9)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_realised_duty_factor_matches_the_command(name: str) -> None:
    parameters = gait(name)
    times = _midpoint_times(parameters, cycles=4, per_cycle=2000)
    realised = sampled_duty_factors(contact_schedule(parameters, times))
    assert realised.shape == (LEG_COUNT,)
    np.testing.assert_allclose(realised, parameters.duty_factor, atol=DUTY_TOLERANCE)


@pytest.mark.parametrize("duty_factor", [0.35, 0.5, 0.62, 0.75, 0.88, 1.0])
def test_realised_duty_factor_tracks_an_overridden_command(duty_factor: float) -> None:
    parameters = gait("walk", duty_factor=duty_factor)
    times = _midpoint_times(parameters, cycles=4, per_cycle=2000)
    realised = sampled_duty_factors(contact_schedule(parameters, times))
    np.testing.assert_allclose(realised, duty_factor, atol=DUTY_TOLERANCE)


def test_trot_has_exactly_two_feet_in_stance_at_mid_phase() -> None:
    """A trot alternates diagonal pairs, so two feet carry the robot at all times."""
    parameters = gait("trot")
    times = _midpoint_times(parameters, cycles=3, per_cycle=512)
    schedule = contact_schedule(parameters, times)
    assert set(np.unique(schedule.sum(axis=1)).tolist()) == {2}
    mid_stance = parameters.period * 0.25
    state = parameters.contact_state(mid_stance)
    assert state.stance_count == 2
    assert set(state.stance_legs) == {LegId.FRONT_LEFT, LegId.HIND_RIGHT}


def test_trot_stance_pairs_are_diagonal() -> None:
    parameters = gait("trot")
    first = parameters.contact_state(parameters.period * 0.25).stance_legs
    second = parameters.contact_state(parameters.period * 0.75).stance_legs
    assert set(first) == {LegId.FRONT_LEFT, LegId.HIND_RIGHT}
    assert set(second) == {LegId.FRONT_RIGHT, LegId.HIND_LEFT}


def test_pace_stance_pairs_are_lateral() -> None:
    parameters = gait("pace")
    assert set(parameters.contact_state(parameters.period * 0.25).stance_legs) == {
        LegId.FRONT_LEFT,
        LegId.HIND_LEFT,
    }


def test_bound_stance_pairs_are_fore_and_hind() -> None:
    parameters = gait("bound")
    assert set(parameters.contact_state(parameters.period * 0.2).stance_legs) == {
        LegId.FRONT_LEFT,
        LegId.FRONT_RIGHT,
    }
    assert set(parameters.contact_state(parameters.period * 0.7).stance_legs) == {
        LegId.HIND_LEFT,
        LegId.HIND_RIGHT,
    }


def test_walk_keeps_at_least_three_feet_in_stance_throughout() -> None:
    """A quarter duty swing window per leg tiles the cycle without overlap."""
    parameters = gait("walk")
    times = _midpoint_times(parameters, cycles=3, per_cycle=1024)
    counts = contact_schedule(parameters, times).sum(axis=1)
    assert int(counts.min()) >= 3


@pytest.mark.parametrize("duty_factor", [0.75, 0.80, 0.85, 0.90])
def test_walk_support_count_follows_the_duty_factor(duty_factor: float) -> None:
    parameters = gait("walk", duty_factor=duty_factor)
    times = _midpoint_times(parameters, cycles=2, per_cycle=1024)
    counts = contact_schedule(parameters, times).sum(axis=1)
    assert int(counts.min()) >= 3
    assert float(counts.mean()) == pytest.approx(4.0 * duty_factor, abs=1e-2)


def test_bound_has_an_aerial_phase() -> None:
    parameters = gait("bound")
    times = _midpoint_times(parameters, cycles=2, per_cycle=1024)
    counts = contact_schedule(parameters, times).sum(axis=1)
    assert int(counts.min()) == 0
    assert float((counts == 0).mean()) == pytest.approx(0.2, abs=1e-2)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_touchdown_and_liftoff_bracket_the_current_time(name: str) -> None:
    parameters = gait(name)
    times = _midpoint_times(parameters, cycles=2, per_cycle=311)
    for time in times.tolist():
        for leg_id in LegId:
            touchdown = parameters.last_touchdown_time(time, leg_id)
            liftoff = parameters.last_liftoff_time(time, leg_id)
            assert touchdown <= time
            assert liftoff <= time
            assert time - touchdown < parameters.period + 1e-9


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_swing_and_stance_progress_are_complementary(name: str) -> None:
    parameters = gait(name)
    times = _midpoint_times(parameters, cycles=1, per_cycle=401)
    for time in times.tolist():
        for leg_id in LegId:
            loaded = parameters.in_stance(time, leg_id)
            swing = parameters.swing_phase(time, leg_id)
            stance = parameters.stance_phase(time, leg_id)
            assert 0.0 <= swing < 1.0
            assert 0.0 <= stance < 1.0
            assert (swing == 0.0) if loaded else (stance == 0.0)


def test_phase_offsets_match_the_published_footfall_orders() -> None:
    """Walk is lateral sequence, trot is diagonal, pace is lateral, bound is transverse."""
    assert GAIT_LIBRARY["walk"].phase_offsets == (0.0, 0.5, 0.75, 0.25)
    assert GAIT_LIBRARY["trot"].phase_offsets == (0.0, 0.5, 0.5, 0.0)
    assert GAIT_LIBRARY["pace"].phase_offsets == (0.0, 0.5, 0.0, 0.5)
    assert GAIT_LIBRARY["bound"].phase_offsets == (0.0, 0.0, 0.5, 0.5)


def test_walk_footfall_order_is_lateral_sequence() -> None:
    """Touchdowns follow hind-left, front-left, hind-right, front-right."""
    offsets = GAIT_LIBRARY["walk"].phase_offsets
    order = sorted(LegId, key=lambda leg: offsets[int(leg)])
    assert order == [LegId.FRONT_LEFT, LegId.HIND_RIGHT, LegId.FRONT_RIGHT, LegId.HIND_LEFT]


def test_gait_library_is_read_only() -> None:
    with pytest.raises(TypeError):
        GAIT_LIBRARY["walk"] = GAIT_LIBRARY["trot"]  # type: ignore[index]


def test_unknown_gait_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown gait"):
        gait("gallop")


@pytest.mark.parametrize(
    ("period", "duty_factor", "offsets"),
    [
        (0.0, 0.5, (0.0, 0.5, 0.5, 0.0)),
        (1.0, 0.0, (0.0, 0.5, 0.5, 0.0)),
        (1.0, 1.5, (0.0, 0.5, 0.5, 0.0)),
        (1.0, 0.5, (0.0, 0.5, 0.5, 1.0)),
        (1.0, 0.5, (0.0, 0.5, 0.5, -0.1)),
    ],
)
def test_invalid_parameters_are_rejected(
    period: float, duty_factor: float, offsets: tuple[float, float, float, float]
) -> None:
    with pytest.raises(ValueError):
        GaitParameters(name="bad", period=period, duty_factor=duty_factor, phase_offsets=offsets)


def test_schedule_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        sampled_duty_factors(np.zeros((5, 3), dtype=bool))
    with pytest.raises(ValueError, match="at least one sample"):
        sampled_duty_factors(np.zeros((0, LEG_COUNT), dtype=bool))


def test_retiming_preserves_the_schedule_shape() -> None:
    slow = gait("trot", period=2.0)
    fast = gait("trot", period=0.25)
    times_slow = _midpoint_times(slow, cycles=1, per_cycle=64)
    times_fast = _midpoint_times(fast, cycles=1, per_cycle=64)
    np.testing.assert_array_equal(
        contact_schedule(slow, times_slow), contact_schedule(fast, times_fast)
    )


# The closed form contact schedule. These tests are the ones that would fail if the
# realised duty factor went back to being counted from samples.


@pytest.mark.parametrize("name", GAIT_NAMES)
@pytest.mark.parametrize("cycles", [1, 2, 7])
def test_exact_duty_factor_over_whole_cycles_is_the_command(name: str, cycles: int) -> None:
    """Over a whole number of cycles the stance fraction is the duty factor exactly."""
    parameters = gait(name)
    exact = exact_duty_factors(parameters, 0.0, cycles * parameters.period)
    assert exact.shape == (LEG_COUNT,)
    np.testing.assert_allclose(exact, parameters.duty_factor, atol=EXACT_TOLERANCE)


@pytest.mark.parametrize("duty_factor", [0.35, 0.5, 0.62, 0.75, 0.88, 1.0])
def test_exact_duty_factor_tracks_an_overridden_command(duty_factor: float) -> None:
    parameters = gait("walk", duty_factor=duty_factor)
    exact = exact_duty_factors(parameters, 0.0, 3.0 * parameters.period)
    np.testing.assert_allclose(exact, duty_factor, atol=EXACT_TOLERANCE)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_exact_duty_factor_is_the_limit_the_sampled_one_approaches(name: str) -> None:
    """Refining the sampling rate drives the counted fraction onto the closed form."""
    parameters = gait(name)
    window = 3.0 * parameters.period
    exact = exact_duty_factors(parameters, 0.0, window)
    errors = []
    for per_cycle in (50, 500, 5000):
        times = _midpoint_times(parameters, cycles=3, per_cycle=per_cycle)
        counted = sampled_duty_factors(contact_schedule(parameters, times))
        errors.append(float(np.max(np.abs(counted - exact))))
        assert errors[-1] <= 1.5 / per_cycle
    assert errors[-1] <= errors[0]


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_exact_duty_factor_is_additive_over_a_split_window(name: str) -> None:
    """A measure is additive, so a window split anywhere must not change the total."""
    parameters = gait(name)
    start, middle, end = 0.37, 1.11, 2.93
    for leg_id in LegId:
        whole = parameters.stance_measure(leg_id, start, end)
        parts = parameters.stance_measure(leg_id, start, middle) + parameters.stance_measure(
            leg_id, middle, end
        )
        assert whole == pytest.approx(parts, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_exact_duty_factor_is_invariant_under_a_whole_cycle_shift(name: str) -> None:
    parameters = gait(name)
    period = parameters.period
    for leg_id in LegId:
        here = parameters.stance_measure(leg_id, 0.23, 1.61)
        later = parameters.stance_measure(leg_id, 0.23 + 4.0 * period, 1.61 + 4.0 * period)
        assert here == pytest.approx(later, abs=EXACT_TOLERANCE)


def test_stance_measure_is_computed_by_hand_on_a_partial_window() -> None:
    """A window that starts and ends mid stance, checked against the geometry."""
    parameters = GaitParameters(
        name="probe", period=1.0, duty_factor=0.4, phase_offsets=(0.0, 0.25, 0.5, 0.75)
    )
    # Leg 0 is loaded on [0.0, 0.4] of each cycle. Over [0.2, 1.3] that is
    # [0.2, 0.4] and [1.0, 1.3], which is 0.2 + 0.3 = 0.5 seconds.
    assert parameters.stance_measure(LegId.FRONT_LEFT, 0.2, 1.3) == pytest.approx(0.5)
    # The same window seen by leg 3, loaded on [0.75, 1.15] of each cycle, gives
    # [0.75, 1.15] intersected with [0.2, 1.3], which is 0.4 seconds.
    assert parameters.stance_measure(LegId.HIND_RIGHT, 0.2, 1.3) == pytest.approx(0.4)
    assert parameters.stance_fraction(LegId.FRONT_LEFT, 0.2, 1.3) == pytest.approx(0.5 / 1.1)


def test_stance_measure_spans_a_window_that_starts_before_zero() -> None:
    """The antiderivative must stay correct for negative times."""
    parameters = gait("walk", duty_factor=0.5)
    for leg_id in LegId:
        measure = parameters.stance_measure(leg_id, -2.0, 2.0)
        assert measure == pytest.approx(2.0, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_stance_intervals_agree_with_the_sampled_contact_flags(name: str) -> None:
    """Every sample falls inside an interval exactly when its contact flag is set."""
    parameters = gait(name)
    window = 2.0 * parameters.period
    times = _midpoint_times(parameters, cycles=2, per_cycle=257)
    schedule = contact_schedule(parameters, times)
    for leg_id in LegId:
        intervals = parameters.stance_intervals(leg_id, 0.0, window)
        for index, time in enumerate(times.tolist()):
            inside = any(lower <= time < upper for lower, upper in intervals)
            assert inside == bool(schedule[index, int(leg_id)])


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_stance_intervals_total_the_exact_stance_measure(name: str) -> None:
    parameters = gait(name)
    window = 3.0 * parameters.period
    for leg_id in LegId:
        intervals = parameters.stance_intervals(leg_id, 0.0, window)
        total = sum(upper - lower for lower, upper in intervals)
        assert total == pytest.approx(
            parameters.stance_measure(leg_id, 0.0, window), abs=EXACT_TOLERANCE
        )


def test_stance_intervals_are_clipped_to_the_requested_window() -> None:
    parameters = gait("walk")
    intervals = parameters.stance_intervals(LegId.FRONT_LEFT, 0.3, 1.4)
    assert intervals
    assert all(0.3 <= lower < upper <= 1.4 for lower, upper in intervals)
    assert intervals[0][0] == pytest.approx(0.3)
    assert intervals[-1][1] == pytest.approx(1.4)


def test_an_empty_or_reversed_window_is_rejected() -> None:
    parameters = gait("walk")
    with pytest.raises(ValueError, match="must not precede"):
        parameters.stance_measure(LegId.FRONT_LEFT, 1.0, 0.5)
    with pytest.raises(ValueError, match="must not precede"):
        parameters.stance_intervals(LegId.FRONT_LEFT, 1.0, 0.5)
    with pytest.raises(ValueError, match="non-empty"):
        parameters.stance_fraction(LegId.FRONT_LEFT, 1.0, 1.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("walk", (3, 3)), ("trot", (2, 2)), ("pace", (2, 2)), ("bound", (0, 2))],
)
def test_stance_count_extrema_match_the_published_footfall_patterns(
    name: str, expected: tuple[int, int]
) -> None:
    """A walk never drops below three, a trot holds two, a bound reaches zero."""
    assert stance_count_extrema(gait(name)) == expected


@pytest.mark.parametrize("duty_factor", [0.76, 0.80, 0.90])
def test_a_walk_above_the_threshold_gains_a_four_foot_interval(duty_factor: float) -> None:
    assert stance_count_extrema(gait("walk", duty_factor=duty_factor)) == (3, 4)


def test_stance_count_extrema_bracket_a_dense_sampling() -> None:
    """The exact range must contain every count a dense sampling can find."""
    for name in GAIT_NAMES:
        parameters = gait(name)
        low, high = stance_count_extrema(parameters)
        times = _midpoint_times(parameters, cycles=1, per_cycle=4001)
        counts = contact_schedule(parameters, times).sum(axis=1)
        assert low <= int(counts.min())
        assert int(counts.max()) <= high


# The closed form support histogram. These are the counterpart of the sampled
# histogram a report prints, measured on the event partition of the cycle.


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_stance_count_durations_partition_the_cycle(name: str) -> None:
    """Every instant has exactly one stance count, so the durations total the period."""
    parameters = gait(name)
    durations = stance_count_durations(parameters)
    assert len(durations) == LEG_COUNT + 1
    assert all(value >= 0.0 for value in durations)
    assert sum(durations) == pytest.approx(parameters.period, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("walk", (0.0, 0.0, 0.0, 1.0, 0.0)),
        ("trot", (0.0, 0.0, 1.0, 0.0, 0.0)),
        ("pace", (0.0, 0.0, 1.0, 0.0, 0.0)),
        ("bound", (0.2, 0.0, 0.8, 0.0, 0.0)),
    ],
)
def test_stance_count_durations_match_the_published_footfall_patterns(
    name: str, expected: tuple[float, ...]
) -> None:
    """A walk holds three feet, a trot and a pace two, a bound is airborne a fifth of the time."""
    parameters = gait(name)
    durations = stance_count_durations(parameters)
    for count, fraction in enumerate(expected):
        assert durations[count] == pytest.approx(fraction * parameters.period, abs=EXACT_TOLERANCE)


def test_the_reference_walk_spends_a_fifth_of_its_cycle_on_four_feet() -> None:
    """Swing windows of 0.20 cycles spaced by 0.25 leave four gaps of 0.05 with nothing airborne."""
    parameters = gait("walk", duty_factor=0.80)
    durations = stance_count_durations(parameters)
    assert durations[3] == pytest.approx(0.8 * parameters.period, abs=EXACT_TOLERANCE)
    assert durations[4] == pytest.approx(0.2 * parameters.period, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_stance_count_durations_reproduce_the_mean_stance_count(name: str) -> None:
    """The first moment of the histogram is four times the duty factor, exactly."""
    parameters = gait(name)
    durations = stance_count_durations(parameters)
    weighted = sum(count * value for count, value in enumerate(durations)) / parameters.period
    assert weighted == pytest.approx(LEG_COUNT * parameters.duty_factor, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_only_counts_inside_the_exact_range_have_a_duration(name: str) -> None:
    """The two views of the event partition have to agree about which counts occur."""
    parameters = gait(name)
    low, high = stance_count_extrema(parameters)
    for count, value in enumerate(stance_count_durations(parameters)):
        if low <= count <= high:
            continue
        assert value == 0.0


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_stance_count_durations_are_the_limit_the_sampled_histogram_approaches(name: str) -> None:
    """Refining the sampling rate drives the counted histogram onto the closed form.

    The duty factor is nudged off the published value first. The library ones put
    every support transition on a cell edge of the grids sampled here, where a counted
    histogram is already exact and would say nothing about a limit.
    """
    parameters = gait(name, duty_factor=gait(name).duty_factor + 0.0121)
    exact = np.asarray(stance_count_durations(parameters)) / parameters.period
    errors = []
    for per_cycle in (53, 501, 5003):
        times = _midpoint_times(parameters, cycles=1, per_cycle=per_cycle)
        counts = contact_schedule(parameters, times).sum(axis=1)
        counted = np.bincount(counts, minlength=LEG_COUNT + 1) / counts.size
        errors.append(float(np.max(np.abs(counted - exact))))
        assert errors[-1] <= 2.0 / per_cycle
    assert errors[-1] <= errors[0]


def test_stance_count_durations_are_computed_by_hand_on_a_probe_gait() -> None:
    """An irregular gait whose event partition is short enough to enumerate."""
    parameters = GaitParameters(
        name="probe", period=1.0, duty_factor=0.6, phase_offsets=(0.0, 0.2, 0.5, 0.7)
    )
    # The four legs are loaded on [0.0, 0.6), [0.2, 0.8), [0.5, 1.1) and [0.7, 1.3) of
    # each cycle. Reading the count off that list on the eight cells the touchdowns
    # and lift offs cut the cycle into gives 3, 2, 3, 2, 3, 2, 3, 2 over cells of
    # length 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, so three feet are down for 0.4 of
    # the cycle and two for the remaining 0.6.
    durations = stance_count_durations(parameters)
    assert durations == pytest.approx((0.0, 0.0, 0.6, 0.4, 0.0), abs=EXACT_TOLERANCE)
    assert exact_supported_fraction(parameters) == pytest.approx(0.4, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize(
    ("name", "expected"), [("walk", 1.0), ("trot", 0.0), ("pace", 0.0), ("bound", 0.0)]
)
def test_exact_supported_fraction_of_the_library_gaits(name: str, expected: float) -> None:
    """Only the walk ever has three feet down, which is why only it has a margin at all."""
    assert exact_supported_fraction(gait(name)) == pytest.approx(expected, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize("duty_factor", [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0])
def test_exact_supported_fraction_of_a_walk_follows_the_quarter_spacing_formula(
    duty_factor: float,
) -> None:
    """Adjacent swing windows overlap by ``0.75 - beta``, and four of them do.

    The unsupported measure is therefore ``4 (0.75 - beta)`` while that is positive,
    which puts the supported fraction at ``4 beta - 2`` and sends it to one exactly
    at the three quarter threshold of McGhee and Frank (1968).
    """
    parameters = gait("walk", duty_factor=duty_factor)
    expected = min(max(4.0 * duty_factor - 2.0, 0.0), 1.0)
    assert exact_supported_fraction(parameters) == pytest.approx(expected, abs=EXACT_TOLERANCE)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_exact_supported_fraction_is_the_measure_of_the_supported_counts(name: str) -> None:
    parameters = gait(name)
    durations = stance_count_durations(parameters)
    supported = (durations[3] + durations[4]) / parameters.period
    assert exact_supported_fraction(parameters) == pytest.approx(supported, abs=EXACT_TOLERANCE)
