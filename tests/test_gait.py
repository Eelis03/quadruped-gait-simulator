"""Invariant tests for the gait scheduler."""

from __future__ import annotations

import numpy as np
import pytest

from quadruped_gait.algorithm.gait import (
    GAIT_LIBRARY,
    GAIT_NAMES,
    GaitParameters,
    contact_schedule,
    gait,
    leg_phases,
    realised_duty_factors,
)
from quadruped_gait.model.geometry import LEG_COUNT, LegId

DUTY_TOLERANCE = 2e-3


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
    realised = realised_duty_factors(contact_schedule(parameters, times))
    assert realised.shape == (LEG_COUNT,)
    np.testing.assert_allclose(realised, parameters.duty_factor, atol=DUTY_TOLERANCE)


@pytest.mark.parametrize("duty_factor", [0.35, 0.5, 0.62, 0.75, 0.88, 1.0])
def test_realised_duty_factor_tracks_an_overridden_command(duty_factor: float) -> None:
    parameters = gait("walk", duty_factor=duty_factor)
    times = _midpoint_times(parameters, cycles=4, per_cycle=2000)
    realised = realised_duty_factors(contact_schedule(parameters, times))
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
        realised_duty_factors(np.zeros((5, 3), dtype=bool))
    with pytest.raises(ValueError, match="at least one sample"):
        realised_duty_factors(np.zeros((0, LEG_COUNT), dtype=bool))


def test_retiming_preserves_the_schedule_shape() -> None:
    slow = gait("trot", period=2.0)
    fast = gait("trot", period=0.25)
    times_slow = _midpoint_times(slow, cycles=1, per_cycle=64)
    times_fast = _midpoint_times(fast, cycles=1, per_cycle=64)
    np.testing.assert_array_equal(
        contact_schedule(slow, times_slow), contact_schedule(fast, times_fast)
    )
