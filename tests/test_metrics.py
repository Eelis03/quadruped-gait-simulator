"""Tests for the analysis layer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from quadruped_gait.algorithm.gait import GAIT_NAMES, gait
from quadruped_gait.analysis import (
    contact_intervals,
    contact_summary,
    critical_sample_indices,
    duty_sweep_figure,
    foot_trajectory_figure,
    format_contact_summary,
    format_duty_sweep,
    format_gait_diagram,
    format_report,
    format_round_trip,
    format_summary_table,
    gait_diagram_figure,
    gait_diagram_grid_figure,
    round_trip_summary,
    save_figure,
    stability_figure,
    stability_series,
    stability_summary,
    summarise,
    support_polygon_at,
    support_polygon_figure,
    sweep_stability,
    trace_window,
)
from quadruped_gait.model.geometry import LEG_COUNT, LegId, default_robot
from quadruped_gait.pipeline import (
    STATIC_STABILITY_THRESHOLD,
    BodyCommand,
    SimulationConfig,
    Trace,
    duty_factor_sweep,
    reference_walk,
    simulate,
    threshold_walk,
)

SAMPLES = 121
EXACT_TOLERANCE = 1e-12


def _trace(name: str, **command: float) -> Trace:
    return simulate(
        SimulationConfig(
            robot=default_robot(),
            gait=gait(name),
            command=BodyCommand(**command),
            cycles=2.0,
            samples_per_cycle=SAMPLES,
        )
    )


@pytest.mark.parametrize("name", GAIT_NAMES)
@pytest.mark.parametrize("samples_per_cycle", [17, 200, 1000])
def test_realised_duty_factor_matches_the_command_exactly(
    name: str, samples_per_cycle: int
) -> None:
    """The reported duty factor is closed form, so no sampling rate degrades it.

    This is the test the sampled measurement could not pass. It asserted a tolerance
    proportional to the sampling rate, which meant a coarse run reported a duty
    factor that was simply wrong by up to one sampling interval.
    """
    trace = simulate(
        SimulationConfig(
            robot=default_robot(),
            gait=gait(name),
            command=BodyCommand(forward_velocity=0.3),
            cycles=4.0,
            samples_per_cycle=samples_per_cycle,
        )
    )
    summary = contact_summary(trace)
    assert summary.max_absolute_error <= EXACT_TOLERANCE
    assert summary.mean_stance_count == pytest.approx(
        LEG_COUNT * summary.commanded_duty_factor, abs=EXACT_TOLERANCE
    )


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_the_sampled_duty_factor_cross_checks_the_closed_form_one(name: str) -> None:
    """The counted value is kept because the closed form cannot audit the simulator.

    An exact duty factor read out of the gait parameters would still be reported if
    the simulator recorded contact flags that disagreed with its own scheduler. The
    sampled value is an independent measurement of what the trace holds, and it has
    to land within one sampling interval of the closed form.
    """
    samples_per_cycle = 200
    trace = simulate(
        SimulationConfig(
            robot=default_robot(),
            gait=gait(name),
            command=BodyCommand(forward_velocity=0.3),
            cycles=4.0,
            samples_per_cycle=samples_per_cycle,
        )
    )
    summary = contact_summary(trace)
    assert 0.0 <= summary.sampling_error <= 1.5 / samples_per_cycle
    assert summary.sampled_mean_stance_count == pytest.approx(
        summary.mean_stance_count, abs=LEG_COUNT * 1.5 / samples_per_cycle
    )


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_the_exact_duty_factor_survives_a_partial_final_cycle(name: str) -> None:
    """A window that is not a whole number of cycles has a stance fraction too."""
    trace = simulate(
        SimulationConfig(
            robot=default_robot(),
            gait=gait(name),
            command=BodyCommand(forward_velocity=0.3),
            cycles=2.5,
            samples_per_cycle=64,
        )
    )
    summary = contact_summary(trace)
    start, end = trace_window(trace)
    assert end - start == pytest.approx(2.5 * trace.config.gait.period, abs=1e-9)
    for leg_id in LegId:
        expected = trace.config.gait.stance_fraction(leg_id, start, end)
        assert summary.exact_duty_factors[int(leg_id)] == pytest.approx(
            expected, abs=EXACT_TOLERANCE
        )


def test_the_reported_feet_down_range_is_the_exact_one() -> None:
    trace = simulate(reference_walk(cycles=2.0, samples_per_cycle=SAMPLES))
    summary = contact_summary(trace)
    assert summary.stance_count_range == (3, 4)
    histogram = summary.stance_count_histogram
    assert histogram[0] == histogram[1] == histogram[2] == 0
    assert histogram[3] > 0
    assert histogram[4] > 0


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_stability_series_has_one_row_per_sample(name: str) -> None:
    trace = _trace(name, forward_velocity=0.3)
    series = stability_series(trace)
    assert series.shape == (int(2.0 * SAMPLES), 4)
    assert np.all(series[:, 3] >= 0.0)
    assert np.all(series[:, 3] <= LEG_COUNT)


def test_walk_stability_series_is_defined_everywhere() -> None:
    trace = reference_walk(cycles=2.0, samples_per_cycle=SAMPLES)
    series = stability_series(simulate(trace))
    assert bool(np.isfinite(series[:, 0]).all())
    assert float(series[:, 0].min()) > 0.0


@pytest.mark.parametrize("name", ["trot", "pace", "bound"])
def test_two_beat_gaits_have_no_support_polygon(name: str) -> None:
    """Two feet span a line, so the quasi-static criterion says nothing about them."""
    trace = _trace(name, forward_velocity=0.3)
    summary = stability_summary(trace)
    assert summary.supported_fraction == 0.0
    assert math.isnan(summary.minimum_static)
    assert not summary.statically_stable
    assert support_polygon_at(trace, 0).shape == (0, 2)


def test_contact_intervals_reconstruct_the_duty_factor_exactly() -> None:
    """The intervals come from the schedule, so their total is the exact stance time."""
    trace = _trace("walk", forward_velocity=0.3)
    intervals = contact_intervals(trace)
    assert len(intervals) >= 2 * LEG_COUNT
    total = sum(interval.duration for interval in intervals)
    expected = 0.75 * LEG_COUNT * 2.0 * 1.0
    assert total == pytest.approx(expected, abs=EXACT_TOLERANCE)
    assert {interval.leg_index for interval in intervals} == set(range(LEG_COUNT))


def test_contact_intervals_stay_inside_the_trace_window() -> None:
    trace = _trace("bound", forward_velocity=0.3)
    start, end = trace_window(trace)
    intervals = contact_intervals(trace)
    assert all(start <= item.start_time < item.end_time <= end for item in intervals)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_contact_intervals_agree_with_the_recorded_contact_flags(name: str) -> None:
    """The exact intervals must contain a sample exactly when its flag is set.

    Deriving the gait diagram from the schedule instead of from the trace loses the
    guarantee that it depicts what was recorded. This test restores it.
    """
    trace = _trace(name, forward_velocity=0.3)
    per_leg: dict[int, list[tuple[float, float]]] = {index: [] for index in range(LEG_COUNT)}
    for item in contact_intervals(trace):
        per_leg[item.leg_index].append((item.start_time, item.end_time))
    for index, time in enumerate(trace.times.tolist()):
        for leg_index in range(LEG_COUNT):
            inside = any(lower <= time < upper for lower, upper in per_leg[leg_index])
            assert inside == bool(trace.contacts[index, leg_index])


def test_critical_sample_indices_find_the_least_stable_instant() -> None:
    """The zero margin of the threshold walk is the sample the selector must land on."""
    trace = simulate(threshold_walk(cycles=3.0, samples_per_cycle=200))
    indices = critical_sample_indices(trace)
    series = stability_series(trace)
    assert len(indices) == 4
    assert indices == tuple(sorted(indices))
    centre = indices[2]
    assert series[centre, 0] == pytest.approx(0.0, abs=1e-9)
    assert all(series[index, 0] >= series[centre, 0] for index in indices)


def test_critical_sample_indices_stay_inside_the_trace() -> None:
    trace = simulate(reference_walk(cycles=1.0, samples_per_cycle=8))
    indices = critical_sample_indices(trace, offsets=(-0.5, 0.0, 0.5))
    assert all(0 <= index < len(trace) for index in indices)


def test_critical_sample_indices_fall_back_when_no_margin_is_defined() -> None:
    """A trot has no support polygon anywhere, so there is no least stable instant."""
    trace = _trace("trot", forward_velocity=0.3)
    indices = critical_sample_indices(trace)
    assert len(indices) == 4
    assert all(0 <= index < len(trace) for index in indices)


def test_critical_sample_indices_validate_their_arguments() -> None:
    trace = simulate(reference_walk(cycles=1.0, samples_per_cycle=40))
    with pytest.raises(ValueError, match="at least one entry"):
        critical_sample_indices(trace, offsets=())


def test_the_threshold_walk_sits_exactly_on_the_published_threshold() -> None:
    """McGhee and Frank put the creeping gait threshold at three quarters."""
    config = threshold_walk(cycles=2.0, samples_per_cycle=100)
    assert config.gait.duty_factor == STATIC_STABILITY_THRESHOLD == 0.75
    summary = stability_summary(simulate(config))
    assert summary.supported_fraction == pytest.approx(1.0)
    assert summary.minimum_static == pytest.approx(0.0, abs=1e-9)
    assert not summary.statically_stable


def test_summary_reports_the_expected_fields() -> None:
    report = summarise(simulate(reference_walk(cycles=2.0, samples_per_cycle=SAMPLES)))
    assert report.gait_name == "walk"
    assert report.stride_length == pytest.approx(0.30)
    assert report.unreachable_samples == 0
    assert report.stability.statically_stable


def test_round_trip_summary_handles_rejected_targets() -> None:
    errors = np.array([1e-16, 2e-16, np.nan, 3e-16])
    summary = round_trip_summary(errors)
    assert summary.sample_count == 4
    assert summary.solved_count == 3
    assert summary.maximum_error == pytest.approx(3e-16)
    empty = round_trip_summary(np.array([np.nan, np.nan]))
    assert empty.solved_count == 0
    assert math.isnan(empty.mean_error)
    with pytest.raises(ValueError, match="at least one"):
        round_trip_summary(np.array([]))


def test_text_renderings_are_plain_ascii() -> None:
    trace = simulate(reference_walk(cycles=1.0, samples_per_cycle=SAMPLES))
    report = summarise(trace)
    rows = sweep_stability(duty_factor_sweep(reference_walk(1.0, 40), (0.7, 0.8)))
    blocks = [
        format_report(report),
        format_contact_summary(report),
        format_summary_table([report]),
        format_duty_sweep(rows),
        format_round_trip(round_trip_summary(np.array([1e-16])), "label"),
        format_gait_diagram(contact_intervals(trace), trace.config.gait.period, 40),
    ]
    for block in blocks:
        assert block
        assert block.isascii()


def test_gait_diagram_text_marks_the_expected_fraction() -> None:
    trace = _trace("trot", forward_velocity=0.3)
    text = format_gait_diagram(contact_intervals(trace), trace.config.gait.period, 40)
    body = [line for line in text.splitlines() if line.startswith(("FL", "FR", "HL", "HR"))]
    assert len(body) == LEG_COUNT
    for line in body:
        assert line.count("#") == pytest.approx(20, abs=1)


def test_gait_diagram_text_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="period"):
        format_gait_diagram((), 0.0)
    with pytest.raises(ValueError, match="columns"):
        format_gait_diagram((), 1.0, 0)


def test_figures_are_written(tmp_path: Path) -> None:
    trace = simulate(reference_walk(cycles=1.0, samples_per_cycle=40))
    rows = sweep_stability(duty_factor_sweep(reference_walk(1.0, 40), (0.70, 0.75, 0.80)))
    figures = {
        "diagram.png": gait_diagram_figure(trace),
        "grid.png": gait_diagram_grid_figure([gait(name) for name in GAIT_NAMES]),
        "stability.png": stability_figure(trace),
        "sweep.png": duty_sweep_figure(rows),
        "feet.png": foot_trajectory_figure(trace),
        "support.png": support_polygon_figure(trace, (0, 10, 20)),
    }
    for name, figure in figures.items():
        path = save_figure(figure, tmp_path / name, dpi=60)
        assert path.exists()
        assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="at least one"):
        support_polygon_figure(trace, ())


def test_figure_builders_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one duty factor"):
        duty_sweep_figure(())
    with pytest.raises(ValueError, match="at least one gait"):
        gait_diagram_grid_figure([])
    with pytest.raises(ValueError, match="cycles must be positive"):
        gait_diagram_grid_figure([gait("walk")], cycles=0.0)


def test_duty_sweep_figure_draws_one_marker_per_point() -> None:
    """Stable and unstable points are split by marker fill, so both series exist."""
    factors = (0.70, 0.75, 0.80, 0.85)
    rows = sweep_stability(duty_factor_sweep(reference_walk(1.0, 60), factors))
    figure = duty_sweep_figure(rows)
    upper, lower = figure.axes
    # One line for the margin plus two marker only series, split by stability.
    marker_points = sum(
        line.get_xdata().size for line in upper.lines if line.get_linestyle() == "None"
    )
    assert marker_points == len(factors)
    assert lower.get_xlabel() == "duty factor"


def test_gait_diagram_grid_figure_has_one_panel_per_gait() -> None:
    gaits = [gait(name) for name in GAIT_NAMES]
    figure = gait_diagram_grid_figure(gaits, cycles=2.0, title="library gaits")
    assert len(figure.axes) == len(gaits)
    for axes, parameters in zip(figure.axes, gaits, strict=True):
        assert parameters.name in axes.get_title(loc="left")
        assert axes.get_xlim() == (0.0, 2.0)
        assert [label.get_text() for label in axes.get_yticklabels()] == ["HR", "HL", "FR", "FL"]


def test_stability_summary_rejects_an_empty_trace() -> None:
    empty = np.empty((0, 4), dtype=np.float64)
    trace = simulate(reference_walk(cycles=1.0, samples_per_cycle=40))
    with pytest.raises(ValueError, match="no samples"):
        stability_summary(trace, empty)
