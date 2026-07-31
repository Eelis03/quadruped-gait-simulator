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
    foot_trajectory_figure,
    format_contact_summary,
    format_duty_sweep,
    format_gait_diagram,
    format_report,
    format_round_trip,
    format_summary_table,
    gait_diagram_figure,
    round_trip_summary,
    save_figure,
    stability_figure,
    stability_series,
    stability_summary,
    summarise,
    support_polygon_at,
    support_polygon_figure,
    sweep_stability,
)
from quadruped_gait.model.geometry import LEG_COUNT, default_robot
from quadruped_gait.pipeline import (
    BodyCommand,
    SimulationConfig,
    Trace,
    duty_factor_sweep,
    reference_walk,
    simulate,
)

SAMPLES = 121


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
def test_realised_duty_factor_matches_the_command_within_one_sample(name: str) -> None:
    """The sampled stance fraction can only differ by the width of one sample."""
    trace = simulate(
        SimulationConfig(
            robot=default_robot(),
            gait=gait(name),
            command=BodyCommand(forward_velocity=0.3),
            cycles=4.0,
            samples_per_cycle=1000,
        )
    )
    summary = contact_summary(trace)
    assert summary.max_absolute_error <= 1.5 / 1000


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


def test_contact_intervals_reconstruct_the_duty_factor() -> None:
    trace = _trace("walk", forward_velocity=0.3)
    intervals = contact_intervals(trace)
    assert len(intervals) >= 2 * LEG_COUNT
    total = sum(interval.duration for interval in intervals)
    expected = 0.75 * LEG_COUNT * 2.0 * 1.0
    assert total == pytest.approx(expected, abs=0.12)
    assert {interval.leg_index for interval in intervals} == set(range(LEG_COUNT))


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
    figures = {
        "diagram.png": gait_diagram_figure(trace),
        "stability.png": stability_figure(trace),
        "feet.png": foot_trajectory_figure(trace),
        "support.png": support_polygon_figure(trace, (0, 10, 20)),
    }
    for name, figure in figures.items():
        path = save_figure(figure, tmp_path / name, dpi=60)
        assert path.exists()
        assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="at least one"):
        support_polygon_figure(trace, ())


def test_stability_summary_rejects_an_empty_trace() -> None:
    empty = np.empty((0, 4), dtype=np.float64)
    trace = simulate(reference_walk(cycles=1.0, samples_per_cycle=40))
    with pytest.raises(ValueError, match="no samples"):
        stability_summary(trace, empty)
