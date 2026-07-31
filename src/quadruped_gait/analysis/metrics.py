"""Reduction of a simulation trace to contact statistics and stability margins."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.algorithm.stability import stability_margins, support_polygon
from quadruped_gait.model.geometry import LEG_COUNT, LEG_NAMES
from quadruped_gait.pipeline.simulator import Trace
from quadruped_gait.pipeline.sweep import DutySweepRow

__all__ = [
    "ContactInterval",
    "ContactSummary",
    "GaitReport",
    "RoundTripSummary",
    "StabilitySummary",
    "contact_intervals",
    "contact_summary",
    "round_trip_summary",
    "stability_series",
    "stability_summary",
    "summarise",
    "support_polygon_at",
    "sweep_stability",
]


@dataclass(frozen=True, slots=True)
class ContactSummary:
    """How closely the realised contact schedule matched the command.

    Attributes:
        commanded_duty_factor: The duty factor the scheduler was given.
        realised_duty_factors: Measured stance fraction of each leg over the trace.
        max_absolute_error: Largest difference between a realised and the commanded
            duty factor.
        stance_count_histogram: How many samples had zero, one, two, three, or four
            feet loaded.
        mean_stance_count: Mean number of loaded feet per sample.
    """

    commanded_duty_factor: float
    realised_duty_factors: tuple[float, ...]
    max_absolute_error: float
    stance_count_histogram: tuple[int, ...]
    mean_stance_count: float


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    """Quasi-static stability over a whole trace.

    Attributes:
        sample_count: Number of samples examined.
        supported_fraction: Fraction of samples whose support polygon has an interior.
        minimum_static: Smallest static stability margin over the supported samples,
            in metres, or ``nan`` when no sample was supported.
        mean_static: Mean static stability margin over the supported samples.
        minimum_longitudinal: Smallest longitudinal stability margin over the
            supported samples, in metres.
        mean_longitudinal: Mean longitudinal stability margin over the supported samples.
        mean_support_area: Mean support polygon area over the supported samples, in
            square metres.
        statically_stable: Whether every sample was supported with a positive static
            margin.
    """

    sample_count: int
    supported_fraction: float
    minimum_static: float
    mean_static: float
    minimum_longitudinal: float
    mean_longitudinal: float
    mean_support_area: float
    statically_stable: bool


@dataclass(frozen=True, slots=True)
class ContactInterval:
    """One continuous stance interval of one leg, used to draw a gait diagram."""

    leg_index: int
    leg_name: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Length of the interval, in seconds."""
        return self.end_time - self.start_time


@dataclass(frozen=True, slots=True)
class GaitReport:
    """Everything the analysis layer extracts from one trace.

    Attributes:
        gait_name: Name of the gait that produced the trace.
        period: Gait cycle period, in seconds.
        forward_velocity: Commanded forward velocity, in metres per second.
        stride_length: Distance the trunk travels in one gait cycle, in metres.
        contact: Contact schedule statistics.
        stability: Quasi-static stability statistics.
        unreachable_samples: Number of leg samples whose inverse kinematics failed.
    """

    gait_name: str
    period: float
    forward_velocity: float
    stride_length: float
    contact: ContactSummary
    stability: StabilitySummary
    unreachable_samples: int


def contact_summary(trace: Trace) -> ContactSummary:
    """Measure the realised duty factor and support pattern of a trace."""
    contacts = np.asarray(trace.contacts, dtype=np.bool_)
    if contacts.shape[0] == 0:
        raise ValueError("trace contains no samples")
    realised = contacts.mean(axis=0)
    commanded = trace.config.gait.duty_factor
    stance_counts = contacts.sum(axis=1)
    histogram = np.bincount(stance_counts, minlength=LEG_COUNT + 1)
    return ContactSummary(
        commanded_duty_factor=commanded,
        realised_duty_factors=tuple(float(value) for value in realised),
        max_absolute_error=float(np.max(np.abs(realised - commanded))),
        stance_count_histogram=tuple(int(value) for value in histogram),
        mean_stance_count=float(stance_counts.mean()),
    )


def stability_series(trace: Trace) -> NDArray[np.float64]:
    """Return per sample stability data of shape ``(n, 4)``.

    The columns are the static stability margin, the longitudinal stability
    margin, the support polygon area, and the number of loaded feet. The first
    three are ``nan`` for samples whose support polygon has no interior.
    """
    count = len(trace)
    output = np.full((count, 4), np.nan, dtype=np.float64)
    for index in range(count):
        margins = stability_margins(
            trace.foot_positions[index],
            trace.contacts[index],
            trace.com_positions[index],
            trace.travel_direction[index],
        )
        polygon = support_polygon(trace.foot_positions[index], trace.contacts[index])
        output[index, 0] = margins.static
        output[index, 1] = margins.longitudinal
        output[index, 2] = polygon.area if polygon.is_defined else math.nan
        output[index, 3] = float(margins.stance_count)
    return output


def stability_summary(trace: Trace, series: NDArray[np.float64] | None = None) -> StabilitySummary:
    """Reduce the per sample stability data of a trace to scalar statistics."""
    data = stability_series(trace) if series is None else np.asarray(series, dtype=np.float64)
    count = data.shape[0]
    if count == 0:
        raise ValueError("trace contains no samples")
    supported = np.isfinite(data[:, 0])
    supported_count = int(supported.sum())
    if supported_count == 0:
        return StabilitySummary(
            sample_count=count,
            supported_fraction=0.0,
            minimum_static=math.nan,
            mean_static=math.nan,
            minimum_longitudinal=math.nan,
            mean_longitudinal=math.nan,
            mean_support_area=math.nan,
            statically_stable=False,
        )
    static = data[supported, 0]
    longitudinal = data[supported, 1]
    area = data[supported, 2]
    return StabilitySummary(
        sample_count=count,
        supported_fraction=supported_count / count,
        minimum_static=float(np.min(static)),
        mean_static=float(np.mean(static)),
        minimum_longitudinal=float(np.nanmin(longitudinal)),
        mean_longitudinal=float(np.nanmean(longitudinal)),
        mean_support_area=float(np.mean(area)),
        statically_stable=bool(supported_count == count and np.min(static) > 0.0),
    )


def contact_intervals(trace: Trace) -> tuple[ContactInterval, ...]:
    """Return the stance intervals of every leg, the gait diagram in interval form.

    Intervals are closed on the left and open on the right, and an interval that
    is still open at the end of the trace is closed at the final sample time plus
    one timestep.
    """
    intervals: list[ContactInterval] = []
    times = trace.times
    step = trace.config.timestep
    contacts = np.asarray(trace.contacts, dtype=np.bool_)
    for leg_index in range(LEG_COUNT):
        column = contacts[:, leg_index]
        start: float | None = None
        for sample_index in range(column.size):
            loaded = bool(column[sample_index])
            if loaded and start is None:
                start = float(times[sample_index])
            elif not loaded and start is not None:
                intervals.append(
                    ContactInterval(
                        leg_index=leg_index,
                        leg_name=LEG_NAMES[leg_index],
                        start_time=start,
                        end_time=float(times[sample_index]),
                    )
                )
                start = None
        if start is not None:
            intervals.append(
                ContactInterval(
                    leg_index=leg_index,
                    leg_name=LEG_NAMES[leg_index],
                    start_time=start,
                    end_time=float(times[-1]) + step,
                )
            )
    return tuple(intervals)


def support_polygon_at(trace: Trace, index: int) -> NDArray[np.float64]:
    """Return the support polygon vertices at one sample, shape ``(m, 2)``."""
    return support_polygon(trace.foot_positions[index], trace.contacts[index]).vertices


@dataclass(frozen=True, slots=True)
class RoundTripSummary:
    """Accuracy of forward kinematics composed with inverse kinematics.

    Attributes:
        sample_count: Number of targets tested.
        solved_count: Number of targets the inverse kinematics accepted.
        maximum_error: Largest reconstruction error, in metres.
        mean_error: Mean reconstruction error, in metres.
        median_error: Median reconstruction error, in metres.
    """

    sample_count: int
    solved_count: int
    maximum_error: float
    mean_error: float
    median_error: float


def round_trip_summary(errors: NDArray[np.float64]) -> RoundTripSummary:
    """Reduce an array of round trip errors to scalar statistics."""
    array = np.asarray(errors, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("errors must contain at least one entry")
    solved = np.isfinite(array)
    if not solved.any():
        return RoundTripSummary(
            sample_count=int(array.size),
            solved_count=0,
            maximum_error=math.nan,
            mean_error=math.nan,
            median_error=math.nan,
        )
    values = array[solved]
    return RoundTripSummary(
        sample_count=int(array.size),
        solved_count=int(solved.sum()),
        maximum_error=float(np.max(values)),
        mean_error=float(np.mean(values)),
        median_error=float(np.median(values)),
    )


def sweep_stability(
    rows: Sequence[DutySweepRow],
) -> tuple[tuple[float, StabilitySummary], ...]:
    """Pair each duty factor of a sweep with the stability summary it produced."""
    return tuple((row.duty_factor, stability_summary(row.trace)) for row in rows)


def summarise(trace: Trace) -> GaitReport:
    """Reduce a trace to the report printed by the example scripts."""
    series = stability_series(trace)
    command = trace.config.command
    return GaitReport(
        gait_name=trace.config.gait.name,
        period=trace.config.gait.period,
        forward_velocity=command.forward_velocity,
        stride_length=command.forward_velocity * trace.config.gait.period,
        contact=contact_summary(trace),
        stability=stability_summary(trace, series),
        unreachable_samples=int(np.count_nonzero(~trace.reachable)),
    )
