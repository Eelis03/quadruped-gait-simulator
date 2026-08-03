"""Reduction of a simulation trace to contact statistics and stability margins."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.algorithm.gait import exact_duty_factors, stance_count_extrema
from quadruped_gait.algorithm.stability import stability_margins, support_polygon
from quadruped_gait.model.geometry import LEG_COUNT, LEG_NAMES, LegId
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
    "critical_sample_indices",
    "round_trip_summary",
    "stability_series",
    "stability_summary",
    "summarise",
    "support_polygon_at",
    "sweep_stability",
    "trace_window",
]


@dataclass(frozen=True, slots=True)
class ContactSummary:
    """How closely the realised contact schedule matched the command.

    The realised quantities are computed in closed form from the schedule over the
    time window the trace covers, not by counting samples. The sampled counterparts
    are kept beside them, because a value read out of the gait parameters cannot on
    its own detect a simulator that disagrees with its own scheduler, and the
    difference between the two is exactly the discretisation error.

    Attributes:
        commanded_duty_factor: The duty factor the scheduler was given.
        exact_duty_factors: Stance fraction of each leg over the trace window,
            computed in closed form.
        sampled_duty_factors: Stance fraction of each leg counted from the recorded
            contact flags.
        max_absolute_error: Largest difference between an exact and the commanded
            duty factor.
        sampling_error: Largest difference between a sampled and an exact duty
            factor, which is the discretisation error of the recorded trace.
        stance_count_histogram: How many samples had zero, one, two, three, or four
            feet loaded.
        stance_count_range: Smallest and largest number of loaded feet over a gait
            cycle, computed exactly from the event times.
        mean_stance_count: Mean number of loaded feet over the trace window,
            computed in closed form as the sum of the exact duty factors.
        sampled_mean_stance_count: Mean number of loaded feet per recorded sample.
    """

    commanded_duty_factor: float
    exact_duty_factors: tuple[float, ...]
    sampled_duty_factors: tuple[float, ...]
    max_absolute_error: float
    sampling_error: float
    stance_count_histogram: tuple[int, ...]
    stance_count_range: tuple[int, int]
    mean_stance_count: float
    sampled_mean_stance_count: float


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


def trace_window(trace: Trace) -> tuple[float, float]:
    """Return the time interval the samples of ``trace`` stand for.

    A sample at time ``t`` represents the cell ``[t, t + dt)``, so the window the
    trace covers runs from the first sample time to one timestep past the last. This
    is the interval over which an exact contact fraction is comparable with the
    fraction counted from the samples.
    """
    if trace.times.size == 0:
        raise ValueError("trace contains no samples")
    return float(trace.times[0]), float(trace.times[-1]) + trace.config.timestep


def contact_summary(trace: Trace) -> ContactSummary:
    """Measure the realised duty factor and support pattern of a trace."""
    contacts = np.asarray(trace.contacts, dtype=np.bool_)
    if contacts.shape[0] == 0:
        raise ValueError("trace contains no samples")
    parameters = trace.config.gait
    start, end = trace_window(trace)
    exact = exact_duty_factors(parameters, start, end)
    sampled = contacts.mean(axis=0)
    commanded = parameters.duty_factor
    stance_counts = contacts.sum(axis=1)
    histogram = np.bincount(stance_counts, minlength=LEG_COUNT + 1)
    return ContactSummary(
        commanded_duty_factor=commanded,
        exact_duty_factors=tuple(float(value) for value in exact),
        sampled_duty_factors=tuple(float(value) for value in sampled),
        max_absolute_error=float(np.max(np.abs(exact - commanded))),
        sampling_error=float(np.max(np.abs(sampled - exact))),
        stance_count_histogram=tuple(int(value) for value in histogram),
        stance_count_range=stance_count_extrema(parameters),
        mean_stance_count=float(np.sum(exact)),
        sampled_mean_stance_count=float(stance_counts.mean()),
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

    The intervals come from the closed form schedule of the gait, clipped to the
    window the trace covers, so an interval edge lands on the phase boundary itself
    rather than on the nearest sample. Their total duration is therefore the exact
    stance time, and the gait diagram drawn from them does not quantise to the
    sampling grid.
    """
    start, end = trace_window(trace)
    parameters = trace.config.gait
    return tuple(
        ContactInterval(
            leg_index=int(leg_id),
            leg_name=LEG_NAMES[int(leg_id)],
            start_time=lower,
            end_time=upper,
        )
        for leg_id in LegId
        for lower, upper in parameters.stance_intervals(leg_id, start, end)
    )


def support_polygon_at(trace: Trace, index: int) -> NDArray[np.float64]:
    """Return the support polygon vertices at one sample, shape ``(m, 2)``."""
    return support_polygon(trace.foot_positions[index], trace.contacts[index]).vertices


def critical_sample_indices(
    trace: Trace, offsets: Sequence[float] = (-0.10, -0.05, 0.0, 0.05)
) -> tuple[int, ...]:
    """Return the samples lying at cycle ``offsets`` from the least stable instant.

    The least stable instant is the sample with the smallest static stability
    margin, searched over the interior of the trace so that the whole window of
    offsets exists. A trace with no support polygon anywhere, which is every two
    beat gait, has no least stable instant, so the middle of the trace is used.

    Args:
        trace: The run to examine.
        offsets: Positions relative to the critical instant, in gait cycles.

    Raises:
        ValueError: If ``offsets`` is empty or the trace has no samples.
    """
    if not offsets:
        raise ValueError("offsets must contain at least one entry")
    count = len(trace)
    if count == 0:
        raise ValueError("trace contains no samples")
    step = trace.config.timestep
    shifts = [round(offset * trace.config.gait.period / step) for offset in offsets]
    pad = min(max(max(shifts), -min(shifts), 0), (count - 1) // 2)
    margins = stability_series(trace)[:, 0][pad : count - pad]
    if margins.size == 0:
        centre = count // 2
    elif bool(np.isfinite(margins).any()):
        centre = pad + int(np.nanargmin(margins))
    else:
        centre = pad + margins.size // 2
    return tuple(int(np.clip(centre + shift, 0, count - 1)) for shift in shifts)


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
