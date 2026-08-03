"""Periodic gait scheduler parameterised by duty factor and leg phase offsets.

A symmetrical gait is fixed by three numbers per Hildebrand (1965, 1989): the
period, the duty factor, and the relative phase of each leg. Writing ``T`` for
the period, ``beta`` for the duty factor, and ``phi_i`` for the phase offset of
leg ``i``, the normalised cycle position of that leg is::

    p_i(t) = (t / T - phi_i) mod 1

and the leg is in stance while ``p_i < beta``. Touchdown of leg ``i`` therefore
occurs at ``t = phi_i T`` within each cycle, and the duty factor is the fraction
of the cycle spent in stance.

The four gaits below use the standard footfall patterns. The lateral sequence
walk has the footfall order hind-left, front-left, hind-right, front-right with
quarter cycle spacing (Hildebrand, 1965). Trot pairs diagonal legs, pace pairs
lateral legs, and bound pairs the fore legs against the hind legs.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.model.contact import ContactState
from quadruped_gait.model.geometry import LEG_COUNT, LegId

__all__ = [
    "GAIT_LIBRARY",
    "GAIT_NAMES",
    "GaitParameters",
    "contact_schedule",
    "exact_duty_factors",
    "gait",
    "leg_phases",
    "sampled_duty_factors",
    "stance_count_extrema",
]

# Phases that land within this distance of a full cycle are snapped to zero so that
# the schedule stays exactly periodic under floating point time accumulation.
_PHASE_EPSILON = 1e-9


def _wrap_phase(value: float) -> float:
    """Map ``value`` into the half open interval ``[0, 1)``."""
    wrapped = math.fmod(value, 1.0)
    if wrapped < 0.0:
        wrapped += 1.0
    if wrapped >= 1.0 - _PHASE_EPSILON:
        return 0.0
    return wrapped


def _stance_measure_to(cycles: float, duty_factor: float) -> float:
    """Return the measure of ``{u in [0, x] : frac(u) < beta}``, in cycles.

    This is the antiderivative of the stance indicator. Writing ``k`` for
    ``floor(x)``, the interval ``[0, x]`` contains ``k`` whole cycles, each
    contributing ``beta``, followed by a partial cycle of length ``x - k`` that
    contributes ``min(x - k, beta)``. The same expression is correct for negative
    ``x``, where ``k`` is negative and the result is the negated measure of
    ``[x, 0]``, which is what makes a difference of two evaluations valid over any
    interval.
    """
    whole = math.floor(cycles)
    return whole * duty_factor + min(cycles - whole, duty_factor)


@dataclass(frozen=True, slots=True)
class GaitParameters:
    """One periodic gait.

    Attributes:
        name: Identifier of the gait, for example ``"trot"``.
        period: Duration of one complete gait cycle, in seconds.
        duty_factor: Fraction of the cycle each foot spends in stance, in ``(0, 1]``.
        phase_offsets: Touchdown instant of each leg as a fraction of the cycle, in
            the canonical leg ordering.
    """

    name: str
    period: float
    duty_factor: float
    phase_offsets: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.period <= 0.0:
            raise ValueError(f"period must be positive, got {self.period}")
        if not 0.0 < self.duty_factor <= 1.0:
            raise ValueError(f"duty_factor must lie in (0, 1], got {self.duty_factor}")
        if len(self.phase_offsets) != LEG_COUNT:
            raise ValueError(f"expected {LEG_COUNT} phase offsets")
        if any(not 0.0 <= offset < 1.0 for offset in self.phase_offsets):
            raise ValueError(f"phase offsets must lie in [0, 1), got {self.phase_offsets}")

    @property
    def stance_duration(self) -> float:
        """Time one foot spends in stance during a cycle, in seconds."""
        return self.duty_factor * self.period

    @property
    def swing_duration(self) -> float:
        """Time one foot spends in swing during a cycle, in seconds."""
        return (1.0 - self.duty_factor) * self.period

    def with_period(self, period: float) -> GaitParameters:
        """Return a copy of this gait with a different cycle period."""
        return GaitParameters(
            name=self.name,
            period=period,
            duty_factor=self.duty_factor,
            phase_offsets=self.phase_offsets,
        )

    def with_duty_factor(self, duty_factor: float) -> GaitParameters:
        """Return a copy of this gait with a different duty factor."""
        return GaitParameters(
            name=self.name,
            period=self.period,
            duty_factor=duty_factor,
            phase_offsets=self.phase_offsets,
        )

    def leg_phase(self, time: float, leg_id: LegId) -> float:
        """Return the cycle position of ``leg_id`` at ``time``, in ``[0, 1)``."""
        return _wrap_phase(time / self.period - self.phase_offsets[int(leg_id)])

    def phases(self, time: float) -> tuple[float, float, float, float]:
        """Return the cycle position of every leg at ``time``."""
        values = tuple(self.leg_phase(time, leg_id) for leg_id in LegId)
        return (values[0], values[1], values[2], values[3])

    def in_stance(self, time: float, leg_id: LegId) -> bool:
        """Return ``True`` when ``leg_id`` is loaded at ``time``."""
        return self.leg_phase(time, leg_id) < self.duty_factor

    def contact_state(self, time: float) -> ContactState:
        """Return the contact state of all four feet at ``time``."""
        flags = tuple(self.in_stance(time, leg_id) for leg_id in LegId)
        return ContactState(contacts=(flags[0], flags[1], flags[2], flags[3]))

    def swing_phase(self, time: float, leg_id: LegId) -> float:
        """Return the normalised swing progress of ``leg_id``, in ``[0, 1)``.

        The value is zero at lift off and approaches one at touch down. It is
        zero whenever the leg is in stance.
        """
        phase = self.leg_phase(time, leg_id)
        if phase < self.duty_factor:
            return 0.0
        return (phase - self.duty_factor) / (1.0 - self.duty_factor)

    def stance_phase(self, time: float, leg_id: LegId) -> float:
        """Return the normalised stance progress of ``leg_id``, in ``[0, 1)``.

        The value is zero at touch down and approaches one at lift off. It is
        zero whenever the leg is in swing.
        """
        phase = self.leg_phase(time, leg_id)
        if phase >= self.duty_factor:
            return 0.0
        return phase / self.duty_factor

    def last_touchdown_time(self, time: float, leg_id: LegId) -> float:
        """Return the most recent touchdown instant of ``leg_id`` at or before ``time``."""
        phase = self.leg_phase(time, leg_id)
        return time - phase * self.period

    def last_liftoff_time(self, time: float, leg_id: LegId) -> float:
        """Return the most recent lift off instant of ``leg_id`` at or before ``time``."""
        touchdown = self.last_touchdown_time(time, leg_id)
        liftoff = touchdown + self.stance_duration
        if liftoff > time:
            liftoff -= self.period
        return liftoff

    def stance_measure(self, leg_id: LegId, start: float, end: float) -> float:
        """Return the time ``leg_id`` spends loaded within ``[start, end]``, in seconds.

        The contact schedule is a known periodic step function of time, so the
        measure of the stance set is an antiderivative evaluated at the two ends.
        The result carries no discretisation error at all, unlike a stance fraction
        obtained by counting samples, which can be wrong by up to one sampling
        interval.

        Raises:
            ValueError: If ``end`` precedes ``start``.
        """
        if end < start:
            raise ValueError(f"end must not precede start, got start={start}, end={end}")
        offset = self.phase_offsets[int(leg_id)]
        lower = start / self.period - offset
        upper = end / self.period - offset
        cycles = _stance_measure_to(upper, self.duty_factor) - _stance_measure_to(
            lower, self.duty_factor
        )
        return cycles * self.period

    def stance_fraction(self, leg_id: LegId, start: float, end: float) -> float:
        """Return the fraction of ``[start, end]`` that ``leg_id`` spends loaded.

        Over a whole number of gait cycles this is the duty factor exactly. Over a
        partial window it is the exact stance fraction of that window, which is not
        in general the duty factor.

        Raises:
            ValueError: If the window is empty.
        """
        if end <= start:
            raise ValueError(f"the window must be non-empty, got start={start}, end={end}")
        return self.stance_measure(leg_id, start, end) / (end - start)

    def stance_intervals(
        self, leg_id: LegId, start: float, end: float
    ) -> tuple[tuple[float, float], ...]:
        """Return the loaded intervals of ``leg_id``, clipped to ``[start, end]``.

        Touchdown of leg ``i`` happens at ``t = (k + phi_i) T`` for every integer
        ``k``, and the stance that begins there ends one stance duration later. The
        intervals are therefore enumerated directly rather than recovered by
        scanning a sampled contact array, so an interval edge lands on the phase
        boundary itself.

        Raises:
            ValueError: If ``end`` precedes ``start``.
        """
        if end < start:
            raise ValueError(f"end must not precede start, got start={start}, end={end}")
        offset = self.phase_offsets[int(leg_id)]
        first = math.floor(start / self.period - offset - self.duty_factor)
        last = math.ceil(end / self.period - offset)
        intervals: list[tuple[float, float]] = []
        for cycle in range(first, last + 1):
            touchdown = (cycle + offset) * self.period
            lower = max(touchdown, start)
            upper = min(touchdown + self.stance_duration, end)
            if upper > lower:
                intervals.append((lower, upper))
        return tuple(intervals)


_LIBRARY: dict[str, GaitParameters] = {
    # Lateral sequence walk: touchdown order HL, FL, HR, FR at quarter cycle spacing,
    # normalised so that the front left foot touches down at the start of the cycle.
    "walk": GaitParameters(
        name="walk",
        period=1.0,
        duty_factor=0.75,
        phase_offsets=(0.0, 0.5, 0.75, 0.25),
    ),
    # Diagonal pairs move together.
    "trot": GaitParameters(
        name="trot",
        period=0.5,
        duty_factor=0.5,
        phase_offsets=(0.0, 0.5, 0.5, 0.0),
    ),
    # Lateral pairs move together.
    "pace": GaitParameters(
        name="pace",
        period=0.5,
        duty_factor=0.5,
        phase_offsets=(0.0, 0.5, 0.0, 0.5),
    ),
    # Fore pair moves against the hind pair. The duty factor is below one half, so the
    # gait has two aerial phases per cycle and never has three feet on the ground.
    "bound": GaitParameters(
        name="bound",
        period=0.4,
        duty_factor=0.4,
        phase_offsets=(0.0, 0.0, 0.5, 0.5),
    ),
}

GAIT_LIBRARY: Mapping[str, GaitParameters] = MappingProxyType(_LIBRARY)
GAIT_NAMES: tuple[str, ...] = ("walk", "trot", "pace", "bound")


def gait(
    name: str,
    *,
    period: float | None = None,
    duty_factor: float | None = None,
) -> GaitParameters:
    """Return a named gait from the library, optionally retimed or rescaled.

    Args:
        name: One of ``"walk"``, ``"trot"``, ``"pace"``, ``"bound"``.
        period: Override the cycle period, in seconds.
        duty_factor: Override the duty factor.

    Raises:
        KeyError: If ``name`` is not in the library.
    """
    if name not in GAIT_LIBRARY:
        raise KeyError(f"unknown gait {name!r}; available gaits are {GAIT_NAMES}")
    selected = GAIT_LIBRARY[name]
    if period is not None:
        selected = selected.with_period(period)
    if duty_factor is not None:
        selected = selected.with_duty_factor(duty_factor)
    return selected


def leg_phases(parameters: GaitParameters, times: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the cycle position of every leg at every sample time, shape ``(n, 4)``."""
    sample_times = np.asarray(times, dtype=np.float64).reshape(-1)
    return np.array(
        [parameters.phases(float(t)) for t in sample_times], dtype=np.float64
    ).reshape(sample_times.size, LEG_COUNT)


def contact_schedule(parameters: GaitParameters, times: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return the gait diagram as a boolean array of shape ``(n, 4)``.

    Entry ``(k, i)`` is ``True`` when leg ``i`` is in stance at ``times[k]``. This
    array is the numeric form of the gait diagram of Hildebrand (1965).
    """
    phases = leg_phases(parameters, times)
    return np.asarray(phases < parameters.duty_factor, dtype=np.bool_)


def sampled_duty_factors(schedule: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return the stance fraction of each leg counted from a sampled schedule.

    This is the fraction of sampled instants at which each leg was loaded, so it
    carries a discretisation error of up to one sampling interval. Use
    :func:`exact_duty_factors` for the value itself; this function exists as an
    independent measurement of what a trace actually recorded.
    """
    array = np.asarray(schedule, dtype=np.bool_)
    if array.ndim != 2 or array.shape[1] != LEG_COUNT:
        raise ValueError(f"schedule must have shape (n, {LEG_COUNT}), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("schedule must contain at least one sample")
    return np.asarray(array.mean(axis=0), dtype=np.float64)


def exact_duty_factors(
    parameters: GaitParameters, start: float, end: float
) -> NDArray[np.float64]:
    """Return the exact stance fraction of every leg over ``[start, end]``, shape ``(4,)``.

    The values come from the closed form of the schedule, so over a whole number of
    cycles every entry equals the commanded duty factor to floating point rounding.
    """
    return np.array(
        [parameters.stance_fraction(leg_id, start, end) for leg_id in LegId],
        dtype=np.float64,
    )


def stance_count_extrema(parameters: GaitParameters) -> tuple[int, int]:
    """Return the smallest and largest number of loaded feet over one gait cycle.

    The number of loaded feet is piecewise constant and changes only at a touchdown
    or a lift off, so evaluating it once inside each interval between consecutive
    events gives the exact range. The distinction this makes visible is the one the
    quasi-static criterion turns on: a walk never drops below three loaded feet,
    a trot and a pace hold exactly two, and a bound reaches zero.
    """
    events = {0.0, 1.0}
    for offset in parameters.phase_offsets:
        events.add(_wrap_phase(offset))
        events.add(_wrap_phase(offset + parameters.duty_factor))
    ordered = sorted(events)
    counts = [
        parameters.contact_state(0.5 * (lower + upper) * parameters.period).stance_count
        for lower, upper in itertools.pairwise(ordered)
    ]
    return min(counts), max(counts)
