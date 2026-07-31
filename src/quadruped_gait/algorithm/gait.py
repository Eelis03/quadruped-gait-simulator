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
    "gait",
    "leg_phases",
    "realised_duty_factors",
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


def realised_duty_factors(schedule: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return the measured stance fraction of each leg from a contact schedule."""
    array = np.asarray(schedule, dtype=np.bool_)
    if array.ndim != 2 or array.shape[1] != LEG_COUNT:
        raise ValueError(f"schedule must have shape (n, {LEG_COUNT}), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("schedule must contain at least one sample")
    return np.asarray(array.mean(axis=0), dtype=np.float64)
