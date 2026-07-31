"""Swing phase foot trajectories and the stance phase foot constraint.

Two swing profiles are provided. The cycloidal profile follows Sakakibara et al.
(1990): the horizontal displacement is the cycloid ``s - sin(2 pi s) / (2 pi)``,
which starts and ends with zero horizontal velocity, and the vertical
displacement is the raised cosine ``(1 - cos(2 pi s)) / 2``. The Bezier profile
follows the swing leg parameterisation of Hyun et al. (2014): a quintic Bezier
whose first and last control edges are vertical, so the foot leaves and meets the
ground along the surface normal.

Both profiles are parameterised by normalised swing progress ``s`` in ``[0, 1]``,
which the gait scheduler supplies. Stance feet do not use these curves at all:
a loaded foot is stationary in the world frame, so its motion relative to the
trunk is entirely the commanded body motion, which is what
:func:`stance_foot_in_body` evaluates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.model.transforms import BodyPose

__all__ = [
    "BezierSwing",
    "CycloidalSwing",
    "SwingTrajectory",
    "bezier_point",
    "make_swing",
    "stance_foot_in_body",
]

# Relative vertical offsets of the six Bezier control points. The first and last are
# zero so the curve starts and ends on the chord; the repeated interior values give a
# flat apex. The apex of the resulting curve sits at 38 / 32 times the offset scale,
# which is divided out so that the requested clearance is reproduced exactly.
_BEZIER_HEIGHT_SHAPE: tuple[float, ...] = (0.0, 1.0, 1.4, 1.4, 1.0, 0.0)
_BEZIER_APEX_GAIN = 38.0 / 32.0
_BEZIER_ALONG: tuple[float, ...] = (0.0, 0.0, 0.25, 0.75, 1.0, 1.0)


class SwingTrajectory(Protocol):
    """A foot path from lift off to touch down, parameterised by swing progress."""

    def position(self, progress: float) -> NDArray[np.float64]:
        """Return the foot position at normalised swing progress in ``[0, 1]``."""
        ...


def _endpoints(
    lift_off: NDArray[np.float64], touch_down: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    start = np.asarray(lift_off, dtype=np.float64).reshape(-1)
    end = np.asarray(touch_down, dtype=np.float64).reshape(-1)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("lift_off and touch_down must both have shape (3,)")
    return start, end


@dataclass(frozen=True, slots=True, eq=False)
class CycloidalSwing:
    """Cycloidal swing profile with a raised cosine height envelope.

    Attributes:
        lift_off: Foot position at the start of swing, shape ``(3,)``.
        touch_down: Foot position at the end of swing, shape ``(3,)``.
        clearance: Height of the apex above the straight chord between the
            endpoints, in metres.
    """

    lift_off: NDArray[np.float64]
    touch_down: NDArray[np.float64]
    clearance: float

    def __post_init__(self) -> None:
        start, end = _endpoints(self.lift_off, self.touch_down)
        if self.clearance < 0.0:
            raise ValueError("clearance must not be negative")
        object.__setattr__(self, "lift_off", start)
        object.__setattr__(self, "touch_down", end)

    def position(self, progress: float) -> NDArray[np.float64]:
        """Return the foot position at normalised swing progress in ``[0, 1]``."""
        s = min(1.0, max(0.0, progress))
        along = s - math.sin(math.tau * s) / math.tau
        chord = self.lift_off + along * (self.touch_down - self.lift_off)
        height = self.clearance * 0.5 * (1.0 - math.cos(math.tau * s))
        return np.array([chord[0], chord[1], chord[2] + height], dtype=np.float64)

    def velocity(self, progress: float, swing_duration: float) -> NDArray[np.float64]:
        """Return the foot velocity in metres per second at ``progress``."""
        if swing_duration <= 0.0:
            raise ValueError("swing_duration must be positive")
        s = min(1.0, max(0.0, progress))
        along_rate = 1.0 - math.cos(math.tau * s)
        chord_rate = along_rate * (self.touch_down - self.lift_off)
        height_rate = self.clearance * 0.5 * math.tau * math.sin(math.tau * s)
        return (
            np.array([chord_rate[0], chord_rate[1], chord_rate[2] + height_rate], dtype=np.float64)
            / swing_duration
        )


def bezier_point(
    control_points: NDArray[np.float64], parameter: float
) -> NDArray[np.float64]:
    """Evaluate a Bezier curve by the de Casteljau algorithm.

    Args:
        control_points: Control polygon of shape ``(n, d)`` with ``n >= 1``.
        parameter: Curve parameter in ``[0, 1]``.

    Returns:
        The curve point, shape ``(d,)``.
    """
    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError(f"control_points must have shape (n, d) with n >= 1, got {points.shape}")
    t = min(1.0, max(0.0, parameter))
    current = points.copy()
    while current.shape[0] > 1:
        current = (1.0 - t) * current[:-1] + t * current[1:]
    return np.asarray(current[0], dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False)
class BezierSwing:
    """Quintic Bezier swing profile with vertical lift off and touch down.

    Attributes:
        control_points: Control polygon of shape ``(6, 3)``.
    """

    control_points: NDArray[np.float64]

    def __post_init__(self) -> None:
        points = np.asarray(self.control_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"control_points must have shape (n, 3), got {points.shape}")
        object.__setattr__(self, "control_points", points)

    @classmethod
    def from_endpoints(
        cls,
        lift_off: NDArray[np.float64],
        touch_down: NDArray[np.float64],
        clearance: float,
    ) -> BezierSwing:
        """Build the standard six point control polygon for one swing.

        The horizontal control points are placed at the fractions
        ``(0, 0, 0.25, 0.75, 1, 1)`` of the chord, which repeats the endpoints and
        therefore makes the first and last control edges purely vertical. The
        vertical offsets are scaled so that the apex sits exactly ``clearance``
        above the chord midpoint.
        """
        start, end = _endpoints(lift_off, touch_down)
        if clearance < 0.0:
            raise ValueError("clearance must not be negative")
        scale = clearance / _BEZIER_APEX_GAIN
        points = np.empty((len(_BEZIER_ALONG), 3), dtype=np.float64)
        for index, (along, shape) in enumerate(
            zip(_BEZIER_ALONG, _BEZIER_HEIGHT_SHAPE, strict=True)
        ):
            base = start + along * (end - start)
            points[index] = base + np.array([0.0, 0.0, scale * shape], dtype=np.float64)
        return cls(control_points=points)

    def position(self, progress: float) -> NDArray[np.float64]:
        """Return the foot position at normalised swing progress in ``[0, 1]``."""
        return bezier_point(self.control_points, progress)


def make_swing(
    profile: str,
    lift_off: NDArray[np.float64],
    touch_down: NDArray[np.float64],
    clearance: float,
) -> SwingTrajectory:
    """Build a swing trajectory by profile name.

    Args:
        profile: Either ``"cycloidal"`` or ``"bezier"``.
        lift_off: Foot position at the start of swing, shape ``(3,)``.
        touch_down: Foot position at the end of swing, shape ``(3,)``.
        clearance: Apex height above the chord, in metres.

    Raises:
        KeyError: If ``profile`` is not a known swing profile.
    """
    if profile == "cycloidal":
        return CycloidalSwing(lift_off=lift_off, touch_down=touch_down, clearance=clearance)
    if profile == "bezier":
        return BezierSwing.from_endpoints(lift_off, touch_down, clearance)
    raise KeyError(f"unknown swing profile {profile!r}; expected 'cycloidal' or 'bezier'")


def stance_foot_in_body(
    pose: BodyPose, foot_in_world: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the trunk frame position of a stationary loaded foot.

    A foot in stance does not slip, so its world position is constant and all of
    its apparent motion in the trunk frame comes from the commanded body twist.
    This function is the whole stance phase model.
    """
    return pose.inverse_apply(np.asarray(foot_in_world, dtype=np.float64))
