"""Forward and inverse kinematics of one three degree of freedom leg.

The chain is hip roll about x, a fixed lateral offset, hip pitch about y, the
thigh, knee pitch about y, and the shank. Writing ``q1, q2, q3`` for the three
joint angles, ``d`` for the signed lateral offset, and ``l2, l3`` for the thigh
and shank lengths, the foot position in the hip frame is::

    x =  -(l2 sin q2 + l3 sin(q2 + q3))
    y =   d cos q1 - z_s sin q1
    z =   d sin q1 + z_s cos q1
    z_s = -(l2 cos q2 + l3 cos(q2 + q3))

The inverse is closed form. The lateral component fixes ``q1`` up to the choice
of foot above or below the hip, the planar two link subchain fixes ``q3`` up to
the knee sign, and ``q2`` then follows from a single two argument arctangent.
The derivation is the standard planar two link solution of Siciliano et al.
(2009), section 2.12, applied after the roll rotation has been removed.

Two branch choices are therefore needed to make the inverse single valued. The
knee sign is exposed as the ``knee_forward`` argument. The other choice is fixed
by convention: :func:`inverse_kinematics` always returns the solution with
``z_s < 0``, that is with the foot below the hip pitch axis, which is the only
branch a standing or walking robot uses. :func:`foot_below_hip_axis` reports
whether a given joint triple lies on that branch, and the inverse is a left
inverse of the forward map exactly on the triples for which it returns ``True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.model.geometry import LegGeometry

__all__ = [
    "JointAngles",
    "UnreachableTargetError",
    "foot_below_hip_axis",
    "forward_kinematics",
    "inverse_kinematics",
    "is_reachable",
    "reach_interval",
]

_TOLERANCE = 1e-12


class UnreachableTargetError(ValueError):
    """Raised when a commanded foot position lies outside the leg workspace."""


@dataclass(frozen=True, slots=True)
class JointAngles:
    """Joint angles of one leg, in radians.

    Attributes:
        hip_roll: Rotation about the trunk x axis, positive by the right hand rule.
        hip_pitch: Rotation of the thigh about the hip y axis, positive swings the foot back.
        knee_pitch: Rotation of the shank about the knee y axis, relative to the thigh.
    """

    hip_roll: float
    hip_pitch: float
    knee_pitch: float

    def as_array(self) -> NDArray[np.float64]:
        """Return the angles as an array of shape ``(3,)``."""
        return np.array([self.hip_roll, self.hip_pitch, self.knee_pitch], dtype=np.float64)

    @classmethod
    def from_array(cls, values: NDArray[np.float64]) -> JointAngles:
        """Build joint angles from an array of shape ``(3,)``."""
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (3,):
            raise ValueError(f"expected three joint angles, got shape {array.shape}")
        return cls(hip_roll=float(array[0]), hip_pitch=float(array[1]), knee_pitch=float(array[2]))


def forward_kinematics(
    leg: LegGeometry, angles: JointAngles, *, lateral_sign: float
) -> NDArray[np.float64]:
    """Return the foot position in the hip frame for the given joint angles.

    Args:
        leg: Link lengths of the leg.
        angles: Hip roll, hip pitch, and knee pitch, in radians.
        lateral_sign: ``+1.0`` for a left leg, ``-1.0`` for a right leg. It sets the
            direction of the fixed abduction offset.

    Returns:
        Foot position in hip coordinates, shape ``(3,)``.
    """
    offset = lateral_sign * leg.abduction_offset
    pitch_sum = angles.hip_pitch + angles.knee_pitch
    sagittal_x = -(
        leg.thigh_length * math.sin(angles.hip_pitch) + leg.shank_length * math.sin(pitch_sum)
    )
    sagittal_z = -(
        leg.thigh_length * math.cos(angles.hip_pitch) + leg.shank_length * math.cos(pitch_sum)
    )
    cos_roll = math.cos(angles.hip_roll)
    sin_roll = math.sin(angles.hip_roll)
    return np.array(
        [
            sagittal_x,
            cos_roll * offset - sin_roll * sagittal_z,
            sin_roll * offset + cos_roll * sagittal_z,
        ],
        dtype=np.float64,
    )


def foot_below_hip_axis(leg: LegGeometry, angles: JointAngles) -> bool:
    """Return ``True`` when the foot lies below the hip pitch axis.

    This is the branch that :func:`inverse_kinematics` returns. The test is on the
    sagittal component of the hip to foot vector, taken before the hip roll
    rotation, so it does not depend on the roll angle.
    """
    sagittal_z = -(
        leg.thigh_length * math.cos(angles.hip_pitch)
        + leg.shank_length * math.cos(angles.hip_pitch + angles.knee_pitch)
    )
    return sagittal_z < 0.0


def reach_interval(leg: LegGeometry) -> tuple[float, float]:
    """Return the closed interval of hip to foot distances the planar subchain spans."""
    return leg.min_sagittal_reach, leg.max_sagittal_reach


def _sagittal_radius(
    leg: LegGeometry, foot_in_hip: NDArray[np.float64], lateral_sign: float
) -> tuple[float, float] | None:
    """Return ``(radius, lateral_leg)`` for the planar subchain, or ``None`` if invalid.

    ``lateral_leg`` is the length of the projection of the hip to foot vector onto
    the sagittal plane of the leg after the roll rotation has been removed.
    """
    target = np.asarray(foot_in_hip, dtype=np.float64).reshape(-1)
    if target.shape != (3,):
        raise ValueError(f"foot_in_hip must have shape (3,), got {target.shape}")
    offset = lateral_sign * leg.abduction_offset
    lateral_square = float(target[1]) ** 2 + float(target[2]) ** 2 - offset**2
    if lateral_square < -_TOLERANCE:
        return None
    lateral_leg = math.sqrt(max(lateral_square, 0.0))
    radius = math.hypot(float(target[0]), lateral_leg)
    return radius, lateral_leg


def is_reachable(
    leg: LegGeometry, foot_in_hip: NDArray[np.float64], *, lateral_sign: float
) -> bool:
    """Return ``True`` when the closed form inverse kinematics has a solution."""
    resolved = _sagittal_radius(leg, foot_in_hip, lateral_sign)
    if resolved is None:
        return False
    radius, _ = resolved
    lower, upper = reach_interval(leg)
    return lower - _TOLERANCE <= radius <= upper + _TOLERANCE


def inverse_kinematics(
    leg: LegGeometry,
    foot_in_hip: NDArray[np.float64],
    *,
    lateral_sign: float,
    knee_forward: bool = False,
) -> JointAngles:
    """Return the joint angles that place the foot at ``foot_in_hip``.

    Args:
        leg: Link lengths of the leg.
        foot_in_hip: Desired foot position in hip coordinates, shape ``(3,)``.
        lateral_sign: ``+1.0`` for a left leg, ``-1.0`` for a right leg.
        knee_forward: Select the elbow-up branch of the two link solution. The
            default selects the knee-backward branch, the posture in which the
            shank trails the thigh.

    Returns:
        The joint angles that reproduce ``foot_in_hip`` under
        :func:`forward_kinematics`.

    Raises:
        UnreachableTargetError: If the target lies inside the cylinder swept by the
            abduction offset, or outside the annulus spanned by the thigh and shank.
    """
    target = np.asarray(foot_in_hip, dtype=np.float64).reshape(-1)
    resolved = _sagittal_radius(leg, target, lateral_sign)
    if resolved is None:
        raise UnreachableTargetError(
            "target lies inside the cylinder of radius "
            f"{leg.abduction_offset} swept by the abduction offset: {target.tolist()}"
        )
    radius, lateral_leg = resolved
    lower, upper = reach_interval(leg)
    if radius > upper + _TOLERANCE:
        raise UnreachableTargetError(
            f"target is {radius:.6f} m from the hip pitch axis, beyond the "
            f"maximum reach of {upper:.6f} m"
        )
    if radius < lower - _TOLERANCE:
        raise UnreachableTargetError(
            f"target is {radius:.6f} m from the hip pitch axis, inside the "
            f"minimum reach of {lower:.6f} m"
        )

    offset = lateral_sign * leg.abduction_offset
    hip_roll = math.atan2(float(target[2]), float(target[1])) - math.atan2(-lateral_leg, offset)
    hip_roll = math.remainder(hip_roll, math.tau)

    cosine = (radius**2 - leg.thigh_length**2 - leg.shank_length**2) / (
        2.0 * leg.thigh_length * leg.shank_length
    )
    cosine = min(1.0, max(-1.0, cosine))
    knee_pitch = math.acos(cosine) if knee_forward else -math.acos(cosine)

    reach_along = leg.thigh_length + leg.shank_length * math.cos(knee_pitch)
    reach_across = leg.shank_length * math.sin(knee_pitch)
    sagittal_x = float(target[0])
    hip_pitch = math.atan2(
        -reach_along * sagittal_x - reach_across * lateral_leg,
        -reach_across * sagittal_x + reach_along * lateral_leg,
    )
    return JointAngles(hip_roll=hip_roll, hip_pitch=hip_pitch, knee_pitch=knee_pitch)
