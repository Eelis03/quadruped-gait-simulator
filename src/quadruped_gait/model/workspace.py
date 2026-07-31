"""Sampling of the reachable leg workspace and round trip verification.

The functions here exist so that the accuracy of the closed form inverse
kinematics can be measured on a population of targets rather than on a single
hand picked posture. Targets are produced by sampling joint angles inside the
declared limits and running forward kinematics, so every sampled target is
reachable by construction.

The default limits are chosen so that every sampled posture keeps the foot below
the hip pitch axis, which is the branch the inverse kinematics returns. Sampling
outside that region still gives an exact round trip in position, because the
inverse then returns the mirrored posture that reaches the same point, but the
joint angles themselves would no longer be recovered.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.model.geometry import LegGeometry
from quadruped_gait.model.kinematics import (
    JointAngles,
    UnreachableTargetError,
    forward_kinematics,
    inverse_kinematics,
)

__all__ = [
    "DEFAULT_JOINT_LIMITS",
    "round_trip_errors",
    "sample_joint_angles",
    "workspace_points",
]

# Plausible joint travel for a robot of this size, in radians: hip roll, hip pitch,
# and knee pitch. The knee range stays strictly inside the knee-backward branch and
# away from both the fully folded and the fully extended singular configurations, and
# the hip pitch range keeps the foot below the hip pitch axis for every combination.
DEFAULT_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-0.70, 0.70),
    (-0.55, 1.20),
    (-1.70, -0.30),
)


def sample_joint_angles(
    count: int,
    rng: np.random.Generator,
    limits: tuple[tuple[float, float], ...] = DEFAULT_JOINT_LIMITS,
) -> NDArray[np.float64]:
    """Draw ``count`` joint triples uniformly from ``limits``, shape ``(count, 3)``."""
    if count < 1:
        raise ValueError("count must be at least one")
    if len(limits) != 3:
        raise ValueError("limits must contain one interval per joint")
    lower = np.array([interval[0] for interval in limits], dtype=np.float64)
    upper = np.array([interval[1] for interval in limits], dtype=np.float64)
    if np.any(upper <= lower):
        raise ValueError("every joint limit must have a positive width")
    return np.asarray(rng.uniform(lower, upper, size=(count, 3)), dtype=np.float64)


def workspace_points(
    leg: LegGeometry, angles: NDArray[np.float64], *, lateral_sign: float
) -> NDArray[np.float64]:
    """Map joint triples to foot positions in the hip frame, shape ``(n, 3)``."""
    array = np.asarray(angles, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"angles must have shape (n, 3), got {array.shape}")
    return np.stack(
        [
            forward_kinematics(leg, JointAngles.from_array(row), lateral_sign=lateral_sign)
            for row in array
        ]
    )


def round_trip_errors(
    leg: LegGeometry,
    points: NDArray[np.float64],
    *,
    lateral_sign: float,
    knee_forward: bool = False,
) -> NDArray[np.float64]:
    """Return the position error of forward kinematics composed with the inverse.

    Each target is passed through :func:`inverse_kinematics` and back through
    :func:`forward_kinematics`; the returned array holds the Euclidean distance
    between the target and the reconstruction, in metres. A target that the
    inverse kinematics rejects contributes ``nan``.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {array.shape}")
    errors = np.full(array.shape[0], np.nan, dtype=np.float64)
    for index, target in enumerate(array):
        try:
            angles = inverse_kinematics(
                leg, target, lateral_sign=lateral_sign, knee_forward=knee_forward
            )
        except UnreachableTargetError:
            continue
        reconstructed = forward_kinematics(leg, angles, lateral_sign=lateral_sign)
        errors[index] = float(np.linalg.norm(reconstructed - target))
    return errors
