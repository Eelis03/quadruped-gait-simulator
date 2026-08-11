"""Rigid body transforms in SE(3) for the floating base and the hip frames.

Conventions follow Siciliano et al. (2009), section 2.2: a pose is a rotation
matrix in SO(3) paired with a translation, roll-pitch-yaw angles are intrinsic
ZYX Euler angles, and the body frame has x forward, y to the left, and z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BodyPose",
    "Matrix3",
    "Vector3",
    "rotation_rpy",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "rpy_from_rotation",
]

Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]


def rotation_x(angle: float) -> Matrix3:
    """Return the rotation matrix for ``angle`` radians about the x axis."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]], dtype=np.float64)


def rotation_y(angle: float) -> Matrix3:
    """Return the rotation matrix for ``angle`` radians about the y axis."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]], dtype=np.float64)


def rotation_z(angle: float) -> Matrix3:
    """Return the rotation matrix for ``angle`` radians about the z axis."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_rpy(roll: float, pitch: float, yaw: float) -> Matrix3:
    """Return the intrinsic ZYX rotation ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def rpy_from_rotation(rotation: Matrix3) -> tuple[float, float, float]:
    """Recover roll, pitch, and yaw from an intrinsic ZYX rotation matrix.

    The decomposition is singular at ``pitch = +/- pi/2``; there roll is set to
    zero and the whole rotation about the vertical is assigned to yaw.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    sin_pitch = -float(matrix[2, 0])
    sin_pitch = min(1.0, max(-1.0, sin_pitch))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) > 1.0 - 1e-12:
        roll = 0.0
        yaw = math.atan2(-float(matrix[0, 1]), float(matrix[1, 1]))
    else:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return roll, pitch, yaw


@dataclass(frozen=True, slots=True, eq=False)
class BodyPose:
    """The pose of the trunk frame expressed in the world frame.

    Attributes:
        position: Trunk origin in world coordinates, shape ``(3,)``.
        rotation: Rotation from trunk coordinates to world coordinates, shape ``(3, 3)``.
    """

    position: Vector3
    rotation: Matrix3

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(-1)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)

    @classmethod
    def identity(cls) -> BodyPose:
        """Return the pose with the trunk frame coincident with the world frame."""
        return cls(position=np.zeros(3, dtype=np.float64), rotation=np.eye(3, dtype=np.float64))

    @classmethod
    def from_rpy(
        cls,
        position: Vector3 | tuple[float, float, float],
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> BodyPose:
        """Build a pose from a translation and intrinsic ZYX Euler angles."""
        return cls(
            position=np.asarray(position, dtype=np.float64),
            rotation=rotation_rpy(roll, pitch, yaw),
        )

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Return the homogeneous 4 by 4 transform from trunk to world."""
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self.rotation
        out[:3, 3] = self.position
        return out

    @property
    def rpy(self) -> tuple[float, float, float]:
        """Return the roll, pitch, and yaw of this pose in radians."""
        return rpy_from_rotation(self.rotation)

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``points`` from trunk coordinates to world coordinates.

        Accepts a single point of shape ``(3,)`` or a stack of shape ``(n, 3)``
        and returns an array of the same shape.
        """
        array = np.asarray(points, dtype=np.float64)
        return array @ self.rotation.T + self.position

    def inverse_apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``points`` from world coordinates to trunk coordinates."""
        array = np.asarray(points, dtype=np.float64)
        return (array - self.position) @ self.rotation

    def inverse(self) -> BodyPose:
        """Return the pose that maps world coordinates to trunk coordinates."""
        rotation = self.rotation.T
        return BodyPose(position=-(rotation @ self.position), rotation=rotation)

    def compose(self, other: BodyPose) -> BodyPose:
        """Return ``self`` composed with ``other``, that is ``self * other``."""
        return BodyPose(
            position=self.position + self.rotation @ other.position,
            rotation=self.rotation @ other.rotation,
        )
