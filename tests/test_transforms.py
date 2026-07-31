"""Property tests for the SE(3) helpers."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from quadruped_gait.model.transforms import (
    BodyPose,
    rotation_rpy,
    rotation_x,
    rotation_y,
    rotation_z,
    rpy_from_rotation,
)


@pytest.mark.parametrize("builder", [rotation_x, rotation_y, rotation_z])
@pytest.mark.parametrize("angle", [-2.0, -0.3, 0.0, 0.7, 1.9])
def test_elementary_rotations_are_orthonormal(
    builder: Callable[[float], NDArray[np.float64]], angle: float
) -> None:
    rotation = builder(angle)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize(
    "rpy",
    [(0.0, 0.0, 0.0), (0.2, -0.3, 1.1), (-0.9, 0.4, -2.5), (0.05, 1.2, 0.0)],
)
def test_rpy_round_trip(rpy: tuple[float, float, float]) -> None:
    rotation = rotation_rpy(*rpy)
    recovered = rpy_from_rotation(rotation)
    np.testing.assert_allclose(rotation_rpy(*recovered), rotation, atol=1e-12)


def test_rpy_recovers_the_original_angles_away_from_the_singularity() -> None:
    rpy = (0.2, -0.3, 1.1)
    np.testing.assert_allclose(rpy_from_rotation(rotation_rpy(*rpy)), rpy, atol=1e-12)


def test_pose_apply_and_inverse_apply_are_inverses() -> None:
    pose = BodyPose.from_rpy((0.4, -1.2, 0.5), roll=0.1, pitch=-0.2, yaw=0.9)
    rng = np.random.default_rng(7)
    points = rng.normal(size=(17, 3))
    np.testing.assert_allclose(pose.inverse_apply(pose.apply(points)), points, atol=1e-12)


def test_pose_inverse_composes_to_identity() -> None:
    pose = BodyPose.from_rpy((1.0, 2.0, 3.0), roll=-0.6, pitch=0.3, yaw=2.0)
    composed = pose.compose(pose.inverse())
    np.testing.assert_allclose(composed.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(composed.position, np.zeros(3), atol=1e-12)


def test_pose_matrix_matches_apply() -> None:
    pose = BodyPose.from_rpy((0.1, 0.2, 0.3), yaw=0.7)
    point = np.array([0.5, -0.25, 0.75])
    homogeneous = pose.matrix @ np.append(point, 1.0)
    np.testing.assert_allclose(homogeneous[:3], pose.apply(point), atol=1e-14)


def test_identity_pose_is_the_identity() -> None:
    pose = BodyPose.identity()
    point = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose.apply(point), point, atol=0.0)


def test_yaw_only_rotation_matches_rotation_z() -> None:
    np.testing.assert_allclose(rotation_rpy(0.0, 0.0, 0.8), rotation_z(0.8), atol=1e-15)


def test_pose_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="position"):
        BodyPose(position=np.zeros(2), rotation=np.eye(3))
    with pytest.raises(ValueError, match="rotation"):
        BodyPose(position=np.zeros(3), rotation=np.eye(4))


def test_pitch_singularity_is_handled() -> None:
    rotation = rotation_rpy(0.0, math.pi / 2.0, 0.4)
    roll, pitch, _ = rpy_from_rotation(rotation)
    assert roll == 0.0
    assert pitch == pytest.approx(math.pi / 2.0, abs=1e-7)
