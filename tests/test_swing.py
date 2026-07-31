"""Property tests for the swing trajectory generators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quadruped_gait.algorithm.swing import (
    BezierSwing,
    CycloidalSwing,
    bezier_point,
    make_swing,
    stance_foot_in_body,
)
from quadruped_gait.model.transforms import BodyPose

LIFT_OFF = np.array([0.10, 0.23, 0.0])
TOUCH_DOWN = np.array([0.35, 0.23, 0.0])
CLEARANCE = 0.08


@pytest.mark.parametrize("profile", ["cycloidal", "bezier"])
def test_swing_starts_and_ends_at_the_commanded_footholds(profile: str) -> None:
    trajectory = make_swing(profile, LIFT_OFF, TOUCH_DOWN, CLEARANCE)
    np.testing.assert_allclose(trajectory.position(0.0), LIFT_OFF, atol=1e-12)
    np.testing.assert_allclose(trajectory.position(1.0), TOUCH_DOWN, atol=1e-12)


@pytest.mark.parametrize("profile", ["cycloidal", "bezier"])
def test_swing_reaches_the_requested_clearance_at_mid_swing(profile: str) -> None:
    trajectory = make_swing(profile, LIFT_OFF, TOUCH_DOWN, CLEARANCE)
    assert float(trajectory.position(0.5)[2]) == pytest.approx(CLEARANCE, abs=1e-12)


@pytest.mark.parametrize("profile", ["cycloidal", "bezier"])
def test_swing_never_dips_below_the_ground(profile: str) -> None:
    trajectory = make_swing(profile, LIFT_OFF, TOUCH_DOWN, CLEARANCE)
    heights = [float(trajectory.position(s / 200.0)[2]) for s in range(201)]
    assert min(heights) >= -1e-12
    assert max(heights) == pytest.approx(CLEARANCE, abs=1e-9)


@pytest.mark.parametrize("profile", ["cycloidal", "bezier"])
def test_swing_advances_monotonically_along_the_chord(profile: str) -> None:
    trajectory = make_swing(profile, LIFT_OFF, TOUCH_DOWN, CLEARANCE)
    x = np.array([float(trajectory.position(s / 400.0)[0]) for s in range(401)])
    assert bool(np.all(np.diff(x) >= -1e-15))


@pytest.mark.parametrize("profile", ["cycloidal", "bezier"])
def test_swing_progress_is_clamped_outside_the_unit_interval(profile: str) -> None:
    trajectory = make_swing(profile, LIFT_OFF, TOUCH_DOWN, CLEARANCE)
    np.testing.assert_allclose(trajectory.position(-0.5), trajectory.position(0.0), atol=0.0)
    np.testing.assert_allclose(trajectory.position(1.5), trajectory.position(1.0), atol=0.0)


def test_cycloidal_swing_has_zero_horizontal_velocity_at_both_ends() -> None:
    trajectory = CycloidalSwing(lift_off=LIFT_OFF, touch_down=TOUCH_DOWN, clearance=CLEARANCE)
    for progress in (0.0, 1.0):
        velocity = trajectory.velocity(progress, swing_duration=0.25)
        np.testing.assert_allclose(velocity, np.zeros(3), atol=1e-12)


def test_cycloidal_swing_velocity_peaks_at_mid_swing() -> None:
    trajectory = CycloidalSwing(lift_off=LIFT_OFF, touch_down=TOUCH_DOWN, clearance=CLEARANCE)
    speeds = [
        float(np.linalg.norm(trajectory.velocity(s / 100.0, 0.25))) for s in range(101)
    ]
    assert int(np.argmax(speeds)) == 50


def test_cycloidal_swing_rejects_a_non_positive_duration() -> None:
    trajectory = CycloidalSwing(lift_off=LIFT_OFF, touch_down=TOUCH_DOWN, clearance=CLEARANCE)
    with pytest.raises(ValueError, match="swing_duration"):
        trajectory.velocity(0.5, swing_duration=0.0)


def test_bezier_lifts_off_and_touches_down_vertically() -> None:
    """The repeated endpoint control points make the first and last edges vertical."""
    trajectory = BezierSwing.from_endpoints(LIFT_OFF, TOUCH_DOWN, CLEARANCE)
    points = trajectory.control_points
    assert points.shape == (6, 3)
    np.testing.assert_allclose(points[1, :2], points[0, :2], atol=1e-15)
    np.testing.assert_allclose(points[4, :2], points[5, :2], atol=1e-15)
    assert float(points[1, 2] - points[0, 2]) > 0.0
    assert float(points[4, 2] - points[5, 2]) > 0.0


def test_bezier_point_matches_the_bernstein_expansion() -> None:
    rng = np.random.default_rng(5)
    control = rng.normal(size=(6, 3))
    degree = control.shape[0] - 1
    for parameter in (0.0, 0.17, 0.5, 0.83, 1.0):
        expected = np.zeros(3)
        for index in range(control.shape[0]):
            weight = (
                math.comb(degree, index)
                * parameter**index
                * (1.0 - parameter) ** (degree - index)
            )
            expected += weight * control[index]
        np.testing.assert_allclose(bezier_point(control, parameter), expected, atol=1e-12)


def test_bezier_point_of_a_single_control_point_is_constant() -> None:
    control = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(bezier_point(control, 0.4), [1.0, 2.0, 3.0], atol=0.0)


def test_swing_rejects_bad_input() -> None:
    with pytest.raises(KeyError, match="unknown swing profile"):
        make_swing("spline", LIFT_OFF, TOUCH_DOWN, CLEARANCE)
    with pytest.raises(ValueError, match="shape"):
        CycloidalSwing(lift_off=np.zeros(2), touch_down=TOUCH_DOWN, clearance=CLEARANCE)
    with pytest.raises(ValueError, match="clearance"):
        CycloidalSwing(lift_off=LIFT_OFF, touch_down=TOUCH_DOWN, clearance=-0.01)
    with pytest.raises(ValueError, match="clearance"):
        BezierSwing.from_endpoints(LIFT_OFF, TOUCH_DOWN, -0.01)
    with pytest.raises(ValueError, match="shape"):
        bezier_point(np.zeros(3), 0.5)


def test_stance_foot_in_body_undoes_the_body_pose() -> None:
    pose = BodyPose.from_rpy((1.2, -0.4, 0.42), yaw=0.6)
    world = np.array([1.35, -0.2, 0.0])
    body = stance_foot_in_body(pose, world)
    np.testing.assert_allclose(pose.apply(body), world, atol=1e-12)


def test_stance_foot_moves_backward_in_the_body_frame_as_the_trunk_advances() -> None:
    """A loaded foot is fixed in the world, so it trails the advancing trunk."""
    world = np.array([0.25, 0.23, 0.0])
    first = stance_foot_in_body(BodyPose.from_rpy((0.0, 0.0, 0.42)), world)
    second = stance_foot_in_body(BodyPose.from_rpy((0.10, 0.0, 0.42)), world)
    assert float(second[0]) < float(first[0])
    assert float(first[0] - second[0]) == pytest.approx(0.10, abs=1e-12)
