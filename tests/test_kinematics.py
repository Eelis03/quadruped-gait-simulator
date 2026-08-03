"""Property and invariant tests for the leg kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quadruped_gait.model.contact import ContactState
from quadruped_gait.model.geometry import LegGeometry, LegId, RobotModel, default_robot
from quadruped_gait.model.kinematics import (
    JointAngles,
    UnreachableTargetError,
    foot_below_hip_axis,
    forward_kinematics,
    inverse_kinematics,
    is_reachable,
    reach_interval,
)
from quadruped_gait.model.workspace import (
    DEFAULT_JOINT_LIMITS,
    round_trip_errors,
    sample_joint_angles,
    workspace_points,
)

ROUND_TRIP_TOLERANCE = 1e-9


@pytest.mark.parametrize("lateral_sign", [1.0, -1.0])
def test_forward_then_inverse_recovers_the_foot_position(lateral_sign: float) -> None:
    """Forward kinematics composed with the inverse is the identity on positions."""
    leg = default_robot().leg
    rng = np.random.default_rng(1234)
    angles = sample_joint_angles(4000, rng)
    points = workspace_points(leg, angles, lateral_sign=lateral_sign)
    errors = round_trip_errors(leg, points, lateral_sign=lateral_sign)
    assert np.isfinite(errors).all()
    assert float(np.max(errors)) < ROUND_TRIP_TOLERANCE


def test_the_default_sampling_limits_stay_on_the_selected_branch() -> None:
    """Every sampled posture keeps the foot below the hip pitch axis."""
    leg = default_robot().leg
    rng = np.random.default_rng(42)
    angles = sample_joint_angles(20000, rng, DEFAULT_JOINT_LIMITS)
    assert all(foot_below_hip_axis(leg, JointAngles.from_array(row)) for row in angles)


@pytest.mark.parametrize("lateral_sign", [1.0, -1.0])
def test_inverse_recovers_the_joint_angles_on_the_same_branch(lateral_sign: float) -> None:
    """On the knee-backward branch the inverse returns the very angles used."""
    leg = default_robot().leg
    rng = np.random.default_rng(99)
    angles = sample_joint_angles(2000, rng)
    points = workspace_points(leg, angles, lateral_sign=lateral_sign)
    for row, target in zip(angles, points, strict=True):
        solved = inverse_kinematics(leg, target, lateral_sign=lateral_sign)
        np.testing.assert_allclose(solved.as_array(), row, atol=1e-9)


def test_the_mirrored_branch_reaches_the_same_point_with_different_angles() -> None:
    """A posture with the foot above the hip axis is mapped to its mirror image."""
    leg = default_robot().leg
    folded = JointAngles(hip_roll=0.3, hip_pitch=-1.1, knee_pitch=-1.9)
    assert not foot_below_hip_axis(leg, folded)
    target = forward_kinematics(leg, folded, lateral_sign=1.0)
    solved = inverse_kinematics(leg, target, lateral_sign=1.0)
    assert foot_below_hip_axis(leg, solved)
    assert solved.hip_roll != pytest.approx(folded.hip_roll)
    np.testing.assert_allclose(
        forward_kinematics(leg, solved, lateral_sign=1.0), target, atol=1e-12
    )


def test_both_knee_branches_reach_the_same_point() -> None:
    leg = default_robot().leg
    target = np.array([0.05, 0.08, -0.40])
    for knee_forward in (False, True):
        solved = inverse_kinematics(leg, target, lateral_sign=1.0, knee_forward=knee_forward)
        reconstructed = forward_kinematics(leg, solved, lateral_sign=1.0)
        np.testing.assert_allclose(reconstructed, target, atol=1e-12)
    backward = inverse_kinematics(leg, target, lateral_sign=1.0, knee_forward=False)
    forward = inverse_kinematics(leg, target, lateral_sign=1.0, knee_forward=True)
    assert backward.knee_pitch < 0.0 < forward.knee_pitch


def test_nominal_posture_is_symmetric_between_the_sides() -> None:
    robot = default_robot()
    left = inverse_kinematics(
        robot.leg, robot.nominal_foot_in_hip(LegId.FRONT_LEFT), lateral_sign=1.0
    )
    right = inverse_kinematics(
        robot.leg, robot.nominal_foot_in_hip(LegId.FRONT_RIGHT), lateral_sign=-1.0
    )
    assert left.hip_roll == pytest.approx(0.0, abs=1e-12)
    assert right.hip_roll == pytest.approx(0.0, abs=1e-12)
    assert left.hip_pitch == pytest.approx(right.hip_pitch, abs=1e-12)
    assert left.knee_pitch == pytest.approx(right.knee_pitch, abs=1e-12)


def test_nominal_posture_matches_the_hand_computed_angles() -> None:
    """With equal 0.30 m links and a 0.42 m drop the knee closes by a known amount."""
    robot = default_robot()
    solved = inverse_kinematics(
        robot.leg, robot.nominal_foot_in_hip(LegId.FRONT_LEFT), lateral_sign=1.0
    )
    cosine = (0.42**2 - 0.30**2 - 0.30**2) / (2.0 * 0.30 * 0.30)
    assert solved.knee_pitch == pytest.approx(-math.acos(cosine), abs=1e-12)
    assert solved.hip_pitch == pytest.approx(math.acos(cosine) / 2.0, abs=1e-12)


def test_zero_angles_place_the_foot_straight_below_the_hip_pitch_axis() -> None:
    leg = LegGeometry(abduction_offset=0.08, thigh_length=0.30, shank_length=0.30)
    foot = forward_kinematics(
        leg, JointAngles(hip_roll=0.0, hip_pitch=0.0, knee_pitch=0.0), lateral_sign=1.0
    )
    np.testing.assert_allclose(foot, [0.0, 0.08, -0.60], atol=1e-15)


def test_target_beyond_the_maximum_reach_is_rejected() -> None:
    leg = default_robot().leg
    _, upper = reach_interval(leg)
    target = np.array([0.0, leg.abduction_offset, -(upper + 0.01)])
    assert not is_reachable(leg, target, lateral_sign=1.0)
    with pytest.raises(UnreachableTargetError, match="beyond the"):
        inverse_kinematics(leg, target, lateral_sign=1.0)


def test_target_inside_the_abduction_cylinder_is_rejected() -> None:
    leg = default_robot().leg
    target = np.array([0.2, 0.0, -0.01])
    assert not is_reachable(leg, target, lateral_sign=1.0)
    with pytest.raises(UnreachableTargetError, match="cylinder"):
        inverse_kinematics(leg, target, lateral_sign=1.0)


def test_target_inside_the_minimum_reach_is_rejected() -> None:
    leg = LegGeometry(abduction_offset=0.05, thigh_length=0.40, shank_length=0.20)
    lower, _ = reach_interval(leg)
    assert lower == pytest.approx(0.20)
    target = np.array([0.0, 0.05, -0.10])
    assert not is_reachable(leg, target, lateral_sign=1.0)
    with pytest.raises(UnreachableTargetError, match="inside the minimum reach"):
        inverse_kinematics(leg, target, lateral_sign=1.0)


def test_reach_boundary_is_accepted() -> None:
    leg = default_robot().leg
    _, upper = reach_interval(leg)
    target = np.array([0.0, leg.abduction_offset, -upper])
    assert is_reachable(leg, target, lateral_sign=1.0)
    solved = inverse_kinematics(leg, target, lateral_sign=1.0)
    np.testing.assert_allclose(
        forward_kinematics(leg, solved, lateral_sign=1.0), target, atol=1e-12
    )


def test_hip_roll_rotates_the_foot_about_the_trunk_x_axis() -> None:
    leg = default_robot().leg
    base = forward_kinematics(
        leg, JointAngles(hip_roll=0.0, hip_pitch=0.3, knee_pitch=-0.9), lateral_sign=1.0
    )
    rolled = forward_kinematics(
        leg, JointAngles(hip_roll=0.4, hip_pitch=0.3, knee_pitch=-0.9), lateral_sign=1.0
    )
    assert float(np.linalg.norm(base)) == pytest.approx(float(np.linalg.norm(rolled)), abs=1e-12)
    assert base[0] == pytest.approx(rolled[0], abs=1e-12)


def test_lateral_sign_mirrors_the_solution() -> None:
    leg = default_robot().leg
    angles = JointAngles(hip_roll=0.25, hip_pitch=0.4, knee_pitch=-1.1)
    left = forward_kinematics(leg, angles, lateral_sign=1.0)
    mirrored = JointAngles(hip_roll=-0.25, hip_pitch=0.4, knee_pitch=-1.1)
    right = forward_kinematics(leg, mirrored, lateral_sign=-1.0)
    np.testing.assert_allclose(left, [right[0], -right[1], right[2]], atol=1e-14)


def test_geometry_rejects_invalid_link_lengths() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LegGeometry(abduction_offset=0.05, thigh_length=0.0, shank_length=0.3)
    with pytest.raises(ValueError, match="must not be negative"):
        LegGeometry(abduction_offset=-0.05, thigh_length=0.3, shank_length=0.3)


def test_joint_angles_array_round_trip() -> None:
    angles = JointAngles(hip_roll=0.1, hip_pitch=-0.2, knee_pitch=-1.5)
    assert JointAngles.from_array(angles.as_array()) == angles
    with pytest.raises(ValueError, match="three joint angles"):
        JointAngles.from_array(np.zeros(4))


def test_hip_offsets_follow_the_leg_ordering() -> None:
    robot = default_robot()
    offsets = robot.hip_offsets()
    assert offsets.shape == (4, 3)
    np.testing.assert_allclose(offsets[int(LegId.FRONT_LEFT)], [0.25, 0.15, 0.0])
    np.testing.assert_allclose(offsets[int(LegId.FRONT_RIGHT)], [0.25, -0.15, 0.0])
    np.testing.assert_allclose(offsets[int(LegId.HIND_LEFT)], [-0.25, 0.15, 0.0])
    np.testing.assert_allclose(offsets[int(LegId.HIND_RIGHT)], [-0.25, -0.15, 0.0])
    assert RobotModel.lateral_sign(LegId.HIND_RIGHT) == -1.0
    assert RobotModel.longitudinal_sign(LegId.HIND_RIGHT) == -1.0


def test_robot_model_rejects_a_degenerate_trunk() -> None:
    leg = LegGeometry(abduction_offset=0.08, thigh_length=0.3, shank_length=0.3)
    with pytest.raises(ValueError, match="must be positive"):
        RobotModel(hip_half_length=0.0, hip_half_width=0.15, leg=leg, nominal_height=0.42)
    with pytest.raises(ValueError, match="nominal_height"):
        RobotModel(hip_half_length=0.25, hip_half_width=0.15, leg=leg, nominal_height=0.0)


def test_nominal_feet_sit_below_and_outboard_of_their_hips() -> None:
    robot = default_robot()
    feet = robot.nominal_feet_in_body()
    assert feet.shape == (4, 3)
    for leg_id in LegId:
        hip = robot.hip_offset(leg_id)
        foot = feet[int(leg_id)]
        assert foot[2] == pytest.approx(-robot.nominal_height)
        assert abs(foot[1]) > abs(hip[1])
        assert np.sign(foot[1]) == RobotModel.lateral_sign(leg_id)


def test_contact_state_derives_its_views_from_the_flags() -> None:
    state = ContactState(contacts=(True, False, True, True))
    assert state.stance_legs == (LegId.FRONT_LEFT, LegId.HIND_LEFT, LegId.HIND_RIGHT)
    assert state.swing_legs == (LegId.FRONT_RIGHT,)
    assert state.stance_count == 3
    np.testing.assert_array_equal(state.as_array(), [True, False, True, True])
    assert str(state) == "FL .. HL HR"


def test_contact_state_accepts_any_four_truthy_values() -> None:
    from_array = ContactState.from_iterable(np.array([1, 0, 0, 1]))
    assert from_array.contacts == (True, False, False, True)
    assert ContactState.from_iterable([0, 0, 0, 0]).stance_count == 0


def test_contact_state_rejects_the_wrong_number_of_flags() -> None:
    with pytest.raises(ValueError, match="contact flags"):
        ContactState(contacts=(True, False, True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contact flags"):
        ContactState.from_iterable([True, False, True])


def test_joint_angle_sampling_validates_its_limits() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="at least one"):
        sample_joint_angles(0, rng)
    with pytest.raises(ValueError, match="one interval per joint"):
        sample_joint_angles(4, rng, limits=((-1.0, 1.0), (-1.0, 1.0)))
    with pytest.raises(ValueError, match="positive width"):
        sample_joint_angles(4, rng, limits=((-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)))


def test_workspace_helpers_validate_their_array_shapes() -> None:
    leg = LegGeometry(abduction_offset=0.08, thigh_length=0.3, shank_length=0.3)
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        workspace_points(leg, np.zeros((5, 2)), lateral_sign=1.0)
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        round_trip_errors(leg, np.zeros((5, 2)), lateral_sign=1.0)


def test_round_trip_errors_report_nan_for_a_rejected_target() -> None:
    """An unreachable target contributes nan rather than a wrong number."""
    leg = LegGeometry(abduction_offset=0.08, thigh_length=0.3, shank_length=0.3)
    reachable = np.array([[0.0, 0.08, -0.42]], dtype=np.float64)
    unreachable = np.array([[0.0, 0.08, -2.0]], dtype=np.float64)
    points = np.vstack([reachable, unreachable])
    errors = round_trip_errors(leg, points, lateral_sign=1.0)
    assert errors[0] < ROUND_TRIP_TOLERANCE
    assert math.isnan(errors[1])
