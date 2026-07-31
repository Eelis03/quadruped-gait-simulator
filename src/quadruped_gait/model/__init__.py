"""Model layer: pure geometry, kinematics, and contact data with no input or output."""

from __future__ import annotations

from quadruped_gait.model.contact import ContactState
from quadruped_gait.model.geometry import (
    LEG_COUNT,
    LEG_NAMES,
    LegGeometry,
    LegId,
    RobotModel,
    default_robot,
)
from quadruped_gait.model.kinematics import (
    JointAngles,
    UnreachableTargetError,
    foot_below_hip_axis,
    forward_kinematics,
    inverse_kinematics,
    is_reachable,
    reach_interval,
)
from quadruped_gait.model.transforms import (
    BodyPose,
    rotation_rpy,
    rotation_x,
    rotation_y,
    rotation_z,
    rpy_from_rotation,
)
from quadruped_gait.model.workspace import (
    DEFAULT_JOINT_LIMITS,
    round_trip_errors,
    sample_joint_angles,
    workspace_points,
)

__all__ = [
    "DEFAULT_JOINT_LIMITS",
    "LEG_COUNT",
    "LEG_NAMES",
    "BodyPose",
    "ContactState",
    "JointAngles",
    "LegGeometry",
    "LegId",
    "RobotModel",
    "UnreachableTargetError",
    "default_robot",
    "foot_below_hip_axis",
    "forward_kinematics",
    "inverse_kinematics",
    "is_reachable",
    "reach_interval",
    "rotation_rpy",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "round_trip_errors",
    "rpy_from_rotation",
    "sample_joint_angles",
    "workspace_points",
]
