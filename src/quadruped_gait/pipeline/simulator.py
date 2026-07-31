"""Kinematic gait simulator producing a structured trace.

The simulator is closed form in time. The trunk follows the commanded planar
twist exactly, footholds are placed at the neutral point under the hip at mid
stance (Raibert, 1986, chapter 2), a loaded foot is fixed in the world, and a
swinging foot follows the configured swing profile between the foothold it has
just left and the foothold it is about to reach. Joint angles come from the
closed form inverse kinematics of each leg.

Because every quantity is a function of absolute time only, the trace has no
accumulated integration error and any sample can be recomputed independently.
There is no rigid body dynamics and no contact force model here; see
``docs/design-notes.md`` for what that excludes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.algorithm.gait import GaitParameters, gait
from quadruped_gait.algorithm.swing import make_swing
from quadruped_gait.model.geometry import LEG_COUNT, LegId, RobotModel, default_robot
from quadruped_gait.model.kinematics import (
    UnreachableTargetError,
    inverse_kinematics,
)
from quadruped_gait.model.transforms import BodyPose, rotation_z

__all__ = ["BodyCommand", "SimulationConfig", "Trace", "TraceSample", "simulate"]

_YAW_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class BodyCommand:
    """The commanded trunk motion.

    Attributes:
        forward_velocity: Trunk velocity along its own x axis, in metres per second.
        lateral_velocity: Trunk velocity along its own y axis, in metres per second.
        yaw_rate: Trunk angular rate about the world z axis, in radians per second.
        height: Height of the trunk origin above the ground plane, in metres.
        sway_amplitude: Amplitude of the lateral trunk offset applied once per gait
            cycle, in metres. A lateral shift toward the supporting side is the
            standard way to keep the projected centre of mass inside the support
            triangle of a crawl gait (Song and Waldron, 1989, chapter 6).
        sway_phase: Phase of the lateral offset within the gait cycle, in radians.
    """

    forward_velocity: float = 0.3
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    height: float = 0.42
    sway_amplitude: float = 0.0
    sway_phase: float = 0.0

    def __post_init__(self) -> None:
        if self.height <= 0.0:
            raise ValueError("height must be positive")
        if self.sway_amplitude < 0.0:
            raise ValueError("sway_amplitude must not be negative")

    @property
    def heading_direction(self) -> NDArray[np.float64]:
        """Return the unit horizontal travel direction in trunk coordinates."""
        speed = math.hypot(self.forward_velocity, self.lateral_velocity)
        if speed < _YAW_TOLERANCE:
            return np.array([1.0, 0.0], dtype=np.float64)
        return np.array(
            [self.forward_velocity / speed, self.lateral_velocity / speed], dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Everything needed to reproduce one run.

    Attributes:
        robot: Trunk layout and leg geometry.
        gait: Gait parameters driving the contact schedule.
        command: Commanded trunk motion.
        cycles: Number of complete gait cycles to simulate.
        samples_per_cycle: Number of trace samples per gait cycle.
        swing_clearance: Apex height of the swing arc above the chord, in metres.
        swing_profile: Either ``"cycloidal"`` or ``"bezier"``.
        knee_forward: Inverse kinematics branch selection.
    """

    robot: RobotModel = field(default_factory=default_robot)
    gait: GaitParameters = field(default_factory=lambda: gait("walk"))
    command: BodyCommand = field(default_factory=BodyCommand)
    cycles: float = 3.0
    samples_per_cycle: int = 200
    swing_clearance: float = 0.08
    swing_profile: str = "cycloidal"
    knee_forward: bool = False

    def __post_init__(self) -> None:
        if self.cycles <= 0.0:
            raise ValueError("cycles must be positive")
        if self.samples_per_cycle < 2:
            raise ValueError("samples_per_cycle must be at least 2")
        if self.swing_clearance < 0.0:
            raise ValueError("swing_clearance must not be negative")

    @property
    def duration(self) -> float:
        """Total simulated time, in seconds."""
        return self.cycles * self.gait.period

    @property
    def sample_count(self) -> int:
        """Number of samples in the resulting trace."""
        return round(self.cycles * self.samples_per_cycle)

    @property
    def timestep(self) -> float:
        """Time between consecutive samples, in seconds."""
        return self.gait.period / self.samples_per_cycle


@dataclass(frozen=True, slots=True, eq=False)
class TraceSample:
    """One recorded instant of a simulation.

    Attributes:
        time: Simulated time, in seconds.
        pose: Trunk pose in the world frame.
        phases: Cycle position of each leg.
        contacts: Contact flag of each leg.
        foot_positions: Foot positions in world coordinates, shape ``(4, 3)``.
        joint_angles: Joint angles per leg, shape ``(4, 3)``, ``nan`` where the
            inverse kinematics had no solution.
        reachable: Whether the inverse kinematics succeeded for each leg.
        com_position: Centre of mass in world coordinates, shape ``(3,)``.
    """

    time: float
    pose: BodyPose
    phases: NDArray[np.float64]
    contacts: NDArray[np.bool_]
    foot_positions: NDArray[np.float64]
    joint_angles: NDArray[np.float64]
    reachable: NDArray[np.bool_]
    com_position: NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Trace:
    """A complete recorded run, stored column wise.

    Attributes:
        config: The configuration that produced this trace.
        times: Sample times, shape ``(n,)``.
        body_positions: Trunk origin in world coordinates, shape ``(n, 3)``.
        body_yaws: Trunk heading in the world frame, shape ``(n,)``.
        phases: Cycle position of every leg, shape ``(n, 4)``.
        contacts: Contact flags, shape ``(n, 4)``.
        foot_positions: Foot positions in world coordinates, shape ``(n, 4, 3)``.
        joint_angles: Joint angles, shape ``(n, 4, 3)``.
        reachable: Inverse kinematics success flags, shape ``(n, 4)``.
        com_positions: Centre of mass in world coordinates, shape ``(n, 3)``.
        travel_direction: Unit horizontal direction of travel in world coordinates
            at each sample, shape ``(n, 2)``.
    """

    config: SimulationConfig
    times: NDArray[np.float64]
    body_positions: NDArray[np.float64]
    body_yaws: NDArray[np.float64]
    phases: NDArray[np.float64]
    contacts: NDArray[np.bool_]
    foot_positions: NDArray[np.float64]
    joint_angles: NDArray[np.float64]
    reachable: NDArray[np.bool_]
    com_positions: NDArray[np.float64]
    travel_direction: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def sample_count(self) -> int:
        """Number of samples in this trace."""
        return int(self.times.size)

    def sample(self, index: int) -> TraceSample:
        """Return one sample as a structured record."""
        pose = BodyPose(
            position=self.body_positions[index],
            rotation=rotation_z(float(self.body_yaws[index])),
        )
        return TraceSample(
            time=float(self.times[index]),
            pose=pose,
            phases=self.phases[index],
            contacts=self.contacts[index],
            foot_positions=self.foot_positions[index],
            joint_angles=self.joint_angles[index],
            reachable=self.reachable[index],
            com_position=self.com_positions[index],
        )


def _body_pose_at(config: SimulationConfig, time: float) -> BodyPose:
    """Return the commanded trunk pose at ``time`` in closed form.

    The planar twist is constant in the trunk frame, so the trunk traces a
    straight line when the yaw rate is zero and a circular arc otherwise. The
    lateral sway offset is applied in the trunk frame after that motion.
    """
    command = config.command
    yaw = command.yaw_rate * time
    if abs(command.yaw_rate) < _YAW_TOLERANCE:
        x = command.forward_velocity * time
        y = command.lateral_velocity * time
    else:
        rate = command.yaw_rate
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        x = (command.forward_velocity * sin_yaw + command.lateral_velocity * (cos_yaw - 1.0)) / rate
        y = (command.forward_velocity * (1.0 - cos_yaw) + command.lateral_velocity * sin_yaw) / rate

    rotation = rotation_z(yaw)
    sway = command.sway_amplitude * math.sin(
        math.tau * time / config.gait.period + command.sway_phase
    )
    lateral = rotation @ np.array([0.0, sway, 0.0], dtype=np.float64)
    position = np.array([x, y, command.height], dtype=np.float64) + lateral
    return BodyPose(position=position, rotation=rotation)


def _neutral_foothold(config: SimulationConfig, leg_id: LegId, time: float) -> NDArray[np.float64]:
    """Return the ground point under the nominal foot position of ``leg_id`` at ``time``."""
    pose = _body_pose_at(config, time)
    world = pose.apply(config.robot.nominal_foot_in_body(leg_id))
    return np.array([world[0], world[1], 0.0], dtype=np.float64)


def _planned_foothold(
    config: SimulationConfig, leg_id: LegId, touchdown_time: float
) -> NDArray[np.float64]:
    """Return the foothold used for the stance that begins at ``touchdown_time``.

    The foothold is the neutral point under the hip at the middle of the coming
    stance phase. Placing it there makes the stance sweep symmetric about the hip,
    which is Raibert's symmetry condition for steady locomotion.
    """
    mid_stance = touchdown_time + 0.5 * config.gait.stance_duration
    return _neutral_foothold(config, leg_id, mid_stance)


def _foot_position(
    config: SimulationConfig, leg_id: LegId, time: float
) -> NDArray[np.float64]:
    """Return the world position of one foot at ``time``."""
    parameters = config.gait
    phase = parameters.leg_phase(time, leg_id)
    touchdown_time = parameters.last_touchdown_time(time, leg_id)
    if phase < parameters.duty_factor:
        return _planned_foothold(config, leg_id, touchdown_time)

    liftoff_time = touchdown_time + parameters.stance_duration
    lift_off = _planned_foothold(config, leg_id, touchdown_time)
    next_touchdown_time = liftoff_time + parameters.swing_duration
    touch_down = _planned_foothold(config, leg_id, next_touchdown_time)
    progress = (phase - parameters.duty_factor) / (1.0 - parameters.duty_factor)
    trajectory = make_swing(config.swing_profile, lift_off, touch_down, config.swing_clearance)
    return trajectory.position(progress)


def simulate(config: SimulationConfig) -> Trace:
    """Run the gait forward and return the recorded trace.

    Args:
        config: The run to perform.

    Returns:
        A :class:`Trace` with one entry per sample. Samples are spaced uniformly
        over ``config.cycles`` gait cycles starting at ``t = 0``.
    """
    count = config.sample_count
    times = np.arange(count, dtype=np.float64) * config.timestep

    body_positions = np.empty((count, 3), dtype=np.float64)
    body_yaws = np.empty(count, dtype=np.float64)
    phases = np.empty((count, LEG_COUNT), dtype=np.float64)
    contacts = np.empty((count, LEG_COUNT), dtype=np.bool_)
    foot_positions = np.empty((count, LEG_COUNT, 3), dtype=np.float64)
    joint_angles = np.full((count, LEG_COUNT, 3), np.nan, dtype=np.float64)
    reachable = np.zeros((count, LEG_COUNT), dtype=np.bool_)
    com_positions = np.empty((count, 3), dtype=np.float64)
    travel_direction = np.empty((count, 2), dtype=np.float64)

    heading_body = config.command.heading_direction

    for index in range(count):
        time = float(times[index])
        pose = _body_pose_at(config, time)
        body_positions[index] = pose.position
        body_yaws[index] = pose.rpy[2]
        phases[index] = config.gait.phases(time)
        contacts[index] = config.gait.contact_state(time).as_array()
        com_positions[index] = pose.apply(config.robot.com_in_body())
        travel_direction[index] = pose.rotation[:2, :2] @ heading_body

        for leg_id in LegId:
            world = _foot_position(config, leg_id, time)
            foot_positions[index, int(leg_id)] = world
            in_hip = pose.inverse_apply(world) - config.robot.hip_offset(leg_id)
            try:
                angles = inverse_kinematics(
                    config.robot.leg,
                    in_hip,
                    lateral_sign=RobotModel.lateral_sign(leg_id),
                    knee_forward=config.knee_forward,
                )
            except UnreachableTargetError:
                continue
            joint_angles[index, int(leg_id)] = angles.as_array()
            reachable[index, int(leg_id)] = True

    return Trace(
        config=config,
        times=times,
        body_positions=body_positions,
        body_yaws=body_yaws,
        phases=phases,
        contacts=contacts,
        foot_positions=foot_positions,
        joint_angles=joint_angles,
        reachable=reachable,
        com_positions=com_positions,
        travel_direction=travel_direction,
    )
