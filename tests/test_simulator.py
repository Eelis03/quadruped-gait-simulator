"""Invariant tests for the simulator pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from quadruped_gait.algorithm.gait import GAIT_NAMES, gait
from quadruped_gait.model.geometry import LEG_COUNT, LegId, default_robot
from quadruped_gait.model.kinematics import JointAngles, forward_kinematics
from quadruped_gait.model.transforms import rotation_z
from quadruped_gait.pipeline import (
    DUTY_SWEEP_FACTORS,
    BodyCommand,
    SimulationConfig,
    duty_factor_sweep,
    reference_gaits,
    reference_walk,
    run_gaits,
    simulate,
)

CYCLES = 2.0
# An odd sample count keeps every sample time away from the exact phase boundaries of
# the library gaits, so no test depends on how a boundary sample is rounded.
SAMPLES = 121


def _config(name: str, **command: float) -> SimulationConfig:
    return SimulationConfig(
        robot=default_robot(),
        gait=gait(name),
        command=BodyCommand(**command),
        cycles=CYCLES,
        samples_per_cycle=SAMPLES,
    )


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_trace_shapes_are_consistent(name: str) -> None:
    trace = simulate(_config(name, forward_velocity=0.3))
    count = int(CYCLES * SAMPLES)
    assert len(trace) == count
    assert trace.times.shape == (count,)
    assert trace.body_positions.shape == (count, 3)
    assert trace.phases.shape == (count, LEG_COUNT)
    assert trace.contacts.shape == (count, LEG_COUNT)
    assert trace.foot_positions.shape == (count, LEG_COUNT, 3)
    assert trace.joint_angles.shape == (count, LEG_COUNT, 3)
    assert trace.com_positions.shape == (count, 3)
    assert trace.travel_direction.shape == (count, 2)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_every_commanded_foot_position_is_reachable(name: str) -> None:
    trace = simulate(_config(name, forward_velocity=0.3))
    assert bool(trace.reachable.all())
    assert bool(np.isfinite(trace.joint_angles).all())


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_loaded_feet_rest_on_the_ground_plane(name: str) -> None:
    trace = simulate(_config(name, forward_velocity=0.3))
    loaded = trace.foot_positions[:, :, 2][trace.contacts]
    np.testing.assert_allclose(loaded, 0.0, atol=1e-12)


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_loaded_feet_do_not_slip(name: str) -> None:
    """A foot that is loaded at two consecutive samples must not have moved."""
    trace = simulate(_config(name, forward_velocity=0.4, lateral_velocity=0.1))
    held = trace.contacts[:-1] & trace.contacts[1:]
    displacement = np.linalg.norm(trace.foot_positions[1:] - trace.foot_positions[:-1], axis=2)
    assert float(np.max(displacement[held], initial=0.0)) < 1e-12


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_swinging_feet_stay_above_the_ground(name: str) -> None:
    trace = simulate(_config(name, forward_velocity=0.3))
    swinging = trace.foot_positions[:, :, 2][~trace.contacts]
    assert float(np.min(swinging, initial=0.0)) >= -1e-12


@pytest.mark.parametrize("name", GAIT_NAMES)
def test_feet_advance_one_stride_per_cycle(name: str) -> None:
    """Steady locomotion moves every foothold forward by exactly one stride length."""
    parameters = gait(name)
    trace = simulate(_config(name, forward_velocity=0.3))
    stride = 0.3 * parameters.period
    shift = SAMPLES
    delta = trace.foot_positions[shift:, :, 0] - trace.foot_positions[:-shift, :, 0]
    np.testing.assert_allclose(delta, stride, atol=1e-9)


def test_trunk_advances_at_the_commanded_velocity() -> None:
    trace = simulate(_config("walk", forward_velocity=0.35))
    np.testing.assert_allclose(
        trace.body_positions[:, 0], 0.35 * trace.times, atol=1e-12
    )
    np.testing.assert_allclose(trace.body_positions[:, 2], 0.42, atol=1e-12)


def test_lateral_sway_offsets_the_trunk_without_moving_it_forward() -> None:
    plain = simulate(_config("walk", forward_velocity=0.3))
    swayed = simulate(_config("walk", forward_velocity=0.3, sway_amplitude=0.06))
    np.testing.assert_allclose(swayed.body_positions[:, 0], plain.body_positions[:, 0], atol=1e-12)
    assert float(np.max(np.abs(swayed.body_positions[:, 1]))) == pytest.approx(0.06, abs=1e-3)
    assert float(np.max(np.abs(plain.body_positions[:, 1]))) == pytest.approx(0.0, abs=1e-15)


def test_yaw_rate_turns_the_trunk_along_a_circular_arc() -> None:
    config = SimulationConfig(
        robot=default_robot(),
        gait=gait("walk"),
        command=BodyCommand(forward_velocity=0.3, yaw_rate=0.5),
        cycles=1.0,
        samples_per_cycle=SAMPLES,
    )
    trace = simulate(config)
    np.testing.assert_allclose(trace.body_yaws, 0.5 * trace.times, atol=1e-12)
    radius = 0.3 / 0.5
    centre = np.array([0.0, radius])
    distance = np.linalg.norm(trace.body_positions[:, :2] - centre, axis=1)
    np.testing.assert_allclose(distance, radius, atol=1e-12)


def test_travel_direction_is_a_unit_vector() -> None:
    trace = simulate(_config("trot", forward_velocity=0.3, lateral_velocity=0.15))
    np.testing.assert_allclose(np.linalg.norm(trace.travel_direction, axis=1), 1.0, atol=1e-12)


def test_zero_velocity_gives_a_forward_travel_direction() -> None:
    trace = simulate(_config("walk", forward_velocity=0.0))
    np.testing.assert_allclose(trace.travel_direction, [[1.0, 0.0]] * len(trace), atol=1e-12)


def test_forward_kinematics_reproduces_the_commanded_feet() -> None:
    """The recorded joint angles place the feet exactly where the planner asked."""
    robot = default_robot()
    trace = simulate(_config("walk", forward_velocity=0.3, sway_amplitude=0.05))
    for index in range(0, len(trace), 17):
        rotation = rotation_z(float(trace.body_yaws[index]))
        for leg_id in LegId:
            angles = JointAngles.from_array(trace.joint_angles[index, int(leg_id)])
            in_hip = forward_kinematics(
                robot.leg, angles, lateral_sign=robot.lateral_sign(leg_id)
            )
            world = rotation @ (in_hip + robot.hip_offset(leg_id)) + trace.body_positions[index]
            np.testing.assert_allclose(
                world, trace.foot_positions[index, int(leg_id)], atol=1e-9
            )


@pytest.mark.parametrize("profile", ["cycloidal", "bezier"])
def test_both_swing_profiles_produce_the_same_footholds(profile: str) -> None:
    config = SimulationConfig(
        robot=default_robot(),
        gait=gait("trot"),
        command=BodyCommand(forward_velocity=0.3),
        cycles=CYCLES,
        samples_per_cycle=SAMPLES,
        swing_profile=profile,
    )
    trace = simulate(config)
    reference = simulate(_config("trot", forward_velocity=0.3))
    loaded = trace.contacts
    np.testing.assert_allclose(
        trace.foot_positions[loaded], reference.foot_positions[loaded], atol=1e-12
    )


def test_trace_sample_matches_the_columns() -> None:
    trace = simulate(_config("walk", forward_velocity=0.3))
    sample = trace.sample(31)
    assert sample.time == pytest.approx(float(trace.times[31]))
    np.testing.assert_allclose(sample.pose.position, trace.body_positions[31], atol=0.0)
    np.testing.assert_allclose(sample.foot_positions, trace.foot_positions[31], atol=0.0)
    np.testing.assert_array_equal(sample.contacts, trace.contacts[31])


def test_reference_presets_are_consistent() -> None:
    walk = reference_walk(cycles=1.0, samples_per_cycle=50)
    assert walk.gait.duty_factor == pytest.approx(0.80)
    assert walk.command.sway_amplitude == pytest.approx(0.06)
    assert walk.sample_count == 50
    configs = reference_gaits(cycles=1.0, samples_per_cycle=50)
    assert tuple(config.gait.name for config in configs) == GAIT_NAMES
    assert len(run_gaits(configs)) == len(GAIT_NAMES)


def test_duty_factor_sweep_covers_every_requested_value() -> None:
    base = reference_walk(cycles=1.0, samples_per_cycle=40)
    rows = duty_factor_sweep(base, DUTY_SWEEP_FACTORS)
    assert tuple(row.duty_factor for row in rows) == DUTY_SWEEP_FACTORS
    for row in rows:
        assert row.trace.config.gait.duty_factor == pytest.approx(row.duty_factor)
    with pytest.raises(ValueError, match="must not be empty"):
        duty_factor_sweep(base, [])


def test_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="cycles"):
        SimulationConfig(cycles=0.0)
    with pytest.raises(ValueError, match="samples_per_cycle"):
        SimulationConfig(samples_per_cycle=1)
    with pytest.raises(ValueError, match="swing_clearance"):
        SimulationConfig(swing_clearance=-0.1)
    with pytest.raises(ValueError, match="height"):
        BodyCommand(height=0.0)
    with pytest.raises(ValueError, match="sway_amplitude"):
        BodyCommand(sway_amplitude=-0.01)
