"""Verify the closed form leg inverse kinematics against forward kinematics.

Samples joint angles inside the declared limits, maps them to foot positions,
solves the inverse kinematics, and reports the reconstruction error. Also shows
the nominal standing posture and the rejection of an out of reach target.

Usage:
    uv run python examples/leg_kinematics.py [--samples N] [--seed S]
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np

from quadruped_gait.analysis import format_round_trip, round_trip_summary
from quadruped_gait.model import (
    DEFAULT_JOINT_LIMITS,
    LegId,
    RobotModel,
    UnreachableTargetError,
    default_robot,
    inverse_kinematics,
    reach_interval,
    round_trip_errors,
    sample_joint_angles,
    workspace_points,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for this example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000, help="number of random targets")
    parser.add_argument("--seed", type=int, default=20260731, help="random seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    robot = default_robot()
    lower, upper = reach_interval(robot.leg)

    print("leg geometry")
    print(f"  abduction offset {robot.leg.abduction_offset:.3f} m")
    print(f"  thigh length     {robot.leg.thigh_length:.3f} m")
    print(f"  shank length     {robot.leg.shank_length:.3f} m")
    print(f"  reach interval   [{lower:.3f}, {upper:.3f}] m")
    print(f"  joint limits     {DEFAULT_JOINT_LIMITS}")
    print()

    rng = np.random.default_rng(arguments.seed)
    for leg_id in (LegId.FRONT_LEFT, LegId.FRONT_RIGHT):
        sign = RobotModel.lateral_sign(leg_id)
        angles = sample_joint_angles(arguments.samples, rng)
        points = workspace_points(robot.leg, angles, lateral_sign=sign)
        errors = round_trip_errors(robot.leg, points, lateral_sign=sign)
        print(format_round_trip(round_trip_summary(errors), f"round trip, leg {leg_id.name}"))
        print()

    nominal = robot.nominal_foot_in_hip(LegId.FRONT_LEFT)
    solved = inverse_kinematics(robot.leg, nominal, lateral_sign=1.0)
    print("nominal standing posture, leg FRONT_LEFT")
    print(f"  foot in hip frame {np.array2string(nominal, precision=4)} m")
    print(f"  hip roll          {math.degrees(solved.hip_roll):+.4f} deg")
    print(f"  hip pitch         {math.degrees(solved.hip_pitch):+.4f} deg")
    print(f"  knee pitch        {math.degrees(solved.knee_pitch):+.4f} deg")
    print()

    far = np.array([0.0, robot.leg.abduction_offset, -(upper + 0.05)], dtype=np.float64)
    try:
        inverse_kinematics(robot.leg, far, lateral_sign=1.0)
    except UnreachableTargetError as error:
        print(f"unreachable target rejected: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
