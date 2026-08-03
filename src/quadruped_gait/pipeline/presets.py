"""Named configurations shared by the example scripts and the regression tests.

Keeping the reference configurations in one place means the numbers quoted in
the README, the numbers the examples print, and the numbers the regression test
pins are produced by exactly the same settings.
"""

from __future__ import annotations

from dataclasses import replace

from quadruped_gait.algorithm.gait import GAIT_NAMES, gait
from quadruped_gait.model.geometry import default_robot
from quadruped_gait.pipeline.simulator import BodyCommand, SimulationConfig

__all__ = [
    "DUTY_SWEEP_FACTORS",
    "DUTY_SWEEP_FIGURE_FACTORS",
    "REFERENCE_SWAY_AMPLITUDE",
    "REFERENCE_VELOCITY",
    "REFERENCE_WALK_DUTY_FACTOR",
    "STATIC_STABILITY_THRESHOLD",
    "reference_gaits",
    "reference_walk",
    "threshold_walk",
]

# Commanded forward velocity used by every reference configuration, in metres per second.
REFERENCE_VELOCITY = 0.30
# Duty factor of the reference walk. It is above the three quarter threshold of
# McGhee and Frank (1968), so the gait has an interval of four foot support.
REFERENCE_WALK_DUTY_FACTOR = 0.80
# Lateral trunk offset amplitude of the reference walk, in metres.
REFERENCE_SWAY_AMPLITUDE = 0.06
# Duty factors used by the sweep that locates the static stability threshold.
DUTY_SWEEP_FACTORS: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
# A finer grid over the same interval, used for the published figure. The step is
# chosen so that the three quarter threshold of McGhee and Frank (1968) is a grid
# point rather than something the eye has to interpolate.
DUTY_SWEEP_FIGURE_FACTORS: tuple[float, ...] = tuple(
    round(0.60 + 0.0125 * step, 4) for step in range(25)
)
# The duty factor below which a quadruped creeping gait cannot be statically stable,
# derived analytically by McGhee and Frank (1968). The sweep in the examples
# reproduces it by measurement rather than assuming it.
STATIC_STABILITY_THRESHOLD = 0.75


def reference_walk(cycles: float = 3.0, samples_per_cycle: int = 200) -> SimulationConfig:
    """Return the statically stable reference walk.

    The gait is a lateral sequence walk with a duty factor of 0.80 and a lateral
    trunk offset of 0.06 m applied once per cycle, driven at 0.30 m/s.
    """
    return SimulationConfig(
        robot=default_robot(),
        gait=gait("walk", duty_factor=REFERENCE_WALK_DUTY_FACTOR),
        command=BodyCommand(
            forward_velocity=REFERENCE_VELOCITY,
            sway_amplitude=REFERENCE_SWAY_AMPLITUDE,
        ),
        cycles=cycles,
        samples_per_cycle=samples_per_cycle,
        swing_clearance=0.08,
        swing_profile="cycloidal",
    )


def threshold_walk(cycles: float = 3.0, samples_per_cycle: int = 200) -> SimulationConfig:
    """Return the reference walk retimed to sit exactly on the stability threshold.

    Everything except the duty factor matches :func:`reference_walk`, including the
    lateral trunk offset. The gait keeps three feet loaded for the whole cycle, and
    yet its minimum static margin is zero, because at each support transition the
    outgoing and the incoming support triangle share the edge that the projected
    centre of mass is crossing.
    """
    base = reference_walk(cycles, samples_per_cycle)
    return replace(base, gait=base.gait.with_duty_factor(STATIC_STABILITY_THRESHOLD))


def reference_gaits(
    cycles: float = 3.0, samples_per_cycle: int = 200
) -> tuple[SimulationConfig, ...]:
    """Return one configuration per library gait with a common trunk command.

    No lateral trunk offset is applied, so the four gaits are compared purely on
    their contact schedules.
    """
    return tuple(
        SimulationConfig(
            robot=default_robot(),
            gait=gait(name),
            command=BodyCommand(forward_velocity=REFERENCE_VELOCITY),
            cycles=cycles,
            samples_per_cycle=samples_per_cycle,
            swing_clearance=0.08,
            swing_profile="cycloidal",
        )
        for name in GAIT_NAMES
    )
