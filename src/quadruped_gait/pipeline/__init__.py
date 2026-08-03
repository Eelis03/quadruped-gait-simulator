"""Pipeline layer: stepping a gait over time and recording a structured trace."""

from __future__ import annotations

from quadruped_gait.pipeline.presets import (
    DUTY_SWEEP_FACTORS,
    DUTY_SWEEP_FIGURE_FACTORS,
    REFERENCE_SWAY_AMPLITUDE,
    REFERENCE_VELOCITY,
    REFERENCE_WALK_DUTY_FACTOR,
    STATIC_STABILITY_THRESHOLD,
    reference_gaits,
    reference_walk,
    threshold_walk,
)
from quadruped_gait.pipeline.simulator import (
    BodyCommand,
    SimulationConfig,
    Trace,
    TraceSample,
    simulate,
)
from quadruped_gait.pipeline.sweep import DutySweepRow, duty_factor_sweep, run_gaits

__all__ = [
    "DUTY_SWEEP_FACTORS",
    "DUTY_SWEEP_FIGURE_FACTORS",
    "REFERENCE_SWAY_AMPLITUDE",
    "REFERENCE_VELOCITY",
    "REFERENCE_WALK_DUTY_FACTOR",
    "STATIC_STABILITY_THRESHOLD",
    "BodyCommand",
    "DutySweepRow",
    "SimulationConfig",
    "Trace",
    "TraceSample",
    "duty_factor_sweep",
    "reference_gaits",
    "reference_walk",
    "run_gaits",
    "simulate",
    "threshold_walk",
]
