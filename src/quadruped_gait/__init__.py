"""Quadruped gait generation with support polygon and stability margin analysis.

The package is layered. :mod:`quadruped_gait.model` holds the geometry,
kinematics, and contact data with no input or output.
:mod:`quadruped_gait.algorithm` holds the gait scheduler, the swing trajectory
generators, and the stability analysis. :mod:`quadruped_gait.pipeline` steps a
gait over time and records a trace. :mod:`quadruped_gait.analysis` reduces a
trace to numbers and figures.
"""

from __future__ import annotations

from quadruped_gait.algorithm import (
    GAIT_LIBRARY,
    GAIT_NAMES,
    BezierSwing,
    CycloidalSwing,
    GaitParameters,
    StabilityMargins,
    SupportPolygon,
    contact_schedule,
    exact_duty_factors,
    exact_supported_fraction,
    gait,
    sampled_duty_factors,
    stability_margins,
    stance_count_durations,
    stance_count_extrema,
    support_polygon,
)
from quadruped_gait.analysis import (
    GaitReport,
    format_gait_diagram,
    format_report,
    format_summary_table,
    summarise,
)
from quadruped_gait.model import (
    BodyPose,
    ContactState,
    JointAngles,
    LegGeometry,
    LegId,
    RobotModel,
    UnreachableTargetError,
    default_robot,
    forward_kinematics,
    inverse_kinematics,
    is_reachable,
)
from quadruped_gait.pipeline import BodyCommand, SimulationConfig, Trace, simulate

__all__ = [
    "GAIT_LIBRARY",
    "GAIT_NAMES",
    "BezierSwing",
    "BodyCommand",
    "BodyPose",
    "ContactState",
    "CycloidalSwing",
    "GaitParameters",
    "GaitReport",
    "JointAngles",
    "LegGeometry",
    "LegId",
    "RobotModel",
    "SimulationConfig",
    "StabilityMargins",
    "SupportPolygon",
    "Trace",
    "UnreachableTargetError",
    "__version__",
    "contact_schedule",
    "default_robot",
    "exact_duty_factors",
    "exact_supported_fraction",
    "format_gait_diagram",
    "format_report",
    "format_summary_table",
    "forward_kinematics",
    "gait",
    "inverse_kinematics",
    "is_reachable",
    "sampled_duty_factors",
    "simulate",
    "stability_margins",
    "stance_count_durations",
    "stance_count_extrema",
    "summarise",
    "support_polygon",
]

__version__ = "0.1.0"
