"""Algorithm layer: gait scheduling, swing trajectories, and stability analysis."""

from __future__ import annotations

from quadruped_gait.algorithm.gait import (
    GAIT_LIBRARY,
    GAIT_NAMES,
    GaitParameters,
    contact_schedule,
    gait,
    leg_phases,
    realised_duty_factors,
)
from quadruped_gait.algorithm.stability import (
    StabilityMargins,
    SupportPolygon,
    convex_hull_2d,
    distance_to_boundary,
    longitudinal_stability_margin,
    point_in_convex_polygon,
    stability_margins,
    static_stability_margin,
    support_polygon,
)
from quadruped_gait.algorithm.swing import (
    BezierSwing,
    CycloidalSwing,
    SwingTrajectory,
    bezier_point,
    make_swing,
    stance_foot_in_body,
)

__all__ = [
    "GAIT_LIBRARY",
    "GAIT_NAMES",
    "BezierSwing",
    "CycloidalSwing",
    "GaitParameters",
    "StabilityMargins",
    "SupportPolygon",
    "SwingTrajectory",
    "bezier_point",
    "contact_schedule",
    "convex_hull_2d",
    "distance_to_boundary",
    "gait",
    "leg_phases",
    "longitudinal_stability_margin",
    "make_swing",
    "point_in_convex_polygon",
    "realised_duty_factors",
    "stability_margins",
    "stance_foot_in_body",
    "static_stability_margin",
    "support_polygon",
]
