"""Analysis layer: turning traces into margins, contact statistics, and figures."""

from __future__ import annotations

from quadruped_gait.analysis.figures import (
    foot_trajectory_figure,
    gait_diagram_figure,
    save_figure,
    stability_figure,
    support_polygon_figure,
)
from quadruped_gait.analysis.metrics import (
    ContactInterval,
    ContactSummary,
    GaitReport,
    RoundTripSummary,
    StabilitySummary,
    contact_intervals,
    contact_summary,
    round_trip_summary,
    stability_series,
    stability_summary,
    summarise,
    support_polygon_at,
    sweep_stability,
)
from quadruped_gait.analysis.report import (
    format_contact_summary,
    format_duty_sweep,
    format_gait_diagram,
    format_report,
    format_round_trip,
    format_summary_table,
)

__all__ = [
    "ContactInterval",
    "ContactSummary",
    "GaitReport",
    "RoundTripSummary",
    "StabilitySummary",
    "contact_intervals",
    "contact_summary",
    "foot_trajectory_figure",
    "format_contact_summary",
    "format_duty_sweep",
    "format_gait_diagram",
    "format_report",
    "format_round_trip",
    "format_summary_table",
    "gait_diagram_figure",
    "round_trip_summary",
    "save_figure",
    "stability_figure",
    "stability_series",
    "stability_summary",
    "summarise",
    "support_polygon_at",
    "support_polygon_figure",
    "sweep_stability",
]
