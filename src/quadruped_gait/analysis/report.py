"""Plain text rendering of analysis results for the console and the README."""

from __future__ import annotations

from collections.abc import Sequence

from quadruped_gait.analysis.metrics import (
    ContactInterval,
    GaitReport,
    RoundTripSummary,
    StabilitySummary,
)
from quadruped_gait.model.geometry import LEG_NAMES

__all__ = [
    "format_contact_summary",
    "format_duty_sweep",
    "format_gait_diagram",
    "format_report",
    "format_round_trip",
    "format_summary_table",
]

_NOT_AVAILABLE = "n/a"


def _number(value: float, digits: int = 4) -> str:
    """Render a float, or ``n/a`` when it is not finite."""
    if value != value:
        return _NOT_AVAILABLE
    return f"{value:.{digits}f}"


def format_summary_table(reports: Sequence[GaitReport]) -> str:
    """Render one row per gait with duty factor and stability statistics."""
    header = (
        f"{'gait':<7}{'beta':>7}{'beta_real':>11}{'err':>9}"
        f"{'support':>9}{'ssm_min':>10}{'ssm_mean':>10}{'lsm_min':>10}{'feet':>7}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        contact = report.contact
        stability = report.stability
        realised = sum(contact.exact_duty_factors) / len(contact.exact_duty_factors)
        lines.append(
            f"{report.gait_name:<7}"
            f"{contact.commanded_duty_factor:>7.3f}"
            f"{realised:>11.4f}"
            f"{contact.max_absolute_error:>9.4f}"
            f"{stability.supported_fraction:>9.3f}"
            f"{_number(stability.minimum_static):>10}"
            f"{_number(stability.mean_static):>10}"
            f"{_number(stability.minimum_longitudinal):>10}"
            f"{contact.mean_stance_count:>7.2f}"
        )
    return "\n".join(lines)


def _per_leg(values: Sequence[float]) -> str:
    """Render one value per leg, labelled with the leg name."""
    return "  ".join(f"{name}={value:.4f}" for name, value in zip(LEG_NAMES, values, strict=True))


def format_contact_summary(report: GaitReport) -> str:
    """Render the per leg duty factors and the support pattern histogram.

    The realised rows are the closed form values. The sampled rows below them are
    the same quantities counted from the recorded trace, and the gap between the
    two is the discretisation error of the sampling rate.
    """
    contact = report.contact
    histogram = "  ".join(
        f"{count}feet={value}" for count, value in enumerate(contact.stance_count_histogram)
    )
    low, high = contact.stance_count_range
    return (
        f"gait               {report.gait_name}\n"
        f"period             {report.period:.3f} s\n"
        f"forward velocity   {report.forward_velocity:.3f} m/s\n"
        f"stride length      {report.stride_length:.3f} m\n"
        f"commanded duty     {contact.commanded_duty_factor:.4f}\n"
        f"realised duty      {_per_leg(contact.exact_duty_factors)}\n"
        f"duty error         {contact.max_absolute_error:.2e}\n"
        f"sampled duty       {_per_leg(contact.sampled_duty_factors)}\n"
        f"sampling error     {contact.sampling_error:.2e}\n"
        f"support histogram  {histogram}\n"
        f"feet down range    {low} to {high}\n"
        f"mean stance feet   {contact.mean_stance_count:.4f}\n"
        f"sampled stance     {contact.sampled_mean_stance_count:.4f}\n"
        f"unreachable        {report.unreachable_samples}"
    )


def format_report(report: GaitReport) -> str:
    """Render one gait report as a labelled block."""
    stability = report.stability
    return (
        f"{format_contact_summary(report)}\n"
        f"samples            {stability.sample_count}\n"
        f"supported fraction {stability.supported_fraction:.4f}\n"
        f"static margin      min {_number(stability.minimum_static)} m, "
        f"mean {_number(stability.mean_static)} m\n"
        f"longitudinal       min {_number(stability.minimum_longitudinal)} m, "
        f"mean {_number(stability.mean_longitudinal)} m\n"
        f"support area       mean {_number(stability.mean_support_area)} m^2\n"
        f"statically stable  {stability.statically_stable}"
    )


def format_duty_sweep(entries: Sequence[tuple[float, StabilitySummary]]) -> str:
    """Render a duty factor sweep, one row per duty factor."""
    header = (
        f"{'beta':>7}{'support':>10}{'ssm_min':>11}{'ssm_mean':>11}"
        f"{'lsm_min':>11}{'area_mean':>12}{'stable':>9}"
    )
    lines = [header, "-" * len(header)]
    for duty_factor, stability in entries:
        lines.append(
            f"{duty_factor:>7.3f}"
            f"{stability.supported_fraction:>10.3f}"
            f"{_number(stability.minimum_static, 5):>11}"
            f"{_number(stability.mean_static, 5):>11}"
            f"{_number(stability.minimum_longitudinal, 5):>11}"
            f"{_number(stability.mean_support_area, 5):>12}"
            f"{stability.statically_stable!s:>9}"
        )
    return "\n".join(lines)


def format_round_trip(summary: RoundTripSummary, label: str) -> str:
    """Render the accuracy of forward kinematics composed with the inverse."""
    return (
        f"{label}\n"
        f"  targets tested   {summary.sample_count}\n"
        f"  targets solved   {summary.solved_count}\n"
        f"  max error        {summary.maximum_error:.3e} m\n"
        f"  mean error       {summary.mean_error:.3e} m\n"
        f"  median error     {summary.median_error:.3e} m"
    )


def format_gait_diagram(
    intervals: Sequence[ContactInterval], period: float, columns: int = 60
) -> str:
    """Render the contact schedule of one cycle as fixed width text.

    Each row is one leg, a filled cell marks stance, and the row spans exactly one
    gait cycle. Intervals are folded back into a single cycle before rendering.
    """
    if period <= 0.0:
        raise ValueError("period must be positive")
    if columns < 1:
        raise ValueError("columns must be at least one")
    grid = [[False] * columns for _ in LEG_NAMES]
    for interval in intervals:
        span = (interval.end_time - interval.start_time) / period * columns
        steps = max(1, round(span))
        first = round(interval.start_time / period * columns)
        for offset in range(steps):
            grid[interval.leg_index][(first + offset) % columns] = True
    lines = [f"cycle length {period:.3f} s, one column is {period / columns:.4f} s"]
    for name, row in zip(LEG_NAMES, grid, strict=True):
        lines.append(f"{name} |" + "".join("#" if cell else "." for cell in row) + "|")
    lines.append("   " + " " * 1 + "0" + " " * (columns - 2) + "T")
    return "\n".join(lines)
