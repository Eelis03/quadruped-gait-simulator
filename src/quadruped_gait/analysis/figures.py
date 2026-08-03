"""Figure construction for traces.

The figures are built on :class:`matplotlib.figure.Figure` directly rather than
through ``pyplot``, so no interactive backend is selected and nothing is drawn to
a screen. Each function returns a figure that the caller saves.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from quadruped_gait.algorithm.gait import GaitParameters, stance_count_extrema
from quadruped_gait.algorithm.stability import stability_margins
from quadruped_gait.analysis.metrics import (
    StabilitySummary,
    contact_intervals,
    stability_series,
    support_polygon_at,
)
from quadruped_gait.model.geometry import LEG_COUNT, LEG_NAMES, LegId
from quadruped_gait.pipeline.simulator import Trace

__all__ = [
    "duty_sweep_figure",
    "foot_trajectory_figure",
    "gait_diagram_figure",
    "gait_diagram_grid_figure",
    "save_figure",
    "stability_figure",
    "support_polygon_figure",
]

# One hue per leg, assigned in the canonical leg order and never cycled. The four
# steps clear a colour vision deficiency separation of 8.4 in OKLab across every
# pair, so the legs stay distinguishable under protanopia and deuteranopia. Every
# figure that uses them also labels the legs, so colour is never the only encoding.
_LEG_COLOURS: tuple[str, ...] = ("#0173b2", "#de8f05", "#029e73", "#cc78bc")
_PRIMARY = _LEG_COLOURS[0]
_SECONDARY = _LEG_COLOURS[1]
_INK = "#3c3c3c"
_GRID_ALPHA = 0.25


def save_figure(figure: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Write ``figure`` to ``path``, creating the parent directory if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    return target


def gait_diagram_figure(trace: Trace, title: str | None = None) -> Figure:
    """Draw the contact schedule as one horizontal bar per leg."""
    figure = Figure(figsize=(8.0, 3.0))
    axes = figure.add_subplot(111)
    for interval in contact_intervals(trace):
        axes.barh(
            y=LEG_COUNT - 1 - interval.leg_index,
            width=interval.duration,
            left=interval.start_time,
            height=0.6,
            color=_LEG_COLOURS[interval.leg_index],
            edgecolor="none",
        )
    axes.set_yticks(list(range(LEG_COUNT)))
    axes.set_yticklabels(list(reversed(LEG_NAMES)))
    axes.set_xlabel("time (s)")
    axes.set_xlim(float(trace.times[0]), float(trace.times[-1]) + trace.config.timestep)
    axes.set_title(title or f"{trace.config.gait.name} contact schedule")
    axes.grid(axis="x", alpha=_GRID_ALPHA)
    figure.tight_layout()
    return figure


def stability_figure(trace: Trace, title: str | None = None) -> Figure:
    """Draw the static and longitudinal stability margins against time."""
    series = stability_series(trace)
    figure = Figure(figsize=(8.0, 4.5))
    upper = figure.add_subplot(211)
    upper.plot(trace.times, series[:, 0], label="static margin", color=_PRIMARY, linewidth=1.8)
    upper.plot(
        trace.times, series[:, 1], label="longitudinal margin", color=_SECONDARY, linewidth=1.8
    )
    upper.axhline(0.0, color=_INK, linewidth=0.8)
    upper.set_ylabel("margin (m)")
    upper.legend(loc="upper right", fontsize=8, frameon=False)
    upper.grid(alpha=_GRID_ALPHA)
    upper.set_title(title or f"{trace.config.gait.name} quasi-static stability")

    lower = figure.add_subplot(212, sharex=upper)
    lower.step(trace.times, series[:, 3], where="post", color=_LEG_COLOURS[2], linewidth=1.8)
    lower.set_ylabel("feet in stance")
    lower.set_xlabel("time (s)")
    lower.set_ylim(-0.2, LEG_COUNT + 0.2)
    lower.grid(alpha=_GRID_ALPHA)
    figure.tight_layout()
    return figure


def duty_sweep_figure(
    entries: Sequence[tuple[float, StabilitySummary]],
    threshold: float = 0.75,
    title: str | None = None,
) -> Figure:
    """Draw the minimum static margin and the supported fraction against duty factor.

    The upper panel is the measured minimum static stability margin of a walk at
    each duty factor. The lower panel is the fraction of the cycle that has a
    support polygon at all, which is what makes the region below the threshold
    uncertifiable even where its margin is positive. Both are needed: a reader
    shown only the margin would conclude that a duty factor of 0.60 is safer than
    0.75, when in fact it spends part of the cycle on two feet.

    Raises:
        ValueError: If ``entries`` is empty.
    """
    if not entries:
        raise ValueError("entries must contain at least one duty factor")
    duty = np.array([value for value, _ in entries], dtype=np.float64)
    margin = np.array([summary.minimum_static for _, summary in entries], dtype=np.float64)
    supported = np.array([summary.supported_fraction for _, summary in entries], dtype=np.float64)
    stable = np.array([summary.statically_stable for _, summary in entries], dtype=np.bool_)

    figure = Figure(figsize=(7.2, 5.0))
    upper = figure.add_subplot(211)
    upper.plot(duty, margin, color=_PRIMARY, linewidth=1.8, zorder=2)
    upper.axhline(0.0, color=_INK, linewidth=0.9, zorder=1)
    upper.axvline(threshold, color=_INK, linewidth=0.9, linestyle="--", zorder=1)
    # Status is carried by marker fill and by the legend text, never by hue alone.
    upper.plot(
        duty[stable], margin[stable], linestyle="none", marker="o", markersize=6,
        color=_PRIMARY, label="every sample supported with a positive margin", zorder=3,
    )
    upper.plot(
        duty[~stable], margin[~stable], linestyle="none", marker="o", markersize=6,
        markerfacecolor="white", markeredgecolor=_PRIMARY, markeredgewidth=1.6,
        color=_PRIMARY, label="not certified by the quasi-static criterion", zorder=3,
    )
    upper.set_ylabel("minimum static margin (m)")
    # Headroom so that the legend clears the left branch of the curve.
    ceiling = float(np.nanmax(margin)) if np.isfinite(margin).any() else 1.0
    upper.set_ylim(top=max(ceiling, 1e-6) * 1.55)
    upper.legend(loc="upper left", fontsize=8, frameon=False)
    upper.grid(alpha=_GRID_ALPHA)
    upper.set_title(
        title or "Static stability of a lateral sequence walk against duty factor"
    )

    at_threshold = np.flatnonzero(np.isclose(duty, threshold))
    if at_threshold.size:
        value = float(margin[at_threshold[0]])
        upper.annotate(
            f"minimum margin {value:.5f} m\nat duty factor {threshold:.3f}",
            xy=(threshold, value),
            xytext=(0, 34),
            textcoords="offset points",
            fontsize=8,
            color=_INK,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": _INK, "linewidth": 0.9},
        )

    lower = figure.add_subplot(212, sharex=upper)
    lower.plot(duty, supported, color=_SECONDARY, linewidth=1.8, marker="o", markersize=5)
    lower.axvline(threshold, color=_INK, linewidth=0.9, linestyle="--")
    lower.axhline(1.0, color=_INK, linewidth=0.8, linestyle=":")
    lower.set_ylabel("fraction of cycle\nwith a support polygon")
    lower.set_xlabel("duty factor")
    lower.set_ylim(-0.05, 1.12)
    lower.grid(alpha=_GRID_ALPHA)
    figure.tight_layout()
    return figure


def gait_diagram_grid_figure(
    gaits: Sequence[GaitParameters], cycles: float = 1.0, title: str | None = None
) -> Figure:
    """Draw the contact schedule of several gaits on a common normalised cycle axis.

    One panel per gait, one bar per leg, drawn from the closed form schedule rather
    than from a sampled trace, so every bar edge is the phase boundary itself. The
    panel titles carry the exact range of loaded feet over a cycle, which is the
    quantity the quasi-static criterion turns on.

    Raises:
        ValueError: If ``gaits`` is empty or ``cycles`` is not positive.
    """
    if not gaits:
        raise ValueError("gaits must contain at least one gait")
    if cycles <= 0.0:
        raise ValueError("cycles must be positive")

    figure = Figure(figsize=(7.2, 1.55 * len(gaits) + 0.7))
    axes_list = figure.subplots(len(gaits), 1, sharex=True, squeeze=False)[:, 0]
    for axes, parameters in zip(axes_list, gaits, strict=True):
        span = cycles * parameters.period
        for leg_id in LegId:
            index = int(leg_id)
            for lower, upper in parameters.stance_intervals(leg_id, 0.0, span):
                axes.barh(
                    y=LEG_COUNT - 1 - index,
                    width=(upper - lower) / parameters.period,
                    left=lower / parameters.period,
                    height=0.62,
                    color=_LEG_COLOURS[index],
                    edgecolor="none",
                )
        low, high = stance_count_extrema(parameters)
        axes.set_yticks(list(range(LEG_COUNT)))
        axes.set_yticklabels(list(reversed(LEG_NAMES)), fontsize=8)
        axes.set_ylim(-0.6, LEG_COUNT - 0.4)
        axes.set_xlim(0.0, cycles)
        axes.grid(axis="x", alpha=_GRID_ALPHA)
        axes.set_title(
            f"{parameters.name}: duty factor {parameters.duty_factor:.2f}, "
            f"period {parameters.period:.2f} s, {low} to {high} feet down",
            fontsize=9,
            loc="left",
        )
    axes_list[-1].set_xlabel("gait cycles")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def foot_trajectory_figure(trace: Trace, title: str | None = None) -> Figure:
    """Draw the world path of every foot in the sagittal and horizontal planes."""
    figure = Figure(figsize=(8.0, 5.0))
    sagittal = figure.add_subplot(211)
    horizontal = figure.add_subplot(212)
    for leg_index in range(LEG_COUNT):
        path: NDArray[np.float64] = trace.foot_positions[:, leg_index, :]
        colour = _LEG_COLOURS[leg_index]
        sagittal.plot(path[:, 0], path[:, 2], color=colour, label=LEG_NAMES[leg_index], linewidth=1)
        horizontal.plot(path[:, 0], path[:, 1], color=colour, linewidth=1)
    sagittal.plot(
        trace.body_positions[:, 0], trace.body_positions[:, 2], color=_INK,
        linestyle="--", linewidth=1, label="trunk",
    )
    horizontal.plot(
        trace.com_positions[:, 0], trace.com_positions[:, 1], color=_INK,
        linestyle="--", linewidth=1,
    )
    sagittal.set_ylabel("z (m)")
    sagittal.set_title(title or f"{trace.config.gait.name} foot paths in the world frame")
    sagittal.legend(loc="upper right", fontsize=8, ncol=5, frameon=False)
    sagittal.grid(alpha=_GRID_ALPHA)
    horizontal.set_xlabel("x (m)")
    horizontal.set_ylabel("y (m)")
    horizontal.grid(alpha=_GRID_ALPHA)
    horizontal.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    return figure


def support_polygon_figure(
    trace: Trace, indices: tuple[int, ...], title: str | None = None
) -> Figure:
    """Draw the support polygon and the projected centre of mass at chosen samples.

    A loaded foot is a filled circle, a swinging foot is a cross, and the star is
    the vertically projected centre of mass. Each foot is also named in the legend,
    so the leg identity does not rest on colour alone. Each panel carries its static
    stability margin, which is the distance from the star to the nearest edge.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    if not indices:
        raise ValueError("indices must contain at least one sample index")
    figure = Figure(figsize=(2.6 * len(indices), 3.4))
    for column, index in enumerate(indices):
        axes = figure.add_subplot(1, len(indices), column + 1)
        margins = stability_margins(
            trace.foot_positions[index],
            trace.contacts[index],
            trace.com_positions[index],
            trace.travel_direction[index],
        )
        polygon = support_polygon_at(trace, index)
        feet = trace.foot_positions[index]
        contacts = trace.contacts[index]
        if polygon.shape[0] >= 3:
            closed = np.vstack([polygon, polygon[:1]])
            axes.plot(closed[:, 0], closed[:, 1], color=_PRIMARY, linewidth=1.2)
            axes.fill(closed[:, 0], closed[:, 1], color=_PRIMARY, alpha=0.12)
        for leg_index in range(LEG_COUNT):
            loaded = bool(contacts[leg_index])
            axes.plot(
                feet[leg_index, 0], feet[leg_index, 1],
                marker="o" if loaded else "X",
                color=_LEG_COLOURS[leg_index],
                markersize=7,
                linestyle="none",
                label=LEG_NAMES[leg_index] if column == 0 else None,
            )
        axes.plot(
            trace.com_positions[index, 0], trace.com_positions[index, 1],
            marker="*", color=_INK, markersize=11, linestyle="none",
            label="centre of mass" if column == 0 else None,
        )
        margin = "undefined" if margins.static != margins.static else f"{margins.static:.5f} m"
        axes.set_title(
            f"t = {float(trace.times[index]):.3f} s\n"
            f"{margins.stance_count} feet down, margin {margin}",
            fontsize=8.5,
        )
        axes.set_xlabel("x (m)", fontsize=8)
        axes.tick_params(labelsize=7)
        if column == 0:
            axes.set_ylabel("y (m)", fontsize=8)
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(alpha=_GRID_ALPHA)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=LEG_COUNT + 1, fontsize=8, frameon=False
    )
    figure.suptitle(
        title
        or (
            f"{trace.config.gait.name} support polygons at duty factor "
            f"{trace.config.gait.duty_factor:.2f}, filled marker loaded, cross swinging"
        )
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return figure
