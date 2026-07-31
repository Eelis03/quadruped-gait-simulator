"""Figure construction for traces.

The figures are built on :class:`matplotlib.figure.Figure` directly rather than
through ``pyplot``, so no interactive backend is selected and nothing is drawn to
a screen. Each function returns a figure that the caller saves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from quadruped_gait.analysis.metrics import contact_intervals, stability_series, support_polygon_at
from quadruped_gait.model.geometry import LEG_COUNT, LEG_NAMES
from quadruped_gait.pipeline.simulator import Trace

__all__ = [
    "foot_trajectory_figure",
    "gait_diagram_figure",
    "save_figure",
    "stability_figure",
    "support_polygon_figure",
]

_LEG_COLOURS: tuple[str, ...] = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


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
    axes.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    return figure


def stability_figure(trace: Trace, title: str | None = None) -> Figure:
    """Draw the static and longitudinal stability margins against time."""
    series = stability_series(trace)
    figure = Figure(figsize=(8.0, 4.5))
    upper = figure.add_subplot(211)
    upper.plot(trace.times, series[:, 0], label="static margin", color="#1f77b4")
    upper.plot(trace.times, series[:, 1], label="longitudinal margin", color="#d62728")
    upper.axhline(0.0, color="black", linewidth=0.8)
    upper.set_ylabel("margin (m)")
    upper.legend(loc="upper right", fontsize=8)
    upper.grid(alpha=0.3)
    upper.set_title(title or f"{trace.config.gait.name} quasi-static stability")

    lower = figure.add_subplot(212, sharex=upper)
    lower.step(trace.times, series[:, 3], where="post", color="#2ca02c")
    lower.set_ylabel("feet in stance")
    lower.set_xlabel("time (s)")
    lower.set_ylim(-0.2, LEG_COUNT + 0.2)
    lower.grid(alpha=0.3)
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
        trace.body_positions[:, 0], trace.body_positions[:, 2], color="black",
        linestyle="--", linewidth=1, label="trunk",
    )
    horizontal.plot(
        trace.com_positions[:, 0], trace.com_positions[:, 1], color="black",
        linestyle="--", linewidth=1,
    )
    sagittal.set_ylabel("z (m)")
    sagittal.set_title(title or f"{trace.config.gait.name} foot paths in the world frame")
    sagittal.legend(loc="upper right", fontsize=8, ncol=5)
    sagittal.grid(alpha=0.3)
    horizontal.set_xlabel("x (m)")
    horizontal.set_ylabel("y (m)")
    horizontal.grid(alpha=0.3)
    horizontal.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    return figure


def support_polygon_figure(trace: Trace, indices: tuple[int, ...]) -> Figure:
    """Draw the support polygon and the projected centre of mass at chosen samples."""
    if not indices:
        raise ValueError("indices must contain at least one sample index")
    figure = Figure(figsize=(3.0 * len(indices), 3.4))
    for column, index in enumerate(indices):
        axes = figure.add_subplot(1, len(indices), column + 1)
        polygon = support_polygon_at(trace, index)
        feet = trace.foot_positions[index]
        contacts = trace.contacts[index]
        if polygon.shape[0] >= 3:
            closed = np.vstack([polygon, polygon[:1]])
            axes.plot(closed[:, 0], closed[:, 1], color="#1f77b4", linewidth=1.2)
            axes.fill(closed[:, 0], closed[:, 1], color="#1f77b4", alpha=0.15)
        for leg_index in range(LEG_COUNT):
            marker = "o" if contacts[leg_index] else "x"
            axes.plot(
                feet[leg_index, 0], feet[leg_index, 1],
                marker=marker, color=_LEG_COLOURS[leg_index], markersize=6, linestyle="none",
            )
        axes.plot(
            trace.com_positions[index, 0], trace.com_positions[index, 1],
            marker="*", color="black", markersize=10, linestyle="none",
        )
        axes.set_title(f"t = {float(trace.times[index]):.3f} s", fontsize=9)
        axes.set_xlabel("x (m)")
        if column == 0:
            axes.set_ylabel("y (m)")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(alpha=0.3)
    figure.suptitle(f"{trace.config.gait.name} support polygons")
    figure.tight_layout()
    return figure
