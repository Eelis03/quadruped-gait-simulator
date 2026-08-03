"""Regenerate the three figures committed under docs/figures.

This is the only script that writes into a tracked directory. It produces the
duty factor sweep, the contact schedules of the four library gaits, and the
support polygons of a walk sitting exactly on the stability threshold, at a size
and resolution chosen to keep the whole set inside the repository figure budget.

Usage:
    uv run python examples/docs_figures.py [--samples-per-cycle N] [--no-figures]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quadruped_gait.algorithm import GAIT_NAMES, gait
from quadruped_gait.analysis import (
    critical_sample_indices,
    duty_sweep_figure,
    format_duty_sweep,
    format_report,
    gait_diagram_grid_figure,
    save_figure,
    summarise,
    support_polygon_figure,
    sweep_stability,
)
from quadruped_gait.pipeline import (
    DUTY_SWEEP_FIGURE_FACTORS,
    STATIC_STABILITY_THRESHOLD,
    duty_factor_sweep,
    reference_walk,
    simulate,
    threshold_walk,
)

FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"

# Chosen so that the three files together stay well inside the 250 kB budget the
# portfolio applies to tracked figures. Compression is not involved: the size is
# controlled by the figure dimensions and this resolution alone.
FIGURE_DPI = 110


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for this example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=float, default=3.0, help="gait cycles to simulate")
    parser.add_argument("--samples-per-cycle", type=int, default=200, help="samples per cycle")
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="output resolution")
    parser.add_argument("--no-figures", action="store_true", help="skip figure output")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR, help="figure directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    base = reference_walk(arguments.cycles, arguments.samples_per_cycle)
    rows = sweep_stability(duty_factor_sweep(base, DUTY_SWEEP_FIGURE_FACTORS))
    print(f"duty factor sweep over {len(rows)} points, 0.06 m lateral trunk offset")
    print(format_duty_sweep(rows))
    print()

    critical = simulate(threshold_walk(arguments.cycles, arguments.samples_per_cycle))
    indices = critical_sample_indices(critical)
    print(f"walk at the threshold duty factor {STATIC_STABILITY_THRESHOLD:.2f}")
    print(format_report(summarise(critical)))
    print()
    gaits = [gait(name) for name in GAIT_NAMES]

    if arguments.no_figures:
        return 0

    figures = (
        ("duty_factor_sweep.png", duty_sweep_figure(rows, STATIC_STABILITY_THRESHOLD)),
        ("gait_diagrams.png", gait_diagram_grid_figure(gaits, cycles=2.0)),
        (
            "critical_support_polygon.png",
            support_polygon_figure(
                critical,
                indices,
                title=(
                    "Why the margin is zero at a duty factor of "
                    f"{STATIC_STABILITY_THRESHOLD:.2f}: the outgoing and the incoming "
                    "support triangle share the edge the centre of mass is crossing"
                ),
            ),
        ),
    )
    total = 0
    for name, figure in figures:
        path = save_figure(figure, arguments.figure_dir / name, dpi=arguments.dpi)
        size = path.stat().st_size
        total += size
        print(f"figure written to {path} ({size / 1024:.1f} kB)")
    print(f"total {total / 1024:.1f} kB across {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
