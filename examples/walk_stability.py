"""Run the statically stable reference walk and report its stability margins.

The reference walk is a lateral sequence walk at a duty factor of 0.80 with a
0.06 m lateral trunk offset, driven at 0.30 m/s. The script prints the contact
statistics and the static and longitudinal stability margins, and saves the
stability history, the foot paths, and four support polygons.

Usage:
    uv run python examples/walk_stability.py [--cycles C] [--samples-per-cycle N] [--no-figures]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quadruped_gait.analysis import (
    foot_trajectory_figure,
    format_report,
    save_figure,
    stability_figure,
    summarise,
    support_polygon_figure,
)
from quadruped_gait.pipeline import reference_walk, simulate

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for this example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=float, default=3.0, help="gait cycles to simulate")
    parser.add_argument("--samples-per-cycle", type=int, default=200, help="samples per cycle")
    parser.add_argument("--no-figures", action="store_true", help="skip figure output")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR, help="figure directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    config = reference_walk(arguments.cycles, arguments.samples_per_cycle)
    trace = simulate(config)
    print(format_report(summarise(trace)))
    if arguments.no_figures:
        return 0

    stride = max(1, len(trace) // 4)
    indices = tuple(index * stride for index in range(4))
    for figure, name in (
        (stability_figure(trace), "walk_stability.png"),
        (foot_trajectory_figure(trace), "walk_foot_paths.png"),
        (support_polygon_figure(trace, indices), "walk_support_polygons.png"),
    ):
        print(f"figure written to {save_figure(figure, arguments.figure_dir / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
