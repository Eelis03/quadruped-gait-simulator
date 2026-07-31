"""Print the contact schedule of every library gait and check the duty factor.

For each gait the script renders one cycle of the gait diagram as text, reports
the realised stance fraction of every leg against the commanded duty factor, and
saves the same diagram as a figure.

Usage:
    uv run python examples/gait_diagram.py [--cycles C] [--samples-per-cycle N] [--no-figures]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quadruped_gait.analysis import (
    contact_intervals,
    format_contact_summary,
    format_gait_diagram,
    gait_diagram_figure,
    save_figure,
    summarise,
)
from quadruped_gait.pipeline import reference_gaits, run_gaits

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figures"


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for this example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=float, default=3.0, help="gait cycles to simulate")
    parser.add_argument("--samples-per-cycle", type=int, default=200, help="samples per cycle")
    parser.add_argument("--columns", type=int, default=60, help="text diagram width")
    parser.add_argument("--no-figures", action="store_true", help="skip figure output")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR, help="figure directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example."""
    arguments = build_parser().parse_args(argv)
    configs = reference_gaits(arguments.cycles, arguments.samples_per_cycle)
    for trace in run_gaits(configs):
        report = summarise(trace)
        print("=" * 72)
        print(format_contact_summary(report))
        print()
        print(
            format_gait_diagram(
                contact_intervals(trace), trace.config.gait.period, arguments.columns
            )
        )
        print()
        if not arguments.no_figures:
            path = save_figure(
                gait_diagram_figure(trace),
                arguments.figure_dir / f"gait_diagram_{trace.config.gait.name}.png",
            )
            print(f"figure written to {path}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
