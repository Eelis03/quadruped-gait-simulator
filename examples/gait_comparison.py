"""Compare walk, trot, pace, and bound, then sweep the walk duty factor.

The first table places the four library gaits side by side under a common trunk
command. The second table sweeps the duty factor of the walk and locates the
value below which the quasi-static criterion no longer certifies the gait.

Usage:
    uv run python examples/gait_comparison.py [--cycles C] [--samples-per-cycle N] [--no-figures]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quadruped_gait.analysis import (
    format_duty_sweep,
    format_summary_table,
    save_figure,
    stability_figure,
    summarise,
    sweep_stability,
)
from quadruped_gait.pipeline import (
    DUTY_SWEEP_FACTORS,
    duty_factor_sweep,
    reference_gaits,
    reference_walk,
    run_gaits,
)

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
    traces = run_gaits(reference_gaits(arguments.cycles, arguments.samples_per_cycle))
    print("library gaits at 0.30 m/s, no lateral trunk offset")
    print(format_summary_table([summarise(trace) for trace in traces]))
    print()
    print("columns: beta commanded, beta realised (mean over legs), largest per leg error,")
    print("fraction of samples with a support polygon, static margin, longitudinal margin,")
    print("and the mean number of loaded feet. A value of n/a means no sample had three")
    print("or more non-collinear feet on the ground.")
    print()

    base = reference_walk(arguments.cycles, arguments.samples_per_cycle)
    rows = duty_factor_sweep(base, DUTY_SWEEP_FACTORS)
    print("walk duty factor sweep, 0.06 m lateral trunk offset")
    print(format_duty_sweep(sweep_stability(rows)))
    print()

    if arguments.no_figures:
        return 0
    for trace in traces:
        path = save_figure(
            stability_figure(trace),
            arguments.figure_dir / f"stability_{trace.config.gait.name}.png",
        )
        print(f"figure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
