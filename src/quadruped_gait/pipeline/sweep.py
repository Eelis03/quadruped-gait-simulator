"""Batch runs over a parameter axis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from quadruped_gait.pipeline.simulator import SimulationConfig, Trace, simulate

__all__ = ["DutySweepRow", "duty_factor_sweep", "run_gaits"]


@dataclass(frozen=True, slots=True, eq=False)
class DutySweepRow:
    """One point of a duty factor sweep.

    Attributes:
        duty_factor: The commanded duty factor.
        trace: The trace produced at that duty factor.
    """

    duty_factor: float
    trace: Trace


def run_gaits(configs: Sequence[SimulationConfig]) -> tuple[Trace, ...]:
    """Run every configuration and return the traces in the same order."""
    return tuple(simulate(config) for config in configs)


def duty_factor_sweep(
    base: SimulationConfig, duty_factors: Sequence[float]
) -> tuple[DutySweepRow, ...]:
    """Re-run ``base`` at each duty factor, holding everything else fixed.

    This is the experiment that locates the static stability threshold of a crawl
    gait, which McGhee and Frank (1968) place at a duty factor of three quarters
    for a quadruped.
    """
    if not duty_factors:
        raise ValueError("duty_factors must not be empty")
    rows: list[DutySweepRow] = []
    for duty_factor in duty_factors:
        config = replace(base, gait=base.gait.with_duty_factor(duty_factor))
        rows.append(DutySweepRow(duty_factor=duty_factor, trace=simulate(config)))
    return tuple(rows)
