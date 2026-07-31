"""Regression test pinning one recorded gait cycle to stored numbers.

The reference file holds the trunk pose, foot positions, joint angles, contact
flags, and stability margins of the reference walk at evenly spaced samples,
together with the scalar summary of the run. Running this module as a script
regenerates the file, which should only be done when a change of behaviour has
been reviewed deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quadruped_gait.analysis import stability_series, summarise
from quadruped_gait.pipeline import reference_walk, simulate

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "reference_walk.json"
CYCLES = 1.0
SAMPLES_PER_CYCLE = 200
PROBE_STRIDE = 20

POSITION_TOLERANCE = 1e-9
ANGLE_TOLERANCE = 1e-9
MARGIN_TOLERANCE = 1e-9


def _build_reference() -> dict[str, Any]:
    """Compute the reference record from the current implementation."""
    config = reference_walk(cycles=CYCLES, samples_per_cycle=SAMPLES_PER_CYCLE)
    trace = simulate(config)
    series = stability_series(trace)
    report = summarise(trace)
    probes = list(range(0, len(trace), PROBE_STRIDE))
    return {
        "description": "Reference walk, one cycle, recorded for regression testing.",
        "config": {
            "gait": config.gait.name,
            "period": config.gait.period,
            "duty_factor": config.gait.duty_factor,
            "phase_offsets": list(config.gait.phase_offsets),
            "forward_velocity": config.command.forward_velocity,
            "sway_amplitude": config.command.sway_amplitude,
            "height": config.command.height,
            "cycles": config.cycles,
            "samples_per_cycle": config.samples_per_cycle,
            "swing_clearance": config.swing_clearance,
            "swing_profile": config.swing_profile,
        },
        "summary": {
            "stride_length": report.stride_length,
            "mean_stance_count": report.contact.mean_stance_count,
            "max_duty_error": report.contact.max_absolute_error,
            "supported_fraction": report.stability.supported_fraction,
            "minimum_static": report.stability.minimum_static,
            "mean_static": report.stability.mean_static,
            "minimum_longitudinal": report.stability.minimum_longitudinal,
            "mean_longitudinal": report.stability.mean_longitudinal,
            "mean_support_area": report.stability.mean_support_area,
            "unreachable_samples": report.unreachable_samples,
        },
        "probe_indices": probes,
        "times": trace.times[probes].tolist(),
        "body_positions": trace.body_positions[probes].tolist(),
        "contacts": trace.contacts[probes].astype(int).tolist(),
        "foot_positions": trace.foot_positions[probes].tolist(),
        "joint_angles": trace.joint_angles[probes].tolist(),
        "static_margin": series[probes, 0].tolist(),
        "longitudinal_margin": series[probes, 1].tolist(),
        "support_area": series[probes, 2].tolist(),
    }


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """Load the stored reference record."""
    if not REFERENCE_PATH.exists():
        raise AssertionError(
            f"reference file is missing: {REFERENCE_PATH}. "
            "Regenerate it with 'uv run python tests/test_regression.py'."
        )
    with REFERENCE_PATH.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    """Recompute the record from the current implementation."""
    return _build_reference()


def test_configuration_has_not_drifted(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    assert recorded["config"] == reference["config"]


def test_probe_layout_has_not_drifted(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    assert recorded["probe_indices"] == reference["probe_indices"]
    np.testing.assert_allclose(recorded["times"], reference["times"], atol=1e-12)


def test_body_positions_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    np.testing.assert_allclose(
        recorded["body_positions"], reference["body_positions"], atol=POSITION_TOLERANCE
    )


def test_contacts_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    np.testing.assert_array_equal(recorded["contacts"], reference["contacts"])


def test_foot_positions_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    np.testing.assert_allclose(
        recorded["foot_positions"], reference["foot_positions"], atol=POSITION_TOLERANCE
    )


def test_joint_angles_match_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    np.testing.assert_allclose(
        recorded["joint_angles"], reference["joint_angles"], atol=ANGLE_TOLERANCE
    )


@pytest.mark.parametrize("key", ["static_margin", "longitudinal_margin", "support_area"])
def test_stability_series_matches_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any], key: str
) -> None:
    np.testing.assert_allclose(recorded[key], reference[key], atol=MARGIN_TOLERANCE)


def test_summary_matches_the_reference(
    reference: dict[str, Any], recorded: dict[str, Any]
) -> None:
    assert set(recorded["summary"]) == set(reference["summary"])
    for key, value in reference["summary"].items():
        assert recorded["summary"][key] == pytest.approx(value, abs=MARGIN_TOLERANCE)


def main() -> int:
    """Write the reference file from the current implementation."""
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REFERENCE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_build_reference(), handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
