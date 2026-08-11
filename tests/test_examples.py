"""Integration tests running every example script under a reduced step count."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Every example, with the arguments that shrink it to a fast integration run.
EXAMPLE_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "docs_figures": ("--cycles", "1", "--samples-per-cycle", "12", "--no-figures"),
    "gait_comparison": ("--cycles", "1", "--samples-per-cycle", "24", "--no-figures"),
    "gait_diagram": ("--cycles", "1", "--samples-per-cycle", "24", "--no-figures"),
    "leg_kinematics": ("--samples", "50"),
    "walk_stability": ("--cycles", "1", "--samples-per-cycle", "24", "--no-figures"),
}


def _example_names() -> tuple[str, ...]:
    return tuple(
        sorted(path.stem for path in EXAMPLES_DIR.glob("*.py") if not path.stem.startswith("_"))
    )


def _load(name: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"quadruped_gait_example_{name}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load example {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_example_is_covered() -> None:
    """A new example script must be added to the integration table."""
    assert _example_names() == tuple(sorted(EXAMPLE_ARGUMENTS))


@pytest.mark.parametrize("name", sorted(EXAMPLE_ARGUMENTS))
def test_example_runs_to_completion(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load(name)
    arguments: Sequence[str] = EXAMPLE_ARGUMENTS[name]
    assert module.main(arguments) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
    assert captured.err == ""


@pytest.mark.parametrize("name", sorted(EXAMPLE_ARGUMENTS))
def test_example_exposes_a_parser(name: str) -> None:
    module = _load(name)
    parser = module.build_parser()
    assert parser.description


def test_example_writes_figures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The figure producing path of an example is exercised at least once."""
    module = _load("walk_stability")
    arguments = (
        "--cycles",
        "1",
        "--samples-per-cycle",
        "24",
        "--figure-dir",
        str(tmp_path),
    )
    assert module.main(arguments) == 0
    capsys.readouterr()
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "walk_foot_paths.png",
        "walk_stability.png",
        "walk_support_polygons.png",
    ]


def test_published_figure_script_writes_the_documented_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The regeneration command writes exactly the three figures the README embeds."""
    module = _load("docs_figures")
    arguments = (
        "--cycles",
        "1",
        "--samples-per-cycle",
        "12",
        "--dpi",
        "50",
        "--figure-dir",
        str(tmp_path),
    )
    assert module.main(arguments) == 0
    capsys.readouterr()
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "critical_support_polygon.png",
        "duty_factor_sweep.png",
        "gait_diagrams.png",
    ]
