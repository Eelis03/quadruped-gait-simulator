"""Checks on what the installed distribution delivers, as opposed to what it computes.

The package passes mypy in strict mode, but a type checker running against a
consumer of this package ignores every annotation in it unless PEP 561 marker file
``py.typed`` ships inside the package directory. These tests assert that the marker
is present and in the right place, since its absence is silent.
"""

from __future__ import annotations

from pathlib import Path

import quadruped_gait

PACKAGE_ROOT = Path(quadruped_gait.__file__).resolve().parent
MARKER = PACKAGE_ROOT / "py.typed"


def test_the_typing_marker_is_inside_the_package_directory() -> None:
    """PEP 561 requires the marker to sit beside ``__init__.py``, not above it."""
    assert MARKER.is_file()
    assert MARKER.parent == PACKAGE_ROOT
    assert (PACKAGE_ROOT / "__init__.py").is_file()


def test_the_typing_marker_is_empty() -> None:
    """A partial marker would name modules; this package types all of its own."""
    assert MARKER.read_bytes() == b""


def test_the_package_is_importable_by_the_name_the_marker_covers() -> None:
    assert PACKAGE_ROOT.name == "quadruped_gait"
    assert quadruped_gait.__version__
