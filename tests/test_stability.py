"""Invariant and hand-computed tests for the support polygon and the margins."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quadruped_gait.algorithm.stability import (
    convex_hull_2d,
    distance_to_boundary,
    longitudinal_stability_margin,
    point_in_convex_polygon,
    stability_margins,
    static_stability_margin,
    support_polygon,
)
from quadruped_gait.model.geometry import LEG_COUNT

# Axis aligned unit square, counterclockwise. Every margin against it is exact by hand.
SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
# Right triangle with legs of length one along the axes, counterclockwise.
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _feet(*positions: tuple[float, float]) -> np.ndarray:
    array = np.zeros((LEG_COUNT, 3), dtype=np.float64)
    for index, (x, y) in enumerate(positions):
        array[index, 0] = x
        array[index, 1] = y
    return array


def test_hull_of_a_square_recovers_the_four_corners() -> None:
    scrambled = SQUARE[[2, 0, 3, 1]]
    hull = convex_hull_2d(scrambled)
    assert hull.shape == (4, 2)
    assert {tuple(row) for row in hull.tolist()} == {tuple(row) for row in SQUARE.tolist()}


def test_hull_discards_interior_points() -> None:
    points = np.vstack([SQUARE, [[0.0, 0.0], [0.2, -0.3]]])
    assert convex_hull_2d(points).shape == (4, 2)


def test_hull_is_counterclockwise() -> None:
    hull = convex_hull_2d(SQUARE)
    shifted = np.roll(hull, -1, axis=0)
    cross = hull[:, 0] * shifted[:, 1] - hull[:, 1] * shifted[:, 0]
    assert float(cross.sum()) > 0.0


@pytest.mark.parametrize("count", [0, 1, 2])
def test_hull_is_empty_below_three_points(count: int) -> None:
    assert convex_hull_2d(SQUARE[:count]).shape == (0, 2)


def test_hull_is_empty_for_collinear_points() -> None:
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert convex_hull_2d(collinear).shape == (0, 2)


def test_support_polygon_is_empty_below_three_contacts() -> None:
    """Two feet span a line, which has no interior and therefore no margin."""
    feet = _feet((0.25, 0.23), (0.25, -0.23), (-0.25, 0.23), (-0.25, -0.23))
    for contacts in (
        np.array([True, False, False, False]),
        np.array([True, False, False, True]),
        np.array([False, False, False, False]),
    ):
        polygon = support_polygon(feet, contacts)
        assert not polygon.is_defined
        assert polygon.vertices.shape == (0, 2)
        assert polygon.area == 0.0
        assert math.isnan(static_stability_margin(polygon.vertices, np.zeros(2)))


def test_support_polygon_of_three_contacts_is_a_triangle() -> None:
    feet = _feet((0.25, 0.23), (0.25, -0.23), (-0.25, 0.23), (-0.25, -0.23))
    polygon = support_polygon(feet, np.array([True, True, True, False]))
    assert polygon.is_defined
    assert polygon.vertices.shape == (3, 2)
    assert polygon.stance_count == 3
    assert polygon.area == pytest.approx(0.5 * 0.5 * 0.46)


def test_support_polygon_of_four_contacts_is_a_quadrilateral() -> None:
    feet = _feet((0.25, 0.23), (0.25, -0.23), (-0.25, 0.23), (-0.25, -0.23))
    polygon = support_polygon(feet, np.array([True, True, True, True]))
    assert polygon.vertices.shape == (4, 2)
    assert polygon.area == pytest.approx(0.5 * 0.46)
    np.testing.assert_allclose(polygon.centroid, [0.0, 0.0], atol=1e-15)


def test_static_margin_of_a_square_is_hand_computed() -> None:
    """The centre of a unit square is one metre from every edge."""
    assert static_stability_margin(SQUARE, np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert static_stability_margin(SQUARE, np.array([0.25, 0.0])) == pytest.approx(0.75)
    assert static_stability_margin(SQUARE, np.array([0.0, -0.6])) == pytest.approx(0.4)


def test_static_margin_sign_convention_is_negative_outside() -> None:
    """The sign is positive inside and negative outside, as in McGhee and Frank."""
    assert static_stability_margin(SQUARE, np.array([2.0, 0.0])) == pytest.approx(-1.0)
    assert static_stability_margin(SQUARE, np.array([0.0, -3.5])) == pytest.approx(-2.5)
    assert static_stability_margin(SQUARE, np.array([2.0, 2.0])) == pytest.approx(
        -math.hypot(1.0, 1.0)
    )


def test_static_margin_is_zero_on_the_boundary() -> None:
    assert static_stability_margin(SQUARE, np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-12)
    assert static_stability_margin(SQUARE, np.array([1.0, 1.0])) == pytest.approx(0.0, abs=1e-12)


def test_static_margin_of_a_triangle_is_hand_computed() -> None:
    """The nearest edge to the point one quarter along both axes is a leg of the triangle."""
    assert static_stability_margin(TRIANGLE, np.array([0.25, 0.25])) == pytest.approx(0.25)
    assert static_stability_margin(TRIANGLE, np.array([0.4, 0.4])) == pytest.approx(
        0.2 / math.sqrt(2.0)
    )


def test_point_membership_matches_the_margin_sign() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(-2.0, 2.0, size=(400, 2))
    for point in points:
        inside = point_in_convex_polygon(SQUARE, point)
        margin = static_stability_margin(SQUARE, point)
        assert inside == (margin >= 0.0)


def test_distance_to_boundary_is_unsigned() -> None:
    assert distance_to_boundary(SQUARE, np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert distance_to_boundary(SQUARE, np.array([3.0, 0.0])) == pytest.approx(2.0)


def test_longitudinal_margin_of_a_square_is_hand_computed() -> None:
    """Along x the square extends one metre either side of its centre."""
    assert longitudinal_stability_margin(SQUARE, np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert longitudinal_stability_margin(SQUARE, np.array([0.25, 0.0])) == pytest.approx(0.75)
    assert longitudinal_stability_margin(SQUARE, np.array([0.0, 0.0]), (0.0, 1.0)) == pytest.approx(
        1.0
    )


def test_longitudinal_margin_is_negative_when_the_line_misses_the_point() -> None:
    """A centre of mass two metres ahead is one metre past the leading edge."""
    assert longitudinal_stability_margin(SQUARE, np.array([2.0, 0.0])) == pytest.approx(-1.0)
    assert longitudinal_stability_margin(SQUARE, np.array([-4.0, 0.0])) == pytest.approx(-3.0)


def test_longitudinal_margin_is_undefined_when_the_line_misses_the_hull() -> None:
    assert math.isnan(longitudinal_stability_margin(SQUARE, np.array([0.0, 3.0])))


def test_longitudinal_margin_uses_a_normalised_direction() -> None:
    scaled = longitudinal_stability_margin(SQUARE, np.array([0.0, 0.0]), (7.0, 0.0))
    assert scaled == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-zero"):
        longitudinal_stability_margin(SQUARE, np.array([0.0, 0.0]), (0.0, 0.0))


def test_longitudinal_margin_never_exceeds_the_static_margin_direction() -> None:
    """The static margin is the shortest distance, so no directed distance is smaller."""
    rng = np.random.default_rng(11)
    for point in rng.uniform(-0.9, 0.9, size=(200, 2)):
        static = static_stability_margin(SQUARE, point)
        longitudinal = longitudinal_stability_margin(SQUARE, point)
        assert longitudinal >= static - 1e-12


def test_margins_are_defined_for_a_statically_stable_triangle() -> None:
    """Three feet around the origin give a positive margin under a centred mass."""
    feet = _feet((0.25, 0.23), (0.25, -0.23), (-0.25, 0.23), (-0.25, -0.23))
    contacts = np.array([True, True, True, False])
    margins = stability_margins(feet, contacts, np.array([0.0, 0.06, 0.42]))
    assert margins.is_defined
    assert margins.stance_count == 3
    # The nearest edge is the diagonal from the front-right to the hind-left foot.
    assert margins.static == pytest.approx(0.03 / math.hypot(0.5, 0.46), abs=1e-12)
    assert margins.longitudinal == pytest.approx(0.0652173913, abs=1e-9)


def test_margins_are_nan_when_the_support_is_degenerate() -> None:
    feet = _feet((0.25, 0.23), (0.25, -0.23), (-0.25, 0.23), (-0.25, -0.23))
    margins = stability_margins(feet, np.array([True, False, False, True]), np.zeros(3))
    assert not margins.is_defined
    assert math.isnan(margins.static)
    assert math.isnan(margins.longitudinal)


def test_input_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        convex_hull_2d(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="shape"):
        support_polygon(np.zeros((3, 3)), np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        support_polygon(np.zeros((4, 3)), np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        point_in_convex_polygon(SQUARE, np.zeros(3))


def test_non_finite_points_produce_an_empty_hull() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]])
    assert convex_hull_2d(points).shape == (0, 2)
