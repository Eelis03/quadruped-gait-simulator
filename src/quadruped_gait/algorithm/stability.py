"""Support polygon construction and quasi-static stability margins.

The support polygon is the convex hull of the loaded feet projected onto the
horizontal ground plane. The static stability margin is the shortest distance
from the vertically projected centre of mass to the boundary of that polygon,
signed positive inside and negative outside (McGhee and Frank, 1968). The
longitudinal stability margin is the same quantity measured only along the
direction of travel: the smaller of the distances forward and backward from the
projected centre of mass to the boundary, again signed.

Both margins are undefined when fewer than three feet are loaded, or when the
loaded feet are collinear, because there is then no polygon with interior. Those
cases return ``nan`` rather than a number, and callers are expected to check
:attr:`SupportPolygon.is_defined` before drawing conclusions.

The hull itself is computed with Qhull through :mod:`scipy.spatial` (Barber et
al., 1996). Degenerate inputs are filtered before the call, and the residual
Qhull failures are converted into an empty polygon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError

from quadruped_gait.model.geometry import LEG_COUNT

__all__ = [
    "StabilityMargins",
    "SupportPolygon",
    "convex_hull_2d",
    "distance_to_boundary",
    "longitudinal_stability_margin",
    "point_in_convex_polygon",
    "stability_margins",
    "static_stability_margin",
    "support_polygon",
]

_AREA_TOLERANCE = 1e-10
_GEOMETRIC_TOLERANCE = 1e-12
_EMPTY_POLYGON: NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)


def convex_hull_2d(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the convex hull of planar ``points`` as counterclockwise vertices.

    Args:
        points: Array of shape ``(n, 2)``.

    Returns:
        Array of shape ``(m, 2)`` listing the hull vertices counterclockwise, or an
        empty ``(0, 2)`` array when the input has fewer than three points or spans
        no area.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {array.shape}")
    if array.shape[0] < 3 or not np.isfinite(array).all():
        return _EMPTY_POLYGON.copy()
    try:
        hull = ConvexHull(array)
    except QhullError:
        return _EMPTY_POLYGON.copy()
    area = float(hull.volume)
    if area < _AREA_TOLERANCE:
        return _EMPTY_POLYGON.copy()
    order = np.asarray(hull.vertices, dtype=np.intp)
    return np.asarray(array[order], dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False)
class SupportPolygon:
    """The convex hull of the loaded feet on the ground plane.

    Attributes:
        vertices: Counterclockwise hull vertices of shape ``(m, 2)``. The array is
            empty when fewer than three feet are loaded or the loaded feet are
            collinear.
        stance_count: How many feet were loaded when the polygon was built.
    """

    vertices: NDArray[np.float64]
    stance_count: int

    def __post_init__(self) -> None:
        array = np.asarray(self.vertices, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"vertices must have shape (m, 2), got {array.shape}")
        object.__setattr__(self, "vertices", array)

    @property
    def is_defined(self) -> bool:
        """Return ``True`` when the polygon has a non-degenerate interior."""
        return bool(self.vertices.shape[0] >= 3)

    @property
    def area(self) -> float:
        """Return the polygon area in square metres, or ``0.0`` when undefined."""
        if not self.is_defined:
            return 0.0
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def centroid(self) -> NDArray[np.float64]:
        """Return the vertex centroid, or an array of ``nan`` when undefined."""
        if not self.is_defined:
            return np.full(2, np.nan, dtype=np.float64)
        return np.asarray(self.vertices.mean(axis=0), dtype=np.float64)


def support_polygon(
    foot_positions: NDArray[np.float64], contacts: NDArray[np.bool_]
) -> SupportPolygon:
    """Build the support polygon from foot positions and contact flags.

    Args:
        foot_positions: Foot positions in world coordinates, shape ``(4, 3)``. Only
            the first two columns are used; the ground is taken to be the plane
            ``z = 0``.
        contacts: Contact flags of shape ``(4,)``.
    """
    positions = np.asarray(foot_positions, dtype=np.float64)
    flags = np.asarray(contacts, dtype=np.bool_).reshape(-1)
    if positions.shape != (LEG_COUNT, 3):
        raise ValueError(f"foot_positions must have shape ({LEG_COUNT}, 3)")
    if flags.shape != (LEG_COUNT,):
        raise ValueError(f"contacts must have shape ({LEG_COUNT},)")
    loaded = positions[flags][:, :2]
    return SupportPolygon(vertices=convex_hull_2d(loaded), stance_count=int(flags.sum()))


def _inward_normals(
    polygon: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the edge start points and the inward normals of a counterclockwise hull."""
    starts = polygon
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    return starts, normals


def point_in_convex_polygon(polygon: NDArray[np.float64], point: NDArray[np.float64]) -> bool:
    """Return ``True`` when ``point`` lies inside or on a counterclockwise convex hull."""
    hull = np.asarray(polygon, dtype=np.float64)
    query = np.asarray(point, dtype=np.float64).reshape(-1)
    if hull.shape[0] < 3:
        return False
    if query.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {query.shape}")
    starts, normals = _inward_normals(hull)
    signed = np.einsum("ij,ij->i", query - starts, normals)
    return bool(np.all(signed >= -_GEOMETRIC_TOLERANCE))


def distance_to_boundary(polygon: NDArray[np.float64], point: NDArray[np.float64]) -> float:
    """Return the unsigned distance from ``point`` to the polygon boundary."""
    hull = np.asarray(polygon, dtype=np.float64)
    query = np.asarray(point, dtype=np.float64).reshape(-1)
    if hull.shape[0] < 2:
        return math.nan
    if query.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {query.shape}")
    starts = hull
    ends = np.roll(hull, -1, axis=0)
    edges = ends - starts
    lengths_squared = np.einsum("ij,ij->i", edges, edges)
    safe = np.where(lengths_squared > _GEOMETRIC_TOLERANCE, lengths_squared, 1.0)
    projection = np.einsum("ij,ij->i", query - starts, edges) / safe
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts + projection[:, None] * edges
    return float(np.min(np.linalg.norm(query - closest, axis=1)))


def static_stability_margin(polygon: NDArray[np.float64], point: NDArray[np.float64]) -> float:
    """Return the signed static stability margin of ``point`` against a hull.

    The magnitude is the shortest distance to the boundary. The sign is positive
    when the point lies inside the hull and negative when it lies outside, which
    is the convention of McGhee and Frank (1968). Returns ``nan`` when the hull
    has fewer than three vertices.
    """
    hull = np.asarray(polygon, dtype=np.float64)
    if hull.shape[0] < 3:
        return math.nan
    distance = distance_to_boundary(hull, point)
    return distance if point_in_convex_polygon(hull, point) else -distance


def longitudinal_stability_margin(
    polygon: NDArray[np.float64],
    point: NDArray[np.float64],
    direction: NDArray[np.float64] | tuple[float, float] = (1.0, 0.0),
) -> float:
    """Return the signed longitudinal stability margin along ``direction``.

    The infinite line through ``point`` along ``direction`` is clipped against the
    hull, giving an interval ``[t0, t1]`` of signed distances. The margin is
    ``min(t1, -t0)``: when the point is inside the hull this is the smaller of the
    distances forward and backward to the boundary and is positive, and when the
    point is outside but the line still crosses the hull it is the negative of the
    distance to the nearer crossing. Returns ``nan`` when the hull is degenerate or
    the line misses it entirely.
    """
    hull = np.asarray(polygon, dtype=np.float64)
    query = np.asarray(point, dtype=np.float64).reshape(-1)
    heading = np.asarray(direction, dtype=np.float64).reshape(-1)
    if hull.shape[0] < 3:
        return math.nan
    if query.shape != (2,) or heading.shape != (2,):
        raise ValueError("point and direction must both have shape (2,)")
    norm = float(np.linalg.norm(heading))
    if norm < _GEOMETRIC_TOLERANCE:
        raise ValueError("direction must be a non-zero vector")
    unit = heading / norm

    starts, normals = _inward_normals(hull)
    along = normals @ unit
    offset = np.einsum("ij,ij->i", query - starts, normals)

    lower = -math.inf
    upper = math.inf
    for slope, constant in zip(along.tolist(), offset.tolist(), strict=True):
        if abs(slope) <= _GEOMETRIC_TOLERANCE:
            if constant < -_GEOMETRIC_TOLERANCE:
                return math.nan
            continue
        bound = -constant / slope
        if slope > 0.0:
            lower = max(lower, bound)
        else:
            upper = min(upper, bound)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return math.nan
    return min(upper, -lower)


@dataclass(frozen=True, slots=True)
class StabilityMargins:
    """Quasi-static stability of one instant.

    Attributes:
        static: Signed shortest distance from the projected centre of mass to the
            support polygon boundary, in metres, or ``nan`` when undefined.
        longitudinal: Signed distance measured along the direction of travel, in
            metres, or ``nan`` when undefined.
        stance_count: Number of loaded feet.
        is_defined: Whether the support polygon has an interior.
    """

    static: float
    longitudinal: float
    stance_count: int
    is_defined: bool


def stability_margins(
    foot_positions: NDArray[np.float64],
    contacts: NDArray[np.bool_],
    com_position: NDArray[np.float64],
    direction: NDArray[np.float64] | tuple[float, float] = (1.0, 0.0),
) -> StabilityMargins:
    """Return the static and longitudinal margins for one instant.

    Args:
        foot_positions: Foot positions in world coordinates, shape ``(4, 3)``.
        contacts: Contact flags of shape ``(4,)``.
        com_position: Centre of mass in world coordinates, shape ``(3,)`` or ``(2,)``.
            Only the horizontal components are used, which is the vertical
            projection assumed by the quasi-static criterion.
        direction: Horizontal direction of travel used for the longitudinal margin.
    """
    polygon = support_polygon(foot_positions, contacts)
    com = np.asarray(com_position, dtype=np.float64).reshape(-1)[:2]
    return StabilityMargins(
        static=static_stability_margin(polygon.vertices, com),
        longitudinal=longitudinal_stability_margin(polygon.vertices, com, direction),
        stance_count=polygon.stance_count,
        is_defined=polygon.is_defined,
    )
