"""Static geometry of the quadruped: leg link lengths and trunk hip layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LEG_COUNT",
    "LEG_NAMES",
    "LegGeometry",
    "LegId",
    "RobotModel",
    "default_robot",
]


class LegId(IntEnum):
    """Index of a leg in the canonical ordering used throughout the package.

    The ordering is front-left, front-right, hind-left, hind-right. Every array
    with a leg axis uses this ordering.
    """

    FRONT_LEFT = 0
    FRONT_RIGHT = 1
    HIND_LEFT = 2
    HIND_RIGHT = 3


LEG_COUNT: int = len(LegId)
LEG_NAMES: tuple[str, ...] = ("FL", "FR", "HL", "HR")


@dataclass(frozen=True, slots=True)
class LegGeometry:
    """Link lengths of one three degree of freedom leg.

    The kinematic chain is hip roll about the trunk x axis, then a fixed lateral
    offset to the hip pitch joint, then hip pitch and knee pitch about parallel y
    axes. This is the layout used by ANYmal (Hutter et al., 2016) and by the MIT
    Cheetah family (Bledt et al., 2018).

    Attributes:
        abduction_offset: Distance from the hip roll axis to the hip pitch axis,
            measured laterally, in metres.
        thigh_length: Distance from the hip pitch axis to the knee axis, in metres.
        shank_length: Distance from the knee axis to the foot contact point, in metres.
    """

    abduction_offset: float
    thigh_length: float
    shank_length: float

    def __post_init__(self) -> None:
        if self.abduction_offset < 0.0:
            raise ValueError("abduction_offset must not be negative")
        if self.thigh_length <= 0.0 or self.shank_length <= 0.0:
            raise ValueError("thigh_length and shank_length must be positive")

    @property
    def max_sagittal_reach(self) -> float:
        """Largest distance from the hip pitch axis to the foot."""
        return self.thigh_length + self.shank_length

    @property
    def min_sagittal_reach(self) -> float:
        """Smallest distance from the hip pitch axis to the foot."""
        return abs(self.thigh_length - self.shank_length)


@dataclass(frozen=True, slots=True)
class RobotModel:
    """Trunk layout, leg geometry, and the nominal standing posture.

    Attributes:
        hip_half_length: Half the fore-aft distance between the hip roll axes, in metres.
        hip_half_width: Half the lateral distance between the hip roll axes, in metres.
        leg: Link lengths shared by all four legs.
        nominal_height: Height of the trunk origin above flat ground when standing, in metres.
        com_offset: Centre of mass position expressed in trunk coordinates, in metres.
    """

    hip_half_length: float
    hip_half_width: float
    leg: LegGeometry
    nominal_height: float
    com_offset: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        if self.hip_half_length <= 0.0 or self.hip_half_width <= 0.0:
            raise ValueError("hip_half_length and hip_half_width must be positive")
        if self.nominal_height <= 0.0:
            raise ValueError("nominal_height must be positive")

    @staticmethod
    def lateral_sign(leg_id: LegId) -> float:
        """Return ``+1.0`` for a left leg and ``-1.0`` for a right leg."""
        return 1.0 if leg_id in (LegId.FRONT_LEFT, LegId.HIND_LEFT) else -1.0

    @staticmethod
    def longitudinal_sign(leg_id: LegId) -> float:
        """Return ``+1.0`` for a front leg and ``-1.0`` for a hind leg."""
        return 1.0 if leg_id in (LegId.FRONT_LEFT, LegId.FRONT_RIGHT) else -1.0

    def hip_offset(self, leg_id: LegId) -> NDArray[np.float64]:
        """Return the hip roll axis origin of ``leg_id`` in trunk coordinates."""
        return np.array(
            [
                self.longitudinal_sign(leg_id) * self.hip_half_length,
                self.lateral_sign(leg_id) * self.hip_half_width,
                0.0,
            ],
            dtype=np.float64,
        )

    def hip_offsets(self) -> NDArray[np.float64]:
        """Return all four hip origins in trunk coordinates, shape ``(4, 3)``."""
        return np.stack([self.hip_offset(leg_id) for leg_id in LegId])

    def nominal_foot_in_hip(self, leg_id: LegId) -> NDArray[np.float64]:
        """Return the standing foot position of ``leg_id`` in its hip frame."""
        return np.array(
            [
                0.0,
                self.lateral_sign(leg_id) * self.leg.abduction_offset,
                -self.nominal_height,
            ],
            dtype=np.float64,
        )

    def nominal_foot_in_body(self, leg_id: LegId) -> NDArray[np.float64]:
        """Return the standing foot position of ``leg_id`` in trunk coordinates."""
        return self.hip_offset(leg_id) + self.nominal_foot_in_hip(leg_id)

    def nominal_feet_in_body(self) -> NDArray[np.float64]:
        """Return all four standing foot positions in trunk coordinates, shape ``(4, 3)``."""
        return np.stack([self.nominal_foot_in_body(leg_id) for leg_id in LegId])

    def com_in_body(self) -> NDArray[np.float64]:
        """Return the centre of mass in trunk coordinates, shape ``(3,)``."""
        return np.asarray(self.com_offset, dtype=np.float64)


def default_robot() -> RobotModel:
    """Return the reference robot used by the examples and the regression tests.

    The dimensions are of the same order as ANYmal B (Hutter et al., 2016): a
    0.50 m fore-aft hip spacing, a 0.30 m lateral hip spacing, equal 0.30 m
    thigh and shank links, and a 0.42 m standing trunk height.
    """
    return RobotModel(
        hip_half_length=0.25,
        hip_half_width=0.15,
        leg=LegGeometry(abduction_offset=0.08, thigh_length=0.30, shank_length=0.30),
        nominal_height=0.42,
    )
