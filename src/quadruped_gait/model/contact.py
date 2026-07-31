"""Contact state of the four feet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quadruped_gait.model.geometry import LEG_COUNT, LEG_NAMES, LegId

__all__ = ["ContactState"]


@dataclass(frozen=True, slots=True)
class ContactState:
    """Which feet are loaded, in the canonical leg ordering.

    Attributes:
        contacts: One flag per leg, ``True`` when the foot is in stance.
    """

    contacts: tuple[bool, bool, bool, bool]

    def __post_init__(self) -> None:
        if len(self.contacts) != LEG_COUNT:
            raise ValueError(f"expected {LEG_COUNT} contact flags, got {len(self.contacts)}")

    @classmethod
    def from_iterable(cls, values: object) -> ContactState:
        """Build a contact state from any iterable of four truthy values."""
        flags = tuple(bool(value) for value in np.asarray(values, dtype=bool).reshape(-1))
        if len(flags) != LEG_COUNT:
            raise ValueError(f"expected {LEG_COUNT} contact flags, got {len(flags)}")
        return cls(contacts=(flags[0], flags[1], flags[2], flags[3]))

    @property
    def stance_legs(self) -> tuple[LegId, ...]:
        """Return the legs currently in stance."""
        return tuple(leg for leg in LegId if self.contacts[int(leg)])

    @property
    def swing_legs(self) -> tuple[LegId, ...]:
        """Return the legs currently in swing."""
        return tuple(leg for leg in LegId if not self.contacts[int(leg)])

    @property
    def stance_count(self) -> int:
        """Return how many feet are in stance."""
        return sum(self.contacts)

    def as_array(self) -> NDArray[np.bool_]:
        """Return the contact flags as a boolean array of shape ``(4,)``."""
        return np.asarray(self.contacts, dtype=np.bool_)

    def __str__(self) -> str:
        pairs = zip(LEG_NAMES, self.contacts, strict=True)
        return " ".join(name if loaded else "." * len(name) for name, loaded in pairs)
