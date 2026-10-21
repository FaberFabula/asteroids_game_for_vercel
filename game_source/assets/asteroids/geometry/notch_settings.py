"""Validated settings for the optional shape irregularity of a cut."""

import dataclasses
from math import isfinite

from .profiles import NOTCH_PROFILES

@dataclasses.dataclass(frozen=True)
class NotchSettings:
    """Select a notch profile and optional spaced random boundary displacement."""

    notch_style : str = "triangle"
    jaggedness : float = 0.0
    roughness : float = 0.0
    spacing_frac : float | None = None
    offset : float | None = None

    def __post_init__(self) -> None:
        """Reject invalid profiles and numeric ranges before constructing geometry."""
        if self.notch_style not in NOTCH_PROFILES:
            raise ValueError(f"Invalid notch style: {self.notch_style}. Must be one of {list(NOTCH_PROFILES)}")
        if not isfinite(self.jaggedness):
            raise ValueError("jaggedness must be a finite number")
        if not (0 <= self.roughness <= 1):
            raise ValueError("roughness must be between 0 and 1 inclusive")
        if self.offset is not None and not (0 <= self.offset < 1):
            raise ValueError("offset must be in [0, 1)")
        if self.spacing_frac is not None and not (0 < self.spacing_frac < 1):
            raise ValueError("spacing_frac must be between 0 and 1 exclusive")
