"""Unit-depth, unit-width notch outlines; carving transforms these locally."""

import math

NOTCH_PROFILES = {
    "triangle": ((0.0, -0.5), (1.0, 0.0), (0.0, 0.5)),
    "rounded": tuple(
        (math.cos(theta), 0.5 * math.sin(theta))
        for theta in (-math.pi / 2 + math.pi * t / 10 for t in range(11))
    ),
    "rectangle": ((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)),
}
