"""Pure accounting for a cut, shared by entities and geometry-only tests."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from .operations import difference, union_all
from shapely.geometry import Polygon

from .polygon_cleanup import PolygonCleanup


@dataclass(frozen=True)
class GeometryCarveResult:
    """Survivors and removed material, all in the input polygon's coordinates."""

    pieces: tuple[Polygon, ...]
    destroyed_area: float
    removed_material: tuple[Polygon, ...] = ()


def calculate_carve_result(
    hull: Polygon, pieces: Iterable[Polygon]
) -> GeometryCarveResult:
    """Validate a partition and measure material lost, never surviving fragments.

    A split is only worth its removed strip. Reject overlapping or newly added
    material, because either would let repeated hits create area or score twice.
    The tolerance covers floating-point overlay noise, not gameplay features.
    """
    pieces = tuple(pieces)
    if any(
        not isinstance(piece, Polygon)
        or piece.is_empty or not piece.is_valid or piece.area <= 0
        for piece in pieces
    ):
        raise ValueError("Carving must return valid polygons with positive area")
    survivors = union_all(pieces)
    tolerance = max(1.0, hull.area) * 1e-9
    if difference(survivors, hull).area > tolerance:
        raise ValueError("A cut cannot add material outside the original hull")
    if sum(piece.area for piece in pieces) - survivors.area > tolerance:
        raise ValueError("Surviving pieces must not overlap")
    removed = difference(hull, survivors)
    return GeometryCarveResult(
        pieces, removed.area, tuple(PolygonCleanup._polygon_parts(removed))
    )
