"""Consistent precision for polygon overlays, independent of game entities.

Near-coincident cut edges can make GEOS 3.12's floating-point overlay throw a
native exception in WebAssembly. A shared 1e-12 world-unit grid gives every
boolean operation the same precision, far below a pixel or Box2D's weld limit.
"""
from __future__ import annotations

from collections.abc import Iterable

import shapely
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry

OVERLAY_GRID = 1e-12


class GeometryResolutionError(ValueError):
    """A valid outline cannot be partitioned within the geometry tolerance."""


def _overlay_input(geometry: BaseGeometry) -> BaseGeometry:
    """Remove lower-dimensional dust from collections before a grid overlay.

    A collapsed cut edge can leave polygons mixed with lines or points. GEOS
    rejects mixed-dimensional grid inputs. Keep the highest-dimensional parts;
    ordinary LineString operands still work for the cleanup chord-length check.
    """
    if not isinstance(geometry, GeometryCollection):
        return geometry
    parts = []
    pending = list(geometry.geoms)
    while pending:
        part = pending.pop()
        if isinstance(part, GeometryCollection):
            pending.extend(part.geoms)
        elif not part.is_empty:
            parts.append(part)
    if not parts:
        return GeometryCollection()
    dimension = max(shapely.get_dimensions(part) for part in parts)
    return shapely.union_all(
        [part for part in parts if shapely.get_dimensions(part) == dimension],
        grid_size=OVERLAY_GRID,
    )


def intersection(first: BaseGeometry, second: BaseGeometry) -> BaseGeometry:
    """Intersect material on the common grid, avoiding almost-coincident edges."""
    return shapely.intersection(_overlay_input(first), _overlay_input(second), grid_size=OVERLAY_GRID)


def difference(first: BaseGeometry, second: BaseGeometry) -> BaseGeometry:
    """Subtract material using the same precision as intersections and unions."""
    return shapely.difference(_overlay_input(first), _overlay_input(second), grid_size=OVERLAY_GRID)


def union_all(polygons: Iterable[BaseGeometry]) -> BaseGeometry:
    """Join material without changing precision between successive cut stages."""
    return _overlay_input(GeometryCollection(list(polygons)))
