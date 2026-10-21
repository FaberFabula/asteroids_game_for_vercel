"""Cut polygons and account for lost area without touching a physics world."""

from __future__ import annotations

import random
from math import isfinite

import shapely
from pygame.math import Vector2

from .generation import Generation
from .operations import difference, intersection
from .results import GeometryCarveResult, calculate_carve_result
from .notch_settings import NotchSettings
from .polygon_cleanup import DestroyCallback, PolygonCleanup
from .profiles import NOTCH_PROFILES


class Carving:
    """Convert a damage-like strength into a local polygon notch.

    ``impulse_magnitude`` is a historical geometry API name. The game supplies
    weapon damage as strength; this class never applies a physical impulse.
    """

    def __init__(
        self, depth_per_impulse: float, mouth_width: float,
        break_thickness: float, minimum_piece_area: float,
    ) -> None:
        """Configure notch depth, width, fracture width, and surviving piece area."""
        if not isfinite(depth_per_impulse) or depth_per_impulse < 0:
            raise ValueError("depth_per_impulse must be a finite non-negative number")
        self.depth_per_impulse = depth_per_impulse

        if not isfinite(mouth_width) or mouth_width < 0:
            raise ValueError("mouth_width must be a finite non-negative number")
        self.mouth_width = mouth_width

        if not isfinite(break_thickness) or break_thickness < 0:
            raise ValueError("break_thickness must be a finite non-negative number")
        self.break_thickness = break_thickness

        if not isfinite(minimum_piece_area) or minimum_piece_area < 0:
            raise ValueError("minimum_piece_area must be a finite non-negative number")
        self.minimum_piece_area = minimum_piece_area

        self.cleaner = PolygonCleanup(minimum_piece_area)

    def _make_notch(
        self,
        POI: Vector2,
        direction_of_impact: Vector2,
        smooth_depth: float,
        smooth_width: float,
        settings: NotchSettings,
    ) -> shapely.Polygon:
        """Transform a unit profile along the impact and optional jagged boundary."""
        if not isfinite(smooth_depth) or smooth_depth <= 0:
            raise ValueError("smooth_depth must be a finite positive number")
        if not isfinite(smooth_width) or smooth_width <= 0:
            raise ValueError("smooth_width must be a finite positive number")

        normalized_DOI = direction_of_impact / direction_of_impact.length()
        perpendicular_DOI = Vector2(-normalized_DOI.y, normalized_DOI.x)
        base_points = [
            POI
            + normalized_DOI * smooth_depth * profile[0]
            + perpendicular_DOI * smooth_width * profile[1]
            for profile in NOTCH_PROFILES[settings.notch_style]
        ]

        if settings.spacing_frac is not None:
            new_points_intervals = []
            cutting_perimeter = sum(
                (base_points[i + 1] - base_points[i]).length()
                for i in range(len(base_points) - 1)
            )
            spacing = cutting_perimeter * settings.spacing_frac
            distance_to_next_vertex = (
                spacing * settings.offset if settings.offset is not None else 0
            )

            for i in range(len(base_points) - 1):
                p1 = base_points[i]
                p2 = base_points[i + 1]
                boundary_vector = p2 - p1
                boundary_length = boundary_vector.length()
                if boundary_length == 0:
                    new_points_intervals.append([])
                    continue

                perpendicular_boundary = (
                    Vector2(-boundary_vector.y, boundary_vector.x) / boundary_length
                )
                new_points = []
                while distance_to_next_vertex < boundary_length:
                    if random.random() < settings.roughness:
                        t = distance_to_next_vertex / boundary_length
                        new_point = (
                            p1
                            + boundary_vector * t
                            + perpendicular_boundary
                            * settings.jaggedness
                            * 2
                            * (random.random() - 0.5)
                        )
                        new_points.append(new_point)
                    distance_to_next_vertex += spacing

                distance_to_next_vertex -= boundary_length
                new_points_intervals.append(new_points)

            new_base_points = []
            for i in range(len(base_points) - 1):
                new_base_points.append(base_points[i])
                new_base_points.extend(new_points_intervals[i])
            new_base_points.append(base_points[-1])
            base_points = new_base_points

        polygon = shapely.Polygon(base_points)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise ValueError("Generated notch must be a valid polygon with positive area")
        return polygon

    def carve(
        self,
        hull: shapely.Polygon,
        POI: Vector2,
        direction_of_impact: Vector2,
        impulse_magnitude: float,
        settings: NotchSettings | None = None,
        *,
        on_destroy: DestroyCallback | None = None,
        influence_distance: float | None = None,
    ) -> list[shapely.Polygon]:
        """Subtract a notch, clean locally, then discard sub-physics-scale slivers."""
        if hull.is_empty or not hull.is_valid or hull.area <= 0:
            raise ValueError("hull must be a valid polygon with positive area")
        if not all(
            isfinite(value)
            for value in (POI.x, POI.y, direction_of_impact.x, direction_of_impact.y)
        ):
            raise ValueError("Impact point and direction must have finite coordinates")
        if direction_of_impact.length() == 0:
            raise ValueError("direction_of_impact must be a non-zero vector")
        if not isfinite(impulse_magnitude) or impulse_magnitude < 0:
            raise ValueError("impulse_magnitude must be a finite non-negative number")

        if settings is None:
            settings = NotchSettings()

        smooth_depth = self.depth_per_impulse * impulse_magnitude
        smooth_width = self.mouth_width
        if smooth_depth == 0 or smooth_width == 0:
            return [hull]

        notch_polygon = self._make_notch(
            POI=POI,
            direction_of_impact=direction_of_impact,
            smooth_depth=smooth_depth,
            smooth_width=smooth_width,
            settings=settings,
        )
        removed_notch = intersection(hull, notch_polygon)
        if removed_notch.area <= 0:
            return [hull]

        carved_hull = difference(hull, notch_polygon)
        survivors = self.cleaner.cleanup(
            carved_hull,
            minimum_width=self.break_thickness,
            notch=removed_notch,
            influence_distance=influence_distance,
            on_destroy=on_destroy,
        )
        # Native fixtures need a practical minimum feature size. Subtraction
        # here makes any dust explicit geometry loss, so scoring stays honest.
        prepared = []
        for survivor in survivors:
            for piece in Generation.remove_small_features(survivor):
                if piece.area >= self.minimum_piece_area:
                    prepared.append(piece)
                elif on_destroy is not None:
                    on_destroy(piece, "area")
        return prepared

    def carve_with_result(
        self,
        hull: shapely.Polygon,
        POI: Vector2,
        direction_of_impact: Vector2,
        impulse_magnitude: float,
        settings: NotchSettings | None = None,
        *,
        on_destroy: DestroyCallback | None = None,
        influence_distance: float | None = None,
    ) -> GeometryCarveResult:
        """Return local surviving and removed polygons with conserved area."""
        pieces = self.carve(
            hull, POI, direction_of_impact, impulse_magnitude, settings,
            on_destroy=on_destroy, influence_distance=influence_distance,
        )
        return calculate_carve_result(hull, pieces)
