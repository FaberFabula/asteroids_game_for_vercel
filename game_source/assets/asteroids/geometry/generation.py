"""Polygon generation and conservative preparation for convex fixtures."""

from __future__ import annotations

import math
import random
import struct
from collections import deque

import shapely
from shapely.geometry import Polygon
from shapely.ops import triangulate

from .operations import GeometryResolutionError, intersection, union_all
from .polygon_cleanup import PolygonCleanup


class Generation:
    """Geometry helpers; the eight-vertex limit applies to each fixture only."""

    # Legacy Box2D welds vertices closer than 0.0025 world units. A small safety
    # margin also avoids almost-collinear triangles in its float32 calculations.
    MINIMUM_FEATURE = 0.006

    @staticmethod
    def _triangles(polygon: Polygon) -> list[Polygon]:
        """Use exact constrained triangulation, or a Shapely 2.0 web fallback.

        Browser wheels currently carry Shapely 2.0. Clipping every Delaunay cell
        to the original material preserves concavities and holes, rather than
        discarding boundary-crossing triangles and leaving unfilled gaps.
        """
        constrained = getattr(shapely, "constrained_delaunay_triangles", None)
        if constrained is not None:
            return list(constrained(polygon).geoms)
        # Nearly duplicate input vertices can make GEOS's unconstrained cells
        # overlap. Retry only that ambiguous case with a tiny Delaunay snapping
        # tolerance. Every cell is still clipped to the unchanged input outline.
        for tolerance in (0.0, max(1.0, math.sqrt(polygon.area)) * 1e-12):
            result = Generation._web_cells(polygon, tolerance)
            covered_area = sum(part.area for part in result)
            if abs(covered_area - polygon.area) <= max(1, polygon.area) * 1e-9:
                return result
        raise GeometryResolutionError("Web triangulation must preserve all material")

    @staticmethod
    def _web_cells(polygon: Polygon, tolerance: float) -> list[Polygon]:
        """Clip Delaunay cells conservatively, discarding only numerical dust."""
        result = []
        # Repeated floating-point cuts can leave vertices almost coincident.
        # Delaunay clipping then emits cells with effectively zero area whose
        # last-bit noise may fail exact convexity tests. Discard only this dust;
        # never fill a substantive concavity by replacing it with a convex hull.
        dust_area = max(1.0, polygon.area) * 1e-14
        for triangle in triangulate(polygon, tolerance=tolerance):
            for part in PolygonCleanup._polygon_parts(intersection(triangle, polygon)):
                if part.area <= dust_area:
                    continue
                if part.interiors or not part.equals(part.convex_hull):
                    # A clipped cell can retain a last-bit reflex corner next
                    # to almost identical vertices. Subdivide it, keeping only
                    # triangles wholly inside the actual material. The area
                    # check rejects a substantive gap; using a convex hull here
                    # would instead risk filling a real notch or hole.
                    interior = [
                        cell for cell in triangulate(part)
                        if cell.area > dust_area and part.covers(cell)
                    ]
                    covered_area = sum(cell.area for cell in interior)
                    if abs(covered_area - part.area) > max(1, part.area) * 1e-12:
                        raise GeometryResolutionError("Web triangulation must preserve each clipped cell")
                    result.extend(interior)
                    continue
                result.append(part)
        return result

    @staticmethod
    def convex_decomposition(polygon: Polygon) -> list[Polygon]:
        """Cover the exact outline and holes with convex parts of <=8 vertices.

        Constrained triangulation respects concavities. Merging neighboring
        triangles reduces fixture count and avoids unnecessary narrow fixtures;
        testing convexity never replaces the cut outline with its convex hull.
        """
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise ValueError("polygon must be valid with positive area")
        polygon = polygon.simplify(0)
        if polygon.equals(polygon.convex_hull) and len(polygon.exterior.coords) <= 9:
            return [polygon]
        # Only neighbors can merge. An edge map avoids trying every pair again
        # after every successful merge, which becomes costly after many cuts.
        parts = dict(enumerate(Generation._triangles(polygon)))
        boundaries = {}
        owners = {}
        for index, part in parts.items():
            points = list(part.exterior.coords)
            edges = {tuple(sorted((a, b))) for a, b in zip(points, points[1:])}
            boundaries[index] = edges
            for edge in edges:
                owners.setdefault(edge, set()).add(index)
        candidates = deque(tuple(sorted(pair)) for pair in owners.values() if len(pair) == 2)
        next_index = len(parts)
        while candidates:
            first_id, second_id = candidates.popleft()
            if first_id not in parts or second_id not in parts:
                continue
            merged = union_all((parts[first_id], parts[second_id])).simplify(0)
            if not (isinstance(merged, Polygon) and not merged.interiors
                    and len(merged.exterior.coords) <= 9
                    and merged.equals(merged.convex_hull)):
                continue
            # Keep the original triangulation's boundary segments. Simplifying
            # collinear vertices for Box2D must not lose neighboring edge IDs.
            edges = boundaries[first_id] ^ boundaries[second_id]
            for old_id in (first_id, second_id):
                for edge in boundaries.pop(old_id):
                    owners[edge].discard(old_id)
                del parts[old_id]
            parts[next_index] = merged
            boundaries[next_index] = edges
            neighbors = set()
            for edge in edges:
                neighbors.update(owners[edge])
                owners[edge].add(next_index)
            candidates.extend((neighbor, next_index) for neighbor in sorted(neighbors))
            next_index += 1
        return list(parts.values())

    @staticmethod
    def remove_small_features(
        polygon: Polygon, minimum_feature: float = MINIMUM_FEATURE
    ) -> list[Polygon]:
        """Discard sub-tolerance slivers without filling any carved-out notch.

        Keep a safely representable convex decomposition as-is. If tiny edges
        or heights remain, discard only unsafe *triangles*, then reunite the
        survivors. This conservative subtraction is part of the cut and is
        included in destroyed-area accounting. It is smaller than one pixel at
        the game's scale, but prevents native Box2D assertion failures.
        """
        if minimum_feature <= 0 or not math.isfinite(minimum_feature):
            raise ValueError("minimum_feature must be finite and positive")
        if all(
            Generation._fixture_is_stable(part, minimum_feature)
            for part in Generation.convex_decomposition(polygon)
        ):
            return [polygon]
        triangles = Generation._triangles(polygon)
        stable = [
            triangle for triangle in triangles
            if Generation._fixture_is_stable(triangle, minimum_feature)
        ]
        return PolygonCleanup._polygon_parts(union_all(stable))

    @staticmethod
    def _fixture_is_stable(polygon: Polygon, minimum_feature: float) -> bool:
        """Check vertex spacing and minimum altitude before native float32 use."""
        points = list(polygon.exterior.coords)[:-1]
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            edge_length = math.dist(start, end)
            if edge_length < minimum_feature:
                return False
            dx, dy = end[0] - start[0], end[1] - start[1]
            height = max(
                abs(dx * (point[1] - start[1]) - dy * (point[0] - start[0]))
                / edge_length for point in points
            )
            if height < minimum_feature:
                return False
        return True

    @staticmethod
    def fixture_vertices(polygon: Polygon) -> list[tuple[float, float]]:
        """Prepare one convex part at the float32 precision used by Box2D.

        Rounding a nearly straight corner can make a convex part reflex. Legacy
        Box2D's hull builder can then crash instead of raising an exception.
        Round first, remove only numerical corner noise, and check the result
        before entering native code. This affects the collision fixture only;
        the precise carved outline remains authoritative for rendering/scoring.
        Concave outlines must be decomposed before calling this helper.
        """
        if (polygon.is_empty or not polygon.is_valid or polygon.interiors
                or not polygon.equals(polygon.convex_hull)
                or not 3 <= len(polygon.exterior.coords) - 1 <= 8):
            raise ValueError("A fixture must be a valid convex polygon of 3 to 8 vertices")
        try:
            points = [
                struct.unpack("ff", struct.pack("ff", x, y))
                for x, y in list(polygon.exterior.coords)[:-1]
            ]
        except (OverflowError, struct.error) as error:
            raise ValueError("Fixture coordinates exceed float32 range") from error
        if not all(math.isfinite(value) for point in points for value in point):
            raise ValueError("Fixture coordinates must be finite")
        # This convex hull applies only to a part already proven convex, at its
        # final storage precision. It never fills a notch in the asteroid hull.
        rounded = Polygon(points).convex_hull
        if not isinstance(rounded, Polygon):
            raise ValueError("Fixture collapsed when rounded to float32")
        points = list(rounded.exterior.coords)[:-1]
        tolerance = max(1.0, *(abs(value) for point in points for value in point)) * 2**-20
        while len(points) > 3:
            for index, point in enumerate(points):
                before, after = points[index - 1], points[(index + 1) % len(points)]
                chord = math.dist(before, after)
                cross = ((point[0] - before[0]) * (after[1] - before[1])
                         - (point[1] - before[1]) * (after[0] - before[0]))
                if chord == 0 or abs(cross) <= tolerance * chord:
                    points.pop(index)
                    break
            else:
                break
        prepared = Polygon(points)
        if (not prepared.is_valid or not prepared.equals(prepared.convex_hull)
                or not Generation._fixture_is_stable(prepared, Generation.MINIMUM_FEATURE)
                or abs(prepared.area - polygon.area) > max(1, polygon.area) * 1e-5):
            raise ValueError("Fixture cannot be represented safely at float32 precision")
        return points

    @staticmethod
    def generate_random_polygon(num_vertices: int, max_radius: float) -> Polygon:
        """Build a star-shaped, possibly concave outline with well-spaced angles.

        Jittered angular sectors avoid almost coincident vertices in the original
        unrestricted-angle generator. Radius variation retains irregular shapes.
        """
        if not isinstance(num_vertices, int) or num_vertices < 3:
            raise ValueError("num_vertices must be an integer of at least three")
        if not math.isfinite(max_radius) or max_radius <= 0:
            raise ValueError("max_radius must be finite and positive")
        angles = [
            (index + random.uniform(-0.2, 0.2)) * math.tau / num_vertices
            for index in range(num_vertices)
        ]
        radii = [random.uniform(0.5, 1.0) * max_radius for _ in angles]
        return Polygon([
            (radius * math.cos(angle), radius * math.sin(angle))
            for radius, angle in zip(radii, angles)
        ])
