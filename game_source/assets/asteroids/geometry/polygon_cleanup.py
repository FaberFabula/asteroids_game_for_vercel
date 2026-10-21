"""Local removal of narrow sections near material removed by a cut."""

from collections.abc import Callable
from itertools import combinations
from math import isfinite

from pygame.math import Vector2
from .operations import difference, intersection, union_all
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient


DestroyCallback = Callable[[Polygon, str], None]
BoundaryEdge = tuple[Vector2, Vector2]


class PolygonCleanup:
    """A bounded local fracture heuristic, independent of physics and rewards."""

    def __init__(self, minimum_piece_area: float = 0.0) -> None:
        """Remember the minimum area for nearby fragments that survive cleanup."""
        if not isfinite(minimum_piece_area) or minimum_piece_area < 0:
            raise ValueError("minimum_piece_area must be finite and non-negative")
        self.minimum_piece_area = minimum_piece_area

    def cleanup(
        self,
        polygon: BaseGeometry,
        minimum_width: float,
        *,
        notch: BaseGeometry | None = None,
        influence_distance: float | None = None,
        on_destroy: DestroyCallback | None = None,
        max_passes: int = 64,
    ) -> list[Polygon]:
        """Remove thin nearby sections and undersized pieces after a real cut.

        ``notch`` is the material actually removed by the original cut, not the
        entire cutter polygon. Its buffered neighborhood limits this heuristic
        to the damaged region. No notch means no cleanup: explicit whole-object
        cleanup is deliberately left for a future feature.
        """
        if not isinstance(polygon, (Polygon, MultiPolygon, GeometryCollection)):
            raise TypeError("polygon must contain polygonal geometry")
        if not polygon.is_valid:
            raise ValueError("polygon must be valid")
        if not isfinite(minimum_width) or minimum_width < 0:
            raise ValueError("minimum_width must be finite and non-negative")
        if not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        if notch is None or polygon.is_empty:
            return self._polygon_parts(polygon)
        if not isinstance(notch, (Polygon, MultiPolygon, GeometryCollection)):
            raise TypeError("notch must contain polygonal geometry")
        if notch.is_empty:
            return self._polygon_parts(polygon)
        if not notch.is_valid or notch.area <= 0:
            raise ValueError("notch must be valid with positive area")
        if influence_distance is None:
            influence_distance = minimum_width
        if not isfinite(influence_distance) or influence_distance < 0:
            raise ValueError("influence_distance must be finite and non-negative")
        affected_region = notch.buffer(influence_distance) if influence_distance > 0 else notch

        remaining = polygon
        if minimum_width > 0:
            for _ in range(max_passes):
                cutters = self._narrow_cutters(remaining, minimum_width, affected_region)
                if not cutters:
                    break
                cleaned = difference(remaining, union_all(cutters))
                if cleaned.equals(remaining):
                    break
                remaining = cleaned
                if remaining.is_empty:
                    break
            else:
                raise RuntimeError("Width cleanup did not finish within max_passes")

        survivors = []
        destroyed = [
            (part, "width")
            for part in self._polygon_parts(difference(polygon, remaining))
        ]
        for part in self._polygon_parts(remaining):
            if part.area < self.minimum_piece_area and part.intersects(affected_region):
                destroyed.append((part, "area"))
            else:
                survivors.append(part)

        for part, reason in destroyed:
            self.destroy(part, reason=reason, on_destroy=on_destroy)
        return survivors

    @staticmethod
    def destroy(
        polygon: Polygon,
        *,
        reason: str,
        on_destroy: DestroyCallback | None = None,
    ) -> None:
        """Report discarded geometry; callers decide particles, drops, or sound."""
        if on_destroy is not None:
            on_destroy(polygon, reason)

    @staticmethod
    def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
        """Extract positive-area polygons recursively, ignoring lines and points."""
        if isinstance(geometry, Polygon):
            return [geometry] if not geometry.is_empty and geometry.area > 0 else []
        if isinstance(geometry, (MultiPolygon, GeometryCollection)):
            parts = []
            for child in geometry.geoms:
                parts.extend(PolygonCleanup._polygon_parts(child))
            return parts
        return []

    def _narrow_cutters(
        self, geometry: BaseGeometry, minimum_width: float,
        affected_region: BaseGeometry,
    ) -> list[Polygon]:
        """Find opposing walls, then use a corner fallback for each damaged part."""
        cutters = []
        for part in self._polygon_parts(geometry):
            if not part.intersects(affected_region):
                continue
            part = orient(part, sign=1.0)
            if self._whole_piece_is_narrow(part, minimum_width):
                cutters.append(part)
                continue
            part_cutters = []
            edges = []
            locations = []
            reflex_vertices = []
            for ring_index, ring in enumerate((part.exterior, *part.interiors)):
                coordinates = list(ring.coords)
                perimeter = ring.length
                distance = 0.0
                for first, second in zip(coordinates, coordinates[1:]):
                    start, end = Vector2(first), Vector2(second)
                    displacement = end - start
                    if displacement.length_squared() > 0:
                        edges.append((start, end))
                        locations.append((ring_index, distance, perimeter))
                        distance += displacement.length()

                vertices = [Vector2(point) for point in coordinates[:-1]]
                for index, vertex in enumerate(vertices):
                    incoming = vertex - vertices[index - 1]
                    outgoing = vertices[(index + 1) % len(vertices)] - vertex
                    if incoming.cross(outgoing) < -1e-10 * incoming.length() * outgoing.length():
                        reflex_vertices.append(vertex)

            edge_bounds = [
                (min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))
                for a, b in edges
            ]
            # For opposing edges the bisector's gap limit is at most sqrt(2)
            # times minimum_width. Far-apart edge boxes cannot form a neck.
            reach = minimum_width * 2 ** 0.5
            for first_index, second_index in combinations(range(len(edges)), 2):
                a, b = edge_bounds[first_index], edge_bounds[second_index]
                if (a[0] - b[2] > reach or b[0] - a[2] > reach
                        or a[1] - b[3] > reach or b[1] - a[3] > reach):
                    continue
                first, second = edges[first_index], edges[second_index]
                if first[0] in second or first[1] in second:
                    continue
                cutter = self.make_cutter(first, second, minimum_width)
                if cutter is not None:
                    first_location = locations[first_index]
                    second_location = locations[second_index]
                    if first_location[0] == second_location[0]:
                        coordinates = list(cutter.exterior.coords)[:-1]
                        if len(coordinates) == 3:
                            coordinates.append(coordinates[0])
                        boundary_distances = []
                        for first_point, second_point in (
                            (coordinates[0], coordinates[3]),
                            (coordinates[1], coordinates[2]),
                        ):
                            first_distance = first_location[1] + (
                                Vector2(first_point) - first[0]
                            ).length()
                            second_distance = second_location[1] + (
                                Vector2(second_point) - second[0]
                            ).length()
                            separation = abs(second_distance - first_distance)
                            boundary_distances.append(
                                min(separation, first_location[2] - separation)
                            )
                        if max(boundary_distances) < 2 * minimum_width:
                            continue
                    for clipped in self._polygon_parts(intersection(part, cutter)):
                        if clipped.intersects(affected_region):
                            part_cutters.append(clipped)
            if not part_cutters:
                part_cutters.extend(
                    self._vertex_neck_cutters(
                        part, edges, reflex_vertices, minimum_width, affected_region
                    )
                )
            cutters.extend(part_cutters)
        return cutters

    def _vertex_neck_cutters(
        self, polygon: Polygon, edges: list[BoundaryEdge],
        vertices: list[Vector2], minimum_width: float,
        affected_region: BaseGeometry,
    ) -> list[Polygon]:
        """Seed narrow necks at reflex corners when opposing edges find none."""
        cutters = []
        seen = set()
        for vertex in vertices:
            for start, end in edges:
                displacement = end - start
                fraction = (vertex - start).dot(displacement) / displacement.length_squared()
                closest = start + displacement * max(0.0, min(1.0, fraction))
                distance = vertex.distance_to(closest)
                if not minimum_width * 1e-10 < distance < minimum_width * (1 - 1e-10):
                    continue
                midpoint = (vertex + closest) / 2
                if not polygon.contains(Point(tuple(midpoint))):
                    continue
                chord = LineString((tuple(vertex), tuple(closest)))
                if abs(intersection(polygon, chord).length - distance) > minimum_width * 1e-8:
                    continue
                key = tuple(
                    round(value / minimum_width, 8)
                    for point in (vertex, closest)
                    for value in point
                )
                if key in seen:
                    continue
                seen.add(key)
                for cutter in self._profile_cutters(edges, vertex, closest, minimum_width):
                    for clipped in self._polygon_parts(intersection(polygon, cutter)):
                        if clipped.intersects(affected_region):
                            cutters.append(clipped)
        return cutters

    def _profile_cutters(
        self, edges: list[BoundaryEdge], first: Vector2,
        second: Vector2, minimum_width: float,
    ) -> list[Polygon]:
        """Follow the connected narrow-width interval surrounding one neck seed."""
        normal = (second - first).normalize()
        axis = Vector2(normal.y, -normal.x)
        midpoint = (first + second) / 2
        seed_projection = midpoint.dot(axis)
        center_height = midpoint.dot(normal)
        # Sweep along projected vertices, visiting only edges that actually
        # cross each interval. Rechecking every edge at every interval made
        # deeply carved outlines needlessly expensive without changing results.
        starts: dict[float, set[int]] = {}
        ends: dict[float, set[int]] = {}
        for index, edge in enumerate(edges):
            low, high = sorted(point.dot(axis) for point in edge)
            if high > low:
                starts.setdefault(low, set()).add(index)
                ends.setdefault(high, set()).add(index)
        projections = sorted(starts.keys() | ends.keys())
        active: set[int] = set()
        intervals = []
        for start, end in zip(projections, projections[1:]):
            active.difference_update(ends.get(start, ()))
            active.update(starts.get(start, ()))
            if end - start < minimum_width * 1e-10:
                continue
            projection = (start + end) / 2
            crossings = []
            for index in sorted(active):
                edge = edges[index]
                point = self._point_at_projection(edge, axis, projection)
                crossings.append((point.dot(normal), edge))
            crossings.sort(key=lambda crossing: crossing[0])
            for index in range(0, len(crossings) - 1, 2):
                lower_height, lower_edge = crossings[index]
                upper_height, upper_edge = crossings[index + 1]
                if not lower_height <= center_height <= upper_height:
                    continue
                cutter = self.make_cutter(
                    lower_edge, upper_edge, minimum_width,
                    axis=axis, projection_interval=(start, end),
                )
                if cutter is not None:
                    extent = [Vector2(point).dot(axis) for point in cutter.exterior.coords]
                    intervals.append((min(extent), max(extent), cutter))

        selected = []
        left = right = seed_projection
        epsilon = minimum_width * 1e-8
        while intervals:
            unselected = []
            changed = False
            for start, end, cutter in intervals:
                if start <= right + epsilon and end >= left - epsilon:
                    selected.append(cutter)
                    left, right = min(left, start), max(right, end)
                    changed = True
                else:
                    unselected.append((start, end, cutter))
            if not changed:
                break
            intervals = unselected
        return selected

    @staticmethod
    def _whole_piece_is_narrow(polygon: Polygon, minimum_width: float) -> bool:
        """Test minimum support-line width without changing the polygon outline."""
        vertices = [Vector2(point) for point in list(polygon.convex_hull.exterior.coords)[:-1]]
        for index, start in enumerate(vertices):
            displacement = vertices[(index + 1) % len(vertices)] - start
            if displacement.length_squared() == 0:
                continue
            normal = Vector2(-displacement.y, displacement.x).normalize()
            projections = [point.dot(normal) for point in vertices]
            if max(projections) - min(projections) < minimum_width * (1 - 1e-10):
                return True
        return False

    @staticmethod
    def _point_at_projection(
        edge: BoundaryEdge, axis: Vector2, projection: float
    ) -> Vector2:
        """Interpolate the point on an edge at the requested axis coordinate."""
        start, end = edge
        displacement = end - start
        fraction = (projection - start.dot(axis)) / displacement.dot(axis)
        return start + displacement * fraction

    @staticmethod
    def make_cutter(
        first: BoundaryEdge, second: BoundaryEdge, minimum_width: float,
        *, axis: Vector2 | None = None,
        projection_interval: tuple[float, float] | None = None,
    ) -> Polygon | None:
        """Build the strip where two opposing walls are closer than the limit."""
        if not isfinite(minimum_width) or minimum_width <= 0:
            raise ValueError("minimum_width must be finite and positive")
        first_start, first_end = first
        second_start, second_end = second
        first_displacement = first_end - first_start
        second_displacement = second_end - second_start
        if first_displacement.length_squared() == 0 or second_displacement.length_squared() == 0:
            return None
        first_direction = first_displacement.normalize()
        second_direction = second_displacement.normalize()
        automatic_axis = axis is None
        if automatic_axis and first_direction.dot(second_direction) >= -1e-10:
            return None

        axis = (
            (first_direction - second_direction).normalize()
            if axis is None
            else axis.normalize()
        )
        normal = Vector2(-axis.y, axis.x)
        start = max(
            min(first_start.dot(axis), first_end.dot(axis)),
            min(second_start.dot(axis), second_end.dot(axis)),
        )
        end = min(
            max(first_start.dot(axis), first_end.dot(axis)),
            max(second_start.dot(axis), second_end.dot(axis)),
        )
        if projection_interval is not None:
            start = max(start, projection_interval[0])
            end = min(end, projection_interval[1])
        if start >= end:
            return None

        first_a = PolygonCleanup._point_at_projection(first, axis, start)
        first_b = PolygonCleanup._point_at_projection(first, axis, end)
        second_a = PolygonCleanup._point_at_projection(second, axis, start)
        second_b = PolygonCleanup._point_at_projection(second, axis, end)
        gap_a = (second_a - first_a).dot(normal)
        gap_b = (second_b - first_b).dot(normal)
        gap_limit = minimum_width / first_direction.dot(axis) if automatic_axis else minimum_width
        epsilon = gap_limit * 1e-10
        if max(gap_a, gap_b) <= 0 or min(gap_a, gap_b) >= gap_limit - epsilon:
            return None

        change = gap_b - gap_a
        if abs(change) <= epsilon:
            lower, upper = 0.0, 1.0
        else:
            crossings = (-gap_a / change, (gap_limit - gap_a) / change)
            lower = max(0.0, min(crossings))
            upper = min(1.0, max(crossings))
            if lower >= upper:
                return None

        points = (
            first_a.lerp(first_b, lower),
            first_a.lerp(first_b, upper),
            second_a.lerp(second_b, upper),
            second_a.lerp(second_b, lower),
        )
        cutter = Polygon(tuple(tuple(point) for point in points))
        return cutter if cutter.is_valid and cutter.area > 0 else None
