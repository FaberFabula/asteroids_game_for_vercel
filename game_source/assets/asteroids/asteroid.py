"""The Box2D adapter for polygon asteroids and transactional replacement."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import Box2D
import pygame
from pygame.math import Vector2
from shapely.affinity import affine_transform, scale, translate
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from .geometry.carving import Carving
from .geometry.generation import Generation
from .geometry.operations import GeometryResolutionError
from .geometry.notch_settings import NotchSettings
from .geometry.polygon_cleanup import DestroyCallback
from .geometry.results import calculate_carve_result


VectorLike = tuple[float, float] | Vector2 | Box2D.b2Vec2


class FixtureGeometryError(ValueError):
    """A valid outline has features too small for native Box2D fixtures."""


@dataclass(frozen=True)
class CarveResult:
    """Replacement bodies, destroyed area, and removed polygons in world space."""

    pieces: tuple[Asteroid, ...]
    destroyed_area: float
    removed_material: tuple[Polygon, ...] = ()


class Asteroid:
    """Own one compound body; delegate all polygon operations to geometry."""

    def __init__(
        self,
        world: Box2D.b2World,
        hull: Polygon,
        position: VectorLike = (0.0, 0.0),
        velocity: VectorLike = (0.0, 0.0),
        angular_velocity: float = 0.0,
        density: float = 1.0,
        restitution: float = 0.5,
        friction: float = 0.3,
        *,
        angle: float = 0.0,
        hardness: float = 1.0,
        wave_id: int = 0,
    ) -> None:
        """Center a valid outline and build its convex fixtures while unlocked.

        Position denotes the centered hull's centroid, in world units. Hardness
        reduces carving depth independently of density, which controls mass.
        """
        if world.locked:
            raise RuntimeError("Create asteroids after world.Step(), not during callbacks")
        if not isinstance(hull, Polygon):
            raise TypeError("hull must be a Shapely Polygon")
        if hull.is_empty or not hull.is_valid or hull.area <= 0:
            raise ValueError("hull must be a valid polygon with positive area")

        position = Vector2(position)
        velocity = Vector2(velocity)
        values = (*position, *velocity, angle, angular_velocity, density, restitution, friction, hardness)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Motion and material settings must be finite")
        if hardness <= 0:
            raise ValueError("hardness must be positive")
        if not isinstance(wave_id, int) or wave_id < 0:
            raise ValueError("wave_id must be a non-negative integer")
        if density <= 0:
            raise ValueError("density must be positive")
        if not 0 <= restitution <= 1:
            raise ValueError("restitution must be between 0 and 1 inclusive")
        if friction < 0:
            raise ValueError("friction must be non-negative")

        self.world = world
        self.density = density
        self.restitution = restitution
        self.friction = friction
        self.hardness = hardness
        self.wave_id = wave_id

        center = hull.centroid
        self.hull = translate(hull, xoff=-center.x, yoff=-center.y)
        # Subtracting the centroid can round a microscopic gap into a crossing.
        # Treat that as unrepresentable fixture geometry so a cut can roll back
        # without destroying the original body or awarding removed-area points.
        if self.hull.is_empty or not self.hull.is_valid or self.hull.area <= 0:
            raise FixtureGeometryError("Centering the hull produced invalid fixture geometry")
        self.radius = max(Vector2(point).length() for point in list(self.hull.exterior.coords)[:-1])
        parts = Generation.convex_decomposition(self.hull)
        # Recentring may change the decomposition. Check its final parts before
        # calling native shape construction, where degenerate float32 input can
        # abort the process instead of raising a catchable Python exception.
        if not parts or any(
            not Generation._fixture_is_stable(part, Generation.MINIMUM_FEATURE)
            for part in parts
        ):
            raise FixtureGeometryError("Decomposition contains sub-tolerance fixture geometry")
        try:
            shapes = [
                Box2D.b2PolygonShape(vertices=Generation.fixture_vertices(part))
                for part in parts
            ]
            if not shapes or any(
                not shape.Validate() or shape.getMass(1.0).mass <= 0
                for shape in shapes
            ):
                raise ValueError("Decomposition produced an invalid fixture")
        except (AssertionError, ValueError) as error:
            raise FixtureGeometryError("A fixture is too small or degenerate for Box2D") from error

        self.body = world.CreateDynamicBody(
            position=tuple(position),
            angle=angle,
            userData=self,
        )
        try:
            for shape in shapes:
                self.body.CreateFixture(
                    shape=shape,
                    density=density,
                    friction=friction,
                    restitution=restitution,
                )
            self.body.linearVelocity = tuple(velocity)
            self.body.angularVelocity = angular_velocity
        except Exception:
            self.destroy()
            raise

    @property
    def alive(self) -> bool:
        """Whether this object still owns a live body after replacement or removal."""
        return self.body is not None

    @classmethod
    def randomly_generate_asteroid(
        cls,
        world: Box2D.b2World,
        vertices: int,
        max_radius: float,
        position: VectorLike = (0.0, 0.0),
        velocity: VectorLike = (0.0, 0.0),
        angular_velocity: float = 0.0,
        **body_settings: float | int,
    ) -> Asteroid:
        """Generate an irregular hull whose centered extent fits max_radius."""
        # Random outlines occasionally decompose into an unsafe thin fixture.
        # Retry before any body exists, rather than letting a wave crash play.
        last_error = None
        for _ in range(16):
            hull = Generation.generate_random_polygon(
                num_vertices=vertices,
                max_radius=max_radius,
            )
            center = hull.centroid
            extent = max(
                Vector2(x - center.x, y - center.y).length()
                for x, y in list(hull.exterior.coords)[:-1]
            )
            if extent > max_radius:
                factor = max_radius / extent
                hull = scale(hull, xfact=factor, yfact=factor, origin=(center.x, center.y))
            try:
                return cls(
                    world=world,
                    hull=hull,
                    position=position,
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    **body_settings,
                )
            except FixtureGeometryError as error:
                last_error = error
        raise FixtureGeometryError("Could not generate a stable asteroid after 16 attempts") from last_error

    def get_world_vertices(self) -> list[tuple[float, float]]:
        """Return the outer outline transformed by the current body pose."""
        return self.get_world_rings()[0]

    def get_world_rings(self, offset: VectorLike = (0.0, 0.0)) -> list[list[tuple[float, float]]]:
        """Transform the exterior and holes, optionally offset for wrapped drawing."""
        body = self._require_body()
        offset = Vector2(offset)
        return [
            [
                tuple(Vector2(body.GetWorldPoint(vertex)) + offset)
                for vertex in list(ring.coords)[:-1]
            ]
            for ring in (self.hull.exterior, *self.hull.interiors)
        ]

    def draw(
        self,
        surface: pygame.Surface,
        world_to_screen: Callable[[VectorLike], tuple[int, int]],
        *,
        color: tuple[int, int, int] = (170, 180, 195),
        width: int = 2,
        offset: VectorLike = (0.0, 0.0),
    ) -> None:
        """Draw every boundary ring, so cuts and holes remain visible."""
        if not self.alive:
            return
        for ring in self.get_world_rings(offset):
            pygame.draw.lines(
                surface, color, True,
                [world_to_screen(point) for point in ring], width,
            )

    def get_local_hit(
        self, world_point: VectorLike, world_impulse: VectorLike
    ) -> tuple[Vector2, Vector2]:
        """Transform a hit and snap its point to the true outline boundary."""
        body = self._require_body()
        world_point = Vector2(world_point)
        world_impulse = Vector2(world_impulse)
        if not all(math.isfinite(value) for value in (*world_point, *world_impulse)):
            raise ValueError("Hit coordinates and impulse must be finite")
        local_point = Point(tuple(body.GetLocalPoint(tuple(world_point))))
        boundary_point = nearest_points(self.hull.boundary, local_point)[0]
        return (
            Vector2(boundary_point.coords[0]),
            Vector2(body.GetLocalVector(tuple(world_impulse))),
        )

    def carve(
        self,
        local_point: VectorLike,
        local_impulse: VectorLike,
        carver: Carving,
        settings: NotchSettings | None = None,
        *,
        on_destroy: DestroyCallback | None = None,
        influence_distance: float | None = None,
    ) -> list[Asteroid]:
        """Compatibility wrapper returning only the replacement bodies."""
        result = self.carve_with_result(
            local_point,
            local_impulse,
            carver,
            settings,
            on_destroy=on_destroy,
            influence_distance=influence_distance,
        )
        return list(result.pieces)

    def carve_with_result(
        self,
        local_point: VectorLike,
        local_impulse: VectorLike,
        carver: Carving,
        settings: NotchSettings | None = None,
        *,
        on_destroy: DestroyCallback | None = None,
        influence_distance: float | None = None,
    ) -> CarveResult:
        """Carve after Step and replace atomically; removed material is world-space.

        Strength is divided by hardness, with no additional physical impulse:
        Box2D already resolved the impact. A native fixture failure leaves the
        original intact and awards zero destroyed area, instead of crashing or
        silently filling the notch with a convex hull.
        """
        self._require_body()
        if self.world.locked:
            raise RuntimeError("Queue carving until after world.Step()")
        local_point = Vector2(local_point)
        local_impulse = Vector2(local_impulse)
        if not all(math.isfinite(value) for value in (*local_point, *local_impulse)):
            raise ValueError("Hit coordinates and impulse must be finite")
        magnitude = local_impulse.length()
        if magnitude == 0:
            return CarveResult((self,), 0.0)

        notifications = []
        try:
            pieces = carver.carve(
                hull=self.hull,
                POI=local_point,
                direction_of_impact=local_impulse,
                impulse_magnitude=magnitude / self.hardness,
                settings=settings,
                on_destroy=(
                    lambda polygon, reason: notifications.append((polygon, reason))
                ) if on_destroy is not None else None,
                influence_distance=influence_distance,
            )
        except GeometryResolutionError:
            # A numerically unresolved cut leaves material and rewards intact.
            return CarveResult((self,), 0.0)
        if len(pieces) == 1 and pieces[0].equals(self.hull):
            return CarveResult((self,), 0.0)

        geometry_result = calculate_carve_result(self.hull, pieces)
        world_removed = tuple(
            self._world_polygon(part) for part in geometry_result.removed_material
        )
        try:
            replacements = self._replace_with_pieces(list(geometry_result.pieces))
        except (FixtureGeometryError, GeometryResolutionError):
            return CarveResult((self,), 0.0)
        if on_destroy is not None:
            for polygon, reason in notifications:
                on_destroy(polygon, reason)
        return CarveResult(
            tuple(replacements), geometry_result.destroyed_area, world_removed
        )

    def _replace_with_pieces(self, pieces: list[Polygon]) -> list[Asteroid]:
        """Parent is kept until all children are generated or fail; conserves angular momentum"""
        body = self._require_body()
        if any(
            not isinstance(piece, Polygon)
            or piece.is_empty or not piece.is_valid or piece.area <= 0
            for piece in pieces
        ):
            raise ValueError("Carving must return valid polygons with positive area")
        replacements = []
        try:
            for piece in pieces:
                center = piece.centroid
                position = body.GetWorldPoint((center.x, center.y))
                #Box2d does heavy lifting
                velocity = body.GetLinearVelocityFromWorldPoint(position)
                replacements.append(
                    type(self)(
                        world=self.world,
                        hull=piece,
                        position=tuple(position),
                        velocity=tuple(velocity),
                        angular_velocity=body.angularVelocity,
                        density=self.density,
                        restitution=self.restitution,
                        friction=self.friction,
                        angle=body.angle,
                        hardness=self.hardness,
                        wave_id=self.wave_id,
                    )
                )
        except Exception:
            for replacement in replacements:
                replacement.destroy()
            raise
        self.destroy()
        return replacements

    def _world_polygon(self, polygon: Polygon) -> Polygon:
        """Adapt local removed material to the authoritative body transform."""
        body = self._require_body()
        cosine, sine = math.cos(body.angle), math.sin(body.angle)
        return affine_transform(
            polygon,
            [cosine, -sine, sine, cosine, body.position.x, body.position.y],
        )

    def destroy(self) -> None:
        """Idempotently release the native body while the world is unlocked."""
        if not self.alive:
            return
        if self.world.locked:
            raise RuntimeError("Destroy asteroids after world.Step()")
        self.body.userData = None
        self.world.DestroyBody(self.body)
        self.body = None

    def _require_body(self) -> Box2D.b2Body:
        """Fail clearly when an obsolete asteroid is used after replacement."""
        if not self.alive:
            raise RuntimeError("This asteroid has already been destroyed or replaced")
        return self.body
