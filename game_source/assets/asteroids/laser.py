"""Instant, screen-clipped laser queries with simulation-time damage ticks."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import Box2D
import pygame
from pygame.math import Vector2


@dataclass(frozen=True)
class LaserHit:
    """A query result that can safely be consumed after RayCast returns."""
    target: object
    point: tuple[float, float]
    normal: tuple[float, float]
    direction: tuple[float, float]
    damage: float


class _ClosestHit(Box2D.b2RayCastCallback):
    """Shorten the queried segment as nearer non-sensor fixtures are found."""

    def __init__(self, ignore_body: Box2D.b2Body | None) -> None:
        """Exclude the shooter's body."""
        super().__init__()
        self.ignore_body = ignore_body
        self.result = None

    def ReportFixture(self, fixture: Box2D.b2Fixture, point: tuple[float, float],
                      normal: tuple[float, float], fraction: float) -> float:
        """Return -1 to skip a fixture, or its fraction to clip the ray."""
        if fixture.sensor or fixture.body == self.ignore_body:
            return -1.0
        self.result = (fixture.body.userData, tuple(point), tuple(normal))
        return fraction


class LaserWeapon:
    """Maintain tick cadence separately from rendering the current beam."""
    def __init__(
        self,
        world: Box2D.b2World,
        bounds: tuple[float, float, float, float],
        *,
        ticks_per_second: float = 60.0,
        damage_per_tick: float = 0.02,
    ) -> None:
        """Configure finite world bounds, tick frequency, and damage per tick."""
        if len(bounds) != 4 or not all(isfinite(value) for value in bounds):
            raise ValueError("bounds must be finite (left, bottom, right, top) coordinates")
        left, bottom, right, top = bounds
        if left >= right or bottom >= top:
            raise ValueError("bounds must have positive width and height")
        if not isfinite(ticks_per_second) or ticks_per_second <= 0:
            raise ValueError("ticks_per_second must be finite and positive")
        if not isfinite(damage_per_tick) or damage_per_tick < 0:
            raise ValueError("damage_per_tick must be finite and non-negative")
        self.world = world
        self.bounds = tuple(bounds)
        self.tick_interval = 1.0 / ticks_per_second
        self.damage_per_tick = damage_per_tick
        self._until_tick = 0.0
        self.beam = None

    def update(
        self,
        active_dt: float,
        origin: tuple[float, float],
        direction: tuple[float, float],
        *,
        firing: bool,
        ignore_body: Box2D.b2Body | None = None,
        on_hit: Callable[[LaserHit], None] | None = None,
    ) -> int:
        """Use only the active part of dt; recast after each destructive hit."""
        if not isfinite(active_dt) or active_dt < 0:
            raise ValueError("active_dt must be finite and non-negative")
        self.beam = None
        if not firing or active_dt == 0:
            self._until_tick = 0.0
            return 0
        if self.world.locked:
            raise RuntimeError("Fire the laser after world.Step()")

        origin, direction = Vector2(origin), Vector2(direction)
        if not all(isfinite(value) for value in (*origin, *direction)):
            raise ValueError("Laser origin and direction must be finite")
        if direction.length_squared() == 0:
            raise ValueError("Laser direction must be non-zero")
        direction = direction.normalize()
        end = self._screen_edge(origin, direction)
        if end is None or origin.distance_squared_to(end) == 0:
            self._until_tick = 0.0
            return 0

        ticks = 0
        remaining = active_dt
        while remaining > self._until_tick + 1e-12:
            remaining -= self._until_tick
            self._until_tick = self.tick_interval
            hit = self._cast(origin, end, direction, ignore_body)
            ticks += 1
            if hit is not None and on_hit is not None:
                on_hit(hit)
        self._until_tick = max(0.0, self._until_tick - remaining)
        self._cast(origin, end, direction, ignore_body)
        return ticks

    def _screen_edge(self, origin: Vector2, direction: Vector2) -> Vector2 | None:
        """Intersect the forward ray with its nearest visible world boundary."""
        left, bottom, right, top = self.bounds
        if not (left <= origin.x <= right and bottom <= origin.y <= top):
            return None
        distances = []
        if direction.x > 0:
            distances.append((right - origin.x) / direction.x)
        elif direction.x < 0:
            distances.append((left - origin.x) / direction.x)
        if direction.y > 0:
            distances.append((top - origin.y) / direction.y)
        elif direction.y < 0:
            distances.append((bottom - origin.y) / direction.y)
        return origin + direction * min(distances)

    def _cast(self, origin: Vector2, end: Vector2, direction: Vector2,
              ignore_body: Box2D.b2Body | None) -> LaserHit | None:
        """Query the closest body and remember the visible beam segment."""
        callback = _ClosestHit(ignore_body)
        self.world.RayCast(callback, tuple(origin), tuple(end))
        if callback.result is None:
            self.beam = (tuple(origin), tuple(end))
            return None
        target, point, normal = callback.result
        self.beam = (tuple(origin), point)
        return LaserHit(target, point, normal, tuple(direction), self.damage_per_tick)

    def draw(self, surface: pygame.Surface, world_to_screen: Callable) -> None:
        """Render the most recent beam without performing another damage tick."""
        if self.beam is not None:
            start, end = (world_to_screen(point) for point in self.beam)
            pygame.draw.line(surface, (255, 60, 60), start, end, 2)
