"""Collectible sensor bodies and independently testable player effect state."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import cos, isfinite, pi, sin, sqrt

import Box2D
import pygame
from pygame.math import Vector2


class PowerUpKind(Enum):
    """The three visually distinct pickup kinds."""
    UPGRADE = "upgrade"
    SHIELD = "shield"
    LASER = "laser"


@dataclass
class PowerUpState:
    """Currency, collision shields, and additive laser time in simulation seconds."""
    upgrade_points: int = 0
    shield_stacks: int = 0
    laser_remaining: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid initial state before it reaches gameplay."""
        for name in ("upgrade_points", "shield_stacks"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isfinite(self.laser_remaining) or self.laser_remaining < 0:
            raise ValueError("laser_remaining must be finite and non-negative")

    @property
    def laser_active(self) -> bool:
        """Whether any laser duration remains."""
        return self.laser_remaining > 0

    def grant(self, kind: PowerUpKind, *, laser_duration: float = 6.0) -> None:
        """Add exactly one pickup effect; laser durations stack."""
        kind = PowerUpKind(kind)
        if kind is PowerUpKind.UPGRADE:
            self.upgrade_points += 1
        elif kind is PowerUpKind.SHIELD:
            self.shield_stacks += 1
        else:
            if not isfinite(laser_duration) or laser_duration <= 0:
                raise ValueError("laser_duration must be finite and positive")
            self.laser_remaining += laser_duration

    def spend_upgrade_points(self, cost: int = 1) -> bool:
        """Subtract an affordable positive price and report success."""
        if not isinstance(cost, int) or isinstance(cost, bool) or cost <= 0:
            raise ValueError("cost must be a positive integer")
        if self.upgrade_points < cost:
            return False
        self.upgrade_points -= cost
        return True

    def filter_collision_damage(self, amount: float) -> float:
        """Cancel one damaging impact per shield, leaving physics unchanged."""
        if not isfinite(amount) or amount < 0:
            raise ValueError("amount must be finite and non-negative")
        if amount > 0 and self.shield_stacks > 0:
            self.shield_stacks -= 1
            return 0.0
        return amount

    def update(self, dt: float) -> float:
        """Consume active simulation time and return the laser-active part of dt."""
        if not isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        active_time = min(dt, self.laser_remaining)
        self.laser_remaining = max(0.0, self.laser_remaining - dt)
        return active_time


class PowerUp:
    """A static sensor rendered as a green square, blue octagon, or red circle."""
    COLORS = {
        PowerUpKind.UPGRADE: (55, 225, 95),
        PowerUpKind.SHIELD: (65, 150, 255),
        PowerUpKind.LASER: (245, 65, 65),
    }

    def __init__(
        self,
        world: Box2D.b2World,
        kind: PowerUpKind,
        position: tuple[float, float],
        *,
        radius: float = 0.3,
        laser_duration: float = 6.0,
    ) -> None:
        """Create the pickup while Box2D is unlocked."""
        if world.locked:
            raise RuntimeError("Create power-ups after world.Step()")
        self.kind = PowerUpKind(kind)
        position = Vector2(position)
        if not all(isfinite(value) for value in (*position, radius, laser_duration)):
            raise ValueError("Position, radius, and laser duration must be finite")
        if radius <= 0 or laser_duration <= 0:
            raise ValueError("radius and laser_duration must be positive")

        self.world = world
        self.radius = radius
        self.laser_duration = laser_duration
        if self.kind is PowerUpKind.UPGRADE:
            half_side = radius / sqrt(2)
            self.vertices = (
                (-half_side, -half_side), (half_side, -half_side),
                (half_side, half_side), (-half_side, half_side),
            )
        elif self.kind is PowerUpKind.SHIELD:
            self.vertices = tuple(
                (radius * cos(pi / 8 + i * pi / 4),
                 radius * sin(pi / 8 + i * pi / 4))
                for i in range(8)
            )
        else:
            self.vertices = ()

        shape = (
            Box2D.b2PolygonShape(vertices=self.vertices)
            if self.vertices else Box2D.b2CircleShape(radius=radius)
        )
        self.body = world.CreateStaticBody(position=tuple(position), userData=self)
        try:
            self.body.CreateFixture(shape=shape, isSensor=True)
        except Exception:
            self.destroy()
            raise

    @property
    def alive(self) -> bool:
        """Whether the sensor body still exists."""
        return self.body is not None

    def collect(self, state: PowerUpState) -> bool:
        """Grant once and remove the sensor after the physics step."""
        if not self.alive:
            return False
        if self.world.locked:
            raise RuntimeError("Queue collection until after world.Step()")
        state.grant(self.kind, laser_duration=self.laser_duration)
        self.destroy()
        return True

    def destroy(self) -> None:
        """Release the sensor body; repeated calls are harmless."""
        if not self.alive:
            return
        if self.world.locked:
            raise RuntimeError("Destroy power-ups after world.Step()")
        self.body.userData = None
        self.world.DestroyBody(self.body)
        self.body = None

    def draw(self, surface: pygame.Surface, world_to_screen: Callable) -> None:
        """Render the kind's solid-color shape with the shared coordinate map."""
        if not self.alive:
            return
        color = self.COLORS[self.kind]
        if self.vertices:
            points = [
                world_to_screen(tuple(self.body.GetWorldPoint(vertex)))
                for vertex in self.vertices
            ]
            pygame.draw.polygon(surface, color, points)
        else:
            center = Vector2(world_to_screen(tuple(self.body.position)))
            edge = Vector2(world_to_screen(tuple(self.body.GetWorldPoint((self.radius, 0)))))
            pygame.draw.circle(surface, color, center, max(1, round(center.distance_to(edge))))
