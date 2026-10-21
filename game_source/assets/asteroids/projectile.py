"""Short-lived physical projectiles; the game owns their hit consequences."""

from __future__ import annotations

from collections.abc import Callable
from math import hypot, isfinite
from typing import TYPE_CHECKING

import Box2D
import pygame

if TYPE_CHECKING:
    from .ship import Ship


class Projectile:
    """A small bullet body whose momentum is resolved once, by Box2D.

    ``damage`` is a game stat, independent of the body's physical impulse.
    The game uses it for health or converts it into asteroid carving depth.
    Keeping those concepts separate makes damage upgrades predictable.
    """

    def __init__(
        self,
        world: Box2D.b2World,
        position: tuple[float, float],
        direction: tuple[float, float],
        *,
        owner: Ship,
        damage: float,
        speed: float = 25.0,
        lifetime: float = 2.0,
    ) -> None:
        """Spawn at a muzzle, inheriting the owner's velocity and allegiance."""
        if world.locked:
            raise RuntimeError("Create projectiles after world.Step()")
        if len(position) != 2 or len(direction) != 2:
            raise ValueError("position and direction must be two-dimensional")
        if not all(isfinite(value) for value in (*position, *direction, damage, speed, lifetime)):
            raise ValueError("Projectile settings must be finite")
        length = hypot(*direction)
        if length == 0 or speed <= 0 or lifetime <= 0 or damage < 0:
            raise ValueError("Direction, speed and lifetime must be positive; damage cannot be negative")
        if owner.body is None:
            raise ValueError("A projectile needs a living owner")

        self.world = world
        self.owner = owner
        self.team = owner.team
        self.damage = damage
        self.direction = (direction[0] / length, direction[1] / length)
        self.remaining = lifetime
        self.radius = 0.10
        velocity = owner.body.linearVelocity
        self.body = world.CreateDynamicBody(
            position=position,
            linearVelocity=(
                velocity.x + self.direction[0] * speed,
                velocity.y + self.direction[1] * speed,
            ),
            bullet=True,
            userData=self,
        )
        self.body.CreateCircleFixture(
            radius=self.radius, density=1.0, friction=0.0, restitution=0.1,
        )

    @property
    def alive(self) -> bool:
        """Whether a body exists; expiry is checked separately by the game."""
        return self.body is not None

    def tick(self, dt: float) -> None:
        """Consume simulation time; destruction waits until the world unlocks."""
        if not isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        self.remaining = max(0.0, self.remaining - dt)

    def destroy(self) -> None:
        """Release the native body once, outside contact callbacks."""
        if not self.alive:
            return
        if self.world.locked:
            raise RuntimeError("Destroy projectiles after world.Step()")
        self.body.userData = None
        self.world.DestroyBody(self.body)
        self.body = None

    def draw(
        self,
        surface: pygame.Surface,
        world_to_screen: Callable[[tuple[float, float]], tuple[int, int]],
    ) -> None:
        """Draw the physical radius using the game's single coordinate scale."""
        if not self.alive:
            return
        position = tuple(self.body.position)
        center = world_to_screen(position)
        edge = world_to_screen((position[0] + self.radius, position[1]))
        radius = max(2, round(hypot(edge[0] - center[0], edge[1] - center[1])))
        color = (255, 220, 110) if self.team == "player" else (255, 120, 110)
        pygame.draw.circle(surface, color, center, radius)
