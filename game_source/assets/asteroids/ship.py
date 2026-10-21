"""A small Box2D ship with continuous controls and a separately aimed turret.

World coordinates use +x to the right and +y upwards. Angles are radians,
positive counterclockwise; both hull and turret point along +x at angle zero.
Only Box2D integrates the hull's position and rotation.
"""

from __future__ import annotations

from collections.abc import Callable
from math import atan2, cos, exp, hypot, isfinite, pi, sin

import Box2D
import pygame

from .powerups import PowerUpState
from .upgrades import ShipStats
from .config import TURRET_HALF_ARC

Point = tuple[float, float]
WorldToScreen = Callable[[Point], tuple[int, int]]


class Ship:
    """Own one dynamic polygon body, combat state, and a lagged visual turret."""

    HULL: tuple[Point, ...] = ((0.85, 0.0), (-0.45, 0.5), (-0.7, 0.0), (-0.45, -0.5))
    TURRET: tuple[Point, ...] = ((0.95, 0.11), (-0.12, 0.16), (-0.12, -0.16), (0.95, -0.11))
    TURRET_MOUNT: Point = (0.2, 0.0)
    THRUST_ACCELERATION: float = 6.0

    def __init__(
        self,
        world: Box2D.b2World,
        position: Point = (25.0, 17.5),
        *,
        team: str = "player",
        wave_id: int = 0,
        angle: float = 0.0,
    ) -> None:
        """Create a compact convex hull; body state is the sole motion authority."""
        if world.locked:
            raise RuntimeError("Create ships after world.Step()")
        if len(position) != 2 or not all(isfinite(value) for value in (*position, angle)):
            raise ValueError("Position and angle must be finite")
        self.world = world
        self.team = team
        self.wave_id = wave_id
        self.radius = 0.9
        self.max_health = 100.0
        self.health = self.max_health
        self.stats = ShipStats()
        self.powerups = PowerUpState()
        self.upgrade_levels: dict[str, int] = {}
        self.turret_angle = angle
        self.turret_half_arc = TURRET_HALF_ARC
        self.cooldown = 0.0
        self.body: Box2D.b2Body | None = world.CreateDynamicBody(
            position=position,
            angle=angle,
            linearDamping=0.25,
            angularDamping=2.5,
            userData=self,
        )
        try:
            self.body.CreatePolygonFixture(
                vertices=self.HULL, density=1.0, friction=0.2, restitution=0.3,
            )
        except Exception:
            self.destroy()
            raise

    @property
    def alive(self) -> bool:
        """Whether this entity still owns a body; Game handles health-based death."""
        return self.body is not None

    def control(self, dt: float, turn: float, thrust: float, aim: Point) -> None:
        """Apply held controls and turn the turret by a time-based angular fraction.

        Exponential response gives the same aiming lag for different frame sizes.
        Taking the shortest signed angle prevents a full revolution across ±pi.
        """
        if not all(isfinite(value) for value in (dt, turn, thrust, *aim)) or dt < 0:
            raise ValueError("Controls must be finite and dt non-negative")
        if not self.alive or self.health <= 0 or dt == 0:
            return
        body = self.body
        body.ApplyTorque(max(-1.0, min(1.0, turn)) * self.stats.turn_torque, wake=True)
        force = max(-1.0, min(1.0, thrust)) * self.THRUST_ACCELERATION * body.mass
        body.ApplyForceToCenter((force * cos(body.angle), force * sin(body.angle)), wake=True)
        mount = body.GetWorldPoint(self.TURRET_MOUNT)
        dx, dy = aim[0] - mount.x, aim[1] - mount.y
        if hypot(dx, dy) > 1e-9:
            desired_relative = (atan2(dy, dx) - body.angle + pi) % (2 * pi) - pi
            desired = body.angle + max(-self.turret_half_arc, min(self.turret_half_arc, desired_relative))
            difference = (desired - self.turret_angle + pi) % (2 * pi) - pi
            self.turret_angle += difference * (1.0 - exp(-self.stats.turret_response * dt))
        self.constrain_turret()

    def constrain_turret(self) -> None:
        """Keep the barrel inside its mechanical arc as Box2D rotates the hull.

        Call after physics as well as input. A fast hull turn can move the mount
        under a lagging turret; the mechanical stop must still hold before firing.
        """
        if not self.alive:
            return
        relative = (self.turret_angle - self.body.angle + pi) % (2 * pi) - pi
        self.turret_angle = self.body.angle + max(-self.turret_half_arc, min(self.turret_half_arc, relative))

    def tick(self, dt: float) -> None:
        """Advance cooldown and clamp velocity magnitude without changing direction.

        Laser duration belongs to Game: it needs PowerUpState.update's fractional
        active time to avoid granting a whole laser tick after the timer expires.
        """
        if not isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        if not self.alive:
            return
        self.cooldown = max(0.0, self.cooldown - dt)
        velocity = self.body.linearVelocity
        speed = hypot(velocity.x, velocity.y)
        if speed > self.stats.max_speed:
            factor = self.stats.max_speed / speed
            self.body.linearVelocity = (velocity.x * factor, velocity.y * factor)
        self.body.angularVelocity = max(
            -self.stats.max_turn_speed,
            min(self.stats.max_turn_speed, self.body.angularVelocity),
        )

    def muzzle(self) -> Point:
        """Return a world muzzle clear of the hull, even when aiming sideways."""
        if not self.alive:
            raise RuntimeError("A destroyed ship has no muzzle")
        mount = self.body.GetWorldPoint(self.TURRET_MOUNT)
        direction = self.aim_direction()
        # The extra small clearance prevents a shot from starting inside its hull.
        return (mount.x + direction[0] * 1.05, mount.y + direction[1] * 1.05)

    def aim_direction(self) -> Point:
        """Return the normalized world-space direction of the lagged turret."""
        self.constrain_turret()
        return (cos(self.turret_angle), sin(self.turret_angle))

    def ready_to_fire(self) -> bool:
        """Whether this living ship has completed its projectile cooldown."""
        return self.alive and self.health > 0 and self.cooldown <= 1e-9

    def reset_cooldown(self) -> None:
        """Start one projectile interval after Game successfully creates a shot."""
        self.cooldown = self.stats.projectile_interval

    def take_damage(self, amount: float) -> float:
        """Apply already-coalesced collision damage, spending at most one shield.

        The contact listener has already allowed Box2D to apply physical impulse.
        This method changes health only and leaves destruction to Game after Step.
        """
        if not isfinite(amount) or amount < 0:
            raise ValueError("Damage must be finite and non-negative")
        if not self.alive or self.health <= 0:
            return 0.0
        filtered = self.powerups.filter_collision_damage(amount)
        lost = min(self.health, filtered)
        self.health -= lost
        return lost

    def heal(self) -> None:
        """Restore current health to its existing maximum for a live body."""
        if self.alive and self.health > 0:
            self.health = self.max_health

    def destroy(self) -> None:
        """Release the body once, only while the world is unlocked."""
        if not self.alive:
            return
        if self.world.locked:
            raise RuntimeError("Destroy ships after world.Step()")
        self.body.userData = None
        self.world.DestroyBody(self.body)
        self.body = None

    def draw(self, surface: pygame.Surface, world_to_screen: WorldToScreen) -> None:
        """Draw hull and polygon turret with a separate shield ring when protected."""
        if not self.alive:
            return
        self.constrain_turret()
        color = (95, 220, 235) if self.team == "player" else (240, 115, 85)
        hull = [world_to_screen(tuple(self.body.GetWorldPoint(point))) for point in self.HULL]
        pygame.draw.polygon(surface, color, hull)
        pygame.draw.polygon(surface, (225, 240, 250), hull, 1)
        mount = self.body.GetWorldPoint(self.TURRET_MOUNT)
        c, s = cos(self.turret_angle), sin(self.turret_angle)
        turret = [world_to_screen((mount.x + c * x - s * y, mount.y + s * x + c * y))
                  for x, y in self.TURRET]
        pygame.draw.polygon(surface, (255, 225, 125), turret)
        if self.powerups.shield_stacks:
            ring = [world_to_screen((self.body.position.x + 1.05 * cos(i * pi / 4),
                                     self.body.position.y + 1.05 * sin(i * pi / 4)))
                    for i in range(8)]
            pygame.draw.polygon(surface, (70, 155, 255), ring, 2)
