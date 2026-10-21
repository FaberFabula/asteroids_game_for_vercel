"""Simple enemies: a visible awareness circle, a forward cone, and slow shooting."""

from __future__ import annotations

from math import atan2, cos, hypot, isfinite, pi, sin

import Box2D
import pygame

from .ship import Point, Ship, WorldToScreen
from .upgrades import UPGRADE_DEFINITIONS, apply_upgrade


class Enemy(Ship):
    """A ship that chases a visible player and uses the same upgrades as the menu."""

    def __init__(
        self,
        world: Box2D.b2World,
        position: Point,
        *,
        wave_id: int = 0,
        level: int = 1,
        angle: float = 0.0,
    ) -> None:
        """Create an enemy and cycle deterministic combat upgrades in later waves."""
        if not isinstance(level, int) or level < 1:
            raise ValueError("Enemy level must be a positive integer")
        super().__init__(world, position, team="enemy", wave_id=wave_id, angle=angle)
        self.max_health = 45.0
        self.health = self.max_health
        self.awareness_radius = 12.0
        self.aware = False
        self.stats.projectile_interval = 3.0
        self.stats.max_speed = 5.0
        combat_keys = [item.key for item in UPGRADE_DEFINITIONS if item.key != "restore_health"]
        # Capping equipped upgrades also bounds extreme late-wave movement/damage.
        for index in range(min(level - 1, 18)):
            apply_upgrade(self, combat_keys[index % len(combat_keys)])

    @property
    def shot_interval(self) -> float:
        """Keep enemy fire deliberate even after shared fire-rate upgrades."""
        return max(1.5, self.stats.projectile_interval)

    def reset_cooldown(self) -> None:
        """Use the enemy's long interval after a successfully created projectile."""
        self.cooldown = self.shot_interval

    def can_see(self, target_position: Point) -> bool:
        """Test distance and a 120-degree cone, including both ±60-degree edges."""
        if not all(isfinite(value) for value in target_position):
            raise ValueError("Target position must be finite")
        if not self.alive:
            return False
        dx = target_position[0] - self.body.position.x
        dy = target_position[1] - self.body.position.y
        distance = hypot(dx, dy)
        if distance > self.awareness_radius:
            return False
        if distance <= 1e-9:
            return True
        dot = (cos(self.body.angle) * dx + sin(self.body.angle) * dy) / distance
        return dot >= 0.5 - 1e-7

    def update_ai(self, dt: float, player: Ship) -> bool:
        """Chase/aim when aware and return a firing request; Game owns shot creation.

        Timers are advanced by Ship.tick exactly once per simulation step. The
        awareness test intentionally uses ordinary screen positions, not ghosts
        across a wrap seam, matching the game's simple world-level wrapping.
        """
        if not isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        self.aware = bool(self.alive and self.health > 0 and player.alive and player.health > 0
                          and self.can_see(tuple(player.body.position)))
        if not self.aware:
            return False
        target = tuple(player.body.position)
        dx, dy = target[0] - self.body.position.x, target[1] - self.body.position.y
        desired = atan2(dy, dx)
        difference = (desired - self.body.angle + pi) % (2 * pi) - pi
        turn = max(-1.0, min(1.0, difference * 2.0))
        thrust = 0.8 if hypot(dx, dy) > 3.0 else 0.0
        self.control(dt, turn, thrust, target)
        return self.ready_to_fire()

    def draw(self, surface: pygame.Surface, world_to_screen: WorldToScreen) -> None:
        """Draw the awareness radius and cone edges underneath the enemy hull."""
        if not self.alive:
            return
        position = self.body.position
        center = world_to_screen(tuple(position))
        edge = world_to_screen((position.x + self.awareness_radius, position.y))
        radius = max(1, round(hypot(edge[0] - center[0], edge[1] - center[1])))
        color = (92, 57, 48) if self.aware else (44, 47, 58)
        pygame.draw.circle(surface, color, center, radius, 1)
        for offset in (-pi / 3, pi / 3):
            angle = self.body.angle + offset
            end = world_to_screen((position.x + self.awareness_radius * cos(angle),
                                   position.y + self.awareness_radius * sin(angle)))
            pygame.draw.line(surface, color, center, end, 1)
        super().draw(surface, world_to_screen)
