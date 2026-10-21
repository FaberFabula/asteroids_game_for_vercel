"""Coordinate fixed-step gameplay while entities own their own physics bodies.

Contact callbacks only record facts. This coordinator changes bodies after Step,
so carving, collection and deaths cannot mutate Box2D while it is locked.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections.abc import Sequence

import Box2D
import pygame
from pygame.math import Vector2

from . import config
from .asteroid import Asteroid
from .enemy import Enemy
from .geometry.carving import Carving
from .laser import LaserHit, LaserWeapon
from .menu import UpgradeMenu
from .physics import ContactQueue, Impact, wrap_body
from .powerups import PowerUp, PowerUpKind
from .projectile import Projectile
from .ship import Ship


@dataclass(frozen=True)
class Controls:
    """Held input sampled once per rendered frame; aim is in world units."""
    turn: float = 0.0
    thrust: float = 0.0
    aim: tuple[float, float] = (30.0, 17.5)
    firing: bool = False


@dataclass
class Wave:
    """Remember clear rewards even after all original bodies have been replaced."""
    number: int
    spawned_at: float
    cleared: bool = False


class Game:
    """Simulation and UI state, independent of a real display or wall clock."""

    def __init__(self, seed: int = 7, *, spawn_initial: bool = True) -> None:
        """Create a zero-gravity world; tests may opt out of the first wave."""
        self.seed = seed
        self.rng = random.Random(seed)
        self.contacts = ContactQueue()
        self.world = Box2D.b2World(gravity=(0, 0), contactListener=self.contacts)
        self.player = Ship(self.world, (config.WORLD_WIDTH / 2, config.WORLD_HEIGHT / 2))
        self.asteroids: list[Asteroid] = []
        self.enemies: list[Enemy] = []
        self.projectiles: list[Projectile] = []
        self.pickups: list[PowerUp] = []
        self.waves: dict[int, Wave] = {}
        self.wave_number = 0
        self.clear_rewards = 0
        self.simulation_time = 0.0
        self.next_wave_at = config.WAVE_INTERVAL
        self.accumulator = 0.0
        self.state = 'playing'
        self.running = True
        self.countdown_remaining = 0.0
        self.score = 0.0
        self.drop_area = 0.0
        # Game damage becomes notch depth, not a second physical impulse.
        self.projectile_carver = Carving(0.18, 0.5, 0.12, 0.12)
        self.laser_carver = Carving(0.18, 0.18, 0.10, 0.12)
        self.laser = LaserWeapon(self.world, (0, 0, config.WORLD_WIDTH, config.WORLD_HEIGHT))
        self.menu = UpgradeMenu((config.WIDTH, config.HEIGHT))
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.Font(None, 25)
        self.large_font = pygame.font.Font(None, 74)
        if spawn_initial:
            self.spawn_wave()

    @staticmethod
    def world_to_screen(point: Sequence[float]) -> tuple[int, int]:
        """World +y is up; screen +y is down. All entities share this scale."""
        return (round(point[0] * config.PIXELS_PER_UNIT),
                round(config.HEIGHT - point[1] * config.PIXELS_PER_UNIT))

    @staticmethod
    def screen_to_world(point: Sequence[float]) -> tuple[float, float]:
        """Convert mouse coordinates before computing turret aim."""
        return (point[0] / config.PIXELS_PER_UNIT,
                (config.HEIGHT - point[1]) / config.PIXELS_PER_UNIT)

    def handle_event(self, event: pygame.event.Event) -> None:
        """P opens, X starts a frozen countdown, R restarts after death."""
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_p and self.state in ('playing', 'countdown'):
                self.state = 'menu'
                self.accumulator = 0.0
            elif event.key == pygame.K_x and self.state == 'menu':
                self.state = 'countdown'
                self.countdown_remaining = 3.0
                self.accumulator = 0.0
            elif event.key == pygame.K_r and self.state == 'game_over':
                self.restart()
        if self.state == 'menu':
            self.menu.handle_event(event, self.player)

    def advance(self, frame_dt: float, controls: Controls | None = None) -> int:
        """Advance bounded physics steps, or UI-only countdown time when frozen.

        Time spent paused is discarded. The frame ending the countdown also
        discards its remainder, preventing a catch-up burst when play resumes.
        """
        if not math.isfinite(frame_dt) or frame_dt < 0:
            raise ValueError('frame_dt must be finite and non-negative')
        if not self.running:
            return 0
        if self.state == 'countdown':
            self.countdown_remaining = max(0.0, self.countdown_remaining - frame_dt)
            if self.countdown_remaining <= 1e-9:
                self.countdown_remaining = 0.0
                self.state = 'playing'
            self.accumulator = 0.0
            return 0
        if self.state != 'playing':
            self.accumulator = 0.0
            return 0
        self.accumulator += min(frame_dt, config.MAX_FRAME_TIME)
        steps = 0
        while self.accumulator + 1e-12 >= config.FIXED_DT and self.state == 'playing':
            self.step(config.FIXED_DT, controls or Controls())
            self.accumulator = max(0.0, self.accumulator - config.FIXED_DT)
            steps += 1
        return steps

    def step(self, dt: float, controls: Controls) -> None:
        """Run one simulation tick; all destructive work follows world.Step."""
        if self.state != 'playing' or not self.running:
            return
        self.simulation_time += dt
        self.player.tick(dt)
        self.player.control(dt, controls.turn, controls.thrust, controls.aim)
        laser_time = self.player.powerups.update(dt)
        if controls.firing and laser_time == 0:
            self.fire_projectile(self.player)
        for enemy in self.enemies:
            if enemy.alive and enemy.health > 0:
                enemy.tick(dt)
                if enemy.update_ai(dt, self.player):
                    self.fire_projectile(enemy)
        for projectile in self.projectiles:
            projectile.tick(dt)
        self.world.Step(dt, 8, 3)
        self.world.ClearForces()
        for ship in [self.player, *self.enemies]:
            ship.constrain_turret()
        impacts, sensor_pairs = self.contacts.drain()
        for impact in impacts:
            self.handle_impact(impact)
        for first, second in sensor_pairs:
            self.collect_pair(first, second)
        if self.player.alive and self.player.health > 0:
            self.laser.damage_per_tick = self.player.stats.laser_damage
            self.laser.update(
                laser_time, self.player.muzzle(), self.player.aim_direction(),
                firing=controls.firing and laser_time > 0,
                ignore_body=self.player.body, on_hit=self.handle_laser_hit,
            )
        self.remove_dead()
        for entity in [self.player, *self.asteroids, *self.enemies, *self.projectiles]:
            if entity.alive:
                wrap_body(entity.body, config.WORLD_WIDTH, config.WORLD_HEIGHT)
        if self.player.health <= 0:
            self.state = 'game_over'
            self.accumulator = 0.0
            return
        self.check_wave_clears()
        if self.simulation_time + 1e-9 >= self.next_wave_at:
            self.spawn_wave()
            self.next_wave_at += config.WAVE_INTERVAL

    def fire_projectile(self, ship: Ship) -> Projectile | None:
        """Apply the shooter's own damage/cadence and a bounded projectile count."""
        if not ship.alive or ship.health <= 0 or not ship.ready_to_fire():
            return None
        if len(self.projectiles) >= config.MAX_PROJECTILES:
            return None
        projectile = Projectile(self.world, ship.muzzle(), ship.aim_direction(),
                                owner=ship, damage=ship.stats.projectile_damage)
        self.projectiles.append(projectile)
        ship.reset_cooldown()
        return projectile

    def handle_impact(self, impact: Impact) -> None:
        """Apply one logical hit per encounter; native impulses are already applied."""
        first, second = impact.first, impact.second
        if isinstance(first, Projectile) or isinstance(second, Projectile):
            projectile, target = (first, second) if isinstance(first, Projectile) else (second, first)
            if not projectile.alive:
                return
            if isinstance(target, Asteroid):
                self.damage_asteroid(target, impact.point, projectile.direction,
                                     projectile.damage, laser=False,
                                     credit=projectile.team == 'player')
            elif isinstance(target, Ship) and target.alive and target.team != projectile.team:
                target.take_damage(projectile.damage)
            projectile.destroy()
            return
        # The listener coalesces compound fixtures and suppresses persistent
        # contacts. A shield cancels this health damage, never the physics impulse.
        damage = max(0.0, impact.impulse * 4.0 - 0.5)
        for target, other in ((first, second), (second, first)):
            if isinstance(target, Ship) and target.alive and isinstance(other, (Ship, Asteroid)):
                target.take_damage(damage)

    def collect_pair(self, first: object, second: object) -> None:
        """Only the player's ship collects sensors, once and after stepping."""
        pickup = second if first is self.player else first if second is self.player else None
        if isinstance(pickup, PowerUp) and self.player.health > 0:
            pickup.collect(self.player.powerups)

    def handle_laser_hit(self, hit: LaserHit) -> None:
        """Raycasts deal the laser stat; their callbacks run outside world queries."""
        if isinstance(hit.target, Asteroid):
            self.damage_asteroid(hit.target, hit.point, hit.direction, hit.damage,
                                 laser=True, credit=True)
        elif isinstance(hit.target, Ship) and hit.target.alive and hit.target.team != 'player':
            # Shield pickup rules cover damaging collisions, not raycasts.
            hit.target.health = max(0.0, hit.target.health - hit.damage)

    def damage_asteroid(self, asteroid: Asteroid, point: Sequence[float],
                        direction: Sequence[float], damage: float, *,
                        laser: bool, credit: bool) -> float:
        """Carve surviving material and score only area actually removed.

        Two queued hits may name a parent replaced earlier this step. A stale
        hit is consumed without scoring; it must never carve the old hull twice.
        """
        if not asteroid.alive or damage <= 0:
            return 0.0
        local_point, local_direction = asteroid.get_local_hit(point, Vector2(direction) * damage)
        carver = self.laser_carver if laser else self.projectile_carver
        result = asteroid.carve_with_result(local_point, local_direction, carver)
        if result.pieces != (asteroid,):
            self.asteroids = [item for item in self.asteroids if item is not asteroid]
            self.asteroids.extend(result.pieces)
        if credit and result.destroyed_area > 0:
            self.score += result.destroyed_area * config.POINTS_PER_AREA
            self.spawn_drops(result.destroyed_area, point)
        return result.destroyed_area

    def spawn_drops(self, destroyed_area: float, point: Sequence[float]) -> None:
        """One bounded pickup per material budget; splitting alone grants nothing."""
        self.drop_area += destroyed_area
        while self.drop_area >= config.DROP_AREA:
            self.drop_area -= config.DROP_AREA
            if len(self.pickups) >= config.MAX_PICKUPS:
                continue
            kind = self.rng.choices(list(PowerUpKind), weights=(0.60, 0.22, 0.18))[0]
            position = ((point[0] + self.rng.uniform(-0.4, 0.4)) % config.WORLD_WIDTH,
                        (point[1] + self.rng.uniform(-0.4, 0.4)) % config.WORLD_HEIGHT)
            self.pickups.append(PowerUp(self.world, kind, position,
                                       laser_duration=config.LASER_DURATION))

    def remove_dead(self) -> None:
        """Destroy expired entities once, then remove stale list references."""
        for projectile in self.projectiles:
            if projectile.alive and projectile.remaining <= 0:
                projectile.destroy()
        for enemy in self.enemies:
            if enemy.alive and enemy.health <= 0:
                enemy.destroy()
        self.projectiles = [item for item in self.projectiles if item.alive]
        self.enemies = [item for item in self.enemies if item.alive]
        self.asteroids = [item for item in self.asteroids if item.alive]
        self.pickups = [item for item in self.pickups if item.alive]

    def spawn_position(self) -> tuple[float, float]:
        """Spawn away from the player, including near opposite screen edges."""
        player = Vector2(self.player.body.position)
        for _ in range(100):
            candidate = Vector2(self.rng.uniform(1, config.WORLD_WIDTH - 1),
                                self.rng.uniform(1, config.WORLD_HEIGHT - 1))
            dx = min(abs(candidate.x - player.x), config.WORLD_WIDTH - abs(candidate.x - player.x))
            dy = min(abs(candidate.y - player.y), config.WORLD_HEIGHT - abs(candidate.y - player.y))
            if math.hypot(dx, dy) >= 8:
                return tuple(candidate)
        return ((player.x + config.WORLD_WIDTH / 2) % config.WORLD_WIDTH,
                (player.y + config.WORLD_HEIGHT / 2) % config.WORLD_HEIGHT)

    def spawn_wave(self) -> None:
        """Schedule overlapping opponents with explicit IDs and rising hardness."""
        self.wave_number += 1
        number = self.wave_number
        asteroid_count = max(0, min(3 + number, 12, config.MAX_ASTEROIDS - len(self.asteroids)))
        enemy_count = max(0, min(1 + (number - 1) // 2, 5, config.MAX_ENEMIES - len(self.enemies)))
        if asteroid_count + enemy_count == 0:
            return  # A saturated world does not generate empty-wave heal rewards.
        self.waves[number] = Wave(number, self.simulation_time)
        for _ in range(asteroid_count):
            # Generation uses the legacy module's random source. Isolate it so a
            # seeded game stays reproducible without changing other random users.
            previous_random = random.getstate()
            random.seed(self.rng.getrandbits(64))
            try:
                asteroid = Asteroid.randomly_generate_asteroid(
                    self.world, 12, self.rng.uniform(1.0, 1.9),
                    position=self.spawn_position(),
                    velocity=(self.rng.uniform(-1.5, 1.5), self.rng.uniform(-1.5, 1.5)),
                    angular_velocity=self.rng.uniform(-0.5, 0.5),
                    hardness=1 + 0.18 * (number - 1), wave_id=number,
                )
            finally:
                random.setstate(previous_random)
            self.asteroids.append(asteroid)
        for _ in range(enemy_count):
            enemy = Enemy(self.world, self.spawn_position(), wave_id=number,
                          level=number, angle=self.rng.uniform(-math.pi, math.pi))
            self.enemies.append(enemy)

    def remaining_in_wave(self, number: int) -> int:
        """Count live fragment descendants as well as enemy ships."""
        return sum(item.alive and item.wave_id == number
                   for item in [*self.asteroids, *self.enemies])

    def check_wave_clears(self) -> None:
        """Heal once per recorded wave, never once per empty frame."""
        for number, wave in self.waves.items():
            if not wave.cleared and self.remaining_in_wave(number) == 0:
                wave.cleared = True
                self.player.heal()
                self.clear_rewards += 1

    def draw(self, surface: pygame.Surface) -> None:
        """Draw current state, then overlay UI without advancing simulation."""
        surface.fill((8, 12, 22))
        for x in range(25, config.WIDTH, 65):
            for y in range(30, config.HEIGHT, 83):
                pygame.draw.circle(surface, (36, 43, 60), (x, y), 1)
        for entity in [*self.pickups, *self.asteroids, *self.enemies, *self.projectiles, self.player]:
            entity.draw(surface, self.world_to_screen)
        self.laser.draw(surface, self.world_to_screen)
        lines = [
            f'Score {int(self.score)}   Wave {self.wave_number}   Health {max(0, self.player.health):.0f}/{self.player.max_health:.0f}',
            f'Upgrade points {self.player.powerups.upgrade_points}   Shields {self.player.powerups.shield_stacks}   Laser {self.player.powerups.laser_remaining:.1f}s',
            'A/D turn  W/S or arrows thrust  Mouse aim  LMB/Space fire  P upgrades  Esc quit',
        ]
        for index, line in enumerate(lines):
            surface.blit(self.font.render(line, True, (218, 227, 238)), (16, 12 + 25 * index))
        if self.state == 'menu':
            self.menu.draw(surface, self.player)
        elif self.state in ('countdown', 'game_over'):
            veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            veil.fill((8, 12, 22, 140))
            surface.blit(veil, (0, 0))
            text = str(math.ceil(self.countdown_remaining)) if self.state == 'countdown' else 'SHIP LOST'
            label = self.large_font.render(text, True, (245, 220, 140))
            surface.blit(label, label.get_rect(center=(config.WIDTH / 2, config.HEIGHT / 2)))
            reminder = 'Resuming in…  P reopens upgrades' if self.state == 'countdown' else 'R to restart   Esc to quit'
            label = self.font.render(reminder, True, (235, 237, 245))
            surface.blit(label, label.get_rect(center=(config.WIDTH / 2, config.HEIGHT / 2 + 65)))

    def close(self) -> None:
        """Release every body owned by this game, including on restart/quit."""
        self.world.contactListener = None
        for entity in [self.player, *self.asteroids, *self.enemies, *self.projectiles, *self.pickups]:
            entity.destroy()
        self.running = False

    def restart(self) -> None:
        """Destroy the previous world contents and create a clean first wave."""
        seed = self.seed
        self.close()
        self.__init__(seed=seed)
