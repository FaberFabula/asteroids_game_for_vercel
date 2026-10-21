"""One small set of stat upgrades shared by the player's menu and enemy AI.

The effect is separate from payment: waves may equip enemies with the same
definitions without giving them a pretend currency balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ship import Ship


@dataclass
class ShipStats:
    """Independently tunable movement, aiming, and weapon settings in world units."""

    max_speed: float = 8.0
    turn_torque: float = 1.5
    max_turn_speed: float = 2.0
    turret_response: float = 5.0
    projectile_damage: float = 6.0
    laser_damage: float = 0.08
    projectile_interval: float = 0.3


@dataclass(frozen=True)
class UpgradeDefinition:
    """The stable key and player-facing explanation of one purchasable effect."""

    key: str
    name: str
    cost: int
    description: str


UPGRADE_DEFINITIONS: tuple[UpgradeDefinition, ...] = (
    UpgradeDefinition("turn_speed", "Turn Speed", 2, "20% stronger and faster body turning"),
    UpgradeDefinition("speed", "Speed", 2, "20% higher maximum movement speed"),
    UpgradeDefinition("turret_handling", "Turret Handling", 2, "30% quicker turret response"),
    UpgradeDefinition("restore_health", "Restore Health (100%)", 3, "Heal fully; no charge at full health"),
    UpgradeDefinition("projectile_damage", "Projectile Damage", 3, "25% stronger projectile hits"),
    UpgradeDefinition("laser_damage", "Laser Damage", 3, "25% stronger laser ticks"),
    UpgradeDefinition("projectile_fire_rate", "Projectile Fire Rate", 3, "15% shorter interval between shots"),
)

MIN_PROJECTILE_INTERVAL: float = 0.08


def apply_upgrade(ship: Ship, key: str) -> bool:
    """Apply an eligible effect without payment; return whether anything changed.

    Restoration deliberately leaves maximum health and upgrade levels alone.
    The fire interval has a practical minimum to bound projectile creation.
    """
    if not ship.alive or ship.health <= 0:
        return False
    if key not in {definition.key for definition in UPGRADE_DEFINITIONS}:
        return False
    stats = ship.stats
    if key == "restore_health":
        if ship.health >= ship.max_health:
            return False
        ship.heal()
        return True
    if key == "turn_speed":
        stats.turn_torque *= 1.2
        stats.max_turn_speed *= 1.2
    elif key == "speed":
        stats.max_speed *= 1.2
    elif key == "turret_handling":
        stats.turret_response *= 1.3
    elif key == "projectile_damage":
        stats.projectile_damage *= 1.25
    elif key == "laser_damage":
        stats.laser_damage *= 1.25
    elif key == "projectile_fire_rate":
        if stats.projectile_interval <= MIN_PROJECTILE_INTERVAL:
            return False
        stats.projectile_interval = max(MIN_PROJECTILE_INTERVAL, stats.projectile_interval * 0.85)
        # A purchased shorter interval should also help a shot already cooling.
        ship.cooldown = min(ship.cooldown, stats.projectile_interval)
    ship.upgrade_levels[key] = ship.upgrade_levels.get(key, 0) + 1
    return True


def purchase_upgrade(ship: Ship, key: str) -> bool:
    """Apply an affordable upgrade and charge exactly once if its effect succeeds."""
    definition = next((item for item in UPGRADE_DEFINITIONS if item.key == key), None)
    if definition is None or ship.powerups.upgrade_points < definition.cost:
        return False
    if not apply_upgrade(ship, key):
        return False
    ship.powerups.spend_upgrade_points(definition.cost)
    return True


def upgrade_effect(ship: Ship, key: str) -> str:
    """Describe the current effect beside a menu choice in readable units."""
    stats = ship.stats
    effects = {
        "turn_speed": f"{stats.max_turn_speed:.2f} rad/s",
        "speed": f"{stats.max_speed:.1f} units/s",
        "turret_handling": f"{stats.turret_response:.1f}/s response",
        "restore_health": f"{ship.health:.0f}/{ship.max_health:.0f} health",
        "projectile_damage": f"{stats.projectile_damage:.2f} damage",
        "laser_damage": f"{stats.laser_damage:.3f} damage/tick",
        "projectile_fire_rate": f"{stats.projectile_interval:.2f} s/shot",
    }
    if key == "restore_health":
        return effects[key]
    return f"Lv {ship.upgrade_levels.get(key, 0)} · {effects.get(key, 'Unknown upgrade')}"
