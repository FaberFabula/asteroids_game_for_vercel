"""Small world helpers: ordinary wrapping and deferred contact outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import Box2D

from .projectile import Projectile


@dataclass(frozen=True)
class Impact:
    """One physical encounter; ``normal`` points from ``first`` to ``second``.

    The impulse already changed both bodies' velocities. It is only a damage
    measurement here and must never be applied to the bodies a second time.
    """

    first: object
    second: object
    point: tuple[float, float]
    normal: tuple[float, float]
    impulse: float


@dataclass
class _Encounter:
    """Track all touching fixtures so a compound body spends one shield."""

    fixtures: set[tuple[int, int]] = field(default_factory=set)
    reported: bool = False
    sensor_reported: bool = False


class ContactQueue(Box2D.b2ContactListener):
    """Observe contacts while locked, then let the game handle their effects.

    Fixture callbacks belong to a body-pair encounter until every fixture
    separates. We report its strongest solved contact once. Resting contacts
    therefore cannot consume a shield again on every simulation step.
    """

    def __init__(self) -> None:
        """Start with no active encounters or pending gameplay events."""
        super().__init__()
        self._encounters: dict[tuple[int, int], _Encounter] = {}
        self._impacts: dict[tuple[int, int], Impact] = {}
        self._sensors: dict[tuple[int, int], tuple[object, object]] = {}

    @staticmethod
    def _keys(contact: Box2D.b2Contact) -> tuple[tuple[int, int], tuple[int, int]]:
        """Use native pointer hashes, stable across SWIG wrapper instances."""
        first, second = contact.fixtureA, contact.fixtureB
        body_key = tuple(sorted((hash(first.body), hash(second.body))))
        fixture_key = tuple(sorted((hash(first), hash(second))))
        return body_key, fixture_key

    @staticmethod
    def _ignored(first: object, second: object) -> bool:
        """Keep bullets clear of their owner, allies and other projectiles."""
        if isinstance(first, Projectile) and isinstance(second, Projectile):
            return True
        for bullet, target in ((first, second), (second, first)):
            if isinstance(bullet, Projectile):
                if bullet.owner is target:
                    return True
                if bullet.team is not None and bullet.team == getattr(target, "team", None):
                    return True
        return False

    def BeginContact(self, contact: Box2D.b2Contact) -> None:
        """Count fixture contacts and queue each sensor encounter once."""
        body_key, fixture_key = self._keys(contact)
        encounter = self._encounters.setdefault(body_key, _Encounter())
        encounter.fixtures.add(fixture_key)
        first = contact.fixtureA.body.userData
        second = contact.fixtureB.body.userData
        if contact.fixtureA.sensor or contact.fixtureB.sensor:
            if not encounter.sensor_reported and first is not None and second is not None:
                self._sensors.setdefault(body_key, (first, second))
                encounter.sensor_reported = True

    def EndContact(self, contact: Box2D.b2Contact) -> None:
        """Permit a new damaging encounter after complete separation."""
        body_key, fixture_key = self._keys(contact)
        encounter = self._encounters.get(body_key)
        if encounter is not None:
            encounter.fixtures.discard(fixture_key)
            if not encounter.fixtures:
                del self._encounters[body_key]

    def PreSolve(self, contact: Box2D.b2Contact, old_manifold: Box2D.b2Manifold) -> None:
        """Disable only filtered bullets; shield protection never reaches here."""
        first = contact.fixtureA.body.userData
        second = contact.fixtureB.body.userData
        if self._ignored(first, second):
            contact.enabled = False

    def PostSolve(self, contact: Box2D.b2Contact, impulse: Box2D.b2ContactImpulse) -> None:
        """Capture valid manifold points and retain the strongest callback."""
        count = contact.manifold.pointCount
        if not count:
            return
        strength = sum(impulse.normalImpulses[:count])
        if strength <= 1e-7:
            return
        body_key, fixture_key = self._keys(contact)
        encounter = self._encounters.setdefault(body_key, _Encounter())
        encounter.fixtures.add(fixture_key)
        if encounter.reported and body_key not in self._impacts:
            return
        first = contact.fixtureA.body.userData
        second = contact.fixtureB.body.userData
        if first is None or second is None or self._ignored(first, second):
            return
        previous = self._impacts.get(body_key)
        if previous is None or strength > previous.impulse:
            manifold = contact.worldManifold
            points = manifold.points[:count]
            point = (
                sum(value[0] for value in points) / count,
                sum(value[1] for value in points) / count,
            )
            self._impacts[body_key] = Impact(
                first, second, point, tuple(manifold.normal), strength,
            )
        encounter.reported = True

    def drain(self) -> tuple[list[Impact], list[tuple[object, object]]]:
        """Take queued results after Step, retaining persistent-contact state."""
        impacts, sensors = list(self._impacts.values()), list(self._sensors.values())
        self._impacts.clear()
        self._sensors.clear()
        return impacts, sensors


def wrap_body(body: Box2D.b2Body, width: float, height: float) -> None:
    """Wrap a body's center into the world without changing velocity or angle.

    This intentionally uses ordinary collision space. Opposite edges have no
    shared collision bodies, which keeps the implementation easy to inspect.
    """
    if not all(isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("World width and height must be finite and positive")
    if body.world.locked:
        raise RuntimeError("Wrap bodies after world.Step()")
    x, y = body.position
    if not (0 <= x < width and 0 <= y < height):
        body.position = (x % width, y % height)
