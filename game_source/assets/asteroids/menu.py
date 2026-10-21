"""A pause-screen view of shared upgrades; the game owns pause and resume state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from .upgrades import UPGRADE_DEFINITIONS, purchase_upgrade, upgrade_effect

if TYPE_CHECKING:
    from .ship import Ship


class UpgradeMenu:
    """Render seven upgrades and translate clicks or numbered keys into purchases.

    Keeping P/X transitions outside this view gives the game one place to freeze
    simulation and enforce its resume countdown. This class only changes ship
    upgrades when the game explicitly sends it a purchase event.
    """

    def __init__(self, size: tuple[int, int] = (1000, 700)) -> None:
        """Prepare an adaptive layout; prices and effects come from shared rules."""
        pygame.font.init()
        self.size = size
        self.rows: list[pygame.Rect] = []
        self.keys = [definition.key for definition in UPGRADE_DEFINITIONS]
        self.feedback = "Click an upgrade or press its number."
        self._title_font = pygame.font.Font(None, 38)
        self._name_font = pygame.font.Font(None, 28)
        self._detail_font = pygame.font.Font(None, 21)
        self._layout(size)

    def _layout(self, size: tuple[int, int]) -> None:
        """Allocate equal rows between the header and purchase feedback."""
        self.size = size
        width, height = size
        margin = max(12, round(width * 0.08))
        top = round(height * 0.19)
        bottom = height - round(height * 0.09)
        row_slot = (bottom - top) / len(UPGRADE_DEFINITIONS)
        gap = max(3, round(row_slot * 0.10))
        self.rows = [
            pygame.Rect(margin, round(top + index * row_slot),
                        width - 2 * margin, max(1, round(row_slot) - gap))
            for index in range(len(UPGRADE_DEFINITIONS))
        ]

    def handle_event(self, event: pygame.event.Event, ship: Ship) -> bool:
        """Buy on left click or 1–7; return whether stats or health changed.

        P and X deliberately have no menu-local meaning. The game handles them
        even during countdown or game over, without a second toggle mechanism.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, row in enumerate(self.rows):
                if row.collidepoint(event.pos):
                    return self._purchase(ship, index)
        if event.type == pygame.KEYDOWN:
            number_keys = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                           pygame.K_5, pygame.K_6, pygame.K_7)
            if event.key in number_keys:
                return self._purchase(ship, number_keys.index(event.key))
        return False

    def _purchase(self, ship: Ship, index: int) -> bool:
        """Delegate payment and effects to the same rules used by enemy upgrades."""
        definition = UPGRADE_DEFINITIONS[index]
        if purchase_upgrade(ship, definition.key):
            self.feedback = f"Purchased: {definition.name}."
            return True
        if ship.powerups.upgrade_points < definition.cost:
            self.feedback = "Not enough green-square points for that upgrade."
        elif definition.key == "restore_health":
            self.feedback = "Health is already full; no points spent."
        else:
            self.feedback = "That upgrade cannot increase further; no points spent."
        return False

    def draw(self, surface: pygame.Surface, ship: Ship) -> None:
        """Draw a translucent solid-color panel over the frozen game surface."""
        if surface.get_size() != self.size:
            self._layout(surface.get_size())
        width, height = self.size
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        overlay.fill((8, 13, 22, 212))
        margin = self.rows[0].left

        title = self._title_font.render("UPGRADES — PAUSED", True, (235, 241, 250))
        overlay.blit(title, (margin, round(height * 0.035)))
        instructions = self._detail_font.render(
            "P opens this menu  •  X resumes after a 3-second countdown",
            True, (192, 203, 220),
        )
        overlay.blit(instructions, (margin, round(height * 0.10)))
        balance = self._detail_font.render(
            f"Green-square points: {ship.powerups.upgrade_points}", True, (102, 230, 136),
        )
        overlay.blit(balance, (margin, round(height * 0.145)))

        for index, (definition, row) in enumerate(zip(UPGRADE_DEFINITIONS, self.rows)):
            affordable = ship.powerups.upgrade_points >= definition.cost
            background = (34, 52, 47, 232) if affordable else (34, 38, 48, 232)
            pygame.draw.rect(overlay, background, row, border_radius=5)
            color = (230, 239, 247) if affordable else (165, 173, 187)
            name = self._name_font.render(f"{index + 1}. {definition.name}", True, color)
            overlay.blit(name, (row.x + 12, row.y + 7))
            price = self._detail_font.render(f"Price: {definition.cost}", True, color)
            overlay.blit(price, (row.right - price.get_width() - 12, row.y + 9))
            effect = self._detail_font.render(
                upgrade_effect(ship, definition.key),
                True, (181, 195, 213),
            )
            overlay.blit(effect, (row.x + 12, row.y + row.height // 2 + 4))

        feedback = self._detail_font.render(self.feedback, True, (231, 218, 156))
        overlay.blit(feedback, (margin, height - round(height * 0.065)))
        surface.blit(overlay, (0, 0))
