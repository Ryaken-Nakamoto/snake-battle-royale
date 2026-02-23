"""Player interface and implementations for Snake Battle Royale."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pygame

from snake import Action

if TYPE_CHECKING:
    pass


class Player(ABC):
    """Abstract base class for snake controllers."""

    @abstractmethod
    def get_action(self, game_state: dict[str, Any]) -> Action:
        """Return the action for this tick given the current game state."""
        ...


class RandomPlayer(Player):
    """Picks a random valid action every tick."""

    def get_action(self, game_state: dict[str, Any]) -> Action:
        return random.choice(list(Action))


class HumanPlayer(Player):
    """Keyboard-controlled player.

    Controls:
        A       – turn left
        D       – turn right
        Space   – boost (hold) in current direction
        W       – boost straight (alternative)
        No key  – go straight
    """

    def __init__(self) -> None:
        self._buffered_action: Action = Action.STRAIGHT

    def buffer_key(self, key: int) -> None:
        """Buffer a keypress event for the next tick."""
        if key == pygame.K_a:
            self._buffered_action = Action.TURN_LEFT
        elif key == pygame.K_d:
            self._buffered_action = Action.TURN_RIGHT
        elif key == pygame.K_w:
            self._buffered_action = Action.BOOST_STRAIGHT

    def update_held_keys(self, keys: pygame.key.ScancodeWrapper) -> None:
        """Check held keys (for boost via spacebar)."""
        if keys[pygame.K_SPACE]:
            # Space modifies current buffered action to boost variant
            if self._buffered_action == Action.STRAIGHT:
                self._buffered_action = Action.BOOST_STRAIGHT
            elif self._buffered_action == Action.TURN_LEFT:
                self._buffered_action = Action.BOOST_LEFT
            elif self._buffered_action == Action.TURN_RIGHT:
                self._buffered_action = Action.BOOST_RIGHT

    def get_action(self, game_state: dict[str, Any]) -> Action:
        action = self._buffered_action
        self._buffered_action = Action.STRAIGHT  # reset for next tick
        return action
