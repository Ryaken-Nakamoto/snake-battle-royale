"""Player interface and implementations for Snake Battle Royale."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pygame

from snake import Action, Direction, turn_left, turn_right

if TYPE_CHECKING:
    pass

_KEY_TO_DIR = {
    pygame.K_w: Direction.NORTH,
    pygame.K_UP: Direction.NORTH,
    pygame.K_s: Direction.SOUTH,
    pygame.K_DOWN: Direction.SOUTH,
    pygame.K_a: Direction.WEST,
    pygame.K_LEFT: Direction.WEST,
    pygame.K_d: Direction.EAST,
    pygame.K_RIGHT: Direction.EAST,
}

_OPPOSITE = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


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

    Controls use absolute directions — the active keys change based on
    which way the snake is currently facing:

        Moving horizontally (East/West):  W = North,  S = South
        Moving vertically (North/South):  A = West,   D = East
        Space (hold)                    – boost in current direction
        No key                          – go straight
    """

    def __init__(self, snake_id: int = 0) -> None:
        self.snake_id = snake_id
        self._buffered_dir: Direction | None = None
        self._boost: bool = False

    def buffer_key(self, key: int) -> None:
        """Buffer a keydown event for the next tick."""
        if key in _KEY_TO_DIR:
            self._buffered_dir = _KEY_TO_DIR[key]

    def update_held_keys(self, keys: pygame.key.ScancodeWrapper) -> None:
        """Check held keys each tick (overrides single-tap buffer)."""
        for k, d in _KEY_TO_DIR.items():
            if keys[k]:
                self._buffered_dir = d
                break
        self._boost = bool(keys[pygame.K_SPACE])

    def get_action(self, game_state: dict[str, Any]) -> Action:
        # Find this snake's current direction from game state
        current_dir = Direction.NORTH  # fallback (should always be found)
        for snake in game_state["snakes"]:
            if snake["id"] == self.snake_id:
                current_dir = snake["direction"]
                break

        desired = self._buffered_dir
        self._buffered_dir = None  # reset for next tick

        # Convert absolute desired direction → relative action
        if desired is None or desired == current_dir or desired == _OPPOSITE[current_dir]:
            base = Action.STRAIGHT
        elif desired == turn_left(current_dir):
            base = Action.TURN_LEFT
        else:
            base = Action.TURN_RIGHT

        if self._boost:
            return {
                Action.STRAIGHT: Action.BOOST_STRAIGHT,
                Action.TURN_LEFT: Action.BOOST_LEFT,
                Action.TURN_RIGHT: Action.BOOST_RIGHT,
            }[base]

        return base
