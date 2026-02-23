"""Snake entity for Snake Battle Royale."""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import GameConfig


class Direction(Enum):
    """Cardinal directions."""
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)


class Action(Enum):
    """Possible player actions per tick."""
    STRAIGHT = "straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    BOOST_STRAIGHT = "boost_straight"
    BOOST_LEFT = "boost_left"
    BOOST_RIGHT = "boost_right"

    @property
    def is_boost(self) -> bool:
        return self.name.startswith("BOOST")

    @property
    def turn_component(self) -> str:
        """Return 'left', 'right', or 'straight'."""
        if "LEFT" in self.name:
            return "left"
        if "RIGHT" in self.name:
            return "right"
        return "straight"


# Clockwise ordering for turn calculations
_CW = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]


def turn_left(d: Direction) -> Direction:
    return _CW[(_CW.index(d) - 1) % 4]


def turn_right(d: Direction) -> Direction:
    return _CW[(_CW.index(d) + 1) % 4]


def apply_turn(d: Direction, action: Action) -> Direction:
    """Return the new direction after applying the turn component of an action."""
    component = action.turn_component
    if component == "left":
        return turn_left(d)
    if component == "right":
        return turn_right(d)
    return d


Pos = tuple[int, int]


class Snake:
    """A single snake on the grid."""

    def __init__(
        self,
        snake_id: int,
        name: str,
        positions: list[Pos],
        direction: Direction,
        color: tuple[int, int, int],
        config: GameConfig,
    ) -> None:
        self.snake_id = snake_id
        self.name = name
        self.positions: list[Pos] = list(positions)  # head = positions[0]
        self.direction = direction
        self.color = color
        self.alive = True
        self._config = config

        # Stamina
        self.stamina: float = self.max_stamina
        self._grow_pending: int = 0  # tiles to grow (tail stays)

    # -- properties -----------------------------------------------------------

    @property
    def head(self) -> Pos:
        return self.positions[0]

    @property
    def length(self) -> int:
        return len(self.positions)

    @property
    def max_stamina(self) -> float:
        return math.floor(self._config.max_stamina_factor * math.sqrt(self.length))

    # -- stamina helpers ------------------------------------------------------

    def _recovery_per_tick(self) -> float:
        ms = self.max_stamina
        if ms <= 0:
            return 0.0
        denom = self._config.stamina_recovery_factor * math.sqrt(self.length)
        if denom <= 0:
            return ms
        return ms / denom

    def drain_stamina(self) -> None:
        """Drain stamina for one boost tick."""
        self.stamina = max(0.0, self.stamina - self._config.stamina_drain_rate)

    def recover_stamina(self) -> None:
        """Recover stamina for one non-boost tick."""
        self.stamina = min(self.max_stamina, self.stamina + self._recovery_per_tick())

    def can_boost(self) -> bool:
        return self.stamina >= self._config.stamina_drain_rate

    # -- movement -------------------------------------------------------------

    def compute_new_heads(self, action: Action) -> list[Pos]:
        """Return the list of new head positions for this tick (1 or 2)."""
        new_dir = apply_turn(self.direction, action)
        dx, dy = new_dir.value
        hx, hy = self.head

        steps = 1
        boosting = action.is_boost and self.can_boost()
        if boosting:
            steps = int(self._config.boost_multiplier)

        heads: list[Pos] = []
        for i in range(1, steps + 1):
            heads.append((hx + dx * i, hy + dy * i))
        return heads

    def apply_move(self, new_heads: list[Pos], action: Action) -> None:
        """Move the snake by prepending new heads and removing tail(s)."""
        self.direction = apply_turn(self.direction, action)

        # Insert new heads at the front
        for h in reversed(new_heads):
            self.positions.insert(0, h)

        # Remove from tail: we added len(new_heads) segments, remove same minus grow
        remove_count = len(new_heads) - self._grow_pending
        if remove_count > 0:
            self.positions = self.positions[: -remove_count] if remove_count < len(self.positions) else self.positions[:1]
            self._grow_pending = 0
        else:
            self._grow_pending -= len(new_heads)

    def schedule_grow(self, amount: int = 1) -> None:
        """Queue growth: tail won't be removed for the next `amount` moves."""
        self._grow_pending += amount

    def kill(self) -> None:
        self.alive = False
