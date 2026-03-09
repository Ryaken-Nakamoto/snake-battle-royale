"""Grid management for Snake Battle Royale."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from snake import Pos

if TYPE_CHECKING:
    from config import GameConfig
    from snake import Snake


class Grid:
    """Manages the NxN tile grid, apple positions, and spatial queries."""

    def __init__(self, config: GameConfig) -> None:
        self.size = config.grid_size
        self.apples: set[Pos] = set()
        self._config = config

    # -- helpers --------------------------------------------------------------

    def in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def occupied_tiles(self, snakes: list[Snake]) -> set[Pos]:
        """Return all tiles currently occupied by living snakes."""
        tiles: set[Pos] = set()
        for s in snakes:
            if s.alive:
                tiles.update(s.positions)
        return tiles

    def random_free_tile(self, blocked: set[Pos]) -> Pos | None:
        """Return a random tile not in blocked or apples. None if grid is full."""
        # For very dense grids, enumerate free tiles; for sparse grids, random probe.
        total = self.size * self.size
        used = len(blocked) + len(self.apples)
        if used >= total:
            return None

        # Random probing is efficient when usage < 50 %
        if used < total * 0.5:
            for _ in range(1000):
                pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if pos not in blocked and pos not in self.apples:
                    return pos

        # Fall back to enumeration
        all_free = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in blocked and (x, y) not in self.apples
        ]
        return random.choice(all_free) if all_free else None

    # -- apple management -----------------------------------------------------

    def spawn_apples(self, count: int, snakes: list[Snake]) -> None:
        """Spawn `count` apples on free tiles."""
        blocked = self.occupied_tiles(snakes)
        for _ in range(count):
            pos = self.random_free_tile(blocked)
            if pos is None:
                break
            self.apples.add(pos)
            blocked.add(pos)  # don't double-place

    def maintain_apple_count(self, target: int, snakes: list[Snake]) -> None:
        """Spawn apples until we reach `target` (or grid is full)."""
        deficit = target - len(self.apples)
        if deficit > 0:
            self.spawn_apples(deficit, snakes)

    def get_apple_positions(self) -> set[Pos]:
        return self.apples
    
    def get_grid_size(self) -> int:
        return self.size
