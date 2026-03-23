
"""Configuration loading and validation for Snake Battle Royale."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameConfig:
    """Immutable game configuration loaded from JSON."""

    grid_size: int
    tile_size: int
    num_snakes: int
    initial_snake_length: int
    apple_density: int
    win_length: int
    boost_multiplier: float
    max_stamina_factor: float
    stamina_recovery_factor: float
    stamina_drain_rate: float
    fps: int
    limit_scope: bool = False
    max_moves_flag: bool = False
    max_moves: int = 1

    @property
    def total_tiles(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def initial_apple_count(self) -> int:
        return math.floor(self.total_tiles / self.apple_density)

    def validate(self) -> None:
        """Raise ValueError if the configuration is invalid."""
        if self.grid_size < 20:
            raise ValueError("grid_size must be at least 20")
        if self.num_snakes < 1:
            raise ValueError("num_snakes must be at least 1")
        if self.initial_snake_length < 1:
            raise ValueError("initial_snake_length must be at least 1")
        if self.max_moves < 1:
            raise ValueError("max_moves must be at least 1")

        tiles_for_apples = self.initial_apple_count
        tiles_for_snakes = self.num_snakes * self.initial_snake_length
        if tiles_for_apples + tiles_for_snakes > self.total_tiles:
            raise ValueError(
                "Grid too small to fit all snakes and apples. "
                "Increase grid_size or reduce apple_density / num_snakes."
            )


def load_config(path: str | Path = "config.json") -> GameConfig:
    """Load and validate game config from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    cfg = GameConfig(**data)
    cfg.validate()
    return cfg
