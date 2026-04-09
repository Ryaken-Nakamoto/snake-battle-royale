from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import Game
    from snake import Snake


class FitnessFunction(ABC):
    @abstractmethod
    def score(self, snake: "Snake", game: "Game") -> float:
        """Return a fitness score for this snake after the game ends."""
        ...


class LengthFitness(FitnessFunction):
    """Original fitness: final snake length."""

    def score(self, snake: "Snake", game: "Game") -> float:
        return float(snake.length)


class GrowthEfficiencyFitness(FitnessFunction):
    """Rewards ONLY how quickly the snake grew.

    Formula: (growth / ticks)

    - growth  = apples eaten (length - initial_snake_length)
    - ticks   = game.tick (ticks elapsed)

    """

    SCALE: float = 10.0

    def score(self, snake: "Snake", game: "Game") -> float:
        growth = snake.length - game.config.initial_snake_length
        ticks = max(game.tick, 1)
        rate = growth / ticks
        return rate * 100 # because at first performs very poorly, rounding errors
