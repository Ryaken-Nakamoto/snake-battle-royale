"""Entry point for Snake Battle Royale."""

from __future__ import annotations

import sys

import pygame

from config import load_config
from game import Game
from player import HumanPlayer, Player, RandomPlayer
from renderer import Renderer
from data import get_grid_data


def build_players(num_snakes: int, human_index: int = 0) -> list[Player]:
    """Create the player list: one HumanPlayer, rest RandomPlayers."""
    players: list[Player] = []
    for i in range(num_snakes):
        if i == human_index:
            players.append(HumanPlayer())
        else:
            players.append(RandomPlayer())
    return players


def main() -> None:
    config = load_config("config.json")
    human_index = 0
    players = build_players(config.num_snakes, human_index)
    game = Game(config, players)
    renderer = Renderer(config, human_index)
    human_player: HumanPlayer = players[human_index]  # type: ignore[assignment]

    clock = pygame.time.Clock()
    paused = False

    while True:
        # -- event handling ---------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.cleanup()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    renderer.cleanup()
                    sys.exit()
                elif event.key == pygame.K_p:
                    paused = not paused
                else:
                    human_player.buffer_key(event.key)

        if not paused and not game.game_over:
            # Check held keys for boost
            keys = pygame.key.get_pressed()
            human_player.update_held_keys(keys)

            # Advance game
            game.progress_game()

        renderer.draw(game, paused)
        clock.tick(config.fps)
        data = get_grid_data(game)
        for row in data:
            print("".join(f"{item:<2}" for item in row))


if __name__ == "__main__":
    main()
