"""Entry point for Snake Battle Royale."""

from __future__ import annotations

import sys

import pygame

from config import load_config
from game import Game
from player import HumanPlayer, Player, SafeRandomPlayer
from renderer import Renderer
from genetic import GeneticPlayer, load_previous_weights
from data import get_features
import csv

# build snakes that make random actions
def build_players(num_snakes: int, human_index: int = 0) -> list[Player]:
    """Create the player list: one HumanPlayer, rest RandomPlayers."""
    players: list[Player] = []
    for i in range(num_snakes):
        if i == human_index:
           player = HumanPlayer()
        else:
            player = SafeRandomPlayer()
            player.snake_id = i 
        players.append(player)
    return players

# build snakes that make decisions based on prev found weights
def build_genetic_players(num_snakes: int, genome: list[float], human_index: int | None = None) -> list[Player]:
    """Create the player list: all GeneticPlayers sharing one genome, with optional HumanPlayer."""
    players: list[Player] = []
    for i in range(num_snakes):
        if i == human_index:
            player = HumanPlayer()
        else:
            player = GeneticPlayer(i, genome)
        players.append(player)
    return players


def main(create_human: bool, genetic_game: bool, weight_path: str | None = None) -> None:
    config = load_config("config.json")
    human_index: int | None = 0 if create_human else None

    # use snakes trained by the generic algorithm
    if genetic_game:
        genome = load_previous_weights(weight_path)
        players = build_genetic_players(config.num_snakes, genome, human_index)
    # use SafeRandom player snakes 
    else:
        players = build_players(config.num_snakes, human_index)

    game = Game(config, players)
    renderer = Renderer(config, human_index)
    human_player: HumanPlayer | None = players[human_index] if create_human else None

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
                elif human_player is not None:
                    human_player.buffer_key(event.key)

        if not paused and not game.game_over:
            # Check held keys for boost
            keys = pygame.key.get_pressed()

            if human_player is not None:
                human_player.update_held_keys(keys)

            # Advance game
            game.progress_game()

        renderer.draw(game, paused)
        clock.tick(config.fps)
        for i in range(len(game.players)):
            features = get_features(game.get_game_state(), i)


if __name__ == "__main__":
    args = sys.argv[1:]
    config = load_config("config.json")

    weight_path: str | None = None
    if "--prev-weights" in args:
        idx = args.index("--prev-weights")
        weight_path = args[idx + 1]

    if weight_path and "human" in args:
        print(f"Staring game: 1 human player, {config.num_snakes - 1} AI snake players 🐍")
        main(create_human=True, genetic_game=True, weight_path=weight_path)
    elif weight_path:
        print(f"Staring game: {config.num_snakes} AI snake players 🐍")
        main(create_human=False, genetic_game=True, weight_path=weight_path)
    else:
        print(f"Staring game: 1 human player, {config.num_snakes} random snake players 🐍")
        main(create_human=True, genetic_game=False)
