"""Load a saved genome from CSV and watch it play.

Usage:
    python visualize_genome.py <weights.csv>
    python visualize_genome.py weights/default_results_weights.csv
    python visualize_genome.py weights/default_results_weights.csv --nn two_layer

Controls during the game:
    P     pause / unpause
    SPACE speed up (hold)
    ESC   quit
    Q     quit
"""

from __future__ import annotations

import argparse
import csv
import sys

import pygame

from config import load_config
from game import Game
from genetic import GeneticPlayer
from neural_network import (
    Base_algorithm,
    Basic_Neural_Network,
    Neural_Network,
    Two_Layer_Neural_Network,
)
from player import Player, SafeRandomPlayer
from renderer import Renderer


NN_CLASSES: dict[str, type[Neural_Network]] = {
    "basic":     Basic_Neural_Network,
    "two_layer": Two_Layer_Neural_Network,
    "base":      Base_algorithm,
}


def load_genome(csv_path: str) -> list[float]:
    """Load a genome from a CSV file produced by save_genome().

    The file has a header row 'weight' followed by one float per row.
    """
    weights: list[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row
        for row in reader:
            if not row:
                continue
            weights.append(float(row[0]))
    if not weights:
        raise ValueError(f"No weights found in {csv_path}")
    return weights


def build_players(
    num_snakes: int,
    nn_class: type[Neural_Network],
    genome: list[float],
    showcase_index: int = 0,
) -> list[Player]:
    """One GeneticPlayer at showcase_index running the loaded genome; the rest
    are SafeRandomPlayer opponents. Use num_snakes=1 in config to watch it solo.
    """
    players: list[Player] = []
    for i in range(num_snakes):
        if i == showcase_index:
            players.append(GeneticPlayer(i, genome, nn_class))
        else:
            p = SafeRandomPlayer()
            p.snake_id = i
            players.append(p)
    return players


def run_visual_game(
    csv_path: str,
    nn_name: str = "basic",
    config_path: str = "config.json",
    showcase_index: int = 0,
    base_fps: int | None = None,
) -> None:
    if nn_name not in NN_CLASSES:
        raise ValueError(f"Unknown nn class '{nn_name}'. Choices: {list(NN_CLASSES)}")
    nn_class = NN_CLASSES[nn_name]

    genome = load_genome(csv_path)
    expected = nn_class.genome_length()
    if len(genome) != expected:
        print(
            f"[warn] genome length mismatch: file has {len(genome)} weights, "
            f"{nn_class.__name__} expects {expected}. Loading anyway — this "
            f"usually means the feature count or nn class changed since this "
            f"genome was saved.",
            file=sys.stderr,
        )

    config = load_config(config_path)
    players = build_players(config.num_snakes, nn_class, genome, showcase_index)
    game = Game(config, players)
    renderer = Renderer(config, showcase_index)

    fps = base_fps if base_fps is not None else config.fps
    clock = pygame.time.Clock()
    paused = False

    print(
        f"[visualize] {csv_path}  nn={nn_class.__name__}  "
        f"num_snakes={config.num_snakes}  fps={fps}"
    )
    print("[visualize] P=pause  SPACE=fast-forward (hold)  ESC/Q=quit")

    while True:
        fast = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.cleanup()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    renderer.cleanup()
                    return
                elif event.key == pygame.K_p:
                    paused = not paused

        # Check held keys for fast-forward
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            fast = True

        if not paused and not game.game_over:
            game.progress_game()

            if game.game_over:
                # Print a brief summary once, then stay in the window until the
                # user closes it so they can inspect the final state.
                showcase = game.snakes[showcase_index]
                print(
                    f"[visualize] game over — showcase snake: "
                    f"length={showcase.length}  alive={showcase.alive}  "
                    f"ticks={game.tick}"
                )

        renderer.draw(game, paused)
        clock.tick(fps * 4 if fast else fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a saved genome play.")
    parser.add_argument("csv_path", help="Path to the weights CSV (from save_genome)")
    parser.add_argument(
        "--nn",
        default="basic",
        choices=list(NN_CLASSES),
        help="Which Neural_Network subclass to load the genome into (default: basic)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Game config path (default: config.json)",
    )
    parser.add_argument(
        "--slot",
        type=int,
        default=0,
        help="Which snake slot the showcase agent plays in (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override config fps (e.g. --fps 10 to slow it down)",
    )
    args = parser.parse_args()

    run_visual_game(
        csv_path=args.csv_path,
        nn_name=args.nn,
        config_path=args.config,
        showcase_index=args.slot,
        base_fps=args.fps,
    )


if __name__ == "__main__":
    main()