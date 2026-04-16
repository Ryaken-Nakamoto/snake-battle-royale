"""Load a saved DQN checkpoint and watch the agent play.

Usage:
    python visualize_dqn.py
    python visualize_dqn.py weights/rl_agent_best.pt
    python visualize_dqn.py weights/rl_agent.pt --fps 10
    python visualize_dqn.py weights/rl_agent_best.pt --epsilon 0.05

Controls during the game:
    P     pause / unpause
    SPACE fast-forward (hold)
    ESC   quit
    Q     quit
"""

from __future__ import annotations

import argparse
import sys

import pygame

from config import load_config
from game import Game
from player import Player, RandomPlayer, SafeRandomPlayer
from renderer import Renderer
from rl_agent import ACTIONS, DQNAgent, RLPlayer


DEFAULT_CHECKPOINT = "weights/rl_agent_best.pt"


def build_players(
    num_snakes: int,
    agent: DQNAgent,
    showcase_index: int = 0,
) -> list[Player]:
    """Showcase RLPlayer in slot `showcase_index`, SafeRandomPlayer elsewhere."""
    players: list[Player] = []
    for i in range(num_snakes):
        if i == showcase_index:
            players.append(RLPlayer(i, agent))
        else:
            p = SafeRandomPlayer()
            p.snake_id = i
            players.append(p)
    return players


def run_visual_game(
    checkpoint_path: str,
    config_path: str = "config.json",
    showcase_index: int = 0,
    base_fps: int | None = None,
    epsilon: float = 0.0,
) -> None:
    config = load_config(config_path)

    # Build agent and load weights. DQNAgent's __init__ creates an untrained
    # network; load() then replaces the weights with the checkpoint.
    agent = DQNAgent()
    try:
        agent.load(checkpoint_path)
    except FileNotFoundError:
        print(f"[error] checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print("[hint] train first, or pass a different path as the first argument.", file=sys.stderr)
        return
    except Exception as e:
        print(f"[error] failed to load {checkpoint_path}: {e}", file=sys.stderr)
        return

    # RLPlayer always acts greedily; if you want to inject some exploration
    # noise we use a small wrapper that respects the epsilon argument.
    class _NoisyRLPlayer(Player):
        def __init__(self, pid: int, a: DQNAgent, eps: float) -> None:
            self.player_id = pid
            self._agent = a
            self._eps = eps

        def get_action(self, game_state):
            from data import get_features
            from snake import Action
            features = get_features(game_state, self.player_id)
            if not features:
                return Action.STRAIGHT
            return ACTIONS[self._agent.select_action(features, self._eps)]

    players: list[Player] = []
    for i in range(config.num_snakes):
        if i == showcase_index:
            if epsilon > 0.0:
                players.append(_NoisyRLPlayer(i, agent, epsilon))
            else:
                players.append(RLPlayer(i, agent))
        else:
            p = SafeRandomPlayer()
            p.snake_id = i
            players.append(p)

    game = Game(config, players)
    renderer = Renderer(config, showcase_index)

    fps = base_fps if base_fps is not None else config.fps
    clock = pygame.time.Clock()
    paused = False

    print(
        f"[visualize-dqn] {checkpoint_path}  device={agent.device}  "
        f"num_snakes={config.num_snakes}  fps={fps}  epsilon={epsilon}"
    )
    print("[visualize-dqn] P=pause  SPACE=fast-forward (hold)  ESC/Q=quit")

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

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            fast = True

        if not paused and not game.game_over:
            game.progress_game()

            if game.game_over:
                showcase = game.snakes[showcase_index]
                print(
                    f"[visualize-dqn] game over — showcase snake: "
                    f"length={showcase.length}  alive={showcase.alive}  "
                    f"ticks={game.tick}"
                )

        renderer.draw(game, paused)
        clock.tick(fps * 4 if fast else fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a saved DQN agent play.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to .pt checkpoint (default: {DEFAULT_CHECKPOINT})",
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
        help="Which snake slot the agent plays in (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override config fps (e.g. --fps 10 to slow it down)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration noise while watching (default: 0 = fully greedy)",
    )
    args = parser.parse_args()

    run_visual_game(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        showcase_index=args.slot,
        base_fps=args.fps,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()