from __future__ import annotations

import csv
import os

from config import load_config
from data import get_features
from game import Game
from neural_network import N_FEATURES
from player import RandomPlayer, SafeRandomPlayer
from rl_agent import ACTIONS, DQNAgent

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_SIZE        = 64
LR                 = 1e-3
GAMMA              = 0.99
BUFFER_CAPACITY    = 50_000
BATCH_SIZE         = 1024
TARGET_UPDATE_FREQ = 500    # global steps between target network syncs

EPSILON_START      = 1.0
EPSILON_MIN        = 0.05
EPSILON_DECAY      = 0.99999  # multiplied per episode


REWARD_ALIVE       =  1.0
REWARD_APPLE       =  50.0
REWARD_DIE         = -100.0

RL_SNAKE_ID        = 0
RESULTS_DIR        = "results"
WEIGHTS_DIR        = "weights"
LOG_PATH           = os.path.join(RESULTS_DIR, "rl_training.csv")
CHECKPOINT_PATH    = os.path.join(WEIGHTS_DIR, "rl_agent.pt")


def _init_log(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "length", "ticks", "epsilon", "avg_loss"])


def _append_log(path: str, ep: int, reward: float, length: int, ticks: int, eps: float, loss: float | None) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([ep, round(reward, 2), length, ticks, round(eps, 4), f"{loss:.4f}" if loss is not None else ""])


def train_agent(num_episodes: int = 1_000_000, config_path: str = "config.json") -> DQNAgent:
    config = load_config(config_path)
    agent = DQNAgent(
        hidden=HIDDEN_SIZE,
        lr=LR,
        gamma=GAMMA,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
    )
    _init_log(LOG_PATH)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    epsilon = EPSILON_START
    print(f"[dqn] Training for {num_episodes} episodes  device={agent.device}")

    window_reward = 0.0
    window_length = 0
    window_loss_sum = 0.0
    window_loss_n = 0

    for episode in range(1, num_episodes + 1):
        # Slot 0 is a dummy — its action is always overridden via player_decisions
        opponents = []
        for i in range(1, config.num_snakes):
            p = SafeRandomPlayer()
            p.snake_id = i
            opponents.append(p)
        players = [RandomPlayer()] + opponents
        game = Game(config, players)

        state = get_features(game.get_game_state(), RL_SNAKE_ID)
        prev_len = config.initial_snake_length
        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_loss_n = 0

        while not game.game_over:
            action_idx = agent.select_action(state, epsilon)
            game.progress_game(player_decisions={RL_SNAKE_ID: ACTIONS[action_idx]})

            gs = game.get_game_state()
            rl_snake = next(s for s in gs["snakes"] if s["id"] == RL_SNAKE_ID)
            alive = rl_snake["alive"]

            if not alive:
                reward, done = REWARD_DIE, True
                next_state = [0.0] * N_FEATURES
            else:
                reward = REWARD_ALIVE
                if rl_snake["length"] > prev_len:
                    reward += REWARD_APPLE
                    prev_len = rl_snake["length"]
                done = game.game_over
                next_state = get_features(gs, RL_SNAKE_ID)

            agent.buffer.push(state, action_idx, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                ep_loss_sum += loss
                ep_loss_n += 1

            ep_reward += reward
            state = next_state
            if done or not alive:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        final = next(s for s in gs["snakes"] if s["id"] == RL_SNAKE_ID)
        ep_avg_loss = ep_loss_sum / ep_loss_n if ep_loss_n else None
        _append_log(LOG_PATH, episode, ep_reward, final["length"], game.tick, epsilon, ep_avg_loss)

        window_reward += ep_reward
        window_length += final["length"]
        if ep_avg_loss is not None:
            window_loss_sum += ep_avg_loss
            window_loss_n += 1

        if episode % 100 == 0:
            print(
                f"Ep {episode:>6}/{num_episodes}  "
                f"reward={window_reward / 100:>8.1f}  "
                f"len={window_length / 100:>5.1f}  "
                f"eps={epsilon:.3f}  "
                f"loss={f'{window_loss_sum / window_loss_n:.4f}' if window_loss_n else 'n/a'}"
            )
            window_reward = 0.0
            window_length = 0
            window_loss_sum = 0.0
            window_loss_n = 0
        if episode % 1_000 == 0:
            agent.save(CHECKPOINT_PATH)

    agent.save(CHECKPOINT_PATH)
    print(f"\n[dqn] Done. Checkpoint → {CHECKPOINT_PATH}  Log → {LOG_PATH}")
    return agent


if __name__ == "__main__":
    train_agent()
