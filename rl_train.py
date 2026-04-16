from __future__ import annotations
 
import csv
import os
 
from config import load_config
from data import get_features, N_FEATURES
from game import Game
from player import RandomPlayer, SafeRandomPlayer
from rl_agent import ACTIONS, DQNAgent
 
# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_SIZE        = 128
LR                 = 3e-4
GAMMA              = 0.99
BUFFER_CAPACITY    = 5_000
BATCH_SIZE         = 128
TARGET_UPDATE_FREQ = 500   # gradient steps between target network syncs
GRAD_CLIP          = 10.0
 
EPSILON_START      = 1.0
EPSILON_MIN        = 0.00
EPSILON_DECAY      = 0.999  # multiplied per episode (reaches 0.05 around ep 6000)
 
 
REWARD_ALIVE       =  0.00
REWARD_DIST        =  0.1
REWARD_APPLE       =  1.0
REWARD_DIE         = -1.0
 
RL_SNAKE_ID        = 0
EVAL_EVERY         = 1000   # episodes between greedy showcase games
EVAL_FPS           = 3      # rendering speed for eval games (lower = easier to watch)
PLOT_EVERY         = 10000   # episodes between saving a progress plot (0 to disable)
SMOOTH_WINDOW      = 100    # episodes averaged together for the smoothed curve
RESULTS_DIR        = "results"
WEIGHTS_DIR        = "weights"
GRAPHS_DIR         = "graphs"
LOG_PATH           = os.path.join(RESULTS_DIR, "rl_training.csv")
PLOT_PATH          = os.path.join(GRAPHS_DIR, "rl_training_length.png")
CHECKPOINT_PATH    = os.path.join(WEIGHTS_DIR, "rl_agent.pt")
BEST_CHECKPOINT    = os.path.join(WEIGHTS_DIR, "rl_agent_best.pt")
 
 
def _init_log(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "length", "ticks", "epsilon", "avg_loss"])
 
 
def _append_log(path: str, ep: int, reward: float, length: int, ticks: int, eps: float, loss: float | None) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([ep, round(reward, 2), length, ticks, round(eps, 4), f"{loss:.4f}" if loss is not None else ""])
 
 
def _nearest_apple_dist(gs: dict, snake_id: int) -> float | None:
    snake = next((s for s in gs["snakes"] if s["id"] == snake_id), None)
    if snake is None or not snake["alive"] or not gs["apples"]:
        return None
    hx, hy = snake["positions"][0]
    return min(abs(ax - hx) + abs(ay - hy) for ax, ay in gs["apples"])
 
 
def _plot_training_progress(csv_path: str, plot_path: str, smooth_window: int = SMOOTH_WINDOW) -> None:
    """Read the training CSV and save a PNG of length over time.
 
    Plots per-episode length as a faint line plus a rolling mean over
    smooth_window episodes as the main curve, so the trend stays readable
    even with thousands of noisy episodes.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend so this works during training
    import matplotlib.pyplot as plt
 
    episodes: list[int] = []
    lengths: list[float] = []
    rewards: list[float] = []
 
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                lengths.append(float(row["length"]))
                rewards.append(float(row["reward"]))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[plot] could not read {csv_path}: {e}")
        return
 
    if not episodes:
        return
 
    # Rolling mean for the smoothed curve.
    def _rolling_mean(xs: list[float], window: int) -> list[float]:
        if window <= 1 or len(xs) < window:
            return xs[:]
        out: list[float] = []
        running = sum(xs[:window])
        out.extend([running / window] * window)  # pad the start so lengths line up
        for i in range(window, len(xs)):
            running += xs[i] - xs[i - window]
            out.append(running / window)
        return out
 
    smoothed = _rolling_mean(lengths, smooth_window)
 
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, lengths, color="steelblue", alpha=0.2, linewidth=0.8, label="per-episode length")
    ax.plot(episodes, smoothed, color="steelblue", linewidth=2.0, label=f"{smooth_window}-episode rolling mean")
 
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Snake length at end of episode", fontsize=12)
    ax.set_title("DQN Training Progress — Length Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
 
    # Annotate with the current best smoothed length.
    if smoothed:
        best = max(smoothed)
        best_ep = episodes[smoothed.index(best)]
        ax.axhline(best, color="darkorange", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(
            0.98, 0.02,
            f"best smoothed: {best:.1f} @ ep {best_ep}\nlatest: {smoothed[-1]:.1f} @ ep {episodes[-1]}",
            transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
 
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
 
 
def _run_eval_game(agent: DQNAgent, config, episode: int) -> None:
    """Pop open a pygame window and play one greedy game with the agent.
 
    Uses your existing Renderer. Closes the window when the agent dies, the
    user closes the window, or the user hits ESC. Pause with P, skip the rest
    of the eval with SPACE.
    """
    import pygame
    from renderer import Renderer
 
    # Build a fresh game with the agent in slot 0 and SafeRandomPlayer opponents.
    opponents = []
    for i in range(1, config.num_snakes):
        p = SafeRandomPlayer()
        p.snake_id = i
        opponents.append(p)
    players = [RandomPlayer()] + opponents  # slot 0 dummy, overridden each tick
    game = Game(config, players)
 
    renderer = Renderer(config, RL_SNAKE_ID)
    clock = pygame.time.Clock()
    paused = False
    skip = False
 
    print(f"\n[eval @ ep {episode}] showing greedy game — P=pause  SPACE=skip  ESC=close")
 
    while not game.game_over and not skip:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.cleanup()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    renderer.cleanup()
                    return
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_SPACE:
                    skip = True
 
        if not paused:
            state = get_features(game.get_game_state(), RL_SNAKE_ID)
            if state:
                action_idx = agent.select_action(state, epsilon=0.0)
            else:
                action_idx = 0
            game.progress_game(player_decisions={RL_SNAKE_ID: ACTIONS[action_idx]})
 
            gs = game.get_game_state()
            rl_snake = next(s for s in gs["snakes"] if s["id"] == RL_SNAKE_ID)
            if not rl_snake["alive"]:
                renderer.draw(game, paused)
                pygame.time.wait(800)
                final_len = rl_snake["length"]
                print(f"[eval @ ep {episode}] agent died — length={final_len}  ticks={game.tick}")
                renderer.cleanup()
                return
 
        renderer.draw(game, paused)
        clock.tick(EVAL_FPS)
 
    final_gs = game.get_game_state()
    rl_final = next(s for s in final_gs["snakes"] if s["id"] == RL_SNAKE_ID)
    print(
        f"[eval @ ep {episode}] game ended — length={rl_final['length']}  "
        f"ticks={game.tick}  alive={rl_final['alive']}"
    )
    renderer.cleanup()
 
 
def train_agent(num_episodes: int = 10_000, config_path: str = "config.json") -> DQNAgent:
    config = load_config(config_path)
    agent = DQNAgent(
        hidden=HIDDEN_SIZE,
        lr=LR,
        gamma=GAMMA,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        grad_clip=GRAD_CLIP,
    )
    _init_log(LOG_PATH)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
 
    epsilon = EPSILON_START
    print(f"[dqn] Training for {num_episodes} episodes  device={agent.device}  features={N_FEATURES}")
 
    window_reward = 0.0
    window_length = 0
    window_loss_sum = 0.0
    window_loss_n = 0
    best_window_length = 0.0  # for saving the best checkpoint
 
    for episode in range(1, num_episodes + 1):
        # Slot 0 is a dummy — its action is always overridden via player_decisions
        opponents = []
        for i in range(1, config.num_snakes):
            p = SafeRandomPlayer()
            opponents.append(p)
        players = [RandomPlayer()] + opponents
        game = Game(config, players)
 
        state = get_features(game.get_game_state(), RL_SNAKE_ID)
        prev_len = config.initial_snake_length
        prev_dist = _nearest_apple_dist(game.get_game_state(), RL_SNAKE_ID)
        grid_size = config.grid_size
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
                prev_dist = None
            else:
                reward = REWARD_ALIVE
                ate_apple = rl_snake["length"] > prev_len
                if ate_apple:
                    reward += REWARD_APPLE
                    prev_len = rl_snake["length"]
 
                new_dist = _nearest_apple_dist(gs, RL_SNAKE_ID)
                # Skip shaping on the eat-tick (the apple changing makes the
                # distance jump meaningless and would punish eating).
                if not ate_apple and prev_dist is not None and new_dist is not None:
                    reward += REWARD_DIST * (prev_dist - new_dist) / grid_size
                prev_dist = new_dist
 
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
            avg_len = window_length / 100
            print(
                f"Ep {episode:>6}/{num_episodes}  "
                f"reward={window_reward / 100:>8.1f}  "
                f"len={avg_len:>5.1f}  "
                f"eps={epsilon:.3f}  "
                f"loss={f'{window_loss_sum / window_loss_n:.4f}' if window_loss_n else 'n/a'}"
            )
            # Save best-ever window so a future collapse doesn't destroy our best weights.
            if avg_len > best_window_length:
                best_window_length = avg_len
                agent.save(BEST_CHECKPOINT)
            window_reward = 0.0
            window_length = 0
            window_loss_sum = 0.0
            window_loss_n = 0
 
        if episode % 1_000 == 0:
            agent.save(CHECKPOINT_PATH)
 
        # Periodically save a progress graph to disk.
        if PLOT_EVERY > 0 and episode % PLOT_EVERY == 0:
            try:
                _plot_training_progress(LOG_PATH, PLOT_PATH)
            except Exception as e:
                print(f"[plot @ ep {episode}] failed: {e}")
 
        # Periodically pop a pygame window and watch a greedy game.
        if episode % EVAL_EVERY == 0:
            try:
                _run_eval_game(agent, config, episode)
            except Exception as e:
                print(f"[eval @ ep {episode}] failed: {e}")
 
    agent.save(CHECKPOINT_PATH)
    # Final plot at the end of training.
    try:
        _plot_training_progress(LOG_PATH, PLOT_PATH)
    except Exception as e:
        print(f"[plot final] failed: {e}")
    print(f"\n[dqn] Done. Last → {CHECKPOINT_PATH}  Best → {BEST_CHECKPOINT}")
    print(f"[dqn] Log → {LOG_PATH}  Plot → {PLOT_PATH}")
    return agent
 
 
if __name__ == "__main__":
    train_agent()