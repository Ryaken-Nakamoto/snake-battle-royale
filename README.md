# Snake Battle Royale — AI Training Project

A multiplayer snake game where AI-controlled snakes are trained to compete against each other (and optionally a human player). Two training approaches are supported: a **genetic algorithm** that evolves neural-network weights, and a **Deep Q-Learning (DQN)** agent trained via reinforcement learning.

---

## Requirements

Python 3.10+, plus:

```bash
pip install pygame numpy matplotlib torch
```

---

## How to Play

### Watch trained genetic AI snakes compete

```bash
python main.py --prev-weights weights/my_weights.csv
```

All snakes are genetic AI players using the loaded weights. Bare filenames are looked up in `weights/`.

### Play against trained genetic AI

```bash
python main.py --prev-weights weights/my_weights.csv human
```

Adds you as a human player. Controls: `W`/`S` or `A`/`D` to turn (depending on current direction), `Space` to boost, `P` to pause, `ESC` to quit.

### Watch a trained DQN agent

```bash
python visualize_dqn.py                                    # uses weights/rl_agent_best.pt
python visualize_dqn.py weights/rl_agent.pt
python visualize_dqn.py weights/rl_agent_best.pt --fps 5
python visualize_dqn.py weights/rl_agent_best.pt --epsilon 0.05
python visualize_dqn.py weights/rl_agent_best.pt --slot 2 --config config.json
```

The DQN agent plays against `SafeRandom` opponents. Controls while watching: `P` pause, `Space` (hold) fast-forward 4×, `ESC`/`Q` quit.

| Flag | Default | Description |
|------|---------|-------------|
| positional `checkpoint` | `weights/rl_agent_best.pt` | Path to `.pt` checkpoint |
| `--config` | `config.json` | Game config path |
| `--slot` | `0` | Which snake index the agent controls |
| `--fps` | config value | Override rendering speed |
| `--epsilon` | `0.0` | Exploration noise (0 = fully greedy) |

### Watch a saved genetic genome (standalone)

```bash
python visualize_genome.py weights/default_results_weights.csv
python visualize_genome.py weights/default_results_weights.csv --nn two_layer
python visualize_genome.py weights/my_weights.csv --slot 1 --fps 8
```

Loads a genome CSV and plays it against `SafeRandom` opponents. Same keyboard controls as `visualize_dqn.py`.

| Flag | Default | Description |
|------|---------|-------------|
| positional `csv_path` | required | Path to genome CSV |
| `--nn` | `basic` | NN class: `basic`, `two_layer`, or `base` |
| `--config` | `config.json` | Game config path |
| `--slot` | `0` | Which snake index the genome controls |
| `--fps` | config value | Override rendering speed |

### Play against random snakes (no training required)

```bash
python main.py
```

Snakes make safe random moves (they avoid immediately lethal actions). Good for testing the game itself.

---

## State Representation (25 features)

All AI controllers — both genetic and DQN — see the same 25-dimensional feature vector computed by `data.py`:

| Features | Count | Description |
|----------|-------|-------------|
| 8 directional rays × (normalised distance, is-wall flag) | 16 | Distance to nearest obstacle in 8 directions |
| Nearest apple unit vector + Manhattan distance | 3 | Direction and distance to closest apple |
| Second-nearest apple unit vector | 2 | Direction to next closest apple |
| Stamina fraction | 1 | `stamina / max_stamina` |
| Length fraction | 1 | `length / win_length` |
| Nearest enemy head unit vector | 2 | Direction to closest other snake's head |

---

## Training AI Snakes

### Genetic algorithm

#### Quick single run

```bash
python genetic.py
```

Trains with default hyperparameters. Outputs:

```
results/default_results.csv          ← fitness per generation
graphs/default_fitness.png           ← convergence plot
weights/default_results_weights.csv  ← best genome (load with main.py or visualize_genome.py)
```

#### Resume training from existing weights

```bash
python genetic.py --prev-weights my_weights.csv config.json my_run_name
```

Seeds the population from `weights/my_weights.csv` and continues training. Outputs saved under `my_run_name`. Full paths also work:

```bash
python genetic.py --prev-weights runs/coarse_v1/candidates/0008_mr=0.15,ms=0.30/rep0/weights.csv
```

#### Run a set of named experiments

Define experiments in `experiments.json`:

```json
[
  { "name": "baseline" },
  { "name": "high_mutation", "mutation_rate": 0.3, "mutation_strength": 0.5 },
  { "name": "high_elitism",  "elitism_count": 4 }
]
```

```bash
python genetic.py --sweep
```

Each experiment runs in sequence and saves its own results, plot, and weights.

---

### Deep Q-Learning (DQN)

#### Training

```bash
python rl_train.py
```

No CLI arguments — all hyperparameters are module-level constants at the top of `rl_train.py`. Edit them directly before running:

| Constant | Default | Description |
|----------|---------|-------------|
| `HIDDEN_SIZE` | `128` | Hidden units per layer (two hidden layers) |
| `LR` | `3e-4` | Adam learning rate |
| `GAMMA` | `0.99` | Discount factor |
| `BUFFER_CAPACITY` | `5_000` | Replay buffer size |
| `BATCH_SIZE` | `128` | Samples per gradient step |
| `TARGET_UPDATE_FREQ` | `500` | Gradient steps between target network syncs |
| `EPSILON_START` | `1.0` | Initial exploration rate |
| `EPSILON_MIN` | `0.0` | Minimum exploration rate |
| `EPSILON_DECAY` | `0.999` | Per-episode decay (reaches ~0.05 around episode 6,000) |
| `REWARD_ALIVE` | `0.0` | Reward per tick alive |
| `REWARD_DIST` | `0.1` | Reward scaling for getting closer to apple |
| `REWARD_APPLE` | `1.0` | Reward for eating an apple |
| `REWARD_DIE` | `-1.0` | Reward on death |
| `EVAL_EVERY` | `1000` | Episodes between greedy eval games |
| `PLOT_EVERY` | `10000` | Episodes between progress-plot saves (0 to disable) |

Default training run is 10,000 episodes (`train_agent(num_episodes=10_000)`).

Outputs:

```
weights/rl_agent.pt            ← latest checkpoint (saved every 1,000 episodes)
weights/rl_agent_best.pt       ← best checkpoint by 100-episode rolling mean length
results/rl_training.csv        ← per-episode log: episode, reward, length, ticks, epsilon, avg_loss
graphs/rl_training_length.png  ← training curve
```

Every `EVAL_EVERY` episodes a pygame window pops showing a greedy game. Controls during eval: `P` pause, `Space` skip eval immediately, `ESC` close.

#### Algorithm details

- **Architecture**: Double DQN with a target network. Online and target nets are both 3-layer MLPs (25 → 128 → 128 → 6). Target net synced every 1,000 gradient steps.
- **Exploration**: epsilon-greedy, decaying multiplicatively each episode.
- **Reward shaping**: `+REWARD_DIST * Δ(distance-to-nearest-apple)` each step encourages moving toward apples without being sparse.
- **Checkpoint format**: torch `.pt` dict with keys `"online"` (state_dict) and `"steps"` (gradient step count).

---

## Hyperparameter Sweep

The sweep system runs every combination of hyperparameters you specify (cartesian product), evaluates each one, and generates a heatmap showing which settings performed best.

```bash
python sweep.py                     # uses sweep.json
python sweep.py --config my.json    # custom config
```

`sweep.json` also controls which NN architecture and fitness function are used via the `neural_network` and `fitness_function` keys. Available values:
- `neural_network`: `"Basic_Neural_Network"`, `"Two_Layer_Neural_Network"`, `"Base_algorithm"`
- `fitness_function`: `"LengthFitness"`, `"GrowthEfficiencyFitness"`

See **SWEEP.md** for the full config reference.

### Output structure

Every sweep run is self-contained under `runs/`:

```
runs/
  coarse_v1/                        ← set "run_name" in sweep.json
    sweep_config.json               ← snapshot of the config used
    summary.csv                     ← all candidates + scores
    heatmap.png
    candidates/
      0001_mr=0.05,ms=0.10/
        params.json
        rep0/
          fitness.csv
          fitness.png
          weights.csv               ← load with main.py, genetic.py, or visualize_genome.py
        rep1/ ...
      0002_mr=0.05,ms=0.20/ ...
```

### Minimal `sweep.json`

```json
{
  "run_name": "coarse_v1",
  "n_repeats": 2,
  "param_grid": {
    "mutation_rate":     {"arange": [0.05, 0.30, 0.05]},
    "mutation_strength": {"arange": [0.10, 0.40, 0.10]}
  },
  "base_params": {
    "population_size": 20,
    "num_generations": 10,
    "crossover_rate":  0.8,
    "elitism_count":   3,
    "games_per_genome": 3
  },
  "output": {
    "x_param": "mutation_rate",
    "y_param": "mutation_strength"
  }
}
```

See **SWEEP.md** for the full config reference, seeding from weights, and multi-parameter visualization.

---

## Loading Weights

Two weight formats exist — genetic and DQN — and they are not interchangeable.

### Genetic weights (CSV)

Produced by every `genetic.py` or `sweep.py` run. Contain one float per row, plus an optional `nn_class` header used to auto-select the NN architecture.

```bash
# Watch in the main game
python main.py --prev-weights path/to/weights.csv
python main.py --prev-weights path/to/weights.csv human

# Watch standalone
python visualize_genome.py path/to/weights.csv --nn two_layer

# Continue training
python genetic.py --prev-weights path/to/weights.csv config.json experiment_name

# Seed a sweep
# In sweep.json: "prev_weights": "path/to/weights.csv"
```

Bare filenames are resolved against `weights/`.

### DQN weights (.pt)

Produced by `rl_train.py`. PyTorch checkpoint files — **not** usable with `main.py` or `genetic.py`.

```bash
python visualize_dqn.py weights/rl_agent_best.pt
python visualize_dqn.py weights/rl_agent.pt --fps 5
```

---

## Visualization

### Genetic fitness convergence plot

Each `genetic.py` run produces a PNG showing fitness over generations:
- **Blue line** — average fitness per generation
- **Blue band** — best/worst envelope
- **Annotation box** — all hyperparameters used

### DQN training curve

`rl_train.py` writes `graphs/rl_training_length.png` every `PLOT_EVERY` episodes:
- **Faint line** — per-episode final snake length
- **Bold line** — 100-episode rolling mean
- **Dashed line** — best smoothed length reached so far
- **Annotation box** — best smoothed length and the episode it was achieved

### Sweep heatmap

| # swept params | Visualization |
|---|---|
| 1 | Line plot — param value vs. mean score |
| 2 | Heatmap — x × y grid, colour = mean score |
| 3 | Faceted heatmap — one subplot per value of the 3rd param |
| 4+ | One heatmap file per parameter pair (e.g. `heatmap_mr_x_xr.png`) |

Brighter = better. All panels in a faceted heatmap share the same colour scale.

---

## Configuration (`config.json`)

Controls the game environment for all runners (training, play, visualization):

| Key | Default | Description |
|-----|---------|-------------|
| `grid_size` | `50` | Grid is `grid_size × grid_size` tiles |
| `tile_size` | `1000` | Pixel size of the display window (window = `tile_size × tile_size`) |
| `num_snakes` | `20` | Total snakes per game |
| `initial_snake_length` | `3` | Starting length of each snake |
| `apple_density` | `20` | Apples = `grid_size² / apple_density` |
| `win_length` | `1000` | Segments needed to win |
| `boost_multiplier` | `5.0` | Tiles moved per tick when boosting |
| `max_stamina_factor` | `1.0` | Scales max stamina with snake length |
| `stamina_recovery_factor` | `1.0` | Scales stamina recovery speed |
| `stamina_drain_rate` | `1.0` | Stamina cost per boost tick |
| `fps` | `10` | Game ticks per second |
| `limit_scope` | `true` | Render a zoomed viewport centred on the showcase snake instead of the full grid |
| `max_moves_flag` | `true` | Enable the per-game move cap |
| `max_moves` | `200` | Max ticks per game when `max_moves_flag` is true (prevents infinite games during training) |

---

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Game launcher — human play, AI watch, `NN_REGISTRY` |
| `genetic.py` | Genetic algorithm training + `resume_training()` + `run_experiments()` |
| `sweep.py` | Hyperparameter grid search over `genetic()` |
| `rl_agent.py` | DQN components: `MLP_RL_Agent`, `ReplayBuffer`, `DQNAgent` (Double DQN), `RLPlayer` |
| `rl_train.py` | DQN training loop — episodes, reward shaping, checkpointing, plotting |
| `visualize_dqn.py` | CLI to watch a saved `.pt` DQN checkpoint play |
| `visualize_genome.py` | CLI to watch a saved genome CSV play |
| `neural_network.py` | NN architectures: `Basic_Neural_Network` (single-layer), `Two_Layer_Neural_Network` (8-unit hidden), `Base_algorithm` (hand-coded greedy baseline) |
| `data.py` | 25-feature extraction from game state (`get_features`), shared by all AI types |
| `fitness.py` | Fitness functions for genetic training: `LengthFitness`, `GrowthEfficiencyFitness` |
| `game.py` | Core game logic — tick progression, collisions, apple management |
| `snake.py` | `Snake` entity, `Action`/`Direction` enums, movement math |
| `player.py` | `Player` ABC; `HumanPlayer`, `RandomPlayer`, `SafeRandomPlayer` |
| `grid.py` | Grid bounds and apple spawning |
| `renderer.py` | Pygame rendering — grid, HUD, limited-scope viewport, overlays |
| `config.py` | `GameConfig` dataclass and JSON loader |
| `sweep.json` | Default sweep configuration |
| `experiments.json` | Named experiment definitions for `genetic.py --sweep` |
| `config.json` | Game environment configuration |

### Documentation

| File | Covers |
|------|--------|
| `README.md` | This file — overview and common workflows |
| `GENETIC.md` | Genetic algorithm details, hyperparameter tuning guide |
| `SWEEP.md` | Sweep config reference, heatmap guide, coarse→fine workflow |

---

## Typical Workflows

### Genetic algorithm

```
1. python sweep.py                          # coarse grid search
2. Look at runs/coarse_v1/heatmap.png       # find best region
3. python sweep.py --config sweep_fine.json # zoom in (seeding from coarse best)
4. python genetic.py --prev-weights runs/fine_v1/candidates/0006_.../rep0/weights.csv
                                            # full training run from best weights
5. python main.py --prev-weights runs/.../weights.csv human
                                            # play against your trained snake
```

### Deep Q-Learning

```
1. python rl_train.py                       # train for 10,000 episodes
2. Look at graphs/rl_training_length.png    # check convergence
3. python visualize_dqn.py                 # watch the best checkpoint play
```
