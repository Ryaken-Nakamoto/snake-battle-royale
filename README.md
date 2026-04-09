# Snake Battle Royale — AI Training Project

A multiplayer snake game where AI-controlled snakes are trained via a genetic algorithm to compete against each other (and optionally a human player).

---

## How to Play

### Watch trained AI snakes compete

```bash
python main.py --prev-weights weights/my_weights.csv
```

All snakes are genetic AI players using the loaded weights.

### Play against trained AI

```bash
python main.py --prev-weights weights/my_weights.csv human
```

Adds you as a human player. Controls: `W`/`S` or `A`/`D` to turn (depending on current direction), `Space` to boost, `P` to pause, `ESC` to quit.

### Play against random snakes (no training required)

```bash
python main.py
```

Snakes make safe random moves (they avoid immediately lethal actions). Good for testing the game itself.

---

## Training AI Snakes

### Quick single run

```bash
python genetic.py
```

Trains with default hyperparameters. Outputs:

```
results/default_results.csv       ← fitness per generation
graphs/default_fitness.png        ← convergence plot
weights/default_results_weights.csv  ← best genome (load this with main.py)
```

### Resume training from existing weights

```bash
python genetic.py --prev-weights my_weights.csv config.json my_run_name
```

Seeds the population from `weights/my_weights.csv` and continues training. Outputs saved under `my_run_name`. You can use full paths too:

```bash
python genetic.py --prev-weights runs/coarse_v1/candidates/0008_mr=0.15,ms=0.30/rep0/weights.csv
```

### Run a set of named experiments

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

Each experiment runs in sequence and saves its own results and weights.

---

## Hyperparameter Sweep

The sweep system runs every combination of hyperparameters you specify (cartesian product), evaluates each one, and generates a heatmap showing which settings performed best.

```bash
python sweep.py                     # uses sweep.json
python sweep.py --config my.json    # custom config
```

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
          weights.csv               ← load this with main.py or genetic.py
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

Weights are CSV files containing the trained neural network genome. They're produced by every training or sweep run.

### In the game

```bash
python main.py --prev-weights path/to/weights.csv
python main.py --prev-weights path/to/weights.csv human
```

Bare filenames are looked up in `weights/`. Full or relative paths also work.

### Continuing training

```bash
python genetic.py --prev-weights path/to/weights.csv config.json experiment_name
```

Seeds the population from the saved genome and evolves from there. The best result is saved under `experiment_name`.

### In a sweep

```json
"prev_weights": "path/to/weights.csv"
```

All sweep candidates start from this seed instead of random. You can also pass a list to seed each candidate individually — see SWEEP.md.

---

## Visualization

### Fitness convergence plot

Each training run produces a PNG showing fitness over generations:
- **Blue line** — average fitness per generation
- **Blue band** — best/worst envelope
- **Annotation box** — all hyperparameters used

### Sweep heatmap

| # swept params | Visualization |
|---|---|
| 1 | Line plot — param value vs. mean score |
| 2 | Heatmap — x × y grid, colour = mean score |
| 3 | Faceted heatmap — one subplot per value of the 3rd param |
| 4+ | One heatmap file per parameter pair (e.g. `heatmap_mr_x_xr.png`) |

Brighter = better. All panels in a faceted heatmap share the same colour scale so you can compare them directly.

---

## Configuration (`config.json`)

Controls the game environment for training and play:

| Key | Default | Description |
|-----|---------|-------------|
| `grid_size` | 200 | Grid is `grid_size × grid_size` tiles |
| `num_snakes` | 50 | Total snakes in each game |
| `initial_snake_length` | 3 | Starting length |
| `apple_density` | 100 | Apples = `grid_size² / apple_density` |
| `win_length` | 50 | Segments needed to win |
| `fps` | 10 | Ticks per second (display only) |
| `max_moves` | 1000 | Max ticks per game (prevents infinite games during training) |

---

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Game launcher — human play, AI watch |
| `genetic.py` | Genetic algorithm training + `resume_training()` |
| `sweep.py` | Hyperparameter grid search |
| `sweep.json` | Sweep configuration |
| `experiments.json` | Named experiment definitions for `--sweep` |
| `config.json` | Game environment configuration |
| `neural_network.py` | NN architectures (`Basic_Neural_Network`, etc.) |
| `data.py` | Feature extraction from game state |
| `game.py` | Core game logic |
| `snake.py` | Snake entity, `Action`/`Direction` enums |
| `player.py` | `Player` ABC, `HumanPlayer`, `RandomPlayer` |
| `grid.py` | Grid bounds and apple spawning |
| `renderer.py` | Pygame rendering |
| `config.py` | `GameConfig` dataclass and JSON loader |

### Documentation

| File | Covers |
|------|--------|
| `README.md` | This file — overview and common workflows |
| `GENETIC.md` | Genetic algorithm details, hyperparameter tuning guide |
| `SWEEP.md` | Sweep config reference, heatmap guide, coarse→fine workflow |

---

## Typical Workflow

```
1. python sweep.py                          # coarse grid search
2. Look at runs/coarse_v1/heatmap.png       # find best region
3. python sweep.py --config sweep_fine.json # zoom in (seeding from coarse best)
4. Look at runs/fine_v1/heatmap.png
5. python genetic.py --prev-weights runs/fine_v1/candidates/0006_.../rep0/weights.csv
                                            # full training run from best weights
6. python main.py --prev-weights runs/.../weights.csv human
                                            # play against your trained snake
```
