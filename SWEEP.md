# Hyperparameter Sweep — `sweep.py`

All sweep behaviour is configured through `sweep_config.json`. No code changes are needed for experiments — edit the JSON, run the script, read the graph.

---

## Quick Start

```bash
python sweep.py                        # uses sweep_config.json
python sweep.py --config my.json       # uses a different config file
```

Outputs written automatically based on the `output` block in the config:

```
results/sweep_results.csv     ← all candidates + scores
graphs/sweep_heatmap.png      ← heatmap or line plot
results/sweep/                ← per-run internal CSVs (not for analysis)
```

---

## `sweep_config.json` Reference

```json
{
  "config_path":  "config.json",
  "n_repeats":    2,
  "results_dir":  "results/sweep",
  "verbose":      true,

  "param_grid": {
    "mutation_rate":     {"arange": [0.05, 0.30, 0.05]},
    "mutation_strength": {"arange": [0.10, 0.40, 0.10]}
  },

  "base_params": {
    "population_size":  20,
    "num_generations":  10,
    "crossover_rate":   0.8,
    "elitism_count":    3,
    "games_per_genome": 3
  },

  "output": {
    "results_csv": "results/sweep_results.csv",
    "plot_path":   "graphs/sweep_heatmap.png",
    "x_param":     "mutation_rate",
    "y_param":     "mutation_strength"
  }
}
```

### Top-level keys

| Key | Type | Description |
|-----|------|-------------|
| `config_path` | string | Game config JSON passed to `genetic()` |
| `n_repeats` | int | Runs per candidate — results are averaged to reduce noise |
| `results_dir` | string | Directory for internal per-run CSVs |
| `verbose` | bool | Print progress to stdout |
| `nn_class` | string | Neural network architecture. One of: `Basic_Neural_Network`, `Two_Layer_Neural_Network`, `Base_algorithm`. Defaults to `Basic_Neural_Network`. |

### `param_grid`

Defines which hyperparameters to sweep and what values to try. All combinations are evaluated (cartesian product). Three value formats are supported:

**Explicit list** — provide the values directly:
```json
"population_size": [10, 20, 50, 100]
```

**`arange` spec** — mirrors `np.arange(start, stop, step)`:
```json
"mutation_rate": {"arange": [0.05, 0.30, 0.05]}
```
Produces `[0.05, 0.10, 0.15, 0.20, 0.25]`. Note `stop` is exclusive, same as NumPy.

**`linspace` spec** — mirrors `np.linspace(start, stop, n)`:
```json
"crossover_rate": {"linspace": [0.5, 0.9, 5]}
```
Produces `[0.5, 0.6, 0.7, 0.8, 0.9]`. Both endpoints are inclusive.

Any parameter accepted by `genetic()` can appear here: `mutation_rate`, `mutation_strength`, `crossover_rate`, `population_size`, `elitism_count`, `games_per_genome`, `num_generations`.

### `base_params`

Fixed hyperparameters that are **not** being swept. These are merged into every `genetic()` call unchanged. Any parameter not listed here falls back to the defaults in `genetic.py`.

### `output`

| Key | Description |
|-----|-------------|
| `results_csv` | Path for the final sweep CSV (candidates + scores) |
| `plot_path` | Path for the output PNG |
| `x_param` | Which swept param goes on the x-axis of the heatmap |
| `y_param` | Which swept param goes on the y-axis of the heatmap |

`x_param` and `y_param` are only needed when sweeping 2+ params. For a single-param sweep a line plot is generated automatically and these are ignored.

---

## Step Sizes

Run `python sweep.py --step-config` to print the full guide. Summary:

| Parameter | Coarse first pass | Fine zoom |
|-----------|-------------------|-----------|
| `mutation_rate` | `{"arange": [0.05, 0.30, 0.05]}` | `{"arange": [best-0.04, best+0.05, 0.01]}` |
| `mutation_strength` | `{"arange": [0.10, 0.50, 0.10]}` | `{"arange": [best-0.08, best+0.09, 0.02]}` |
| `crossover_rate` | `{"arange": [0.50, 0.95, 0.10]}` | `{"arange": [best-0.08, best+0.09, 0.02]}` |
| `population_size` | `[10, 20, 50, 100]` | `[best-10, best, best+10]` |
| `elitism_count` | `[1, 2, 3, 5]` | `[best-1, best, best+1]` |
| `games_per_genome` | `[1, 3, 5, 10]` | `[3, 4, 5, 6, 7]` |

---

## Workflow

### 1. Coarse sweep

Start with `mutation_rate` × `mutation_strength` — these interact the most and dominate fitness. Keep `population_size` low (20) and `num_generations` short (10) to keep runtime manageable.

```json
"param_grid": {
  "mutation_rate":     {"arange": [0.05, 0.30, 0.05]},
  "mutation_strength": {"arange": [0.10, 0.50, 0.10]}
},
"base_params": {
  "population_size": 20,
  "num_generations": 10
}
```

### 2. Read the heatmap

The heatmap colour encodes `mean_score` (best fitness averaged over `n_repeats` runs). Look for the brightest cell — that's the best-performing combination. Each cell is also annotated with its numeric score.

Things to look for:
- A clear bright region → the optimum is well-defined, zoom in with a fine grid.
- A gradient across one axis → that parameter matters more; the other can stay fixed.
- Uniform colour → neither parameter is decisive; consider sweeping a different pair.

### 3. Fine sweep

Copy the config, rename it (e.g. `sweep_fine.json`), and zoom in around the best values from step 2:

```json
"param_grid": {
  "mutation_rate":     {"arange": [0.08, 0.18, 0.01]},
  "mutation_strength": {"arange": [0.15, 0.35, 0.02]}
},
"base_params": {
  "population_size": 20,
  "num_generations": 10
}
```

```bash
python sweep.py --config sweep_fine.json
```

### 4. Confirm with full training

Once mutation params are settled, run a shorter sweep over `population_size` and `crossover_rate` with a longer `num_generations` to confirm the findings hold at full scale.

---

## Reading `sweep_results.csv`

| Column | Description |
|--------|-------------|
| `candidate_idx` | Row number (1-indexed) |
| `mean_score` | Average best fitness across `n_repeats` runs |
| `std_score` | Standard deviation across repeats (higher = noisier candidate) |
| `mutation_rate` | Value used for this candidate |
| `mutation_strength` | Value used for this candidate |
| *(any other swept params)* | One column per swept parameter |

Candidates with a high `std_score` relative to their `mean_score` should be treated with caution — increase `n_repeats` before acting on them.

---

## Programmatic Usage

```python
from sweep import GeneticGridSearch, load_sweep_config

# Load from file
gs = GeneticGridSearch.from_config("sweep_config.json")
gs.fit()
gs.save_results()
gs.plot()

# Or construct directly (param_grid values must already be flat lists)
gs = GeneticGridSearch(
    param_grid={
        "mutation_rate":     [0.05, 0.10, 0.15, 0.20],
        "mutation_strength": [0.10, 0.20, 0.30],
    },
    base_params={"population_size": 20, "num_generations": 10},
    n_repeats=3,
)
gs.fit()
print(gs.best_params_)
print(gs.best_score_)
```

---

## Tips

- `n_repeats=2` is a good default. Use `n_repeats=1` for very fast exploratory runs and `n_repeats=5` when you need tight confidence on the winner.
- `mutation_rate` and `mutation_strength` interact strongly — always sweep them together as a pair first.
- Keep `num_generations` low (10–15) during sweeps and only raise it for the final confirmation run.
- Use separate named config files for each experiment phase (`sweep_coarse.json`, `sweep_fine.json`, `sweep_confirm.json`) so results are reproducible.