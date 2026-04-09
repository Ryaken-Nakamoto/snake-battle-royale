# Hyperparameter Sweep — `sweep.py`

All sweep behaviour is configured through a JSON file. No code changes needed for experiments — edit the JSON, run the script, read the results.

---

## Quick Start

```bash
python sweep.py                        # uses sweep.json
python sweep.py --config my.json       # uses a different config file
```

Every sweep run creates a self-contained output directory:

```
runs/
  sweep_2026-04-09_14-30/             ← named by timestamp (or your run_name)
    sweep_config.json                 ← snapshot of the config used
    summary.csv                       ← all candidates + scores
    heatmap.png                       ← visualization (see below)
    candidates/
      0001_mr=0.05,ms=0.10/           ← one folder per hyperparameter combo
        params.json                   ← full params for this candidate
        rep0/
          fitness.csv
          fitness.png
          weights.csv
        rep1/ ...
      0002_mr=0.05,ms=0.20/
        ...
```

---

## `sweep.json` Reference

```json
{
  "nn_class":         "Basic_Neural_Network",
  "fitness_function": "LengthFitness",
  "config_path":      "config.json",
  "n_repeats":        2,
  "verbose":          true,

  "run_name":      null,
  "prev_weights":  null,
  "abbreviations": null,

  "param_grid": {
    "mutation_rate":     {"arange": [0.05, 0.30, 0.05]},
    "mutation_strength": {"arange": [0.10, 0.40, 0.10]}
  },

  "base_params": {
    "population_size":  50,
    "num_generations":  5,
    "crossover_rate":   0.8,
    "elitism_count":    3,
    "games_per_genome": 3
  },

  "output": {
    "x_param":     "mutation_rate",
    "y_param":     "mutation_strength",
    "facet_param": null
  }
}
```

### Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nn_class` | string | `"Basic_Neural_Network"` | Architecture to use. One of: `Basic_Neural_Network`, `Two_Layer_Neural_Network`, `Base_algorithm` |
| `fitness_function` | string \| null | `"LengthFitness"` | Fitness function used to score each snake. See **Fitness Functions** below. |
| `config_path` | string | `"config.json"` | Game config JSON passed to every `genetic()` call |
| `n_repeats` | int | `1` | Runs per candidate — scores averaged to reduce noise |
| `verbose` | bool | `true` | Print progress to stdout |
| `run_name` | string \| null | timestamp | Name for the output directory under `runs/`. Set this to something meaningful so you can identify the run later (e.g. `"coarse_grid_v1"`). Defaults to `sweep_YYYY-MM-DD_HH-MM`. |
| `prev_weights` | string \| list \| null | `null` | Seed the population from preexisting weights instead of random init. See **Seeding from Weights** below. |
| `abbreviations` | object \| null | `null` | Override the default param abbreviations used in candidate folder names. E.g. `{"mutation_rate": "mut"}` |

### `param_grid`

Which hyperparameters to sweep and what values to try. All combinations are evaluated (cartesian product). Three value formats:

**Explicit list:**
```json
"population_size": [10, 20, 50, 100]
```

**`arange` spec** — mirrors `np.arange(start, stop, step)`, stop is exclusive:
```json
"mutation_rate": {"arange": [0.05, 0.30, 0.05]}
```
→ `[0.05, 0.10, 0.15, 0.20, 0.25]`

**`linspace` spec** — mirrors `np.linspace(start, stop, n)`, both endpoints inclusive:
```json
"crossover_rate": {"linspace": [0.5, 0.9, 5]}
```
→ `[0.5, 0.6, 0.7, 0.8, 0.9]`

Any parameter accepted by `genetic()` can go here: `mutation_rate`, `mutation_strength`, `crossover_rate`, `population_size`, `elitism_count`, `games_per_genome`, `num_generations`.

### `base_params`

Fixed hyperparameters that are **not** being swept. Merged into every `genetic()` call. Anything not listed falls back to the defaults in `genetic.py`.

### `output`

| Key | Description |
|-----|-------------|
| `x_param` | Swept param on the x-axis of the heatmap |
| `y_param` | Swept param on the y-axis of the heatmap |
| `facet_param` | For 3-param sweeps: which param to use as panel facets (defaults to the 3rd swept param) |

`x_param` and `y_param` are required when sweeping 2+ params. For single-param sweeps a line plot is generated automatically.

---

## Fitness Functions

The `fitness_function` key selects how each snake is scored after each game. The value must be a string matching one of the registered classes in `fitness.py`.

| Name | Formula | When to use |
|------|---------|-------------|
| `LengthFitness` | `snake.length` | Default. Rewards raw growth, straightforward to interpret. |
| `GrowthEfficiencyFitness` | `snake.length + (growth / ticks) × 10` | Rewards both size and speed of growth. Discourages slow, passive strategies. |

`GrowthEfficiencyFitness` detail: `growth = length − initial_snake_length`, `ticks = game.tick`. The scale factor of 10 keeps the efficiency bonus meaningful relative to raw length while still letting length dominate for long-lived snakes.

To compare fitness functions across the same hyperparameter grid, run two separate sweeps with different `run_name` values and different `fitness_function` settings.

---

## Seeding from Weights

Instead of starting each candidate from random weights, you can seed the population from a previous run's best weights. This is useful for fine-grained sweeps after an initial coarse pass has found a good region.

**Single seed — all candidates start from the same weights:**
```json
"prev_weights": "baseline_results_weights.csv"
```
Bare filenames are looked up in `weights/`. Full paths also work:
```json
"prev_weights": "runs/sweep_2026-04-09_14-30/candidates/0003_mr=0.15,ms=0.25/rep0/weights.csv"
```

**Per-candidate seeds — one weight file per candidate (by cartesian-product order):**
```json
"prev_weights": [
  "runs/coarse/candidates/0003_mr=0.15,ms=0.25/rep0/weights.csv",
  "runs/coarse/candidates/0007_mr=0.20,ms=0.30/rep0/weights.csv"
]
```
The list is indexed by candidate order (0-indexed). If the list is shorter than the number of candidates, remaining candidates fall back to random init (a warning is printed).

**To find the cartesian-product order**, run a sweep with `verbose: true` — each candidate's index is printed before it runs.

---

## Visualization

The heatmap type is chosen automatically based on how many parameters are being swept:

| # swept params | Output |
|---|---|
| 1 | **Line plot** — param value on x-axis, mean score ± std on y-axis |
| 2 | **Heatmap** — x_param × y_param grid, colour = mean score |
| 3 | **Faceted heatmap** — grid of x_param × y_param panels, one per value of `facet_param`. All panels share the same colour scale for valid comparison. |
| 4+ | **All pairwise heatmaps** — one file per parameter pair (e.g. `heatmap_mr_x_xr.png`), saved in the run directory. Others fixed at best-found values. |

### Reading the heatmap

- **Brighter = better.** Each cell is annotated with its numeric score.
- A clear bright region → optimum is well-defined; zoom in with a fine grid.
- A gradient along one axis → that parameter dominates; the other can stay fixed.
- Uniform colour → neither parameter is decisive; consider sweeping a different pair.

For the **faceted heatmap** (3 params), compare panels left-to-right/top-to-bottom to see how the 3rd parameter shifts the relationship between the other two. A panel that is uniformly brighter indicates the best level of the facet param.

For **pairwise heatmaps** (4+ params), each file shows one pair in isolation. Check all files — a parameter pair that looks uniform in isolation may interact differently when other params change.

---

## Reading `summary.csv`

| Column | Description |
|--------|-------------|
| `candidate_idx` | Row number (1-indexed), matches the folder prefix |
| `cand_dir` | Folder name for this candidate under `candidates/` |
| `mean_score` | Average best fitness across `n_repeats` runs |
| `std_score` | Std deviation across repeats — high = noisy candidate |
| `mutation_rate` | (and any other swept params) |

High `std_score` relative to `mean_score` means the candidate is sensitive to random initialisation. Increase `n_repeats` before trusting its ranking.

---

## Workflow

### 1. Coarse sweep

Start with `mutation_rate` × `mutation_strength` — these interact the most. Keep `population_size` and `num_generations` small to keep runtime manageable:

```json
{
  "run_name": "coarse_v1",
  "param_grid": {
    "mutation_rate":     {"arange": [0.05, 0.30, 0.05]},
    "mutation_strength": {"arange": [0.10, 0.50, 0.10]}
  },
  "base_params": { "population_size": 20, "num_generations": 10 }
}
```

### 2. Fine sweep

Copy the config, zoom in around the best cell from step 1, give it a new `run_name`:

```json
{
  "run_name": "fine_v1",
  "prev_weights": "runs/coarse_v1/candidates/0008_mr=0.15,ms=0.30/rep0/weights.csv",
  "param_grid": {
    "mutation_rate":     {"arange": [0.10, 0.22, 0.02]},
    "mutation_strength": {"arange": [0.20, 0.40, 0.02]}
  },
  "base_params": { "population_size": 20, "num_generations": 10 }
}
```

```bash
python sweep.py --config sweep_fine.json
```

### 3. Confirm at full scale

Once params are settled, run a final long training to confirm the result holds:

```bash
python genetic.py --prev-weights runs/fine_v1/candidates/0006_mr=0.12,ms=0.28/rep0/weights.csv config.json confirmed_run
```

---

## Step-Size Reference

| Parameter | Coarse first pass | Fine zoom |
|-----------|-------------------|-----------|
| `mutation_rate` | `{"arange": [0.05, 0.30, 0.05]}` | `{"arange": [best-0.04, best+0.05, 0.01]}` |
| `mutation_strength` | `{"arange": [0.10, 0.50, 0.10]}` | `{"arange": [best-0.08, best+0.09, 0.02]}` |
| `crossover_rate` | `{"arange": [0.50, 0.95, 0.10]}` | `{"arange": [best-0.08, best+0.09, 0.02]}` |
| `population_size` | `[10, 20, 50, 100]` | `[best-10, best, best+10]` |
| `elitism_count` | `[1, 2, 3, 5]` | `[best-1, best, best+1]` |
| `games_per_genome` | `[1, 3, 5, 10]` | `[3, 4, 5, 6, 7]` |

---

## Programmatic Usage

```python
from sweep import GeneticGridSearch

# Load from file (recommended)
gs = GeneticGridSearch.from_config("sweep.json")
gs.fit()
gs.save_results()
gs.plot()

print(gs.best_params_)   # dict of best hyperparameters
print(gs.best_score_)    # float — best mean fitness

# Construct directly (param_grid values must be flat lists)
gs = GeneticGridSearch(
    param_grid={
        "mutation_rate":     [0.05, 0.10, 0.15, 0.20],
        "mutation_strength": [0.10, 0.20, 0.30],
    },
    base_params={"population_size": 20, "num_generations": 10},
    n_repeats=3,
    run_name="my_experiment",
)
gs.fit()
```

---

## Tips

- Use `run_name` always — it's far easier to read `runs/coarse_v1/` than `runs/sweep_2026-04-09_14-30/`.
- `n_repeats=2` is a good default. Use `1` for fast exploratory passes, `5` when you need tight confidence.
- `mutation_rate` and `mutation_strength` interact strongly — sweep them together as a pair first.
- Keep `num_generations` low (5–15) during sweeps; raise it only for the final confirmation run.
- Use separate config files per experiment phase (`sweep_coarse.json`, `sweep_fine.json`) — `run_name` in each file keeps outputs from colliding.
- To reload results from a previous run without re-running it, read `runs/{run_name}/summary.csv` directly.
