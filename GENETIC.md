# Genetic Algorithm Training — `genetic.py`

This module trains neural network weights for snake AI players using a genetic algorithm. All snakes in each game are evolved simultaneously, playing against each other.

---

## Quick Start

### Run a single training session
```bash
python genetic.py
```
Trains with default hyperparameters and saves outputs to:
- `results/default_results.csv` — fitness per generation (internal data)
- `graphs/default_fitness.png` — convergence plot
- `weights/default_results_weights.csv` — best genome weights

### Run multiple experiments (parameter sweep)
```bash
python genetic.py --sweep
```
Reads experiment definitions from `experiments.json`, runs each, and saves all outputs organized by experiment name.

---

### Run an experiment using prev weights
```bash
python genetic.py --prev-weights {weight_path} {optional config_path} {optional experiment_name}
```
You do not need to put weight/{path}, just put the name of the file. If you do not put a config file name, the default "config.json" will be used. If you do not put an experiement name, the results and weights of the experiement will be saved as "results/experiment_YYYY-MM-DD_HH:MM:SS:_results.csv" and "weights/experiment_YYYY-MM-DD_HH:MM:SS:_weights.csv" respectively. 

## Understanding the Algorithm

### Flow
1. **Generate random population** — `population_size` snakes with random genomes
2. **Evaluate each generation** — run `games_per_genome` games and average final snake length (fitness)
3. **Selection & reproduction** — tournament selection picks best parents
4. **Crossover & mutation** — breed children with chance of genetic variation
5. **Elitism** — top `elitism_count` genomes pass unchanged to next generation
6. **Repeat** — for `num_generations` iterations

### Fitness
Fitness = average final snake length across all games a genome plays. Longer snakes = higher fitness.

---

## Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 20 | Number of snakes per generation |
| `num_generations` | 10 | How many generations to evolve |
| `mutation_rate` | 0.1 | Probability each gene mutates (0–1) |
| `mutation_strength` | 0.2 | Std dev of Gaussian mutation (±) |
| `crossover_rate` | 0.7 | Probability of crossover vs. cloning (0–1) |
| `elitism_count` | 2 | Top N genomes carry to next generation |
| `games_per_genome` | 3 | Stability: run each genome N games, average fitness |
| `genome_length` | 30 | Total weights: 15 features × 2 (turn + boost) + 2 biases |

To change defaults globally, edit the constants at the top of `genetic.py`:

```python
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
# ... etc
```

---

## The Genome

Each genome is a list of 30 weights:
- **genome[0:15]** — turn weights (negative = left, positive = right)
- **genome[15:30]** — boost weights (positive = boost)
- **genome[30]** — turn bias
- **genome[31]** — boost bias

These weights are applied to 15 game state features (distance to apple, threat level, etc.) computed by `data.get_features()`.

---

## How to Modify Hyperparameters

### Option 1: Edit defaults in code
```python
POPULATION_SIZE = 50        # larger populations = slower, better search
NUM_GENERATIONS = 50        # more generations = slower, better convergence
MUTATION_RATE = 0.15        # higher = more variation, less convergence
MUTATION_STRENGTH = 0.3     # larger shifts = bolder exploration
CROSSOVER_RATE = 0.8        # higher = more recombination, less cloning
ELITISM_COUNT = 3           # keep more top genomes = less diversity loss
GAMES_PER_GENOME = 5        # more games = stabler fitness estimates, slower
```

### Option 2: Pass arguments to `genetic()` function
```python
from genetic import genetic

best_genome = genetic(
    population_size=50,
    num_generations=100,
    mutation_rate=0.2,
    # ... override any defaults
)
```

### Option 3: Use `experiments.json` for parameter sweeps
Create `experiments.json`:
```json
[
  {
    "name": "baseline",
    "population_size": 20,
    "num_generations": 10
  },
  {
    "name": "large_pop",
    "population_size": 50,
    "num_generations": 10
  },
  {
    "name": "many_gens",
    "population_size": 20,
    "num_generations": 50
  }
]
```

Then run:
```bash
python genetic.py --sweep
```

All experiments run in sequence with their own output graphs and weights.

---

## Output Structure

```
results/
  ├─ default_results.csv           # fitness per generation (internal data)
  ├─ my_exp_results.csv            # from experiments.json
  └─ another_exp_results.csv

graphs/
  ├─ default_fitness.png           # plot titled "Default Run"
  ├─ my_exp_fitness.png            # plot titled "my_exp"
  └─ another_exp_fitness.png

weights/
  ├─ default_results_weights.csv    # best genome from each run
  ├─ my_exp_results_weights.csv
  └─ another_exp_results_weights.csv
```

**Note:** `*_results.csv` files in `results/` are internal data for plotting. Only the graphs are meant for analysis.

---

## Graph Output

Each PNG shows:
- **Blue line** — average fitness per generation
- **Blue shaded band** — best/worst fitness envelope
- **Hyperparameters box** (top-right) — all settings used for that run
- **Title** — experiment name (e.g., "large_pop — Fitness over Generations")

---

## Advanced: Modifying `GeneticPlayer`

The `GeneticPlayer` class reads features and applies genome weights to decide actions:

```python
class GeneticPlayer(Player):
    def get_action(self, game_state: dict[str, Any]) -> Action:
        features = get_features(game_state, self.player_id)
        
        # Compute turn and boost scores
        turn_score  = sum(weights * feature for ...)
        boost_score = sum(weights * feature for ...)
        
        # Decide action based on scores
        # ...
        
        return action
```

To change the decision logic:
1. Modify the thresholds (currently `-0.33` and `0.33` for turn decisions)
2. Change how boost interacts with turn (currently boosts any chosen direction)
3. Add new action types (not recommended without changing `genome_length`)

---

## Tuning Tips

- **Convergence too slow?** Increase `population_size`, `crossover_rate`, or reduce `mutation_strength`
- **Fitness plateaus?** Increase `mutation_rate` or `mutation_strength` to inject variation
- **Results noisy?** Increase `games_per_genome` for stabler fitness estimates
- **Want stability?** Increase `elitism_count` (keep more top genomes)
- **Want exploration?** Decrease `elitism_count`, increase `mutation_rate`

---

## Development Notes

- Genome length is fixed at 30. To add more features, update `GENOME_LENGTH` and `data.get_features()`
- Selection uses tournament selection (two random parents, winner selected). Can be changed in `select_parent()`
- Crossover is single-point. Can be changed to multi-point in `crossover()`
- Fitness is average final snake length. To use wins or other metrics, change `evaluate_population()`
