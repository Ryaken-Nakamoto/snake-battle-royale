from __future__ import annotations
import sys
import csv
import os
import random
from typing import Any

from config import load_config
from game import Game
from player import Player, RandomPlayer
from snake import Action
from data import get_features
import matplotlib.pyplot as plt
from neural_network import Neural_Network, Basic_Neural_Network, Two_Layer_Neural_Network, Base_algorithm, N_FEATURES
from datetime import datetime


GENOME_LENGTH = Two_Layer_Neural_Network.genome_length(N_FEATURES)  # switch based on which nn class u r using

#could be moved into config.json?

POPULATION_SIZE = 50       # number of genomes per generation
NUM_GENERATIONS = 30       # how many generations to run
MUTATION_RATE = 0.15        # probability of mutating each gene
MUTATION_STRENGTH = 0.25    # how much a mutated gene shifts (±)
CROSSOVER_RATE = 0.8       # probability of doing crossover vs. cloning
ELITISM_COUNT = 3         # top N genomes carried unchanged to next gen
GAMES_PER_GENOME = 5       # run multiple games per generation and average fitness for stability

RESULTS_DIR = "results"
GRAPHS_DIR = "graphs"
WEIGHTS_DIR = "weights"


class GeneticPlayer(Player):

    def __init__(self, player_id: int, genome: list[float], nn_class: type[Neural_Network] = Basic_Neural_Network) -> None:
        self.genome = genome
        self.player_id = player_id
        self.neural_network = nn_class(genome)
        self._fallback = RandomPlayer()

    def get_action(self, game_state: dict[str, Any]) -> Action:
        features = get_features(game_state, self.player_id)
        if not features:
            return self._fallback.get_action(game_state)

        return self.neural_network.get_action(features)

    

# Runs games_per_genome games for each genome and averages the snake length as fitness.
def evaluate_population(
    population: list[list[float]],
    config_path: str = "config.json",
    *,
    games_per_genome: int,
    nn_class: type[Neural_Network] = Basic_Neural_Network,
) -> list[float]:

    config = load_config(config_path)
    assert len(population) == config.num_snakes, (
        f"POPULATION_SIZE ({len(population)}) must equal "
        f"config.num_snakes ({config.num_snakes})"
    )

    total_fitnesses = [0.0] * len(population)

    for j in range(games_per_genome):
        players: list[Player] = [GeneticPlayer(i, genome, nn_class) for i, genome in enumerate(population)]
        game = Game(config, players)

        while not game.game_over:
            game.progress_game()

        for i, snake in enumerate(game.snakes):
            total_fitnesses[i] += float(snake.length)

    return [f / games_per_genome for f in total_fitnesses]


def random_genome(length: int) -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(length)]

def crossover(parent_a: list[float], parent_b: list[float], *, crossover_rate: float, genome_length: int) -> list[float]:
    if random.random() > crossover_rate:
        return list(parent_a)  # clone

    point = random.randint(1, genome_length - 1)
    return parent_a[:point] + parent_b[point:]


def mutate(genome: list[float], *, mutation_rate: float, mutation_strength: float) -> list[float]:
    return [
        gene + random.gauss(0, mutation_strength) if random.random() < mutation_rate else gene
        for gene in genome
    ]

def select_parent(population: list[list[float]], fitnesses: list[float]) -> list[float]:
    a, b = random.sample(range(len(population)), 2)
    return population[a] if fitnesses[a] >= fitnesses[b] else population[b]


def _init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "best_fitness",
            "avg_fitness",
            "worst_fitness",     
        ])


def _log_generation(path: str, generation: int, fitnesses: list[float]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            generation,
            round(max(fitnesses), 4),
            round(sum(fitnesses) / len(fitnesses), 4),
            round(min(fitnesses), 4),
        ])

def genetic(
    config_path: str = "config.json",
    csv_path: str = "results/default_results.csv",
    *,
    nn_class: type[Neural_Network] = Basic_Neural_Network,
    population_size: int = POPULATION_SIZE,
    num_generations: int = NUM_GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    mutation_strength: float = MUTATION_STRENGTH,
    crossover_rate: float = CROSSOVER_RATE,
    elitism_count: int = ELITISM_COUNT,
    games_per_genome: int = GAMES_PER_GENOME,
    genome_length: int | None = None,
) -> list[float]:
    if genome_length is None:
        genome_length = nn_class.genome_length(N_FEATURES)

    print(f"[genetic] Starting — {num_generations} generations × {population_size} genomes")
    print(f"[genetic] nn={nn_class.__name__}  genome_length={genome_length}")
    print(f"[genetic] Results will be saved to: {csv_path}\n")

    _init_csv(csv_path)

    # Generation 0: random population
    population: list[list[float]] = [random_genome(genome_length) for _ in range(population_size)]

    best_genome: list[float] = population[0]
    best_ever_fitness: float = float("-inf")

    for generation in range(1, num_generations + 1):

        fitnesses: list[float] = evaluate_population(population, config_path, games_per_genome=games_per_genome, nn_class=nn_class)

        gen_best_idx = fitnesses.index(max(fitnesses))
        if fitnesses[gen_best_idx] > best_ever_fitness:
            best_ever_fitness = fitnesses[gen_best_idx]
            best_genome = list(population[gen_best_idx])

        _log_generation(csv_path, generation, fitnesses)

        print(
            f"Gen {generation:>4}/{num_generations} | "
            f"best={max(fitnesses):.2f}  avg={sum(fitnesses)/len(fitnesses):.2f}  "
            f"worst={min(fitnesses):.2f}"
        )

        # Build next generation - fill the top N with the best genomes, then crossover/mutate the rest
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)

        next_population: list[list[float]] = []

        for i in range(elitism_count):
            next_population.append(list(population[ranked[i]]))

        while len(next_population) < population_size:
            parent_a = select_parent(population, fitnesses)
            parent_b = select_parent(population, fitnesses)
            child = crossover(parent_a, parent_b, crossover_rate=crossover_rate, genome_length=genome_length)
            child = mutate(child, mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            next_population.append(child)

        population = next_population

    print(f"\n[genetic] Done. Best fitness: {best_ever_fitness:.4f}")
    print(f"[genetic] CSV saved to: {csv_path}")

    # Save best genome weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weights_path = os.path.join(WEIGHTS_DIR, os.path.basename(os.path.splitext(csv_path)[0]) + "_weights.csv")
    save_genome(best_genome, weights_path)
    print(f"[genetic] Weights saved to: {weights_path}")

    return best_genome

def resume_training(
    weight_path: str,
    config_path: str = "config.json",
    experiment_name: str = "resumed",
    *,
    nn_class: type[Neural_Network] = Basic_Neural_Network,
    population_size: int = POPULATION_SIZE,
    num_generations: int = NUM_GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    mutation_strength: float = MUTATION_STRENGTH,
    crossover_rate: float = CROSSOVER_RATE,
    elitism_count: int = ELITISM_COUNT,
    games_per_genome: int = GAMES_PER_GENOME,
) -> list[float]:
    """Resume training from a previously saved genome, seeding the population with it.

    The saved genome is placed at index 0; the rest of the population is generated
    by mutating copies of it, giving evolution a head-start from a known good solution.

    Args:
        weight_path: Path to the saved weights CSV
        config_path: Path to game config JSON
        experiment_name: Name used for output file paths

    Returns:
        Best genome found after continued training
    """
    seed_genome = load_previous_weights(weight_path)
    genome_length = len(seed_genome)

    print(f"[resume_training] Seeding population from: {weight_path}")
    print(f"[resume_training] Genome length: {genome_length}")

    # Seed population: keep the original + mutated variants of it
    population: list[list[float]] = [seed_genome]
    while len(population) < population_size:
        mutated = mutate(
            seed_genome,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
        )
        population.append(mutated)

    csv_path = os.path.join(RESULTS_DIR, f"{experiment_name}_results.csv")
    plot_path = os.path.join(GRAPHS_DIR, f"{experiment_name}_fitness.png")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    _init_csv(csv_path)

    best_genome = seed_genome
    best_ever_fitness = float("-inf")

    for generation in range(1, num_generations + 1):
        fitnesses = evaluate_population(
            population,
            config_path,
            games_per_genome=games_per_genome,
            nn_class=nn_class,
        )

        gen_best_idx = fitnesses.index(max(fitnesses))
        if fitnesses[gen_best_idx] > best_ever_fitness:
            best_ever_fitness = fitnesses[gen_best_idx]
            best_genome = list(population[gen_best_idx])

        _log_generation(csv_path, generation, fitnesses)

        print(
            f"Gen {generation:>4}/{num_generations} | "
            f"best={max(fitnesses):.2f}  avg={sum(fitnesses)/len(fitnesses):.2f}  "
            f"worst={min(fitnesses):.2f}"
        )

        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        next_population: list[list[float]] = [list(population[ranked[i]]) for i in range(elitism_count)]

        while len(next_population) < population_size:
            parent_a = select_parent(population, fitnesses)
            parent_b = select_parent(population, fitnesses)
            child = crossover(parent_a, parent_b, crossover_rate=crossover_rate, genome_length=genome_length)
            child = mutate(child, mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            next_population.append(child)

        population = next_population

    print(f"\n[resume_training] Done. Best fitness: {best_ever_fitness:.4f}")

    weights_path = os.path.join(WEIGHTS_DIR, f"{experiment_name}_results_weights.csv")
    save_genome(best_genome, weights_path)
    print(f"[resume_training] Weights saved to: {weights_path}")

    hyperparams = {
        "population_size": population_size,
        "num_generations": num_generations,
        "mutation_rate": mutation_rate,
        "mutation_strength": mutation_strength,
        "crossover_rate": crossover_rate,
        "elitism_count": elitism_count,
        "games_per_genome": games_per_genome,
        "genome_length": genome_length,
    }
    plot_fitness(csv_path=csv_path, plot_path=plot_path, hyperparams=hyperparams, exp_name=experiment_name)
    print(f"[resume_training] Plot saved to: {plot_path}")

    return best_genome


def save_genome(genome: list[float], csv_path: str) -> None:
    """Save a genome (list of weights) to a CSV file.

    Args:
        genome: List of float weights
        csv_path: Path where to save the genome CSV
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weight"])
        for weight in genome:
            writer.writerow([weight])


def plot_fitness(csv_path: str, plot_path: str, *, hyperparams: dict, exp_name: str = "Genetic Algorithm") -> None:
    """Plot fitness convergence from a genetic algorithm CSV output.

    Args:
        csv_path: Path to the results CSV file
        plot_path: Path where to save the PNG plot
        hyperparams: Dict of 8 hyperparameters to annotate on the plot
        exp_name: Experiment name to display in the plot title
    """
    # Read CSV
    generations = []
    avg_fitnesses = []
    best_fitnesses = []
    worst_fitnesses = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row["generation"]))
            avg_fitnesses.append(float(row["avg_fitness"]))
            best_fitnesses.append(float(row["best_fitness"]))
            worst_fitnesses.append(float(row["worst_fitness"]))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot avg fitness line
    ax.plot(generations, avg_fitnesses, linewidth=2, label="avg fitness", color="steelblue")

    # Fill between best/worst band
    ax.fill_between(generations, worst_fitnesses, best_fitnesses, alpha=0.25, color="steelblue", label="best/worst band")

    # Labels and legend
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (avg snake length)", fontsize=12)
    ax.set_title(f"{exp_name} — Fitness over Generations", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Build annotation text from hyperparams
    anno_lines = [
        f"pop={hyperparams['population_size']:<3}  gens={hyperparams['num_generations']:<3}",
        f"mut_rate={hyperparams['mutation_rate']:<5.2f}  mut_str={hyperparams['mutation_strength']:<5.2f}",
        f"xover={hyperparams['crossover_rate']:<5.2f}  elitism={hyperparams['elitism_count']:<2}",
        f"games={hyperparams['games_per_genome']:<2}  genome={hyperparams['genome_length']:<2}",
    ]
    anno_text = "\n".join(anno_lines)

    ax.text(
        0.98, 0.98, anno_text,
        transform=ax.transAxes,
        fontsize=9, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

def load_previous_weights(weight_path: str) -> list[float]:
    """Load a genome from a previously saved weights CSV."""
    genome = []
    with open(f"weights/{weight_path}", "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genome.append(float(row["weight"]))
    print(f"[load_previous_weights] Loaded {len(genome)} weights from {weight_path}")
    return genome


def run_experiments(
    experiments_path: str = "experiments.json",
    config_path: str = "config.json",
    results_dir: str = RESULTS_DIR,
    nn_class: type[Neural_Network] = Basic_Neural_Network,
) -> None:
    """Run a sequence of genetic algorithm experiments with different hyperparameters.

    Args:
        experiments_path: Path to JSON file with experiment definitions
        config_path: Path to game config JSON
        results_dir: Directory to save results CSVs and plots
    """
    import json

    # Read experiments from JSON
    with open(experiments_path, "r") as f:
        experiments = json.load(f)

    # Default hyperparameters
    defaults = {
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "mutation_rate": MUTATION_RATE,
        "mutation_strength": MUTATION_STRENGTH,
        "crossover_rate": CROSSOVER_RATE,
        "elitism_count": ELITISM_COUNT,
        "games_per_genome": GAMES_PER_GENOME,
        "genome_length": GENOME_LENGTH,
    }

    for i, exp in enumerate(experiments, start=1):
        # Get experiment name
        name = exp.get("name", f"run_{i:02d}")

        # Build hyperparams by merging experiment over defaults
        hyperparams = {**defaults, **{k: v for k, v in exp.items() if k != "name"}}

        # Derive output paths
        csv_path = os.path.join(RESULTS_DIR, f"{name}_results.csv")
        plot_path = os.path.join(GRAPHS_DIR, f"{name}_fitness.png")

        print(f"\n[run_experiments] Starting experiment {i}: {name}")

        # Run genetic algorithm
        genetic(config_path=config_path, csv_path=csv_path, nn_class=nn_class, **hyperparams)

        # Plot the results
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        plot_fitness(csv_path=csv_path, plot_path=plot_path, hyperparams=hyperparams, exp_name=name)

        print(f"[run_experiments] Saved CSV → {csv_path}, plot → {plot_path}")

    print(f"\n[run_experiments] All {len(experiments)} experiments complete.")


if __name__ == "__main__":
    time = datetime.now()
    if "--sweep" in sys.argv:
        run_experiments()
    # --prev-weights "weight_csv.path" "config_path" "{experiment_name}"
    elif "--prev-weights" in sys.argv:
        idx = sys.argv.index("--prev-weights")
        weight_path = sys.argv[idx + 1]
        config_path = sys.argv[idx + 2] if len(sys.argv) > idx + 2 else "config.json"
        exp_name    = sys.argv[idx + 3] if len(sys.argv) > idx + 3 else f"experiment_{time.strftime("%Y-%m-%d_%H:%M:%S:")}"
        resume_training(weight_path, config_path, exp_name)
    else:
        csv_path = "results/default_results.csv"
        os.makedirs(GRAPHS_DIR, exist_ok=True)
        plot_path = os.path.join(GRAPHS_DIR, "default_fitness.png")
        nn = Basic_Neural_Network
        best = genetic(csv_path=csv_path, nn_class=nn)
        plot_fitness(
            csv_path=csv_path,
            plot_path=plot_path,
            hyperparams={
                "population_size": POPULATION_SIZE,
                "num_generations": NUM_GENERATIONS,
                "mutation_rate": MUTATION_RATE,
                "mutation_strength": MUTATION_STRENGTH,
                "crossover_rate": CROSSOVER_RATE,
                "elitism_count": ELITISM_COUNT,
                "games_per_genome": GAMES_PER_GENOME,
                "genome_length": nn.genome_length(N_FEATURES),
            },
            exp_name="Default Run",
        )
        print(f"\nBest genome:\n{best}")