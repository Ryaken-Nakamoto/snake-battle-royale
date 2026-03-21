from __future__ import annotations

import csv
import os
import random
from typing import Any

from config import load_config
from game import Game
from player import Player, RandomPlayer
from snake import Action

# TODO: decide what the genome actually encodes (a weight for each feature?)
# rn it is just a placeholder list of floats

GENOME_LENGTH = 16  # TODO: set to match updated representation

#could be moved into config.json?

POPULATION_SIZE = 20       # number of genomes per generation
NUM_GENERATIONS = 50       # how many generations to run
MUTATION_RATE = 0.1        # probability of mutating each gene
MUTATION_STRENGTH = 0.2    # how much a mutated gene shifts (±)
CROSSOVER_RATE = 0.7       # probability of doing crossover vs. cloning
ELITISM_COUNT = 2          # top N genomes carried unchanged to next gen
GAMES_PER_GENOME = 3       # run multiple games per generation and average fitness for stability

CSV_OUTPUT_PATH = os.path.join("results", "genetic_results.csv")

#TODO: implement get_action() so it actually uses the genome list and produces a real decision. 
# For now it just falls back to RandomPlayer.
class GeneticPlayer(Player):

    def __init__(self, genome: list[float]) -> None:
        self.genome = genome
        self._fallback = RandomPlayer()

    def get_action(self, game_state: dict[str, Any]) -> Action:
        return self._fallback.get_action(game_state)

# Runs GAMES_PER_GENOME games for each genome and averages the snake length as fitness.
def evaluate_population(
    population: list[list[float]], config_path: str = "config.json"
) -> list[float]:
    
    config = load_config(config_path)
    assert len(population) == config.num_snakes, (
        f"POPULATION_SIZE ({len(population)}) must equal "
        f"config.num_snakes ({config.num_snakes})"
    )

    total_fitnesses = [0.0] * len(population)

    for j in range(GAMES_PER_GENOME):
        players: list[Player] = [GeneticPlayer(g) for g in population]
        game = Game(config, players)

        while not game.game_over:
            game.progress_game()

        for i, snake in enumerate(game.snakes):
            total_fitnesses[i] += float(snake.length)

    return [f / GAMES_PER_GENOME for f in total_fitnesses]


def random_genome() -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(GENOME_LENGTH)]

#TODO: different crossover strategy?
def crossover(parent_a: list[float], parent_b: list[float]) -> list[float]:
    if random.random() > CROSSOVER_RATE:
        return list(parent_a)  # clone

    point = random.randint(1, GENOME_LENGTH - 1)
    return parent_a[:point] + parent_b[point:]


def mutate(genome: list[float]) -> list[float]:
    return [
        gene + random.gauss(0, MUTATION_STRENGTH) if random.random() < MUTATION_RATE else gene
        for gene in genome
    ]

#TODO: this uses touranment selection might want to change?
def select_parent(population: list[list[float]], fitnesses: list[float]) -> list[float]:
    a, b = random.sample(range(len(population)), 2)
    return population[a] if fitnesses[a] >= fitnesses[b] else population[b]



# TODO: add more columns if needed
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
    csv_path: str = CSV_OUTPUT_PATH,
) -> list[float]:
    print(f"[genetic] Starting — {NUM_GENERATIONS} generations × {POPULATION_SIZE} genomes")
    print(f"[genetic] Results will be saved to: {csv_path}\n")

    _init_csv(csv_path)

    # Generation 0: random population
    population: list[list[float]] = [random_genome() for _ in range(POPULATION_SIZE)]

    best_genome: list[float] = population[0]
    best_ever_fitness: float = float("-inf")

    for generation in range(1, NUM_GENERATIONS + 1):

        fitnesses: list[float] = evaluate_population(population, config_path)

        gen_best_idx = fitnesses.index(max(fitnesses))
        if fitnesses[gen_best_idx] > best_ever_fitness:
            best_ever_fitness = fitnesses[gen_best_idx]
            best_genome = list(population[gen_best_idx])

        _log_generation(csv_path, generation, fitnesses)

        print(
            f"Gen {generation:>4}/{NUM_GENERATIONS} | "
            f"best={max(fitnesses):.2f}  avg={sum(fitnesses)/len(fitnesses):.2f}  "
            f"worst={min(fitnesses):.2f}"
        )

        # Build next generation - fill the top two with the best genomes, then crossover/mutate the rest
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)

        next_population: list[list[float]] = []

        for i in range(ELITISM_COUNT):
            next_population.append(list(population[ranked[i]]))

        while len(next_population) < POPULATION_SIZE:
            parent_a = select_parent(population, fitnesses)
            parent_b = select_parent(population, fitnesses)
            child = crossover(parent_a, parent_b)
            child = mutate(child)
            next_population.append(child)

        population = next_population

    print(f"\n[genetic] Done. Best fitness: {best_ever_fitness:.4f}")
    print(f"[genetic] CSV saved to: {csv_path}")
    return best_genome

if __name__ == "__main__":
    best = genetic()
    print(f"\nBest genome:\n{best}")