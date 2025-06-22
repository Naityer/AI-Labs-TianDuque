import random

import numpy as np
from functions import rastrigin_2d


def set_seed(seed: int) -> None:
    # Set fixed random seed to make the results reproducible
    random.seed(seed)
    np.random.seed(seed)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.bounds = (-5, 5)

    def initialize_population(self):
        return np.random.uniform(
            self.bounds[0], self.bounds[1], size=(self.population_size, 2)
        )

    def evaluate_population(self, population):
        return np.array([rastrigin_2d(ind[0], ind[1]) for ind in population])

    def selection(self, population, fitness_values):
        # Rank Selection
        sorted_indices = np.argsort(fitness_values)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(fitness_values))
        selection_probs = (len(population) - ranks) / sum(range(1, len(population) + 1))
        selected_indices = np.random.choice(
            np.arange(len(population)),
            size=int(self.crossover_rate * len(population)),
            p=selection_probs,
        )
        return population[selected_indices]

    def crossover(self, parents):
        offspring = []
        for _ in range(len(parents) // 2):
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, individuals):
        for ind in individuals:
            if np.random.rand() < self.mutation_rate:
                ind += np.random.normal(0, self.mutation_strength, size=2)
                ind[:] = np.clip(ind, self.bounds[0], self.bounds[1])
        return individuals

    def evolve(self, seed: int):
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            fitness_values = self.evaluate_population(population)
            best_idx = np.argmin(fitness_values)

            best_solutions.append(population[best_idx])
            best_fitness_values.append(fitness_values[best_idx])
            average_fitness_values.append(np.mean(fitness_values))

            parents = self.selection(population, fitness_values)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)

            survivors = population[np.random.choice(
                np.arange(len(population)),
                size=self.population_size - len(offspring),
                replace=False
            )]
            population = np.vstack((survivors, offspring))

        return best_solutions, best_fitness_values, average_fitness_values
