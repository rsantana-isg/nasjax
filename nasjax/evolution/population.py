"""Population management for evolutionary algorithm.

This module handles population initialization, fitness tracking, and
population statistics.
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from ..descriptors.mlp import MLPDescriptor


@dataclass
class Individual:
    """Represents an individual in the population.

    Attributes:
        descriptor: Network descriptor (genotype)
        fitness: Fitness value (lower is better, None if not evaluated)
        age: Number of generations the individual has survived
    """

    descriptor: MLPDescriptor
    fitness: Optional[float] = None
    age: int = 0

    def __lt__(self, other):
        """Compare individuals by fitness (for sorting)."""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness < other.fitness


class Population:
    """Manages a population of individuals for evolution.

    Attributes:
        individuals: List of Individual objects
        generation: Current generation number
        best_fitness_history: History of best fitness per generation
        mean_fitness_history: History of mean fitness per generation
    """

    def __init__(self, individuals: Optional[List[Individual]] = None):
        """Initialize population.

        Args:
            individuals: Optional list of individuals (empty if None)
        """
        self.individuals = individuals if individuals is not None else []
        self.generation = 0
        self.best_fitness_history = []
        self.mean_fitness_history = []

    def __len__(self) -> int:
        """Return population size."""
        return len(self.individuals)

    def __getitem__(self, idx) -> Individual:
        """Get individual by index."""
        return self.individuals[idx]

    def __iter__(self):
        """Iterate over individuals."""
        return iter(self.individuals)

    def add(self, individual: Individual):
        """Add an individual to the population.

        Args:
            individual: Individual to add
        """
        self.individuals.append(individual)

    def get_fitnesses(self) -> List[float]:
        """Get fitness values for all individuals.

        Returns:
            List of fitness values (None values replaced with inf)
        """
        return [
            ind.fitness if ind.fitness is not None else float('inf')
            for ind in self.individuals
        ]

    def get_best(self, n: int = 1) -> List[Individual]:
        """Get the n best individuals.

        Args:
            n: Number of individuals to return

        Returns:
            List of n best individuals (sorted by fitness)
        """
        # Filter out individuals without fitness
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]

        # Sort by fitness (lower is better)
        sorted_inds = sorted(evaluated, key=lambda x: x.fitness)

        return sorted_inds[:n]

    def get_best_descriptor(self) -> Optional[MLPDescriptor]:
        """Get descriptor of best individual.

        Returns:
            Descriptor of best individual, or None if population empty
        """
        best = self.get_best(1)
        if best:
            return best[0].descriptor
        return None

    def get_statistics(self) -> Dict[str, float]:
        """Compute population statistics.

        Returns:
            Dictionary with statistics (min, max, mean, std, median fitness)
        """
        fitnesses = [f for f in self.get_fitnesses() if f != float('inf')]

        if not fitnesses:
            return {
                "min": float('inf'),
                "max": float('inf'),
                "mean": float('inf'),
                "std": 0.0,
                "median": float('inf'),
            }

        return {
            "min": float(np.min(fitnesses)),
            "max": float(np.max(fitnesses)),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "median": float(np.median(fitnesses)),
        }

    def update_history(self):
        """Update fitness history for current generation."""
        stats = self.get_statistics()
        self.best_fitness_history.append(stats["min"])
        self.mean_fitness_history.append(stats["mean"])

    def replace(self, new_individuals: List[Individual], elitism: int = 0):
        """Replace population with new individuals.

        Args:
            new_individuals: New individuals to use
            elitism: Number of best individuals to keep from current population
        """
        if elitism > 0:
            # Keep best individuals from current population
            elite = self.get_best(elitism)

            # Combine elite with new individuals (remove worst new individuals)
            new_individuals = elite + new_individuals[: -elitism]

        self.individuals = new_individuals
        self.generation += 1


def initialize_population(
    pop_size: int,
    input_dim: int,
    output_dim: int,
    max_num_layers: int,
    max_num_neurons: int,
    key: jax.random.PRNGKey,
    **kwargs,
) -> Population:
    """Initialize a random population.

    Args:
        pop_size: Population size
        input_dim: Input dimension for networks
        output_dim: Output dimension for networks
        max_num_layers: Maximum layers per network
        max_num_neurons: Maximum neurons per layer
        key: JAX PRNG key
        **kwargs: Additional arguments for descriptor initialization

    Returns:
        Population with randomly initialized individuals

    Example:
        >>> pop = initialize_population(
        ...     pop_size=20,
        ...     input_dim=784,
        ...     output_dim=10,
        ...     max_num_layers=5,
        ...     max_num_neurons=128,
        ...     key=jax.random.PRNGKey(0)
        ... )
    """
    keys = jax.random.split(key, pop_size)

    individuals = []
    for k in keys:
        descriptor = MLPDescriptor.random_init(
            input_dim=input_dim,
            output_dim=output_dim,
            max_num_layers=max_num_layers,
            max_num_neurons=max_num_neurons,
            key=k,
            **kwargs,
        )
        individuals.append(Individual(descriptor=descriptor))

    return Population(individuals)


def tournament_selection(
    population: Population, tournament_size: int, key: jax.random.PRNGKey
) -> Individual:
    """Select individual using tournament selection.

    Args:
        population: Population to select from
        tournament_size: Number of individuals in tournament
        key: JAX PRNG key

    Returns:
        Selected individual (winner of tournament)
    """
    # Limit tournament size to population size
    actual_tournament_size = min(tournament_size, len(population))

    # Randomly select tournament participants
    indices = jax.random.choice(key, len(population), (actual_tournament_size,), replace=False)
    tournament = [population[int(i)] for i in indices]

    # Return best individual from tournament
    return min(tournament, key=lambda x: x.fitness if x.fitness is not None else float('inf'))


def select_parents(
    population: Population,
    n_parents: int,
    method: str = "tournament",
    tournament_size: int = 3,
    key: jax.random.PRNGKey = None,
) -> List[Individual]:
    """Select parents for reproduction.

    Args:
        population: Population to select from
        n_parents: Number of parents to select
        method: Selection method ('tournament', 'best', 'random')
        tournament_size: Tournament size (for tournament selection)
        key: JAX PRNG key

    Returns:
        List of selected individuals

    Example:
        >>> parents = select_parents(
        ...     population, n_parents=10,
        ...     method='tournament',
        ...     key=jax.random.PRNGKey(0)
        ... )
    """
    if method == "best":
        # Select best individuals
        return population.get_best(n_parents)

    elif method == "random":
        # Random selection
        if key is None:
            key = jax.random.PRNGKey(0)
        indices = jax.random.choice(key, len(population), (n_parents,), replace=True)
        return [population[int(i)] for i in indices]

    elif method == "tournament":
        # Tournament selection
        if key is None:
            key = jax.random.PRNGKey(0)

        parents = []
        for i in range(n_parents):
            key, subkey = jax.random.split(key)
            parent = tournament_selection(population, tournament_size, subkey)
            parents.append(parent)

        return parents

    else:
        raise ValueError(f"Unknown selection method: {method}")
