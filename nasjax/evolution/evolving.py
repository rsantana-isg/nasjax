"""Main evolution loop for NASJAX.

This module implements the evolutionary algorithm for neural architecture search.
"""

from typing import Optional, Callable, List, Tuple, Dict
import jax
import jax.numpy as jnp
from tqdm import tqdm

from ..descriptors.mlp import MLPDescriptor
from ..evaluation.evaluator import Evaluator
from .population import (
    Population,
    Individual,
    initialize_population,
    select_parents,
)
from .mutation import apply_random_mutation, MUTATION_OPERATORS


class EvolvingConfig:
    """Configuration for evolution."""

    def __init__(
        self,
        pop_size: int = 20,
        n_generations: int = 10,
        mutation_prob: float = 0.8,
        crossover_prob: float = 0.2,
        elitism: int = 2,
        selection_method: str = "tournament",
        tournament_size: int = 3,
    ):
        """Initialize evolution configuration.

        Args:
            pop_size: Population size
            n_generations: Number of generations
            mutation_prob: Probability of mutation
            crossover_prob: Probability of crossover
            elitism: Number of best individuals to preserve
            selection_method: Parent selection method
            tournament_size: Tournament size (for tournament selection)
        """
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.elitism = elitism
        self.selection_method = selection_method
        self.tournament_size = tournament_size


class Evolving:
    """Main class for evolutionary neural architecture search.

    This class orchestrates the evolutionary algorithm:
    1. Initialize random population
    2. Evaluate fitness
    3. Select parents
    4. Apply genetic operators (mutation, crossover)
    5. Replace population
    6. Repeat

    Attributes:
        config: Evolution configuration
        evaluator: Fitness evaluator
        input_dim: Input dimension for networks
        output_dim: Output dimension for networks
        max_num_layers: Maximum layers
        max_num_neurons: Maximum neurons per layer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_num_layers: int = 5,
        max_num_neurons: int = 128,
        config: Optional[EvolvingConfig] = None,
        evaluator: Optional[Evaluator] = None,
        use_crossover: bool = False,
    ):
        """Initialize evolutionary algorithm.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            max_num_layers: Maximum layers per network
            max_num_neurons: Maximum neurons per layer
            config: Evolution configuration
            evaluator: Fitness evaluator
            use_crossover: Whether to use crossover operators
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_num_layers = max_num_layers
        self.max_num_neurons = max_num_neurons

        self.config = config if config is not None else EvolvingConfig()
        self.evaluator = evaluator if evaluator is not None else Evaluator()
        self.use_crossover = use_crossover

        self.population = None
        self.best_individual = None

    def evolve(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_test: jnp.ndarray,
        key: jax.random.PRNGKey,
        verbose: bool = True,
    ) -> Tuple[Population, Dict]:
        """Run the evolutionary algorithm.

        Args:
            x_train: Training inputs
            y_train: Training targets
            x_test: Test inputs
            y_test: Test targets
            key: JAX PRNG key
            verbose: Whether to print progress

        Returns:
            Tuple of (final_population, evolution_log)
            evolution_log contains statistics for each generation

        Example:
            >>> evolving = Evolving(
            ...     input_dim=784, output_dim=10,
            ...     max_num_layers=5, max_num_neurons=128
            ... )
            >>> population, log = evolving.evolve(
            ...     x_train, y_train, x_test, y_test,
            ...     jax.random.PRNGKey(0)
            ... )
        """
        # Initialize population
        key, subkey = jax.random.split(key)
        self.population = initialize_population(
            pop_size=self.config.pop_size,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_num_layers=self.max_num_layers,
            max_num_neurons=self.max_num_neurons,
            key=subkey,
        )

        # Evolution log
        log = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "std_fitness": [],
        }

        # Evolution loop
        iterator = range(self.config.n_generations)
        if verbose:
            iterator = tqdm(iterator, desc="Evolution")

        for gen in iterator:
            # Evaluate population
            key, subkey = jax.random.split(key)
            self._evaluate_population(x_train, y_train, x_test, y_test, subkey)

            # Update statistics
            self.population.update_history()
            stats = self.population.get_statistics()

            # Update best individual
            best = self.population.get_best(1)
            if best:
                self.best_individual = best[0]

            # Log statistics
            log["generation"].append(gen)
            log["best_fitness"].append(stats["min"])
            log["mean_fitness"].append(stats["mean"])
            log["std_fitness"].append(stats["std"])

            if verbose:
                tqdm.write(
                    f"Gen {gen}: Best={stats['min']:.4f}, "
                    f"Mean={stats['mean']:.4f}, Std={stats['std']:.4f}"
                )

            # Generate offspring for next generation
            if gen < self.config.n_generations - 1:
                key, subkey = jax.random.split(key)
                offspring = self._generate_offspring(subkey)

                # Replace population (with elitism)
                self.population.replace(offspring, elitism=self.config.elitism)

        return self.population, log

    def _evaluate_population(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_test: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Evaluate fitness of all individuals in population.

        Args:
            x_train: Training inputs
            y_train: Training targets
            x_test: Test inputs
            y_test: Test targets
            key: JAX PRNG key
        """
        keys = jax.random.split(key, len(self.population))

        for individual, k in zip(self.population, keys):
            if individual.fitness is None:  # Only evaluate if not already evaluated
                fitness = self.evaluator.evaluate(
                    individual.descriptor, x_train, y_train, x_test, y_test, k
                )
                individual.fitness = fitness

    def _generate_offspring(self, key: jax.random.PRNGKey) -> List[Individual]:
        """Generate offspring for next generation.

        Args:
            key: JAX PRNG key

        Returns:
            List of offspring individuals
        """
        offspring = []

        # Generate pop_size offspring
        for i in range(self.config.pop_size):
            key, k1, k2, k3 = jax.random.split(key, 4)

            # Select parents
            if self.use_crossover:
                # Select two parents for crossover
                parents = select_parents(
                    self.population,
                    n_parents=2,
                    method=self.config.selection_method,
                    tournament_size=self.config.tournament_size,
                    key=k1,
                )

                # Apply crossover with probability
                if jax.random.uniform(k2) < self.config.crossover_prob:
                    child_desc = self._crossover(
                        parents[0].descriptor, parents[1].descriptor, k2
                    )
                else:
                    # No crossover, just copy parent
                    child_desc = parents[0].descriptor
            else:
                # Mutation-only: select single parent
                parents = select_parents(
                    self.population,
                    n_parents=1,
                    method=self.config.selection_method,
                    tournament_size=self.config.tournament_size,
                    key=k1,
                )
                child_desc = parents[0].descriptor

            # Apply mutation with probability
            if jax.random.uniform(k3) < self.config.mutation_prob:
                child_desc = apply_random_mutation(child_desc, k3)

            # Create offspring individual
            offspring.append(Individual(descriptor=child_desc))

        return offspring

    def _crossover(
        self,
        parent1: MLPDescriptor,
        parent2: MLPDescriptor,
        key: jax.random.PRNGKey,
    ) -> MLPDescriptor:
        """Apply crossover between two parents.

        This is a protected crossover that ensures valid offspring.

        Args:
            parent1: First parent descriptor
            parent2: Second parent descriptor
            key: JAX PRNG key

        Returns:
            Child descriptor
        """
        # Import crossover functions (will be implemented later)
        from .crossover import uniform_crossover

        return uniform_crossover(parent1, parent2, key)

    def get_best_descriptor(self) -> Optional[MLPDescriptor]:
        """Get the best descriptor found during evolution.

        Returns:
            Best descriptor, or None if evolution hasn't run
        """
        if self.best_individual is not None:
            return self.best_individual.descriptor
        return None


def evolve_architecture(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_test: jnp.ndarray,
    y_test: jnp.ndarray,
    input_dim: int,
    output_dim: int,
    key: jax.random.PRNGKey,
    pop_size: int = 20,
    n_generations: int = 10,
    max_num_layers: int = 5,
    max_num_neurons: int = 128,
    n_epochs: int = 10,
    batch_size: int = 32,
    loss_fn: str = "mse",
    use_crossover: bool = False,
    verbose: bool = True,
) -> Tuple[MLPDescriptor, Dict]:
    """Convenience function to evolve neural architecture.

    Args:
        x_train: Training inputs
        y_train: Training targets
        x_test: Test inputs
        y_test: Test targets
        input_dim: Input dimension
        output_dim: Output dimension
        key: JAX PRNG key
        pop_size: Population size
        n_generations: Number of generations
        max_num_layers: Maximum layers
        max_num_neurons: Maximum neurons
        n_epochs: Training epochs per evaluation
        batch_size: Training batch size
        loss_fn: Loss function name
        use_crossover: Whether to use crossover
        verbose: Print progress

    Returns:
        Tuple of (best_descriptor, evolution_log)

    Example:
        >>> best_desc, log = evolve_architecture(
        ...     x_train, y_train, x_test, y_test,
        ...     input_dim=784, output_dim=10,
        ...     key=jax.random.PRNGKey(0),
        ...     pop_size=15, n_generations=5
        ... )
    """
    # Create config
    config = EvolvingConfig(pop_size=pop_size, n_generations=n_generations)

    # Create evaluator
    evaluator = Evaluator(
        loss_fn=loss_fn, n_epochs=n_epochs, batch_size=batch_size
    )

    # Create evolution instance
    evolving = Evolving(
        input_dim=input_dim,
        output_dim=output_dim,
        max_num_layers=max_num_layers,
        max_num_neurons=max_num_neurons,
        config=config,
        evaluator=evaluator,
        use_crossover=use_crossover,
    )

    # Run evolution
    population, log = evolving.evolve(
        x_train, y_train, x_test, y_test, key, verbose=verbose
    )

    # Get best descriptor
    best_descriptor = evolving.get_best_descriptor()

    return best_descriptor, log
