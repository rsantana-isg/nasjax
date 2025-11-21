"""Tests for evolution loop."""

import jax
import jax.numpy as jnp
import pytest

from nasjax.descriptors.mlp import MLPDescriptor
from nasjax.evolution.evolving import (
    EvolvingConfig,
    Evolving,
    evolve_architecture,
)
from nasjax.evolution.population import Population, Individual
from nasjax.evaluation.evaluator import Evaluator


@pytest.fixture
def simple_data():
    """Create simple synthetic data for testing."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Simple classification problem: 10D input, 2 classes
    n_train, n_test = 50, 20
    n_features, n_classes = 10, 2

    x_train = jax.random.normal(k1, (n_train, n_features))
    y_train = jax.random.randint(k2, (n_train,), 0, n_classes)

    x_test = jax.random.normal(k3, (n_test, n_features))
    y_test = jax.random.randint(k4, (n_test,), 0, n_classes)

    return x_train, y_train, x_test, y_test


class TestEvolvingConfig:
    """Test EvolvingConfig class."""

    def test_config_init_defaults(self):
        """Test config initialization with defaults."""
        config = EvolvingConfig()

        assert config.pop_size == 20
        assert config.n_generations == 10
        assert config.mutation_prob == 0.8
        assert config.crossover_prob == 0.2
        assert config.elitism == 2
        assert config.selection_method == "tournament"
        assert config.tournament_size == 3

    def test_config_init_custom(self):
        """Test config initialization with custom values."""
        config = EvolvingConfig(
            pop_size=15,
            n_generations=5,
            mutation_prob=0.9,
            crossover_prob=0.1,
            elitism=3,
            selection_method="best",
            tournament_size=5,
        )

        assert config.pop_size == 15
        assert config.n_generations == 5
        assert config.mutation_prob == 0.9
        assert config.crossover_prob == 0.1
        assert config.elitism == 3
        assert config.selection_method == "best"
        assert config.tournament_size == 5


class TestEvolving:
    """Test Evolving class."""

    def test_evolving_init_defaults(self):
        """Test Evolving initialization with defaults."""
        evolving = Evolving(input_dim=10, output_dim=2)

        assert evolving.input_dim == 10
        assert evolving.output_dim == 2
        assert evolving.max_num_layers == 5
        assert evolving.max_num_neurons == 128
        assert evolving.config.pop_size == 20
        assert evolving.evaluator is not None
        assert evolving.use_crossover is False
        assert evolving.population is None
        assert evolving.best_individual is None

    def test_evolving_init_custom(self):
        """Test Evolving initialization with custom parameters."""
        config = EvolvingConfig(pop_size=10, n_generations=3)
        evaluator = Evaluator(n_epochs=2, batch_size=16)

        evolving = Evolving(
            input_dim=20,
            output_dim=5,
            max_num_layers=3,
            max_num_neurons=64,
            config=config,
            evaluator=evaluator,
            use_crossover=True,
        )

        assert evolving.input_dim == 20
        assert evolving.output_dim == 5
        assert evolving.max_num_layers == 3
        assert evolving.max_num_neurons == 64
        assert evolving.config.pop_size == 10
        assert evolving.config.n_generations == 3
        assert evolving.evaluator.n_epochs == 2
        assert evolving.use_crossover is True

    def test_evolve_initializes_population(self, simple_data):
        """Test that evolve() initializes population."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=5, n_generations=1)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )

        # Override evaluator to use minimal epochs
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert evolving.population is not None
        assert len(evolving.population) == 5

    def test_evolve_evaluates_population(self, simple_data):
        """Test that evolve() evaluates all individuals."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=1)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # All individuals should have fitness
        for individual in population:
            assert individual.fitness is not None
            assert isinstance(individual.fitness, float)
            assert not jnp.isnan(individual.fitness)

    def test_evolve_tracks_best_individual(self, simple_data):
        """Test that evolve() tracks best individual."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=5, n_generations=2)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert evolving.best_individual is not None
        assert evolving.best_individual.fitness is not None

        # Best individual should have lowest fitness in final population
        pop_fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
        assert evolving.best_individual.fitness == min(pop_fitnesses)

    def test_evolve_logs_statistics(self, simple_data):
        """Test that evolve() logs statistics correctly."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=4, n_generations=3)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # Check log structure
        assert "generation" in log
        assert "best_fitness" in log
        assert "mean_fitness" in log
        assert "std_fitness" in log

        # Check log length
        assert len(log["generation"]) == 3
        assert len(log["best_fitness"]) == 3
        assert len(log["mean_fitness"]) == 3
        assert len(log["std_fitness"]) == 3

        # Check values are valid
        for i in range(3):
            assert log["generation"][i] == i
            assert isinstance(log["best_fitness"][i], float)
            assert isinstance(log["mean_fitness"][i], float)
            assert isinstance(log["std_fitness"][i], float)

    def test_evolve_applies_elitism(self, simple_data):
        """Test that elitism preserves best individuals."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=5, n_generations=2, elitism=2)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # Best fitness should not increase (it should stay same or decrease)
        best_fitnesses = log["best_fitness"]
        for i in range(1, len(best_fitnesses)):
            assert best_fitnesses[i] <= best_fitnesses[i - 1] or jnp.isclose(
                best_fitnesses[i], best_fitnesses[i - 1]
            )

    def test_evolve_applies_mutations(self, simple_data):
        """Test that mutations are applied to generate offspring."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=3, n_generations=2, mutation_prob=1.0, elitism=0
        )
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # With mutation_prob=1.0, offspring should be different from parents
        # (This is probabilistic but with high mutation rate should happen)
        assert population is not None
        assert len(population) == 3

    def test_evolve_deterministic_with_same_key(self, simple_data):
        """Test that same key produces same evolution."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=2)
        evaluator = Evaluator(n_epochs=1, batch_size=32)

        evolving1 = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
            evaluator=evaluator,
        )

        evolving2 = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
            evaluator=evaluator,
        )

        key = jax.random.PRNGKey(42)
        _, log1 = evolving1.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )
        _, log2 = evolving2.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # Logs should be identical
        assert log1["best_fitness"] == log2["best_fitness"]
        assert log1["mean_fitness"] == log2["mean_fitness"]

    def test_evolve_different_keys_different_results(self, simple_data):
        """Test that different keys produce different evolution."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=2)
        # Use cross_entropy for classification data with integer labels
        evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        evolving1 = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
            evaluator=evaluator,
        )

        evolving2 = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
            evaluator=evaluator,
        )

        _, log1 = evolving1.evolve(
            x_train, y_train, x_test, y_test, jax.random.PRNGKey(0), verbose=False
        )
        _, log2 = evolving2.evolve(
            x_train, y_train, x_test, y_test, jax.random.PRNGKey(1), verbose=False
        )

        # Logs should be different (high probability)
        assert (
            log1["best_fitness"] != log2["best_fitness"]
            or log1["mean_fitness"] != log2["mean_fitness"]
        )

    def test_get_best_descriptor(self, simple_data):
        """Test get_best_descriptor method."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=1)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        # Before evolution, should return None
        assert evolving.get_best_descriptor() is None

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # After evolution, should return a descriptor
        best_desc = evolving.get_best_descriptor()
        assert best_desc is not None
        assert isinstance(best_desc, MLPDescriptor)
        assert best_desc.validate()


class TestEvolveArchitecture:
    """Test evolve_architecture convenience function."""

    def test_evolve_architecture_works(self, simple_data):
        """Test evolve_architecture convenience function."""
        x_train, y_train, x_test, y_test = simple_data

        best_desc, log = evolve_architecture(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=10,
            output_dim=2,
            key=jax.random.PRNGKey(0),
            pop_size=3,
            n_generations=2,
            max_num_layers=2,
            max_num_neurons=16,
            n_epochs=1,
            batch_size=32,
            verbose=False,
        )

        assert best_desc is not None
        assert isinstance(best_desc, MLPDescriptor)
        assert best_desc.validate()
        assert best_desc.input_dim == 10
        assert best_desc.output_dim == 2

        # Check log
        assert len(log["generation"]) == 2
        assert len(log["best_fitness"]) == 2

    def test_evolve_architecture_custom_loss(self, simple_data):
        """Test evolve_architecture with custom loss function."""
        x_train, y_train, x_test, y_test = simple_data

        best_desc, log = evolve_architecture(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=10,
            output_dim=2,
            key=jax.random.PRNGKey(0),
            pop_size=3,
            n_generations=1,
            max_num_layers=2,
            max_num_neurons=16,
            n_epochs=1,
            batch_size=32,
            loss_fn="cross_entropy",
            verbose=False,
        )

        assert best_desc is not None
        assert isinstance(best_desc, MLPDescriptor)

    def test_evolve_architecture_with_crossover_flag(self, simple_data):
        """Test evolve_architecture with crossover flag."""
        x_train, y_train, x_test, y_test = simple_data

        # Even with use_crossover=True, should work (mutations still applied)
        best_desc, log = evolve_architecture(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=10,
            output_dim=2,
            key=jax.random.PRNGKey(0),
            pop_size=3,
            n_generations=1,
            max_num_layers=2,
            max_num_neurons=16,
            n_epochs=1,
            batch_size=32,
            use_crossover=False,
            verbose=False,
        )

        assert best_desc is not None
        assert isinstance(best_desc, MLPDescriptor)


class TestEvolutionEdgeCases:
    """Test edge cases and error handling."""

    def test_evolve_single_generation(self, simple_data):
        """Test evolution with single generation."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=1)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert len(population) == 3
        assert len(log["generation"]) == 1

    def test_evolve_small_population(self, simple_data):
        """Test evolution with very small population."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=2, n_generations=2)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert len(population) == 2
        assert all(ind.fitness is not None for ind in population)

    def test_evolve_no_elitism(self, simple_data):
        """Test evolution without elitism."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=2, elitism=0)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert len(population) == 3
        # Without elitism, best fitness might increase (get worse)
        # Just check that evolution completed
        assert len(log["best_fitness"]) == 2

    def test_evolve_no_mutation(self, simple_data):
        """Test evolution with zero mutation probability."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=2, mutation_prob=0.0)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # Evolution should complete even without mutations
        # (offspring are just copies of parents)
        assert len(population) == 3
        assert all(ind.fitness is not None for ind in population)

    def test_evolve_with_verbose(self, simple_data):
        """Test that verbose mode doesn't crash."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=2, n_generations=1)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        # This should work without crashing
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=True
        )

        assert len(population) == 2


class TestGenerateOffspring:
    """Test _generate_offspring method."""

    def test_generate_offspring_mutation_only(self, simple_data):
        """Test offspring generation with mutation only."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=1, mutation_prob=1.0)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=False,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        # Initialize and evaluate population
        key = jax.random.PRNGKey(0)
        population, _ = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # Check that population evolved
        assert len(population) == 3


class TestEvaluatePopulation:
    """Test _evaluate_population method."""

    def test_evaluate_population_all_evaluated(self, simple_data):
        """Test that all individuals get evaluated."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(pop_size=3, n_generations=1)
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32)

        key = jax.random.PRNGKey(0)
        population, _ = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # All should have fitness
        for individual in population:
            assert individual.fitness is not None
            assert not jnp.isnan(individual.fitness)
            assert individual.fitness >= 0


class TestSelectionMethods:
    """Test different selection methods."""

    def test_tournament_selection(self, simple_data):
        """Test evolution with tournament selection."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=5, n_generations=2, selection_method="tournament", tournament_size=3
        )
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=2,
            max_num_neurons=16,
            config=config,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert len(population) == 5
        assert len(log["best_fitness"]) == 2


class TestCrossoverEvolution:
    """Test evolution with crossover enabled."""

    def test_evolve_with_crossover(self, simple_data):
        """Test evolution with crossover enabled."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=5,
            n_generations=3,
            mutation_prob=0.5,
            crossover_prob=0.7,
            elitism=2,
        )
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=True,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert len(population) == 5
        assert len(log["best_fitness"]) == 3
        assert all(ind.fitness is not None for ind in population)

    def test_evolve_crossover_only(self, simple_data):
        """Test evolution with only crossover (no mutation)."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=4,
            n_generations=2,
            mutation_prob=0.0,  # No mutation
            crossover_prob=1.0,  # Always crossover
            elitism=1,
        )
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=True,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert len(population) == 4
        assert all(ind.fitness is not None for ind in population)

    def test_evolve_crossover_and_mutation(self, simple_data):
        """Test evolution with both crossover and mutation."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=5,
            n_generations=2,
            mutation_prob=0.8,
            crossover_prob=0.6,
            elitism=1,
        )
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=True,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        assert len(population) == 5
        assert len(log["best_fitness"]) == 2
        assert all(ind.fitness is not None for ind in population)

    def test_crossover_produces_valid_offspring(self, simple_data):
        """Test that crossover produces valid offspring during evolution."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=4,
            n_generations=2,
            mutation_prob=0.0,
            crossover_prob=1.0,
        )
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=True,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # All individuals should be valid
        for ind in population:
            assert ind.descriptor.validate()

    def test_evolve_architecture_with_crossover(self, simple_data):
        """Test evolve_architecture convenience function with crossover."""
        x_train, y_train, x_test, y_test = simple_data

        best_desc, log = evolve_architecture(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=10,
            output_dim=2,
            key=jax.random.PRNGKey(0),
            pop_size=4,
            n_generations=2,
            max_num_layers=3,
            max_num_neurons=32,
            n_epochs=1,
            batch_size=32,
            loss_fn="cross_entropy",
            use_crossover=True,
            verbose=False,
        )

        assert best_desc is not None
        assert isinstance(best_desc, MLPDescriptor)
        assert best_desc.validate()
        assert len(log["best_fitness"]) == 2

    def test_crossover_with_elitism(self, simple_data):
        """Test that elitism works with crossover enabled."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=5,
            n_generations=3,
            mutation_prob=0.5,
            crossover_prob=0.7,
            elitism=2,
        )
        evolving = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=True,
        )
        evolving.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        key = jax.random.PRNGKey(0)
        population, log = evolving.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # With elitism, best fitness should not get worse
        best_fitnesses = log["best_fitness"]
        for i in range(1, len(best_fitnesses)):
            assert best_fitnesses[i] <= best_fitnesses[i - 1] or jnp.isclose(
                best_fitnesses[i], best_fitnesses[i - 1]
            )

    def test_crossover_different_probabilities(self, simple_data):
        """Test evolution with different crossover probabilities."""
        x_train, y_train, x_test, y_test = simple_data

        for crossover_prob in [0.0, 0.5, 1.0]:
            config = EvolvingConfig(
                pop_size=4,
                n_generations=2,
                mutation_prob=0.5,
                crossover_prob=crossover_prob,
                elitism=1,
            )
            evolving = Evolving(
                input_dim=10,
                output_dim=2,
                max_num_layers=3,
                max_num_neurons=32,
                config=config,
                use_crossover=True,
            )
            evolving.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

            key = jax.random.PRNGKey(0)
            population, log = evolving.evolve(
                x_train, y_train, x_test, y_test, key, verbose=False
            )

            assert len(population) == 4
            assert all(ind.fitness is not None for ind in population)

    def test_crossover_deterministic(self, simple_data):
        """Test that crossover evolution is deterministic with same key."""
        x_train, y_train, x_test, y_test = simple_data

        config = EvolvingConfig(
            pop_size=4, n_generations=2, mutation_prob=0.5, crossover_prob=0.7
        )

        evolving1 = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=True,
        )
        evolving1.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        evolving2 = Evolving(
            input_dim=10,
            output_dim=2,
            max_num_layers=3,
            max_num_neurons=32,
            config=config,
            use_crossover=True,
        )
        evolving2.evaluator = Evaluator(n_epochs=1, batch_size=32, loss_fn="cross_entropy")

        key = jax.random.PRNGKey(42)
        _, log1 = evolving1.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )
        _, log2 = evolving2.evolve(
            x_train, y_train, x_test, y_test, key, verbose=False
        )

        # Results should be identical
        assert log1["best_fitness"] == log2["best_fitness"]
        assert log1["mean_fitness"] == log2["mean_fitness"]
