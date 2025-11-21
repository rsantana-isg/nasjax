"""Tests for fitness evaluator."""

import jax
import jax.numpy as jnp
import pytest

from nasjax.descriptors.mlp import MLPDescriptor
from nasjax.evaluation.evaluator import Evaluator, evaluate_descriptor, evaluate_population


@pytest.fixture
def simple_data():
    """Create simple synthetic data for testing."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Simple classification problem: 2D input, 2 classes
    n_train, n_test = 100, 50
    n_features, n_classes = 10, 2

    x_train = jax.random.normal(k1, (n_train, n_features))
    y_train = jax.random.randint(k2, (n_train,), 0, n_classes)

    x_test = jax.random.normal(k3, (n_test, n_features))
    y_test = jax.random.randint(k4, (n_test,), 0, n_classes)

    return x_train, y_train, x_test, y_test


@pytest.fixture
def simple_descriptor():
    """Create a simple descriptor for testing."""
    return MLPDescriptor(
        input_dim=10,
        output_dim=2,
        dims=(16, 8),
        act_functions=("relu", "relu"),
        init_functions=("glorot_normal", "glorot_normal"),
        dropout_probs=(0.0, 0.0),
        use_dropout=False,
        use_batch_norm=False,
        max_num_layers=5,
        max_num_neurons=64,
    )


class TestEvaluator:
    """Test Evaluator class."""

    def test_evaluator_init(self):
        """Test evaluator initialization."""
        evaluator = Evaluator(
            loss_fn="mse",
            optimizer="adam",
            learning_rate=0.01,
            n_epochs=5,
            batch_size=32,
            metric="loss",
        )

        assert evaluator.loss_fn_name == "mse"
        assert evaluator.optimizer == "adam"
        assert evaluator.learning_rate == 0.01
        assert evaluator.n_epochs == 5
        assert evaluator.batch_size == 32
        assert evaluator.metric == "loss"

    def test_evaluator_default_params(self):
        """Test evaluator with default parameters."""
        evaluator = Evaluator()

        assert evaluator.loss_fn_name == "mse"
        assert evaluator.optimizer == "adam"
        assert evaluator.n_epochs == 10

    def test_evaluate_returns_fitness(self, simple_descriptor, simple_data):
        """Test that evaluate returns a fitness value."""
        x_train, y_train, x_test, y_test = simple_data

        evaluator = Evaluator(n_epochs=2, batch_size=32, metric="loss")

        key = jax.random.PRNGKey(0)
        fitness = evaluator.evaluate(
            simple_descriptor, x_train, y_train, x_test, y_test, key
        )

        assert isinstance(fitness, float)
        assert fitness >= 0  # Fitness should be non-negative
        assert not jnp.isnan(fitness)

    def test_evaluate_with_cross_entropy(self, simple_descriptor, simple_data):
        """Test evaluate with cross-entropy loss."""
        x_train, y_train, x_test, y_test = simple_data

        evaluator = Evaluator(
            loss_fn="cross_entropy", n_epochs=2, batch_size=32, metric="loss"
        )

        key = jax.random.PRNGKey(0)
        fitness = evaluator.evaluate(
            simple_descriptor, x_train, y_train, x_test, y_test, key
        )

        assert isinstance(fitness, float)
        assert not jnp.isnan(fitness)

    def test_evaluate_with_accuracy_metric(self, simple_descriptor, simple_data):
        """Test evaluate with accuracy metric."""
        x_train, y_train, x_test, y_test = simple_data

        evaluator = Evaluator(
            loss_fn="cross_entropy", n_epochs=2, batch_size=32, metric="accuracy"
        )

        key = jax.random.PRNGKey(0)
        fitness = evaluator.evaluate(
            simple_descriptor, x_train, y_train, x_test, y_test, key
        )

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0  # Should be 1 - accuracy
        assert not jnp.isnan(fitness)

    def test_evaluate_invalid_descriptor_raises(self, simple_data):
        """Test that invalid descriptor raises error."""
        x_train, y_train, x_test, y_test = simple_data

        # Create invalid descriptor (too many layers)
        invalid_desc = MLPDescriptor(
            input_dim=10,
            output_dim=2,
            dims=(16, 8, 4, 2, 1, 1),  # 6 layers
            act_functions=("relu",) * 6,
            init_functions=("normal",) * 6,
            dropout_probs=(0.0,) * 6,
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=3,  # Max is 3!
            max_num_neurons=64,
        )

        evaluator = Evaluator(n_epochs=2)

        with pytest.raises(ValueError):
            evaluator.evaluate(
                invalid_desc,
                x_train,
                y_train,
                x_test,
                y_test,
                jax.random.PRNGKey(0),
            )

    def test_evaluate_deterministic(self, simple_descriptor, simple_data):
        """Test that same key gives same result."""
        x_train, y_train, x_test, y_test = simple_data

        evaluator = Evaluator(n_epochs=2, batch_size=32)

        key = jax.random.PRNGKey(42)
        fitness1 = evaluator.evaluate(
            simple_descriptor, x_train, y_train, x_test, y_test, key
        )
        fitness2 = evaluator.evaluate(
            simple_descriptor, x_train, y_train, x_test, y_test, key
        )

        assert jnp.isclose(fitness1, fitness2)

    def test_evaluate_different_keys_different_results(self, simple_descriptor, simple_data):
        """Test that different keys give different results (due to weight init)."""
        x_train, y_train, x_test, y_test = simple_data

        evaluator = Evaluator(n_epochs=2, batch_size=32)

        fitness1 = evaluator.evaluate(
            simple_descriptor,
            x_train,
            y_train,
            x_test,
            y_test,
            jax.random.PRNGKey(0),
        )
        fitness2 = evaluator.evaluate(
            simple_descriptor,
            x_train,
            y_train,
            x_test,
            y_test,
            jax.random.PRNGKey(1),
        )

        # Different initializations should give different results
        # (high probability, not guaranteed)
        assert fitness1 != fitness2


class TestEvaluateDescriptor:
    """Test evaluate_descriptor convenience function."""

    def test_evaluate_descriptor_works(self, simple_descriptor, simple_data):
        """Test evaluate_descriptor convenience function."""
        x_train, y_train, x_test, y_test = simple_data

        fitness = evaluate_descriptor(
            simple_descriptor,
            x_train,
            y_train,
            x_test,
            y_test,
            jax.random.PRNGKey(0),
            loss_fn="mse",
            n_epochs=2,
            batch_size=32,
        )

        assert isinstance(fitness, float)
        assert not jnp.isnan(fitness)

    def test_evaluate_descriptor_custom_params(self, simple_descriptor, simple_data):
        """Test evaluate_descriptor with custom parameters."""
        x_train, y_train, x_test, y_test = simple_data

        fitness = evaluate_descriptor(
            simple_descriptor,
            x_train,
            y_train,
            x_test,
            y_test,
            jax.random.PRNGKey(0),
            loss_fn="cross_entropy",
            n_epochs=3,
            batch_size=16,
            metric="accuracy",
        )

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0


class TestEvaluatePopulation:
    """Test evaluate_population function."""

    def test_evaluate_population_multiple_descriptors(self, simple_data):
        """Test evaluating multiple descriptors."""
        x_train, y_train, x_test, y_test = simple_data

        # Create a small population
        population = []
        for i in range(3):
            desc = MLPDescriptor.random_init(
                10, 2, 3, 32, jax.random.PRNGKey(i), dropout=False
            )
            population.append(desc)

        key = jax.random.PRNGKey(42)
        evaluator = Evaluator(n_epochs=2, batch_size=32)

        fitnesses = evaluate_population(
            population, x_train, y_train, x_test, y_test, key, evaluator
        )

        assert len(fitnesses) == 3
        assert all(isinstance(f, float) for f in fitnesses)
        assert all(not jnp.isnan(f) for f in fitnesses)

    def test_evaluate_population_creates_evaluator(self, simple_data):
        """Test that evaluate_population creates evaluator if None."""
        x_train, y_train, x_test, y_test = simple_data

        population = [
            MLPDescriptor.random_init(10, 2, 3, 32, jax.random.PRNGKey(i))
            for i in range(2)
        ]

        fitnesses = evaluate_population(
            population,
            x_train,
            y_train,
            x_test,
            y_test,
            jax.random.PRNGKey(0),
            evaluator=None,  # Should create default evaluator
        )

        assert len(fitnesses) == 2

    def test_evaluate_population_fitness_variation(self, simple_data):
        """Test that different descriptors get different fitnesses."""
        x_train, y_train, x_test, y_test = simple_data

        # Create population with varied architectures
        population = [
            MLPDescriptor.random_init(10, 2, 5, 64, jax.random.PRNGKey(i))
            for i in range(5)
        ]

        fitnesses = evaluate_population(
            population,
            x_train,
            y_train,
            x_test,
            y_test,
            jax.random.PRNGKey(0),
        )

        # At least some should be different
        unique_fitnesses = set(fitnesses)
        assert len(unique_fitnesses) > 1


class TestEvaluatorEdgeCases:
    """Test edge cases and error handling."""

    def test_evaluate_with_small_dataset(self, simple_descriptor):
        """Test evaluation with very small dataset."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Very small dataset
        x_train = jax.random.normal(k1, (10, 10))
        y_train = jax.random.randint(k2, (10,), 0, 2)
        x_test = jax.random.normal(k3, (5, 10))
        y_test = jax.random.randint(k4, (5,), 0, 2)

        evaluator = Evaluator(n_epochs=2, batch_size=5)

        fitness = evaluator.evaluate(
            simple_descriptor,
            x_train,
            y_train,
            x_test,
            y_test,
            jax.random.PRNGKey(0),
        )

        assert isinstance(fitness, float)
        assert not jnp.isnan(fitness)

    def test_evaluate_with_single_layer(self, simple_data):
        """Test evaluation with single-layer network."""
        x_train, y_train, x_test, y_test = simple_data

        desc = MLPDescriptor(
            input_dim=10,
            output_dim=2,
            dims=(16,),  # Single layer
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.0,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=64,
        )

        evaluator = Evaluator(n_epochs=2)

        fitness = evaluator.evaluate(
            desc, x_train, y_train, x_test, y_test, jax.random.PRNGKey(0)
        )

        assert isinstance(fitness, float)
        assert not jnp.isnan(fitness)
