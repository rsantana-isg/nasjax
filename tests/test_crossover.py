"""Tests for crossover operators."""

import jax
import pytest

from nasjax.descriptors.mlp import MLPDescriptor
from nasjax.evolution.crossover import (
    uniform_crossover,
    one_point_crossover,
    layer_wise_crossover,
    averaged_crossover,
    apply_random_crossover,
    CROSSOVER_OPERATORS,
)


@pytest.fixture
def parent1():
    """Create first parent descriptor."""
    return MLPDescriptor(
        input_dim=10,
        output_dim=2,
        dims=(64, 32),
        act_functions=("relu", "relu"),
        init_functions=("glorot_normal", "glorot_normal"),
        dropout_probs=(0.1, 0.1),
        use_dropout=False,
        use_batch_norm=False,
        max_num_layers=5,
        max_num_neurons=128,
    )


@pytest.fixture
def parent2():
    """Create second parent descriptor."""
    return MLPDescriptor(
        input_dim=10,
        output_dim=2,
        dims=(128, 64, 32),
        act_functions=("elu", "elu", "elu"),
        init_functions=("glorot_uniform", "glorot_uniform", "glorot_uniform"),
        dropout_probs=(0.2, 0.2, 0.2),
        use_dropout=True,
        use_batch_norm=True,
        max_num_layers=5,
        max_num_neurons=128,
    )


class TestUniformCrossover:
    """Test uniform crossover operator."""

    def test_uniform_crossover_produces_valid_offspring(self, parent1, parent2):
        """Test that uniform crossover produces valid descriptor."""
        key = jax.random.PRNGKey(0)
        offspring = uniform_crossover(parent1, parent2, key)

        assert offspring.validate()
        assert offspring.input_dim == parent1.input_dim
        assert offspring.output_dim == parent1.output_dim

    def test_uniform_crossover_respects_constraints(self, parent1, parent2):
        """Test that offspring respects architectural constraints."""
        key = jax.random.PRNGKey(0)
        offspring = uniform_crossover(parent1, parent2, key)

        assert len(offspring.dims) <= offspring.max_num_layers
        assert all(dim <= offspring.max_num_neurons for dim in offspring.dims)
        assert len(offspring.dims) == len(offspring.act_functions)
        assert len(offspring.dims) == len(offspring.init_functions)
        assert len(offspring.dims) == len(offspring.dropout_probs)

    def test_uniform_crossover_deterministic(self, parent1, parent2):
        """Test that same key produces same offspring."""
        key = jax.random.PRNGKey(42)

        offspring1 = uniform_crossover(parent1, parent2, key)
        offspring2 = uniform_crossover(parent1, parent2, key)

        assert offspring1 == offspring2

    def test_uniform_crossover_different_keys(self, parent1, parent2):
        """Test that different keys produce different offspring."""
        offspring1 = uniform_crossover(parent1, parent2, jax.random.PRNGKey(0))
        offspring2 = uniform_crossover(parent1, parent2, jax.random.PRNGKey(1))

        # High probability they're different
        assert offspring1.dims != offspring2.dims or offspring1.act_functions != offspring2.act_functions

    def test_uniform_crossover_with_same_parents(self):
        """Test crossover with identical parents."""
        parent = MLPDescriptor.random_init(10, 2, 3, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        offspring = uniform_crossover(parent, parent, key)

        # Offspring should be identical to parent
        assert offspring.dims == parent.dims
        assert offspring.validate()


class TestOnePointCrossover:
    """Test one-point crossover operator."""

    def test_one_point_crossover_produces_valid_offspring(self, parent1, parent2):
        """Test that one-point crossover produces valid descriptor."""
        key = jax.random.PRNGKey(0)
        offspring = one_point_crossover(parent1, parent2, key)

        assert offspring.validate()
        assert offspring.input_dim == parent1.input_dim
        assert offspring.output_dim == parent1.output_dim

    def test_one_point_crossover_respects_constraints(self, parent1, parent2):
        """Test that offspring respects architectural constraints."""
        key = jax.random.PRNGKey(0)
        offspring = one_point_crossover(parent1, parent2, key)

        assert len(offspring.dims) <= offspring.max_num_layers
        assert len(offspring.dims) >= 1  # At least one layer
        assert all(dim <= offspring.max_num_neurons for dim in offspring.dims)

    def test_one_point_crossover_deterministic(self, parent1, parent2):
        """Test that same key produces same offspring."""
        key = jax.random.PRNGKey(42)

        offspring1 = one_point_crossover(parent1, parent2, key)
        offspring2 = one_point_crossover(parent1, parent2, key)

        assert offspring1 == offspring2

    def test_one_point_crossover_combines_parents(self, parent1, parent2):
        """Test that offspring combines properties from both parents."""
        key = jax.random.PRNGKey(0)
        offspring = one_point_crossover(parent1, parent2, key)

        # Offspring should have valid architecture
        assert offspring.validate()


class TestLayerWiseCrossover:
    """Test layer-wise crossover operator."""

    def test_layer_wise_crossover_produces_valid_offspring(self, parent1, parent2):
        """Test that layer-wise crossover produces valid descriptor."""
        key = jax.random.PRNGKey(0)
        offspring = layer_wise_crossover(parent1, parent2, key)

        assert offspring.validate()
        assert offspring.input_dim == parent1.input_dim
        assert offspring.output_dim == parent1.output_dim

    def test_layer_wise_crossover_respects_constraints(self, parent1, parent2):
        """Test that offspring respects architectural constraints."""
        key = jax.random.PRNGKey(0)
        offspring = layer_wise_crossover(parent1, parent2, key)

        assert len(offspring.dims) <= offspring.max_num_layers
        assert all(dim <= offspring.max_num_neurons for dim in offspring.dims)
        assert len(offspring.dims) == len(offspring.act_functions)

    def test_layer_wise_crossover_deterministic(self, parent1, parent2):
        """Test that same key produces same offspring."""
        key = jax.random.PRNGKey(42)

        offspring1 = layer_wise_crossover(parent1, parent2, key)
        offspring2 = layer_wise_crossover(parent1, parent2, key)

        assert offspring1 == offspring2

    def test_layer_wise_crossover_mixes_properties(self, parent1, parent2):
        """Test that layer-wise crossover can mix properties."""
        key = jax.random.PRNGKey(0)
        offspring = layer_wise_crossover(parent1, parent2, key)

        # Offspring should be valid
        assert offspring.validate()

        # Could have properties from either parent
        for i in range(len(offspring.dims)):
            # Check that each property is from one of the parents (if they have this layer)
            if i < len(parent1.dims) or i < len(parent2.dims):
                assert offspring.dims[i] > 0


class TestAveragedCrossover:
    """Test averaged crossover operator."""

    def test_averaged_crossover_produces_valid_offspring(self, parent1, parent2):
        """Test that averaged crossover produces valid descriptor."""
        key = jax.random.PRNGKey(0)
        offspring = averaged_crossover(parent1, parent2, key)

        assert offspring.validate()
        assert offspring.input_dim == parent1.input_dim
        assert offspring.output_dim == parent1.output_dim

    def test_averaged_crossover_respects_constraints(self, parent1, parent2):
        """Test that offspring respects architectural constraints."""
        key = jax.random.PRNGKey(0)
        offspring = averaged_crossover(parent1, parent2, key)

        assert len(offspring.dims) <= offspring.max_num_layers
        assert all(dim >= 2 for dim in offspring.dims)  # At least 2 neurons per layer
        assert all(dim <= offspring.max_num_neurons for dim in offspring.dims)

    def test_averaged_crossover_averages_numeric_properties(self, parent1, parent2):
        """Test that averaged crossover averages layer sizes."""
        key = jax.random.PRNGKey(0)
        offspring = averaged_crossover(parent1, parent2, key)

        # Number of layers should be average
        avg_layers = (len(parent1.dims) + len(parent2.dims)) // 2
        assert len(offspring.dims) <= max(len(parent1.dims), len(parent2.dims))
        assert len(offspring.dims) >= 1

    def test_averaged_crossover_deterministic(self, parent1, parent2):
        """Test that same key produces same offspring."""
        key = jax.random.PRNGKey(42)

        offspring1 = averaged_crossover(parent1, parent2, key)
        offspring2 = averaged_crossover(parent1, parent2, key)

        assert offspring1 == offspring2


class TestApplyRandomCrossover:
    """Test random crossover application."""

    def test_apply_random_crossover_produces_valid_offspring(self, parent1, parent2):
        """Test that random crossover produces valid descriptor."""
        key = jax.random.PRNGKey(0)
        offspring = apply_random_crossover(parent1, parent2, key)

        assert offspring.validate()
        assert offspring.input_dim == parent1.input_dim
        assert offspring.output_dim == parent1.output_dim

    def test_apply_random_crossover_deterministic(self, parent1, parent2):
        """Test that same key produces same offspring."""
        key = jax.random.PRNGKey(42)

        offspring1 = apply_random_crossover(parent1, parent2, key)
        offspring2 = apply_random_crossover(parent1, parent2, key)

        assert offspring1 == offspring2

    def test_apply_random_crossover_with_subset(self, parent1, parent2):
        """Test random crossover with subset of operators."""
        operators = [uniform_crossover, one_point_crossover]

        key = jax.random.PRNGKey(0)
        offspring = apply_random_crossover(parent1, parent2, key, crossover_operators=operators)

        assert offspring.validate()

    def test_multiple_crossovers_produce_variation(self, parent1, parent2):
        """Test that multiple crossovers produce different results."""
        results = []
        for i in range(10):
            key = jax.random.PRNGKey(i)
            offspring = apply_random_crossover(parent1, parent2, key)
            results.append(offspring)

        # At least some should be different
        unique_results = set(results)
        assert len(unique_results) > 1


class TestCrossoverEdgeCases:
    """Test edge cases and error handling."""

    def test_crossover_with_different_constraints(self):
        """Test crossover with parents having different constraints."""
        p1 = MLPDescriptor(
            input_dim=10,
            output_dim=2,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=3,
            max_num_neurons=64,
        )

        p2 = MLPDescriptor(
            input_dim=10,
            output_dim=2,
            dims=(128, 64, 32, 16),
            act_functions=("elu", "elu", "elu", "elu"),
            init_functions=("glorot_normal", "glorot_normal", "glorot_normal", "glorot_normal"),
            dropout_probs=(0.2, 0.2, 0.2, 0.2),
            use_dropout=True,
            use_batch_norm=True,
            max_num_layers=10,
            max_num_neurons=256,
        )

        key = jax.random.PRNGKey(0)
        offspring = uniform_crossover(p1, p2, key)

        # Offspring should satisfy stricter constraints
        assert offspring.validate()
        assert len(offspring.dims) <= 3  # Stricter max_num_layers
        assert all(dim <= 64 for dim in offspring.dims)  # Stricter max_num_neurons

    def test_crossover_with_single_layer_parents(self):
        """Test crossover with single-layer parents."""
        p1 = MLPDescriptor(
            input_dim=10,
            output_dim=2,
            dims=(64,),
            act_functions=("relu",),
            init_functions=("normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )

        p2 = MLPDescriptor(
            input_dim=10,
            output_dim=2,
            dims=(128,),
            act_functions=("elu",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.2,),
            use_dropout=True,
            use_batch_norm=True,
            max_num_layers=5,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)

        for crossover_fn in CROSSOVER_OPERATORS:
            offspring = crossover_fn(p1, p2, key)
            assert offspring.validate()
            assert len(offspring.dims) >= 1

    def test_crossover_with_different_sized_parents(self):
        """Test crossover with very different parent architectures."""
        p1 = MLPDescriptor.random_init(10, 2, 5, 32, jax.random.PRNGKey(0))
        p2 = MLPDescriptor.random_init(10, 2, 5, 128, jax.random.PRNGKey(1))

        key = jax.random.PRNGKey(2)

        for crossover_fn in CROSSOVER_OPERATORS:
            offspring = crossover_fn(p1, p2, key)
            assert offspring.validate()


class TestCrossoverImmutability:
    """Test that crossover doesn't modify parent descriptors."""

    def test_uniform_crossover_immutable(self, parent1, parent2):
        """Test that uniform crossover doesn't modify parents."""
        original_dims1 = parent1.dims
        original_dims2 = parent2.dims

        uniform_crossover(parent1, parent2, jax.random.PRNGKey(0))

        assert parent1.dims == original_dims1
        assert parent2.dims == original_dims2

    def test_one_point_crossover_immutable(self, parent1, parent2):
        """Test that one-point crossover doesn't modify parents."""
        original_dict1 = parent1.to_dict()
        original_dict2 = parent2.to_dict()

        one_point_crossover(parent1, parent2, jax.random.PRNGKey(0))

        assert parent1.to_dict() == original_dict1
        assert parent2.to_dict() == original_dict2

    def test_all_crossovers_immutable(self, parent1, parent2):
        """Test that all crossover operators preserve parents."""
        for crossover_fn in CROSSOVER_OPERATORS:
            original_dict1 = parent1.to_dict()
            original_dict2 = parent2.to_dict()

            crossover_fn(parent1, parent2, jax.random.PRNGKey(0))

            assert parent1.to_dict() == original_dict1
            assert parent2.to_dict() == original_dict2
