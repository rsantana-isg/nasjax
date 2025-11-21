"""Tests for mutation operators."""

import jax
import pytest

from nasjax.descriptors.mlp import MLPDescriptor
from nasjax.evolution.mutation import (
    mutate_add_layer,
    mutate_remove_layer,
    mutate_layer_size,
    mutate_activation,
    mutate_initializer,
    mutate_dropout_toggle,
    mutate_dropout_probs,
    mutate_batch_norm_toggle,
    apply_random_mutation,
)


class TestMutateAddLayer:
    """Test add layer mutation."""

    def test_add_layer_increases_count(self):
        """Test that adding a layer increases layer count."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_add_layer(desc, key)

        assert success is True
        assert len(new_desc.dims) == 3  # Was 2, now 3
        assert new_desc.validate()

    def test_add_layer_respects_max_layers(self):
        """Test that add layer respects maximum layer constraint."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32, 16),  # Already at max
            act_functions=("relu", "relu", "relu"),
            init_functions=("normal", "normal", "normal"),
            dropout_probs=(0.1, 0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=3,  # Max is 3
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_add_layer(desc, key)

        assert success is False
        assert new_desc == desc  # Unchanged

    def test_add_layer_maintains_validity(self):
        """Test that descriptor remains valid after adding layer."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc, success = mutate_add_layer(desc, key)

        if success:
            assert new_desc.validate()
            assert len(new_desc.act_functions) == len(new_desc.dims)
            assert len(new_desc.init_functions) == len(new_desc.dims)
            assert len(new_desc.dropout_probs) == len(new_desc.dims)


class TestMutateRemoveLayer:
    """Test remove layer mutation."""

    def test_remove_layer_decreases_count(self):
        """Test that removing a layer decreases layer count."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32, 16),
            act_functions=("relu", "relu", "relu"),
            init_functions=("normal", "normal", "normal"),
            dropout_probs=(0.1, 0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_remove_layer(desc, key)

        assert success is True
        assert len(new_desc.dims) == 2  # Was 3, now 2
        assert new_desc.validate()

    def test_remove_layer_respects_min_layers(self):
        """Test that remove layer respects minimum layer constraint (1)."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64,),  # Only 1 layer
            act_functions=("relu",),
            init_functions=("normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_remove_layer(desc, key)

        assert success is False
        assert new_desc == desc  # Unchanged

    def test_remove_layer_maintains_validity(self):
        """Test that descriptor remains valid after removing layer."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        # Only test if we have more than 1 layer
        if len(desc.dims) > 1:
            key = jax.random.PRNGKey(1)
            new_desc, success = mutate_remove_layer(desc, key)

            if success:
                assert new_desc.validate()
                assert len(new_desc.act_functions) == len(new_desc.dims)
                assert len(new_desc.init_functions) == len(new_desc.dims)
                assert len(new_desc.dropout_probs) == len(new_desc.dims)


class TestMutateLayerSize:
    """Test layer size mutation."""

    def test_layer_size_changes(self):
        """Test that layer size mutation changes neuron count."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_layer_size(desc, key)

        assert success is True
        # At least one dimension should be different
        assert new_desc.dims != desc.dims
        # But length should be same
        assert len(new_desc.dims) == len(desc.dims)
        assert new_desc.validate()

    def test_layer_size_within_bounds(self):
        """Test that new layer size is within valid bounds."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc, success = mutate_layer_size(desc, key)

        assert success is True
        for dim in new_desc.dims:
            assert 2 <= dim <= desc.max_num_neurons


class TestMutateActivation:
    """Test activation function mutation."""

    def test_activation_changes(self):
        """Test that activation mutation changes activation function."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_activation(desc, key)

        assert success is True
        # At least one activation should be different
        assert new_desc.act_functions != desc.act_functions
        assert new_desc.validate()

    def test_activation_maintains_other_properties(self):
        """Test that activation mutation only changes activations."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc, success = mutate_activation(desc, key)

        assert success is True
        assert new_desc.dims == desc.dims
        assert new_desc.init_functions == desc.init_functions
        assert new_desc.dropout_probs == desc.dropout_probs


class TestMutateInitializer:
    """Test weight initializer mutation."""

    def test_initializer_changes(self):
        """Test that initializer mutation changes weight initializer."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_initializer(desc, key)

        assert success is True
        # At least one initializer should be different
        assert new_desc.init_functions != desc.init_functions
        assert new_desc.validate()

    def test_initializer_maintains_other_properties(self):
        """Test that initializer mutation only changes initializers."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc, success = mutate_initializer(desc, key)

        assert success is True
        assert new_desc.dims == desc.dims
        assert new_desc.act_functions == desc.act_functions
        assert new_desc.dropout_probs == desc.dropout_probs


class TestMutateDropoutToggle:
    """Test dropout toggle mutation."""

    def test_dropout_toggle_flips(self):
        """Test that dropout toggle flips the dropout flag."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_dropout_toggle(desc, key)

        assert success is True
        assert new_desc.use_dropout == (not desc.use_dropout)
        assert new_desc.validate()

    def test_dropout_toggle_reversible(self):
        """Test that dropout toggle is reversible."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc, _ = mutate_dropout_toggle(desc, key)
        reversed_desc, _ = mutate_dropout_toggle(new_desc, key)

        assert reversed_desc.use_dropout == desc.use_dropout


class TestMutateDropoutProbs:
    """Test dropout probabilities mutation."""

    def test_dropout_probs_change(self):
        """Test that dropout probabilities mutation changes probabilities."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=True,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_dropout_probs(desc, key)

        assert success is True
        assert new_desc.dropout_probs != desc.dropout_probs
        assert new_desc.validate()

    def test_dropout_probs_within_bounds(self):
        """Test that new dropout probabilities are in valid range."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc, success = mutate_dropout_probs(desc, key)

        assert success is True
        for prob in new_desc.dropout_probs:
            assert 0.0 <= prob <= 0.5


class TestMutateBatchNormToggle:
    """Test batch normalization toggle mutation."""

    def test_batch_norm_toggle_flips(self):
        """Test that batch norm toggle flips the flag."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )

        key = jax.random.PRNGKey(0)
        new_desc, success = mutate_batch_norm_toggle(desc, key)

        assert success is True
        assert new_desc.use_batch_norm == (not desc.use_batch_norm)
        assert new_desc.validate()

    def test_batch_norm_toggle_reversible(self):
        """Test that batch norm toggle is reversible."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc, _ = mutate_batch_norm_toggle(desc, key)
        reversed_desc, _ = mutate_batch_norm_toggle(new_desc, key)

        assert reversed_desc.use_batch_norm == desc.use_batch_norm


class TestApplyRandomMutation:
    """Test random mutation application."""

    def test_random_mutation_succeeds(self):
        """Test that random mutation produces a changed descriptor."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(1)
        new_desc = apply_random_mutation(desc, key)

        # Descriptor should be valid
        assert new_desc.validate()

    def test_random_mutation_deterministic(self):
        """Test that same key produces same mutation."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        key = jax.random.PRNGKey(42)
        new_desc1 = apply_random_mutation(desc, key)
        new_desc2 = apply_random_mutation(desc, key)

        assert new_desc1 == new_desc2

    def test_random_mutation_with_subset(self):
        """Test random mutation with a subset of operators."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        # Only use toggle mutations
        operators = [mutate_dropout_toggle, mutate_batch_norm_toggle]

        key = jax.random.PRNGKey(1)
        new_desc = apply_random_mutation(desc, key, mutation_operators=operators)

        assert new_desc.validate()

    def test_multiple_mutations_produce_variation(self):
        """Test that multiple mutations produce different results."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))

        results = []
        for i in range(10):
            key = jax.random.PRNGKey(i)
            new_desc = apply_random_mutation(desc, key)
            results.append(new_desc)

        # At least some should be different
        unique_results = set(results)
        assert len(unique_results) > 1


class TestMutationImmutability:
    """Test that mutations don't modify original descriptors."""

    def test_add_layer_immutable(self):
        """Test that add layer doesn't modify original."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))
        original_dims = desc.dims

        mutate_add_layer(desc, jax.random.PRNGKey(1))

        assert desc.dims == original_dims

    def test_remove_layer_immutable(self):
        """Test that remove layer doesn't modify original."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))
        original_dims = desc.dims

        mutate_remove_layer(desc, jax.random.PRNGKey(1))

        assert desc.dims == original_dims

    def test_all_mutations_immutable(self):
        """Test that all mutations preserve original descriptor."""
        desc = MLPDescriptor.random_init(100, 10, 5, 64, jax.random.PRNGKey(0))
        original_dict = desc.to_dict()

        # Apply all mutations
        mutations = [
            mutate_add_layer,
            mutate_remove_layer,
            mutate_layer_size,
            mutate_activation,
            mutate_initializer,
            mutate_dropout_toggle,
            mutate_dropout_probs,
            mutate_batch_norm_toggle,
        ]

        for mutation_fn in mutations:
            key = jax.random.PRNGKey(0)
            mutation_fn(desc, key)

        # Original should be unchanged
        assert desc.to_dict() == original_dict
