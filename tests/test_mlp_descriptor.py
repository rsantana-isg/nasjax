"""Tests for MLPDescriptor class."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from nasjax.descriptors.mlp import MLPDescriptor, ACTIVATIONS, INITIALIZATIONS


class TestMLPDescriptorCreation:
    """Test MLPDescriptor creation and initialization."""

    def test_random_init_basic(self):
        """Test basic random initialization."""
        key = jax.random.PRNGKey(42)
        desc = MLPDescriptor.random_init(
            input_dim=784,
            output_dim=10,
            max_num_layers=5,
            max_num_neurons=128,
            key=key,
        )

        assert desc.input_dim == 784
        assert desc.output_dim == 10
        assert desc.max_num_layers == 5
        assert desc.max_num_neurons == 128
        assert 1 <= len(desc.dims) <= 5
        assert all(2 <= d <= 128 for d in desc.dims)

    def test_random_init_with_dropout(self):
        """Test random initialization with dropout enabled."""
        key = jax.random.PRNGKey(0)
        desc = MLPDescriptor.random_init(
            input_dim=100,
            output_dim=5,
            max_num_layers=3,
            max_num_neurons=64,
            key=key,
            dropout=True,
        )

        assert desc.use_dropout is True
        assert len(desc.dropout_probs) == len(desc.dims)
        assert all(0.0 <= p <= 0.5 for p in desc.dropout_probs)

    def test_random_init_with_batch_norm(self):
        """Test random initialization with batch norm enabled."""
        key = jax.random.PRNGKey(1)
        desc = MLPDescriptor.random_init(
            input_dim=50,
            output_dim=10,
            max_num_layers=4,
            max_num_neurons=32,
            key=key,
            batch_norm=True,
        )

        assert desc.use_batch_norm is True

    def test_random_init_reproducibility(self):
        """Test that same key produces same descriptor."""
        key = jax.random.PRNGKey(123)

        desc1 = MLPDescriptor.random_init(784, 10, 5, 100, key)
        desc2 = MLPDescriptor.random_init(784, 10, 5, 100, key)

        assert desc1.dims == desc2.dims
        assert desc1.act_functions == desc2.act_functions
        assert desc1.init_functions == desc2.init_functions

    def test_random_init_different_keys(self):
        """Test that different keys produce different descriptors."""
        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(2)

        desc1 = MLPDescriptor.random_init(784, 10, 5, 100, key1)
        desc2 = MLPDescriptor.random_init(784, 10, 5, 100, key2)

        # At least one of these should be different (very high probability)
        assert (
            desc1.dims != desc2.dims
            or desc1.act_functions != desc2.act_functions
            or desc1.init_functions != desc2.init_functions
        )


class TestMLPDescriptorValidation:
    """Test MLPDescriptor validation."""

    def test_valid_descriptor(self):
        """Test that a valid descriptor passes validation."""
        desc = MLPDescriptor(
            input_dim=784,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("glorot_normal", "glorot_uniform"),
            dropout_probs=(0.1, 0.2),
            use_dropout=True,
            use_batch_norm=False,
            max_num_layers=10,
            max_num_neurons=128,
        )
        assert desc.validate() is True

    def test_too_many_layers(self):
        """Test that too many layers fails validation."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(32, 32, 32, 32, 32, 32),  # 6 layers
            act_functions=("relu",) * 6,
            init_functions=("normal",) * 6,
            dropout_probs=(0.1,) * 6,
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,  # Max is 5
            max_num_neurons=64,
        )
        assert desc.validate() is False

    def test_too_many_neurons(self):
        """Test that too many neurons fails validation."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(200,),  # 200 neurons
            act_functions=("relu",),
            init_functions=("normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,  # Max is 128
        )
        assert desc.validate() is False

    def test_inconsistent_lengths(self):
        """Test that inconsistent list lengths fail validation."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(32, 64),
            act_functions=("relu",),  # Wrong length
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.2),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )
        assert desc.validate() is False

    def test_invalid_activation(self):
        """Test that invalid activation function fails validation."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(32,),
            act_functions=("invalid_activation",),  # Invalid
            init_functions=("normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )
        assert desc.validate() is False

    def test_invalid_dropout_prob(self):
        """Test that invalid dropout probability fails validation."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(32,),
            act_functions=("relu",),
            init_functions=("normal",),
            dropout_probs=(1.5,),  # Invalid (> 1.0)
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )
        assert desc.validate() is False


class TestMLPDescriptorPyTree:
    """Test PyTree functionality of MLPDescriptor."""

    def test_pytree_registration(self):
        """Test that descriptor is properly registered as PyTree."""
        key = jax.random.PRNGKey(0)
        desc = MLPDescriptor.random_init(784, 10, 5, 100, key)

        # Should be able to flatten/unflatten
        leaves, treedef = jtu.tree_flatten(desc)
        assert leaves is not None
        assert treedef is not None

    def test_tree_flatten_unflatten(self):
        """Test that flatten/unflatten roundtrip works."""
        key = jax.random.PRNGKey(42)
        desc = MLPDescriptor.random_init(784, 10, 3, 64, key)

        # Flatten
        leaves, treedef = jtu.tree_flatten(desc)

        # Unflatten
        desc_reconstructed = jtu.tree_unflatten(treedef, leaves)

        # Should be identical
        assert desc == desc_reconstructed

    def test_tree_map_compatibility(self):
        """Test that descriptor works with jax.tree_map."""
        key = jax.random.PRNGKey(0)
        desc = MLPDescriptor.random_init(784, 10, 2, 64, key)

        # Should be able to use tree_map (even if it doesn't change much)
        def identity(x):
            return x

        desc_mapped = jax.tree_map(identity, desc)
        assert desc == desc_mapped


class TestMLPDescriptorSerialization:
    """Test serialization functionality."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        desc = MLPDescriptor(
            input_dim=784,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "sigmoid"),
            init_functions=("glorot_normal", "normal"),
            dropout_probs=(0.1, 0.2),
            use_dropout=True,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )

        data = desc.to_dict()

        assert isinstance(data, dict)
        assert data["input_dim"] == 784
        assert data["output_dim"] == 10
        assert data["dims"] == [64, 32]
        assert data["act_functions"] == ["relu", "sigmoid"]

    def test_from_dict(self):
        """Test reconstruction from dictionary."""
        data = {
            "input_dim": 100,
            "output_dim": 5,
            "dims": [32, 16],
            "act_functions": ["relu", "tanh"],
            "init_functions": ["normal", "uniform"],
            "dropout_probs": [0.1, 0.2],
            "use_dropout": True,
            "use_batch_norm": False,
            "max_num_layers": 10,
            "max_num_neurons": 64,
        }

        desc = MLPDescriptor.from_dict(data)

        assert desc.input_dim == 100
        assert desc.output_dim == 5
        assert desc.dims == (32, 16)
        assert desc.use_dropout is True

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip works."""
        key = jax.random.PRNGKey(123)
        desc_original = MLPDescriptor.random_init(784, 10, 5, 100, key)

        # Serialize and deserialize
        data = desc_original.to_dict()
        desc_reconstructed = MLPDescriptor.from_dict(data)

        # Should be identical
        assert desc_original == desc_reconstructed


class TestMLPDescriptorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_layer(self):
        """Test descriptor with single hidden layer."""
        desc = MLPDescriptor(
            input_dim=10,
            output_dim=5,
            dims=(32,),
            act_functions=("relu",),
            init_functions=("normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=64,
        )
        assert desc.validate() is True
        assert len(desc.dims) == 1

    def test_no_activation(self):
        """Test descriptor with no activation (None)."""
        desc = MLPDescriptor(
            input_dim=10,
            output_dim=5,
            dims=(32,),
            act_functions=(None,),
            init_functions=("normal",),
            dropout_probs=(0.0,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=64,
        )
        assert desc.validate() is True

    def test_repr(self):
        """Test string representation."""
        desc = MLPDescriptor(
            input_dim=784,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=True,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )
        repr_str = repr(desc)
        assert "MLPDescriptor" in repr_str
        assert "layers=2" in repr_str
