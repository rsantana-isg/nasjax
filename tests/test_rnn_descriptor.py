"""Tests for RNNDescriptor class."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from nasjax.descriptors.rnn import (
    RNNDescriptor,
    ACTIVATIONS,
    INITIALIZATIONS,
    SIMPLE_RNN,
    LSTM,
    GRU,
    RNN_LAYER_NAMES,
)


class TestRNNDescriptorCreation:
    """Test RNNDescriptor creation and initialization."""

    def test_random_init_basic(self):
        """Test basic random initialization."""
        key = jax.random.PRNGKey(42)
        desc = RNNDescriptor.random_init(
            input_dim=10,
            output_dim=5,
            max_num_layers=3,
            max_num_units=128,
            key=key,
        )

        assert desc.input_dim == 10
        assert desc.output_dim == 5
        assert desc.max_num_layers == 3
        assert desc.max_num_units == 128
        assert 1 <= len(desc.rnn_types) <= 3
        assert all(2 <= u <= 128 for u in desc.units)

    def test_random_init_with_dropout(self):
        """Test random initialization with dropout enabled."""
        key = jax.random.PRNGKey(0)
        desc = RNNDescriptor.random_init(
            input_dim=20,
            output_dim=10,
            max_num_layers=2,
            max_num_units=64,
            key=key,
            dropout=True,
        )

        assert desc.use_dropout is True
        assert len(desc.dropout_probs) == len(desc.rnn_types)
        assert all(0.0 <= p <= 0.5 for p in desc.dropout_probs)

    def test_random_init_with_batch_norm(self):
        """Test random initialization with batch norm enabled."""
        key = jax.random.PRNGKey(1)
        desc = RNNDescriptor.random_init(
            input_dim=15,
            output_dim=8,
            max_num_layers=3,
            max_num_units=32,
            key=key,
            batch_norm=True,
        )

        assert desc.use_batch_norm is True

    def test_random_init_reproducibility(self):
        """Test that same key produces same descriptor."""
        key = jax.random.PRNGKey(123)

        desc1 = RNNDescriptor.random_init(10, 5, 3, 100, key)
        desc2 = RNNDescriptor.random_init(10, 5, 3, 100, key)

        assert desc1.rnn_types == desc2.rnn_types
        assert desc1.units == desc2.units
        assert desc1.bidirectional == desc2.bidirectional

    def test_random_init_different_keys(self):
        """Test that different keys produce different descriptors."""
        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(2)

        desc1 = RNNDescriptor.random_init(10, 5, 3, 100, key1)
        desc2 = RNNDescriptor.random_init(10, 5, 3, 100, key2)

        # At least one should be different
        assert (
            desc1.rnn_types != desc2.rnn_types
            or desc1.units != desc2.units
            or desc1.bidirectional != desc2.bidirectional
        )

    def test_rnn_types_are_valid(self):
        """Test that generated RNN types are valid."""
        key = jax.random.PRNGKey(42)
        desc = RNNDescriptor.random_init(10, 5, 3, 100, key)

        assert all(rt in [SIMPLE_RNN, LSTM, GRU] for rt in desc.rnn_types)


class TestRNNDescriptorValidation:
    """Test RNNDescriptor validation."""

    def test_valid_descriptor(self):
        """Test that a valid descriptor passes validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM, GRU),
            units=(64, 32),
            bidirectional=(True, False),
            act_functions=("relu", "tanh"),
            init_functions=("glorot_normal", "glorot_uniform"),
            dropout_probs=(0.1, 0.2),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is True

    def test_invalid_too_many_layers(self):
        """Test that too many layers fails validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM,) * 6,  # More than max
            units=(64,) * 6,
            bidirectional=(True,) * 6,
            act_functions=("relu",) * 6,
            init_functions=("glorot_normal",) * 6,
            dropout_probs=(0.1,) * 6,
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is False

    def test_invalid_rnn_type(self):
        """Test that invalid RNN type fails validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(99,),  # Invalid type
            units=(64,),
            bidirectional=(True,),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is False

    def test_invalid_units_too_few(self):
        """Test that too few units fails validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM,),
            units=(1,),  # Less than MIN_NUM_UNITS
            bidirectional=(True,),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is False

    def test_invalid_units_too_many(self):
        """Test that too many units fails validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM,),
            units=(200,),  # More than max_num_units
            bidirectional=(True,),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is False

    def test_invalid_activation_function(self):
        """Test that invalid activation fails validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM,),
            units=(64,),
            bidirectional=(True,),
            act_functions=("invalid_activation",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is False

    def test_invalid_inconsistent_lengths(self):
        """Test that inconsistent list lengths fail validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM, GRU),
            units=(64,),  # Wrong length
            bidirectional=(True, False),
            act_functions=("relu", "tanh"),
            init_functions=("glorot_normal", "glorot_uniform"),
            dropout_probs=(0.1, 0.2),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is False

    def test_invalid_dropout_prob_range(self):
        """Test that dropout prob out of range fails validation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM,),
            units=(64,),
            bidirectional=(True,),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            dropout_probs=(1.5,),  # Out of range
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is False


class TestRNNDescriptorPyTree:
    """Test RNNDescriptor PyTree operations."""

    def test_pytree_registration(self):
        """Test that RNNDescriptor is registered as a PyTree."""
        key = jax.random.PRNGKey(42)
        desc = RNNDescriptor.random_init(10, 5, 3, 100, key)

        # Should be able to flatten and unflatten
        leaves, treedef = jtu.tree_flatten(desc)
        desc_restored = jtu.tree_unflatten(treedef, leaves)

        assert desc == desc_restored

    def test_pytree_map(self):
        """Test that tree_map works on RNNDescriptor."""
        key = jax.random.PRNGKey(42)
        desc = RNNDescriptor.random_init(10, 5, 3, 100, key)

        # This should work without errors
        result = jtu.tree_map(lambda x: x, desc)
        assert result == desc

    def test_tree_structure(self):
        """Test tree structure and leaves."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM,),
            units=(64,),
            bidirectional=(True,),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        leaves, aux_data = desc.tree_flatten()

        # Check leaves contain mutable data
        assert leaves[0] == desc.rnn_types
        assert leaves[1] == desc.units
        assert leaves[2] == desc.dropout_probs

        # Check aux_data contains metadata
        assert aux_data["input_dim"] == desc.input_dim
        assert aux_data["bidirectional"] == desc.bidirectional


class TestRNNDescriptorSerialization:
    """Test RNNDescriptor serialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM,),
            units=(64,),
            bidirectional=(True,),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        data = desc.to_dict()

        assert isinstance(data, dict)
        assert data["input_dim"] == 10
        assert data["rnn_types"] == [LSTM]
        assert data["units"] == [64]

    def test_from_dict(self):
        """Test reconstruction from dictionary."""
        data = {
            "input_dim": 10,
            "output_dim": 5,
            "rnn_types": [LSTM],
            "units": [64],
            "bidirectional": [True],
            "act_functions": ["relu"],
            "init_functions": ["glorot_normal"],
            "dropout_probs": [0.1],
            "use_dropout": False,
            "use_batch_norm": False,
            "max_num_layers": 5,
            "max_num_units": 128,
        }

        desc = RNNDescriptor.from_dict(data)

        assert desc.input_dim == 10
        assert desc.rnn_types == (LSTM,)
        assert desc.units == (64,)

    def test_roundtrip_serialization(self):
        """Test that serialization roundtrip preserves data."""
        key = jax.random.PRNGKey(42)
        original = RNNDescriptor.random_init(10, 5, 3, 100, key)

        data = original.to_dict()
        restored = RNNDescriptor.from_dict(data)

        assert original == restored


class TestRNNDescriptorEdgeCases:
    """Test RNNDescriptor edge cases."""

    def test_single_layer(self):
        """Test descriptor with single layer."""
        key = jax.random.PRNGKey(42)
        desc = RNNDescriptor.random_init(
            10, 5, 1, 100, key  # max_num_layers=1
        )

        assert len(desc.rnn_types) == 1
        assert desc.validate() is True

    def test_all_rnn_types(self):
        """Test descriptor with all RNN types."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(SIMPLE_RNN, LSTM, GRU),
            units=(32, 64, 32),
            bidirectional=(False, True, False),
            act_functions=("relu", "tanh", "relu"),
            init_functions=("glorot_normal", "glorot_uniform", "normal"),
            dropout_probs=(0.1, 0.2, 0.15),
            use_dropout=True,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        assert desc.validate() is True
        assert SIMPLE_RNN in desc.rnn_types
        assert LSTM in desc.rnn_types
        assert GRU in desc.rnn_types

    def test_repr(self):
        """Test string representation."""
        desc = RNNDescriptor(
            input_dim=10,
            output_dim=5,
            rnn_types=(LSTM, GRU),
            units=(64, 32),
            bidirectional=(True, False),
            act_functions=("relu", "tanh"),
            init_functions=("glorot_normal", "glorot_uniform"),
            dropout_probs=(0.1, 0.2),
            use_dropout=True,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_units=128,
        )

        repr_str = repr(desc)
        assert "RNNDescriptor" in repr_str
        assert "layers=2" in repr_str
        assert "LSTM" in repr_str
        assert "GRU" in repr_str

    def test_mixed_bidirectional(self):
        """Test descriptor with mixed bidirectional layers."""
        key = jax.random.PRNGKey(42)
        desc = RNNDescriptor.random_init(10, 5, 3, 100, key)

        # Should have some bidirectional flags
        assert len(desc.bidirectional) == len(desc.rnn_types)
        assert all(isinstance(b, bool) for b in desc.bidirectional)
