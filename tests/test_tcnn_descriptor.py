"""Tests for TCNNDescriptor class."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from nasjax.descriptors.tcnn import (
    TCNNDescriptor,
    ACTIVATIONS,
    INITIALIZATIONS,
    calculate_tcnn_output_shape,
)


class TestTCNNDescriptorCreation:
    """Test TCNNDescriptor creation and initialization."""

    def test_random_init_basic(self):
        """Test basic random initialization."""
        key = jax.random.PRNGKey(42)
        desc = TCNNDescriptor.random_init(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
            key=key,
        )

        assert desc.input_dim == (7, 7, 10)
        assert desc.output_dim == (28, 28, 1)
        assert desc.max_num_layers == 5
        assert desc.max_filter == 5
        assert desc.max_stride == 3
        assert 1 <= len(desc.filters) <= 5
        assert len(desc.strides) == len(desc.filters)

    def test_random_init_with_batch_norm(self):
        """Test random initialization with batch norm enabled."""
        key = jax.random.PRNGKey(0)
        desc = TCNNDescriptor.random_init(
            input_dim=(8, 8, 16),
            output_dim=(32, 32, 3),
            max_num_layers=3,
            max_filter=5,
            max_stride=2,
            key=key,
            batch_norm=True,
        )

        assert desc.use_batch_norm is True

    def test_random_init_reproducibility(self):
        """Test that same key produces same descriptor."""
        key = jax.random.PRNGKey(123)

        desc1 = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 5, 5, 3, key
        )
        desc2 = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 5, 5, 3, key
        )

        assert desc1.filters == desc2.filters
        assert desc1.strides == desc2.strides

    def test_random_init_different_keys(self):
        """Test that different keys produce different descriptors."""
        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(2)

        desc1 = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 5, 5, 3, key1
        )
        desc2 = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 5, 5, 3, key2
        )

        # At least one should be different
        assert (
            desc1.filters != desc2.filters
            or desc1.strides != desc2.strides
        )


class TestTCNNDescriptorValidation:
    """Test TCNNDescriptor validation."""

    def test_valid_descriptor(self):
        """Test that a valid descriptor passes validation."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((4, 4, 16), (4, 4, 1)),
            strides=((2, 2, 1), (2, 2, 1)),
            act_functions=("relu", "relu"),
            init_functions=("glorot_normal", "glorot_uniform"),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        assert desc.validate() is True

    def test_invalid_too_many_layers(self):
        """Test that too many layers fails validation."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((3, 3, 16),) * 6,  # More than max
            strides=((2, 2, 1),) * 6,
            act_functions=("relu",) * 6,
            init_functions=("glorot_normal",) * 6,
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        assert desc.validate() is False

    def test_invalid_filter_size(self):
        """Test that invalid filter size fails validation."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((10, 10, 16),),  # Filter too large
            strides=((2, 2, 1),),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        assert desc.validate() is False

    def test_invalid_stride_size(self):
        """Test that invalid stride size fails validation."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((3, 3, 16),),
            strides=((5, 5, 1),),  # Stride too large
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        assert desc.validate() is False

    def test_invalid_activation_function(self):
        """Test that invalid activation fails validation."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((3, 3, 16),),
            strides=((2, 2, 1),),
            act_functions=("invalid_activation",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        assert desc.validate() is False

    def test_invalid_inconsistent_lengths(self):
        """Test that inconsistent list lengths fail validation."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((3, 3, 16), (3, 3, 1)),
            strides=((2, 2, 1),),  # Wrong length
            act_functions=("relu", "relu"),
            init_functions=("glorot_normal", "glorot_uniform"),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        assert desc.validate() is False


class TestTCNNDescriptorPyTree:
    """Test TCNNDescriptor PyTree operations."""

    def test_pytree_registration(self):
        """Test that TCNNDescriptor is registered as a PyTree."""
        key = jax.random.PRNGKey(42)
        desc = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 5, 5, 3, key
        )

        # Should be able to flatten and unflatten
        leaves, treedef = jtu.tree_flatten(desc)
        desc_restored = jtu.tree_unflatten(treedef, leaves)

        assert desc == desc_restored

    def test_pytree_map(self):
        """Test that tree_map works on TCNNDescriptor."""
        key = jax.random.PRNGKey(42)
        desc = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 5, 5, 3, key
        )

        # This should work without errors
        result = jtu.tree_map(lambda x: x, desc)
        assert result == desc

    def test_tree_structure(self):
        """Test tree structure and leaves."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((3, 3, 16),),
            strides=((2, 2, 1),),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        leaves, aux_data = desc.tree_flatten()

        # Check leaves contain mutable data
        assert leaves[0] == desc.filters
        assert leaves[1] == desc.strides

        # Check aux_data contains metadata
        assert aux_data["input_dim"] == desc.input_dim
        assert aux_data["use_batch_norm"] == desc.use_batch_norm


class TestTCNNDescriptorSerialization:
    """Test TCNNDescriptor serialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((3, 3, 16),),
            strides=((2, 2, 1),),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        data = desc.to_dict()

        assert isinstance(data, dict)
        assert data["input_dim"] == [7, 7, 10]
        assert data["filters"] == [[3, 3, 16]]

    def test_from_dict(self):
        """Test reconstruction from dictionary."""
        data = {
            "input_dim": [7, 7, 10],
            "output_dim": [28, 28, 1],
            "filters": [[3, 3, 16]],
            "strides": [[2, 2, 1]],
            "act_functions": ["relu"],
            "init_functions": ["glorot_normal"],
            "use_batch_norm": False,
            "max_num_layers": 5,
            "max_filter": 5,
            "max_stride": 3,
        }

        desc = TCNNDescriptor.from_dict(data)

        assert desc.input_dim == (7, 7, 10)
        assert desc.filters == ((3, 3, 16),)

    def test_roundtrip_serialization(self):
        """Test that serialization roundtrip preserves data."""
        key = jax.random.PRNGKey(42)
        original = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 5, 5, 3, key
        )

        data = original.to_dict()
        restored = TCNNDescriptor.from_dict(data)

        assert original == restored


class TestTCNNDescriptorEdgeCases:
    """Test TCNNDescriptor edge cases."""

    def test_single_layer(self):
        """Test descriptor with single layer."""
        key = jax.random.PRNGKey(42)
        desc = TCNNDescriptor.random_init(
            (7, 7, 10), (28, 28, 1), 1, 5, 3, key  # max_num_layers=1
        )

        assert len(desc.filters) == 1
        assert desc.validate() is True

    def test_repr(self):
        """Test string representation."""
        desc = TCNNDescriptor(
            input_dim=(7, 7, 10),
            output_dim=(28, 28, 1),
            filters=((3, 3, 16),),
            strides=((2, 2, 1),),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        repr_str = repr(desc)
        assert "TCNNDescriptor" in repr_str
        assert "layers=1" in repr_str


class TestTCNNShapeCalculation:
    """Test TCNN output shape calculation helper function."""

    def test_single_transposed_conv_layer(self):
        """Test shape calculation for single transposed conv layer."""
        input_shape = (7, 7, 10)
        filters = ((4, 4, 16),)
        strides = ((2, 2, 1),)

        output = calculate_tcnn_output_shape(input_shape, filters, strides)

        # output = 7 * 2 + max(4 - 2, 0) = 14 + 2 = 16
        assert output == (16, 16, 16)

    def test_multiple_transposed_conv_layers(self):
        """Test shape calculation for multiple transposed conv layers."""
        input_shape = (7, 7, 10)
        filters = ((3, 3, 16), (4, 4, 1))
        strides = ((2, 2, 1), (2, 2, 1))

        output = calculate_tcnn_output_shape(input_shape, filters, strides)

        # Layer 1: 7 * 2 + max(3 - 2, 0) = 14 + 1 = 15
        # Layer 2: 15 * 2 + max(4 - 2, 0) = 30 + 2 = 32
        assert output == (32, 32, 1)

    def test_no_layers(self):
        """Test shape calculation with no layers."""
        input_shape = (7, 7, 10)
        filters = ()
        strides = ()

        output = calculate_tcnn_output_shape(input_shape, filters, strides)

        assert output == input_shape

    def test_stride_larger_than_filter(self):
        """Test shape calculation when stride > filter."""
        input_shape = (4, 4, 8)
        filters = ((2, 2, 16),)
        strides = ((3, 3, 1),)

        output = calculate_tcnn_output_shape(input_shape, filters, strides)

        # output = 4 * 3 + max(2 - 3, 0) = 12 + 0 = 12
        assert output == (12, 12, 16)

    def test_stride_equals_filter(self):
        """Test shape calculation when stride == filter."""
        input_shape = (5, 5, 8)
        filters = ((3, 3, 16),)
        strides = ((3, 3, 1),)

        output = calculate_tcnn_output_shape(input_shape, filters, strides)

        # output = 5 * 3 + max(3 - 3, 0) = 15 + 0 = 15
        assert output == (15, 15, 16)
