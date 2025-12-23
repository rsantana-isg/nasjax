"""Tests for CNNDescriptor class."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from nasjax.descriptors.cnn import (
    CNNDescriptor,
    ACTIVATIONS,
    INITIALIZATIONS,
    CONV_LAYER,
    MAX_POOL_LAYER,
    AVG_POOL_LAYER,
    calculate_cnn_output_shape,
)


class TestCNNDescriptorCreation:
    """Test CNNDescriptor creation and initialization."""

    def test_random_init_basic(self):
        """Test basic random initialization."""
        key = jax.random.PRNGKey(42)
        desc = CNNDescriptor.random_init(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
            key=key,
        )

        assert desc.input_dim == (28, 28, 1)
        assert desc.output_dim == (7, 7, 10)
        assert desc.max_num_layers == 5
        assert desc.max_filter == 5
        assert desc.max_stride == 3
        assert 1 <= len(desc.layer_types) <= 5
        assert len(desc.filters) == len(desc.layer_types)
        assert len(desc.strides) == len(desc.layer_types)

    def test_random_init_with_batch_norm(self):
        """Test random initialization with batch norm enabled."""
        key = jax.random.PRNGKey(0)
        desc = CNNDescriptor.random_init(
            input_dim=(32, 32, 3),
            output_dim=(8, 8, 16),
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

        desc1 = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 5, 5, 3, key
        )
        desc2 = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 5, 5, 3, key
        )

        assert desc1.layer_types == desc2.layer_types
        assert desc1.filters == desc2.filters
        assert desc1.strides == desc2.strides

    def test_random_init_different_keys(self):
        """Test that different keys produce different descriptors."""
        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(2)

        desc1 = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 5, 5, 3, key1
        )
        desc2 = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 5, 5, 3, key2
        )

        # At least one should be different
        assert (
            desc1.filters != desc2.filters
            or desc1.strides != desc2.strides
            or desc1.layer_types != desc2.layer_types
        )


class TestCNNDescriptorValidation:
    """Test CNNDescriptor validation."""

    def test_valid_descriptor(self):
        """Test that a valid descriptor passes validation."""
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER, CONV_LAYER),
            filters=((3, 3, 16), (3, 3, 10)),
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
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER,) * 6,  # More than max
            filters=((3, 3, 16),) * 6,
            strides=((1, 1, 1),) * 6,
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
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER,),
            filters=((10, 10, 16),),  # Filter too large
            strides=((1, 1, 1),),
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
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER,),
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
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER,),
            filters=((3, 3, 16),),
            strides=((1, 1, 1),),
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
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER, CONV_LAYER),
            filters=((3, 3, 16),),  # Wrong length
            strides=((1, 1, 1), (1, 1, 1)),
            act_functions=("relu", "relu"),
            init_functions=("glorot_normal", "glorot_uniform"),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        assert desc.validate() is False


class TestCNNDescriptorPyTree:
    """Test CNNDescriptor PyTree operations."""

    def test_pytree_registration(self):
        """Test that CNNDescriptor is registered as a PyTree."""
        key = jax.random.PRNGKey(42)
        desc = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 5, 5, 3, key
        )

        # Should be able to flatten and unflatten
        leaves, treedef = jtu.tree_flatten(desc)
        desc_restored = jtu.tree_unflatten(treedef, leaves)

        assert desc == desc_restored

    def test_pytree_map(self):
        """Test that tree_map works on CNNDescriptor."""
        key = jax.random.PRNGKey(42)
        desc = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 5, 5, 3, key
        )

        # This should work without errors
        result = jtu.tree_map(lambda x: x, desc)
        assert result == desc

    def test_tree_structure(self):
        """Test tree structure and leaves."""
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER,),
            filters=((3, 3, 16),),
            strides=((1, 1, 1),),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        leaves, aux_data = desc.tree_flatten()

        # Check leaves contain mutable data
        assert leaves[0] == desc.layer_types
        assert leaves[1] == desc.filters
        assert leaves[2] == desc.strides

        # Check aux_data contains metadata
        assert aux_data["input_dim"] == desc.input_dim
        assert aux_data["use_batch_norm"] == desc.use_batch_norm


class TestCNNDescriptorSerialization:
    """Test CNNDescriptor serialization."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER,),
            filters=((3, 3, 16),),
            strides=((1, 1, 1),),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        data = desc.to_dict()

        assert isinstance(data, dict)
        assert data["input_dim"] == [28, 28, 1]
        assert data["layer_types"] == [CONV_LAYER]
        assert data["filters"] == [[3, 3, 16]]

    def test_from_dict(self):
        """Test reconstruction from dictionary."""
        data = {
            "input_dim": [28, 28, 1],
            "output_dim": [7, 7, 10],
            "layer_types": [CONV_LAYER],
            "filters": [[3, 3, 16]],
            "strides": [[1, 1, 1]],
            "act_functions": ["relu"],
            "init_functions": ["glorot_normal"],
            "use_batch_norm": False,
            "max_num_layers": 5,
            "max_filter": 5,
            "max_stride": 3,
        }

        desc = CNNDescriptor.from_dict(data)

        assert desc.input_dim == (28, 28, 1)
        assert desc.layer_types == (CONV_LAYER,)
        assert desc.filters == ((3, 3, 16),)

    def test_roundtrip_serialization(self):
        """Test that serialization roundtrip preserves data."""
        key = jax.random.PRNGKey(42)
        original = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 5, 5, 3, key
        )

        data = original.to_dict()
        restored = CNNDescriptor.from_dict(data)

        assert original == restored


class TestCNNDescriptorEdgeCases:
    """Test CNNDescriptor edge cases."""

    def test_single_layer(self):
        """Test descriptor with single layer."""
        key = jax.random.PRNGKey(42)
        desc = CNNDescriptor.random_init(
            (28, 28, 1), (7, 7, 10), 1, 5, 3, key  # max_num_layers=1
        )

        assert len(desc.layer_types) == 1
        assert desc.validate() is True

    def test_repr(self):
        """Test string representation."""
        desc = CNNDescriptor(
            input_dim=(28, 28, 1),
            output_dim=(7, 7, 10),
            layer_types=(CONV_LAYER,),
            filters=((3, 3, 16),),
            strides=((1, 1, 1),),
            act_functions=("relu",),
            init_functions=("glorot_normal",),
            use_batch_norm=False,
            max_num_layers=5,
            max_filter=5,
            max_stride=3,
        )

        repr_str = repr(desc)
        assert "CNNDescriptor" in repr_str
        assert "layers=1" in repr_str


class TestCNNShapeCalculation:
    """Test CNN output shape calculation helper function."""

    def test_single_conv_layer(self):
        """Test shape calculation for single conv layer."""
        input_shape = (28, 28, 1)
        filters = ((3, 3, 16),)
        strides = ((1, 1, 1),)

        output = calculate_cnn_output_shape(input_shape, filters, strides)

        # (28 - 3) // 1 + 1 = 26
        assert output == (26, 26, 16)

    def test_multiple_conv_layers(self):
        """Test shape calculation for multiple conv layers."""
        input_shape = (28, 28, 1)
        filters = ((3, 3, 16), (3, 3, 32))
        strides = ((2, 2, 1), (2, 2, 1))

        output = calculate_cnn_output_shape(input_shape, filters, strides)

        # Layer 1: (28 - 3) // 2 + 1 = 13
        # Layer 2: (13 - 3) // 2 + 1 = 6
        assert output == (6, 6, 32)

    def test_no_layers(self):
        """Test shape calculation with no layers."""
        input_shape = (28, 28, 1)
        filters = ()
        strides = ()

        output = calculate_cnn_output_shape(input_shape, filters, strides)

        assert output == input_shape
