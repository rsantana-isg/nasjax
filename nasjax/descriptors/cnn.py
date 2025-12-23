"""CNN (Convolutional Neural Network) descriptor implementation.

This module implements the CNNDescriptor class, which is an immutable PyTree
representing the architecture specification for a convolutional neural network.
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .base import BaseDescriptor

# Valid activation function names
ACTIVATIONS = ["relu", "elu", "sigmoid", "tanh", "softplus", "softsign", None]

# Valid weight initialization function names
INITIALIZATIONS = ["glorot_normal", "glorot_uniform", "normal", "uniform"]

# Layer type constants
CONV_LAYER = 0  # Convolutional layer
MAX_POOL_LAYER = 1  # Max pooling layer
AVG_POOL_LAYER = 2  # Average pooling layer

# Constants for valid ranges
MIN_NUM_FILTERS = 1
MIN_NUM_STRIDES = 1
MIN_NUM_CHANNELS = 3
MAX_NUM_CHANNELS = 65


def calculate_cnn_output_shape(
    input_shape: Tuple[int, int, int],
    filters: Tuple[Tuple[int, int, int], ...],
    strides: Tuple[Tuple[int, int, int], ...],
) -> Tuple[int, int, int]:
    """Calculate the output shape after applying convolutional layers.

    Args:
        input_shape: Input shape (height, width, channels)
        filters: Tuple of filter sizes for each layer (h, w, c)
        strides: Tuple of stride sizes for each layer (h, w, c)

    Returns:
        Output shape (height, width, channels) after all layers
    """
    if len(filters) == 0:
        return input_shape

    h, w, c = input_shape
    fh, fw, fc = filters[0]
    sh, sw, _ = strides[0]

    # Calculate output dimensions: (input - filter) // stride + 1
    out_h = (h - fh) // sh + 1
    out_w = (w - fw) // sw + 1
    out_c = fc

    # Recursively calculate for remaining layers
    if len(filters) > 1:
        return calculate_cnn_output_shape((out_h, out_w, out_c), filters[1:], strides[1:])
    return (out_h, out_w, out_c)


@jtu.register_pytree_node_class
class CNNDescriptor(NamedTuple):
    """Descriptor for Convolutional Neural Network architectures.

    This is an immutable PyTree that represents the genotype of a CNN.
    It can be transformed by JAX operations (jit, vmap, pmap) and used
    with evolutionary operators.

    Attributes:
        input_dim: Input dimensions as (height, width, channels)
        output_dim: Output dimensions as (height, width, channels)
        layer_types: Tuple of layer type indices (0=Conv, 1=MaxPool, 2=AvgPool)
        filters: Tuple of filter sizes for each layer (h, w, out_channels)
        strides: Tuple of stride sizes for each layer (h, w, 1)
        act_functions: Tuple of activation function names for each layer
        init_functions: Tuple of weight initializer names for each layer
        use_batch_norm: Whether batch normalization is enabled
        max_num_layers: Maximum allowed layers (for mutation constraints)
        max_filter: Maximum filter size allowed
        max_stride: Maximum stride size allowed

    Example:
        >>> import jax
        >>> key = jax.random.PRNGKey(42)
        >>> desc = CNNDescriptor.random_init(
        ...     input_dim=(28, 28, 1),
        ...     output_dim=(7, 7, 10),
        ...     max_num_layers=5,
        ...     max_filter=5,
        ...     max_stride=3,
        ...     key=key
        ... )
        >>> print(desc.layer_types)  # e.g., (0, 1, 0)
        >>> print(desc.validate())  # True
    """

    input_dim: Tuple[int, int, int]
    output_dim: Tuple[int, int, int]
    layer_types: Tuple[int, ...]
    filters: Tuple[Tuple[int, int, int], ...]
    strides: Tuple[Tuple[int, int, int], ...]
    act_functions: Tuple[Optional[str], ...]
    init_functions: Tuple[str, ...]
    use_batch_norm: bool
    max_num_layers: int
    max_filter: int
    max_stride: int

    def tree_flatten(self):
        """Flatten descriptor into leaves and auxiliary data for JAX.

        Returns:
            Tuple of (leaves, aux_data)
        """
        # Leaves are the data that can be transformed by JAX
        leaves = (self.layer_types, self.filters, self.strides)

        # Auxiliary data is metadata that doesn't change
        aux_data = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "act_functions": self.act_functions,
            "init_functions": self.init_functions,
            "use_batch_norm": self.use_batch_norm,
            "max_num_layers": self.max_num_layers,
            "max_filter": self.max_filter,
            "max_stride": self.max_stride,
        }
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        """Reconstruct descriptor from flattened form.

        Args:
            aux_data: Dictionary of auxiliary data
            leaves: Tuple of (layer_types, filters, strides)

        Returns:
            Reconstructed CNNDescriptor instance
        """
        layer_types, filters, strides = leaves
        return cls(layer_types=layer_types, filters=filters, strides=strides, **aux_data)

    @staticmethod
    def random_init(
        input_dim: Tuple[int, int, int],
        output_dim: Tuple[int, int, int],
        max_num_layers: int,
        max_filter: int,
        max_stride: int,
        key: jax.random.PRNGKey,
        batch_norm: bool = False,
    ) -> "CNNDescriptor":
        """Create a random CNN descriptor.

        This factory method generates a random CNN architecture with:
        - Random number of layers (1 to max_num_layers)
        - Random layer types (Conv, MaxPool, AvgPool)
        - Random filter sizes and strides
        - Random activation functions
        - Random weight initializers

        Args:
            input_dim: Input dimensions (height, width, channels)
            output_dim: Output dimensions (height, width, channels)
            max_num_layers: Maximum number of layers
            max_filter: Maximum filter size
            max_stride: Maximum stride size
            key: JAX PRNG key for randomness
            batch_norm: Whether to enable batch normalization

        Returns:
            Randomly initialized CNNDescriptor

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> desc = CNNDescriptor.random_init((28, 28, 1), (7, 7, 10), 5, 5, 3, key)
            >>> assert desc.validate()
        """
        k1, k2 = jax.random.split(key, 2)

        # Start with 1-3 layers as a simple initialization
        num_layers = int(jax.random.randint(k1, (), 1, min(4, max_num_layers + 1)))

        # Build layers
        layer_types = []
        filters = []
        strides = []
        act_functions = []
        init_functions = []

        current_shape = input_dim

        for i in range(num_layers):
            k2, k_type, k_filter, k_stride, k_act, k_init = jax.random.split(k2, 6)

            # Always use Conv layer for simplicity in random init
            layer_type = CONV_LAYER

            # Random filter size (between MIN_NUM_FILTERS and max_filter)
            filter_h = int(jax.random.randint(k_filter, (), MIN_NUM_FILTERS, max_filter + 1))
            filter_w = filter_h  # Square filters
            filter_c = int(
                jax.random.randint(k_filter, (), MIN_NUM_CHANNELS, min(MAX_NUM_CHANNELS, 32))
            )

            # Random stride size
            stride_h = int(jax.random.randint(k_stride, (), MIN_NUM_STRIDES, max_stride + 1))
            stride_w = stride_h
            stride_c = 1

            # Check if this layer would produce valid output
            test_h = (current_shape[0] - filter_h) // stride_h + 1
            test_w = (current_shape[1] - filter_w) // stride_w + 1

            # If output would be too small, stop adding layers
            if test_h < 2 or test_w < 2:
                break

            # Add the layer
            layer_types.append(layer_type)
            filters.append((filter_h, filter_w, filter_c))
            strides.append((stride_h, stride_w, stride_c))

            # Random activation and initialization
            act_idx = int(jax.random.randint(k_act, (), 0, len(ACTIVATIONS)))
            act_functions.append(ACTIVATIONS[act_idx])

            init_idx = int(jax.random.randint(k_init, (), 0, len(INITIALIZATIONS)))
            init_functions.append(INITIALIZATIONS[init_idx])

            # Update current shape
            current_shape = (test_h, test_w, filter_c)

        # Ensure we have at least one layer
        if len(layer_types) == 0:
            layer_types = [CONV_LAYER]
            filters = [(3, 3, MIN_NUM_CHANNELS)]
            strides = [(1, 1, 1)]
            act_functions = [ACTIVATIONS[0]]
            init_functions = [INITIALIZATIONS[0]]

        return CNNDescriptor(
            input_dim=input_dim,
            output_dim=output_dim,
            layer_types=tuple(layer_types),
            filters=tuple(filters),
            strides=tuple(strides),
            act_functions=tuple(act_functions),
            init_functions=tuple(init_functions),
            use_batch_norm=batch_norm,
            max_num_layers=max_num_layers,
            max_filter=max_filter,
            max_stride=max_stride,
        )

    def validate(self) -> bool:
        """Validate that the descriptor satisfies all constraints.

        Checks:
        - Number of layers is within [1, max_num_layers]
        - All layer types are valid (0, 1, or 2)
        - All filter dimensions are within bounds
        - All stride dimensions are within bounds
        - All lists have consistent lengths
        - All activation functions are valid
        - All initializers are valid
        - Output shape is achievable

        Returns:
            True if descriptor is valid, False otherwise
        """
        # Check layer count
        num_layers = len(self.layer_types)
        if num_layers > self.max_num_layers or num_layers < 1:
            return False

        # Check list lengths are consistent
        if len(self.filters) != num_layers:
            return False
        if len(self.strides) != num_layers:
            return False
        if len(self.act_functions) != num_layers:
            return False
        if len(self.init_functions) != num_layers:
            return False

        # Check layer types are valid
        if any(lt not in [CONV_LAYER, MAX_POOL_LAYER, AVG_POOL_LAYER] for lt in self.layer_types):
            return False

        # Check filter dimensions
        for fh, fw, fc in self.filters:
            if fh < MIN_NUM_FILTERS or fh > self.max_filter:
                return False
            if fw < MIN_NUM_FILTERS or fw > self.max_filter:
                return False
            if fc < MIN_NUM_CHANNELS or fc > MAX_NUM_CHANNELS:
                return False

        # Check stride dimensions
        for sh, sw, sc in self.strides:
            if sh < MIN_NUM_STRIDES or sh > self.max_stride:
                return False
            if sw < MIN_NUM_STRIDES or sw > self.max_stride:
                return False
            if sc != 1:  # Stride in channel dimension should always be 1
                return False

        # Check activation functions are valid
        if any(act not in ACTIVATIONS for act in self.act_functions):
            return False

        # Check initializers are valid
        if any(init not in INITIALIZATIONS for init in self.init_functions):
            return False

        # Check that output shape is achievable (at least produces positive dimensions)
        try:
            output_shape = calculate_cnn_output_shape(
                self.input_dim, self.filters, self.strides
            )
            if output_shape[0] < 1 or output_shape[1] < 1:
                return False
        except Exception:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary for serialization.

        Returns:
            Dictionary representation of the descriptor
        """
        return {
            "input_dim": list(self.input_dim),
            "output_dim": list(self.output_dim),
            "layer_types": list(self.layer_types),
            "filters": [list(f) for f in self.filters],
            "strides": [list(s) for s in self.strides],
            "act_functions": list(self.act_functions),
            "init_functions": list(self.init_functions),
            "use_batch_norm": self.use_batch_norm,
            "max_num_layers": self.max_num_layers,
            "max_filter": self.max_filter,
            "max_stride": self.max_stride,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CNNDescriptor":
        """Reconstruct descriptor from dictionary.

        Args:
            data: Dictionary containing descriptor data

        Returns:
            Reconstructed CNNDescriptor instance
        """
        return cls(
            input_dim=tuple(data["input_dim"]),
            output_dim=tuple(data["output_dim"]),
            layer_types=tuple(data["layer_types"]),
            filters=tuple(tuple(f) for f in data["filters"]),
            strides=tuple(tuple(s) for s in data["strides"]),
            act_functions=tuple(data["act_functions"]),
            init_functions=tuple(data["init_functions"]),
            use_batch_norm=data["use_batch_norm"],
            max_num_layers=data["max_num_layers"],
            max_filter=data["max_filter"],
            max_stride=data["max_stride"],
        )

    def __repr__(self) -> str:
        """String representation of the descriptor."""
        return (
            f"CNNDescriptor(layers={len(self.layer_types)}, "
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"batch_norm={self.use_batch_norm})"
        )
