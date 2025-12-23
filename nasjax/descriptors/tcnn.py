"""TCNN (Transposed Convolutional Neural Network) descriptor implementation.

This module implements the TCNNDescriptor class, which is an immutable PyTree
representing the architecture specification for a transposed convolutional neural network
(also known as deconvolutional networks), useful for generative tasks.
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

# Constants for valid ranges
MIN_NUM_FILTERS = 1
MIN_NUM_STRIDES = 1
MIN_NUM_CHANNELS = 3
MAX_NUM_CHANNELS = 65


def calculate_tcnn_output_shape(
    input_shape: Tuple[int, int, int],
    filters: Tuple[Tuple[int, int, int], ...],
    strides: Tuple[Tuple[int, int, int], ...],
) -> Tuple[int, int, int]:
    """Calculate the output shape after applying transposed convolutional layers.

    For transposed convolution, the output size is:
        output = input * stride + max(filter - stride, 0)

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

    # Calculate output dimensions for transposed convolution
    out_h = h * sh + max(fh - sh, 0)
    out_w = w * sw + max(fw - sw, 0)
    out_c = fc

    # Recursively calculate for remaining layers
    if len(filters) > 1:
        return calculate_tcnn_output_shape((out_h, out_w, out_c), filters[1:], strides[1:])
    return (out_h, out_w, out_c)


@jtu.register_pytree_node_class
class TCNNDescriptor(NamedTuple):
    """Descriptor for Transposed Convolutional Neural Network architectures.

    This is an immutable PyTree that represents the genotype of a TCNN.
    It can be transformed by JAX operations (jit, vmap, pmap) and used
    with evolutionary operators. TCNNs are useful for generative tasks
    where the goal is to upsample from a lower resolution to a higher one.

    Attributes:
        input_dim: Input dimensions as (height, width, channels)
        output_dim: Output dimensions as (height, width, channels)
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
        >>> desc = TCNNDescriptor.random_init(
        ...     input_dim=(7, 7, 10),
        ...     output_dim=(28, 28, 1),
        ...     max_num_layers=5,
        ...     max_filter=5,
        ...     max_stride=3,
        ...     key=key
        ... )
        >>> print(desc.filters)  # e.g., ((3, 3, 16), (4, 4, 1))
        >>> print(desc.validate())  # True
    """

    input_dim: Tuple[int, int, int]
    output_dim: Tuple[int, int, int]
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
        leaves = (self.filters, self.strides)

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
            leaves: Tuple of (filters, strides)

        Returns:
            Reconstructed TCNNDescriptor instance
        """
        filters, strides = leaves
        return cls(filters=filters, strides=strides, **aux_data)

    @staticmethod
    def random_init(
        input_dim: Tuple[int, int, int],
        output_dim: Tuple[int, int, int],
        max_num_layers: int,
        max_filter: int,
        max_stride: int,
        key: jax.random.PRNGKey,
        batch_norm: bool = False,
    ) -> "TCNNDescriptor":
        """Create a random TCNN descriptor.

        This factory method generates a random TCNN architecture with:
        - Random number of layers (1 to max_num_layers)
        - Random filter sizes and strides
        - Random activation functions
        - Random weight initializers

        The algorithm progressively adds layers until the output dimensions
        are close to or exceed the target output dimensions.

        Args:
            input_dim: Input dimensions (height, width, channels)
            output_dim: Output dimensions (height, width, channels)
            max_num_layers: Maximum number of layers
            max_filter: Maximum filter size
            max_stride: Maximum stride size
            key: JAX PRNG key for randomness
            batch_norm: Whether to enable batch normalization

        Returns:
            Randomly initialized TCNNDescriptor

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> desc = TCNNDescriptor.random_init((7, 7, 10), (28, 28, 1), 5, 5, 3, key)
            >>> assert desc.validate()
        """
        k1, k2 = jax.random.split(key, 2)

        filters = []
        strides = []
        act_functions = []
        init_functions = []

        current_shape = input_dim
        max_attempts = 300  # Prevent infinite loops

        # Keep adding layers until we reach or exceed the output dimensions
        for attempt in range(max_attempts):
            k2, k_filter, k_stride, k_act, k_init = jax.random.split(k2, 5)

            # Stop if we've reached max layers
            if len(filters) >= max_num_layers:
                break

            # Random filter size
            filter_h = int(jax.random.randint(k_filter, (), MIN_NUM_FILTERS, max_filter + 1))
            filter_w = filter_h  # Square filters
            filter_c = int(
                jax.random.randint(k_filter, (), MIN_NUM_CHANNELS, min(MAX_NUM_CHANNELS, 32))
            )

            # Random stride size
            stride_h = int(jax.random.randint(k_stride, (), MIN_NUM_STRIDES, max_stride + 1))
            stride_w = stride_h
            stride_c = 1

            # Calculate what the output would be
            test_h = current_shape[0] * stride_h + max(filter_h - stride_h, 0)
            test_w = current_shape[1] * stride_w + max(filter_w - stride_w, 0)

            # Add the layer
            filters.append((filter_h, filter_w, filter_c))
            strides.append((stride_h, stride_w, stride_c))

            # Random activation and initialization
            act_idx = int(jax.random.randint(k_act, (), 0, len(ACTIVATIONS)))
            act_functions.append(ACTIVATIONS[act_idx])

            init_idx = int(jax.random.randint(k_init, (), 0, len(INITIALIZATIONS)))
            init_functions.append(INITIALIZATIONS[init_idx])

            # Update current shape
            current_shape = (test_h, test_w, filter_c)

            # Check if we've reached or exceeded the target output dimensions
            if test_h >= output_dim[0] and test_w >= output_dim[1]:
                # Adjust the last layer to match output dimensions exactly
                prev_h, prev_w, prev_c = (
                    current_shape[0] // stride_h if stride_h > 0 else current_shape[0],
                    current_shape[1] // stride_w if stride_w > 0 else current_shape[1],
                    filter_c,
                )

                # Calculate the filter size needed to reach exact output dimensions
                desired_filter_h = output_dim[0] - (prev_h - 1) * stride_h
                desired_filter_w = output_dim[1] - (prev_w - 1) * stride_w

                # Update the last filter to target the output dimensions and channels
                if desired_filter_h > 0 and desired_filter_w > 0:
                    filters[-1] = (desired_filter_h, desired_filter_w, output_dim[2])
                break

        # Ensure we have at least one layer
        if len(filters) == 0:
            filters = [(3, 3, output_dim[2])]
            strides = [(2, 2, 1)]
            act_functions = [ACTIVATIONS[0]]
            init_functions = [INITIALIZATIONS[0]]

        return TCNNDescriptor(
            input_dim=input_dim,
            output_dim=output_dim,
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
        - All filter dimensions are within bounds
        - All stride dimensions are within bounds
        - All lists have consistent lengths
        - All activation functions are valid
        - All initializers are valid

        Returns:
            True if descriptor is valid, False otherwise
        """
        # Check layer count
        num_layers = len(self.filters)
        if num_layers > self.max_num_layers or num_layers < 1:
            return False

        # Check list lengths are consistent
        if len(self.strides) != num_layers:
            return False
        if len(self.act_functions) != num_layers:
            return False
        if len(self.init_functions) != num_layers:
            return False

        # Check filter dimensions
        for fh, fw, fc in self.filters:
            if fh < MIN_NUM_FILTERS or fh > self.max_filter:
                return False
            if fw < MIN_NUM_FILTERS or fw > self.max_filter:
                return False
            # For TCNN, allow any positive channel count (including 1 for grayscale output)
            if fc < 1 or fc > MAX_NUM_CHANNELS:
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

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary for serialization.

        Returns:
            Dictionary representation of the descriptor
        """
        return {
            "input_dim": list(self.input_dim),
            "output_dim": list(self.output_dim),
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
    def from_dict(cls, data: Dict[str, Any]) -> "TCNNDescriptor":
        """Reconstruct descriptor from dictionary.

        Args:
            data: Dictionary containing descriptor data

        Returns:
            Reconstructed TCNNDescriptor instance
        """
        return cls(
            input_dim=tuple(data["input_dim"]),
            output_dim=tuple(data["output_dim"]),
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
            f"TCNNDescriptor(layers={len(self.filters)}, "
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"batch_norm={self.use_batch_norm})"
        )
