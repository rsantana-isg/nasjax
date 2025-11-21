"""MLP (Multi-Layer Perceptron) descriptor implementation.

This module implements the MLPDescriptor class, which is an immutable PyTree
representing the architecture specification for a multi-layer perceptron.
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


@jtu.register_pytree_node_class
class MLPDescriptor(NamedTuple):
    """Descriptor for Multi-Layer Perceptron architectures.

    This is an immutable PyTree that represents the genotype of an MLP.
    It can be transformed by JAX operations (jit, vmap, pmap) and used
    with evolutionary operators.

    Attributes:
        input_dim: Input dimension (will be flattened if multidimensional)
        output_dim: Output dimension
        dims: Tuple of hidden layer sizes (e.g., (64, 128, 32))
        act_functions: Tuple of activation function names for each layer
        init_functions: Tuple of weight initializer names for each layer
        dropout_probs: Tuple of dropout probabilities per layer (0.0 to 1.0)
        use_dropout: Whether dropout is enabled globally
        use_batch_norm: Whether batch normalization is enabled
        max_num_layers: Maximum allowed layers (for mutation constraints)
        max_num_neurons: Maximum neurons per layer (for mutation constraints)

    Example:
        >>> import jax
        >>> key = jax.random.PRNGKey(42)
        >>> desc = MLPDescriptor.random_init(
        ...     input_dim=784,
        ...     output_dim=10,
        ...     max_num_layers=5,
        ...     max_num_neurons=128,
        ...     key=key
        ... )
        >>> print(desc.dims)  # e.g., (64, 32, 96)
        >>> print(desc.validate())  # True
    """

    input_dim: int
    output_dim: int
    dims: Tuple[int, ...]
    act_functions: Tuple[Optional[str], ...]
    init_functions: Tuple[str, ...]
    dropout_probs: Tuple[float, ...]
    use_dropout: bool
    use_batch_norm: bool
    max_num_layers: int
    max_num_neurons: int

    def tree_flatten(self):
        """Flatten descriptor into leaves and auxiliary data for JAX.

        This method is required for PyTree registration. It separates the
        descriptor into:
        - Leaves: Arrays/numbers that JAX will transform (dims, dropout_probs)
        - Aux data: Metadata that remains constant (activations, flags, etc.)

        Returns:
            Tuple of (leaves, aux_data)
        """
        # Leaves are the data that can be transformed by JAX
        leaves = (self.dims, self.dropout_probs)

        # Auxiliary data is metadata that doesn't change
        aux_data = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "act_functions": self.act_functions,
            "init_functions": self.init_functions,
            "use_dropout": self.use_dropout,
            "use_batch_norm": self.use_batch_norm,
            "max_num_layers": self.max_num_layers,
            "max_num_neurons": self.max_num_neurons,
        }
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        """Reconstruct descriptor from flattened form.

        This method is required for PyTree registration. It reconstructs
        the descriptor from leaves and auxiliary data.

        Args:
            aux_data: Dictionary of auxiliary data
            leaves: Tuple of (dims, dropout_probs)

        Returns:
            Reconstructed MLPDescriptor instance
        """
        dims, dropout_probs = leaves
        return cls(dims=dims, dropout_probs=dropout_probs, **aux_data)

    @staticmethod
    def random_init(
        input_dim: int,
        output_dim: int,
        max_num_layers: int,
        max_num_neurons: int,
        key: jax.random.PRNGKey,
        dropout: bool = False,
        batch_norm: bool = False,
    ) -> "MLPDescriptor":
        """Create a random MLP descriptor.

        This factory method generates a random MLP architecture with:
        - Random number of layers (1 to max_num_layers)
        - Random neuron counts per layer (2 to max_num_neurons)
        - Random activation functions
        - Random weight initializers
        - Random dropout probabilities (if dropout enabled)

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            max_num_layers: Maximum number of hidden layers
            max_num_neurons: Maximum neurons per layer
            key: JAX PRNG key for randomness
            dropout: Whether to enable dropout
            batch_norm: Whether to enable batch normalization

        Returns:
            Randomly initialized MLPDescriptor

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> desc = MLPDescriptor.random_init(784, 10, 5, 128, key)
            >>> assert desc.validate()
        """
        # Split key for different random operations
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # Random number of layers (at least 1)
        num_layers = int(jax.random.randint(k1, (), 1, max_num_layers + 1))

        # Random layer dimensions (at least 2 neurons per layer)
        dims = tuple(
            int(jax.random.randint(k2, (), 2, max_num_neurons + 1))
            for _ in range(num_layers)
        )

        # Random activation functions
        act_indices = jax.random.randint(k3, (num_layers,), 0, len(ACTIVATIONS))
        act_functions = tuple(ACTIVATIONS[int(i)] for i in act_indices)

        # Random initialization functions
        init_indices = jax.random.randint(k4, (num_layers,), 0, len(INITIALIZATIONS))
        init_functions = tuple(INITIALIZATIONS[int(i)] for i in init_indices)

        # Random dropout probabilities (between 0 and 0.5)
        dropout_probs = tuple(float(p) for p in jax.random.uniform(k5, (num_layers,)) * 0.5)

        return MLPDescriptor(
            input_dim=input_dim,
            output_dim=output_dim,
            dims=dims,
            act_functions=act_functions,
            init_functions=init_functions,
            dropout_probs=dropout_probs,
            use_dropout=dropout,
            use_batch_norm=batch_norm,
            max_num_layers=max_num_layers,
            max_num_neurons=max_num_neurons,
        )

    def validate(self) -> bool:
        """Validate that the descriptor satisfies all constraints.

        Checks:
        - Number of layers is within [1, max_num_layers]
        - All neuron counts are within [2, max_num_neurons]
        - All lists have consistent lengths
        - All activation functions are valid
        - All initializers are valid
        - All dropout probabilities are in [0, 1]

        Returns:
            True if descriptor is valid, False otherwise
        """
        # Check layer count
        num_layers = len(self.dims)
        if num_layers > self.max_num_layers or num_layers < 1:
            return False

        # Check neuron counts
        if any(d < 2 or d > self.max_num_neurons for d in self.dims):
            return False

        # Check list lengths are consistent
        if len(self.act_functions) != num_layers:
            return False
        if len(self.init_functions) != num_layers:
            return False
        if len(self.dropout_probs) != num_layers:
            return False

        # Check activation functions are valid
        if any(act not in ACTIVATIONS for act in self.act_functions):
            return False

        # Check initializers are valid
        if any(init not in INITIALIZATIONS for init in self.init_functions):
            return False

        # Check dropout probabilities are in valid range
        if any(p < 0.0 or p > 1.0 for p in self.dropout_probs):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary for serialization.

        Returns:
            Dictionary representation of the descriptor
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dims": list(self.dims),
            "act_functions": list(self.act_functions),
            "init_functions": list(self.init_functions),
            "dropout_probs": list(self.dropout_probs),
            "use_dropout": self.use_dropout,
            "use_batch_norm": self.use_batch_norm,
            "max_num_layers": self.max_num_layers,
            "max_num_neurons": self.max_num_neurons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLPDescriptor":
        """Reconstruct descriptor from dictionary.

        Args:
            data: Dictionary containing descriptor data

        Returns:
            Reconstructed MLPDescriptor instance
        """
        return cls(
            input_dim=data["input_dim"],
            output_dim=data["output_dim"],
            dims=tuple(data["dims"]),
            act_functions=tuple(data["act_functions"]),
            init_functions=tuple(data["init_functions"]),
            dropout_probs=tuple(data["dropout_probs"]),
            use_dropout=data["use_dropout"],
            use_batch_norm=data["use_batch_norm"],
            max_num_layers=data["max_num_layers"],
            max_num_neurons=data["max_num_neurons"],
        )

    def __repr__(self) -> str:
        """String representation of the descriptor."""
        return (
            f"MLPDescriptor(layers={len(self.dims)}, "
            f"dims={self.dims}, "
            f"dropout={self.use_dropout}, "
            f"batch_norm={self.use_batch_norm})"
        )
