"""RNN (Recurrent Neural Network) descriptor implementation.

This module implements the RNNDescriptor class, which is an immutable PyTree
representing the architecture specification for a recurrent neural network.
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

# RNN layer type constants
SIMPLE_RNN = 0
LSTM = 1
GRU = 2

RNN_LAYER_NAMES = ["SimpleRNN", "LSTM", "GRU"]

# Constants for valid ranges
MIN_NUM_UNITS = 2


@jtu.register_pytree_node_class
class RNNDescriptor(NamedTuple):
    """Descriptor for Recurrent Neural Network architectures.

    This is an immutable PyTree that represents the genotype of an RNN.
    It can be transformed by JAX operations (jit, vmap, pmap) and used
    with evolutionary operators.

    Attributes:
        input_dim: Input dimension (sequence feature dimension)
        output_dim: Output dimension
        rnn_types: Tuple of RNN layer type indices (0=SimpleRNN, 1=LSTM, 2=GRU)
        units: Tuple of number of units per RNN layer
        bidirectional: Tuple of booleans indicating if each layer is bidirectional
        act_functions: Tuple of activation function names for each layer
        init_functions: Tuple of weight initializer names for each layer
        dropout_probs: Tuple of dropout probabilities per layer (0.0 to 1.0)
        use_dropout: Whether dropout is enabled globally
        use_batch_norm: Whether batch normalization is enabled
        max_num_layers: Maximum allowed layers (for mutation constraints)
        max_num_units: Maximum units per layer (for mutation constraints)

    Example:
        >>> import jax
        >>> key = jax.random.PRNGKey(42)
        >>> desc = RNNDescriptor.random_init(
        ...     input_dim=10,
        ...     output_dim=5,
        ...     max_num_layers=3,
        ...     max_num_units=128,
        ...     key=key
        ... )
        >>> print(desc.rnn_types)  # e.g., (1, 1, 2)  # LSTM, LSTM, GRU
        >>> print(desc.validate())  # True
    """

    input_dim: int
    output_dim: int
    rnn_types: Tuple[int, ...]
    units: Tuple[int, ...]
    bidirectional: Tuple[bool, ...]
    act_functions: Tuple[Optional[str], ...]
    init_functions: Tuple[str, ...]
    dropout_probs: Tuple[float, ...]
    use_dropout: bool
    use_batch_norm: bool
    max_num_layers: int
    max_num_units: int

    def tree_flatten(self):
        """Flatten descriptor into leaves and auxiliary data for JAX.

        Returns:
            Tuple of (leaves, aux_data)
        """
        # Leaves are the data that can be transformed by JAX
        leaves = (self.rnn_types, self.units, self.dropout_probs)

        # Auxiliary data is metadata that doesn't change
        aux_data = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "bidirectional": self.bidirectional,
            "act_functions": self.act_functions,
            "init_functions": self.init_functions,
            "use_dropout": self.use_dropout,
            "use_batch_norm": self.use_batch_norm,
            "max_num_layers": self.max_num_layers,
            "max_num_units": self.max_num_units,
        }
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        """Reconstruct descriptor from flattened form.

        Args:
            aux_data: Dictionary of auxiliary data
            leaves: Tuple of (rnn_types, units, dropout_probs)

        Returns:
            Reconstructed RNNDescriptor instance
        """
        rnn_types, units, dropout_probs = leaves
        return cls(rnn_types=rnn_types, units=units, dropout_probs=dropout_probs, **aux_data)

    @staticmethod
    def random_init(
        input_dim: int,
        output_dim: int,
        max_num_layers: int,
        max_num_units: int,
        key: jax.random.PRNGKey,
        dropout: bool = False,
        batch_norm: bool = False,
    ) -> "RNNDescriptor":
        """Create a random RNN descriptor.

        This factory method generates a random RNN architecture with:
        - Random number of layers (1 to max_num_layers)
        - Random RNN types (SimpleRNN, LSTM, GRU)
        - Random unit counts per layer (MIN_NUM_UNITS to max_num_units)
        - Random bidirectional flags
        - Random activation functions
        - Random weight initializers
        - Random dropout probabilities (if dropout enabled)

        Args:
            input_dim: Input dimension (sequence feature dimension)
            output_dim: Output dimension
            max_num_layers: Maximum number of RNN layers
            max_num_units: Maximum units per layer
            key: JAX PRNG key for randomness
            dropout: Whether to enable dropout
            batch_norm: Whether to enable batch normalization

        Returns:
            Randomly initialized RNNDescriptor

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> desc = RNNDescriptor.random_init(10, 5, 3, 128, key)
            >>> assert desc.validate()
        """
        # Split key for different random operations
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        # Random number of layers (at least 1)
        num_layers = int(jax.random.randint(k1, (), 1, max_num_layers + 1))

        # Random RNN types
        rnn_type_indices = jax.random.randint(k2, (num_layers,), 0, 3)  # 0, 1, or 2
        rnn_types = tuple(int(i) for i in rnn_type_indices)

        # Random units per layer (at least MIN_NUM_UNITS)
        units = tuple(
            int(jax.random.randint(k3, (), MIN_NUM_UNITS, max_num_units + 1))
            for _ in range(num_layers)
        )

        # Random bidirectional flags
        bidirectional = tuple(
            bool(jax.random.bernoulli(k4, 0.5, (num_layers,))[i]) for i in range(num_layers)
        )

        # Random activation functions
        act_indices = jax.random.randint(k5, (num_layers,), 0, len(ACTIVATIONS))
        act_functions = tuple(ACTIVATIONS[int(i)] for i in act_indices)

        # Random initialization functions
        init_indices = jax.random.randint(k6, (num_layers,), 0, len(INITIALIZATIONS))
        init_functions = tuple(INITIALIZATIONS[int(i)] for i in init_indices)

        # Random dropout probabilities (between 0 and 0.5)
        dropout_probs = tuple(float(p) for p in jax.random.uniform(k7, (num_layers,)) * 0.5)

        return RNNDescriptor(
            input_dim=input_dim,
            output_dim=output_dim,
            rnn_types=rnn_types,
            units=units,
            bidirectional=bidirectional,
            act_functions=act_functions,
            init_functions=init_functions,
            dropout_probs=dropout_probs,
            use_dropout=dropout,
            use_batch_norm=batch_norm,
            max_num_layers=max_num_layers,
            max_num_units=max_num_units,
        )

    def validate(self) -> bool:
        """Validate that the descriptor satisfies all constraints.

        Checks:
        - Number of layers is within [1, max_num_layers]
        - All RNN types are valid (0, 1, or 2)
        - All unit counts are within [MIN_NUM_UNITS, max_num_units]
        - All lists have consistent lengths
        - All activation functions are valid
        - All initializers are valid
        - All dropout probabilities are in [0, 1]

        Returns:
            True if descriptor is valid, False otherwise
        """
        # Check layer count
        num_layers = len(self.rnn_types)
        if num_layers > self.max_num_layers or num_layers < 1:
            return False

        # Check RNN types are valid
        if any(rt not in [SIMPLE_RNN, LSTM, GRU] for rt in self.rnn_types):
            return False

        # Check unit counts
        if any(u < MIN_NUM_UNITS or u > self.max_num_units for u in self.units):
            return False

        # Check list lengths are consistent
        if len(self.units) != num_layers:
            return False
        if len(self.bidirectional) != num_layers:
            return False
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
            "rnn_types": list(self.rnn_types),
            "units": list(self.units),
            "bidirectional": list(self.bidirectional),
            "act_functions": list(self.act_functions),
            "init_functions": list(self.init_functions),
            "dropout_probs": list(self.dropout_probs),
            "use_dropout": self.use_dropout,
            "use_batch_norm": self.use_batch_norm,
            "max_num_layers": self.max_num_layers,
            "max_num_units": self.max_num_units,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RNNDescriptor":
        """Reconstruct descriptor from dictionary.

        Args:
            data: Dictionary containing descriptor data

        Returns:
            Reconstructed RNNDescriptor instance
        """
        return cls(
            input_dim=data["input_dim"],
            output_dim=data["output_dim"],
            rnn_types=tuple(data["rnn_types"]),
            units=tuple(data["units"]),
            bidirectional=tuple(data["bidirectional"]),
            act_functions=tuple(data["act_functions"]),
            init_functions=tuple(data["init_functions"]),
            dropout_probs=tuple(data["dropout_probs"]),
            use_dropout=data["use_dropout"],
            use_batch_norm=data["use_batch_norm"],
            max_num_layers=data["max_num_layers"],
            max_num_units=data["max_num_units"],
        )

    def __repr__(self) -> str:
        """String representation of the descriptor."""
        rnn_type_names = [RNN_LAYER_NAMES[rt] for rt in self.rnn_types]
        return (
            f"RNNDescriptor(layers={len(self.rnn_types)}, "
            f"types={rnn_type_names}, "
            f"units={self.units}, "
            f"dropout={self.use_dropout}, "
            f"batch_norm={self.use_batch_norm})"
        )
