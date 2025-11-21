"""MLP (Multi-Layer Perceptron) network implementation in Equinox.

This module implements the MLP network class that builds from an MLPDescriptor
and provides the actual neural network computation.
"""

from typing import List, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .base import BaseNetwork
from ..descriptors.mlp import MLPDescriptor


class MLP(BaseNetwork):
    """Multi-Layer Perceptron implemented in Equinox.

    This is the phenotype - the actual executable network built from a
    descriptor (genotype). It uses Equinox modules for efficient computation
    and JIT compilation.

    Attributes:
        layers: List of Equinox Linear layers
        activations: List of activation functions
        dropout_probs: List of dropout probabilities per layer
        use_dropout: Whether dropout is enabled
        use_batch_norm: Whether batch normalization is enabled
        input_dim: Input dimension (for shape validation)
        output_dim: Output dimension (for shape validation)

    Example:
        >>> import jax
        >>> from nasjax.descriptors import MLPDescriptor
        >>> from nasjax.networks import MLP
        >>>
        >>> # Create descriptor
        >>> key = jax.random.PRNGKey(42)
        >>> k1, k2, k3 = jax.random.split(key, 3)
        >>> desc = MLPDescriptor.random_init(784, 10, 5, 128, k1)
        >>>
        >>> # Build network
        >>> network = MLP(desc, k2)
        >>>
        >>> # Forward pass
        >>> x = jax.random.normal(k3, (784,))
        >>> output = network(x, inference=True)
        >>> print(output.shape)  # (10,)
    """

    layers: List[eqx.nn.Linear]
    activations: List[Callable]
    dropout_probs: List[float]
    use_dropout: bool
    use_batch_norm: bool
    input_dim: int
    output_dim: int

    def __init__(self, descriptor: MLPDescriptor, key: jax.random.PRNGKey):
        """Build MLP network from descriptor.

        Args:
            descriptor: MLPDescriptor specifying the architecture
            key: JAX PRNG key for weight initialization

        Raises:
            ValueError: If descriptor is invalid

        Example:
            >>> desc = MLPDescriptor.random_init(784, 10, 5, 128, key)
            >>> network = MLP(desc, jax.random.PRNGKey(0))
        """
        # Validate descriptor
        if not descriptor.validate():
            raise ValueError("Invalid descriptor provided to MLP constructor")

        # Store dimensions for validation
        self.input_dim = descriptor.input_dim
        self.output_dim = descriptor.output_dim

        # Calculate number of layers (hidden + output)
        n_layers = len(descriptor.dims) + 1
        keys = jax.random.split(key, n_layers)

        # Build layers
        self.layers = []
        in_dim = descriptor.input_dim

        # Hidden layers
        for i, (out_dim, init_name) in enumerate(
            zip(descriptor.dims, descriptor.init_functions)
        ):
            # Create layer with Equinox (uses Glorot uniform by default)
            layer = eqx.nn.Linear(in_dim, out_dim, key=keys[i])

            # Optionally re-initialize with custom initializer
            if init_name != "glorot_uniform":
                initializer = self.get_initializer(init_name)
                # Re-initialize weights
                weight_shape = (out_dim, in_dim)
                new_weight = initializer(keys[i], weight_shape)
                # Replace the layer's weight (Equinox uses tree_at for this)
                layer = eqx.tree_at(lambda l: l.weight, layer, new_weight)

            self.layers.append(layer)
            in_dim = out_dim

        # Output layer (uses last init function or default)
        output_layer = eqx.nn.Linear(in_dim, descriptor.output_dim, key=keys[-1])
        self.layers.append(output_layer)

        # Store activation functions
        # Hidden layers use specified activations, output has no activation
        self.activations = [self.get_activation(name) for name in descriptor.act_functions]
        self.activations.append(lambda x: x)  # Identity for output

        # Store dropout settings
        self.dropout_probs = list(descriptor.dropout_probs) + [0.0]  # No dropout on output
        self.use_dropout = descriptor.use_dropout

        # Batch normalization (placeholder for now)
        self.use_batch_norm = descriptor.use_batch_norm
        # Note: Batch norm implementation would go here if needed

    def __call__(
        self,
        x: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None,
        *,
        inference: bool = False,
    ) -> jnp.ndarray:
        """Forward pass through the network.

        Args:
            x: Input array of shape (input_dim,) or (..., input_dim)
            key: PRNG key for dropout (required if use_dropout=True and not inference)
            inference: If True, disable dropout and batch norm stats updates

        Returns:
            Network output of shape (output_dim,) or (..., output_dim)

        Raises:
            ValueError: If key is required but not provided

        Example:
            >>> x = jax.random.normal(jax.random.PRNGKey(0), (784,))
            >>> # Inference mode (no dropout)
            >>> output = network(x, inference=True)
            >>> # Training mode (with dropout)
            >>> output = network(x, key=jax.random.PRNGKey(1), inference=False)
        """
        # Flatten input if multidimensional (for images, etc.)
        original_shape = x.shape
        if x.ndim > 1:
            # Flatten all dimensions except batch
            x = x.reshape(-1)

        # Validate input dimension
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, "
                f"got {x.shape[-1]}"
            )

        # Check if key is needed for dropout
        if self.use_dropout and not inference and key is None:
            raise ValueError("PRNG key required for dropout during training")

        # Forward through layers
        for i, (layer, activation, drop_prob) in enumerate(
            zip(self.layers, self.activations, self.dropout_probs)
        ):
            # Linear transformation
            x = layer(x)

            # Activation function
            x = activation(x)

            # Dropout (only during training and if enabled)
            if self.use_dropout and not inference and drop_prob > 0.0:
                key, subkey = jax.random.split(key)
                x = self.apply_dropout(x, drop_prob, subkey, inference=False)

            # Batch normalization would go here if implemented
            # if self.use_batch_norm and i < len(self.layers) - 1:
            #     x = self.batch_norm_layers[i](x, inference=inference)

        return x

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Total number of parameters in the network

        Example:
            >>> n_params = network.count_parameters()
            >>> print(f"Network has {n_params:,} parameters")
        """
        total = 0
        for layer in self.layers:
            # Each Linear layer has weight (out, in) and bias (out,)
            weight_params = layer.weight.size
            bias_params = layer.bias.size if layer.bias is not None else 0
            total += weight_params + bias_params
        return total

    def __repr__(self) -> str:
        """String representation of the network."""
        layer_sizes = [layer.out_features for layer in self.layers]
        n_params = self.count_parameters()
        return (
            f"MLP(layers={len(self.layers)}, "
            f"architecture={[self.input_dim] + layer_sizes}, "
            f"params={n_params:,}, "
            f"dropout={self.use_dropout})"
        )
