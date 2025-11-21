"""Base network class with common utilities for Equinox networks.

This module provides the base class and utility functions for all network
implementations in NASJAX.
"""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


class BaseNetwork(eqx.Module):
    """Base class for all Equinox network implementations.

    This class provides common utilities for:
    - Mapping activation function names to JAX functions
    - Mapping initializer names to initialization functions
    - Common network operations

    All concrete network classes (MLP, CNN, RNN, etc.) should inherit from this.
    """

    @staticmethod
    def get_activation(name: Optional[str]) -> Callable:
        """Map activation function name to JAX function.

        Args:
            name: Name of activation function. Valid options:
                - 'relu': Rectified Linear Unit
                - 'elu': Exponential Linear Unit
                - 'sigmoid': Sigmoid activation
                - 'tanh': Hyperbolic tangent
                - 'softplus': Softplus activation
                - 'softsign': Softsign activation (x / (1 + |x|))
                - None: Identity (no activation)

        Returns:
            JAX activation function

        Raises:
            ValueError: If activation name is not recognized

        Example:
            >>> relu = BaseNetwork.get_activation('relu')
            >>> x = jnp.array([-1, 0, 1])
            >>> print(relu(x))  # [0, 0, 1]
        """
        # Custom softsign implementation since JAX doesn't provide it
        def softsign(x):
            """Softsign activation: x / (1 + |x|)"""
            return x / (1.0 + jnp.abs(x))

        activations = {
            "relu": jnn.relu,
            "elu": jnn.elu,
            "sigmoid": jnn.sigmoid,
            "tanh": jnn.tanh,
            "softplus": jnn.softplus,
            "softsign": softsign,
            None: lambda x: x,  # Identity
        }

        if name not in activations:
            raise ValueError(
                f"Unknown activation: {name}. "
                f"Valid options: {list(activations.keys())}"
            )

        return activations[name]

    @staticmethod
    def get_initializer(name: str) -> Callable[[jax.random.PRNGKey, tuple], jnp.ndarray]:
        """Map initializer name to initialization function.

        Args:
            name: Name of initializer. Valid options:
                - 'glorot_normal': Glorot/Xavier normal initialization
                - 'glorot_uniform': Glorot/Xavier uniform initialization
                - 'normal': Normal distribution with std=0.05
                - 'uniform': Uniform distribution in [-0.05, 0.05]

        Returns:
            Function that takes (key, shape) and returns initialized array

        Raises:
            ValueError: If initializer name is not recognized

        Example:
            >>> init_fn = BaseNetwork.get_initializer('glorot_normal')
            >>> key = jax.random.PRNGKey(0)
            >>> weights = init_fn(key, (10, 5))
            >>> print(weights.shape)  # (10, 5)
        """

        def glorot_normal(key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
            """Glorot/Xavier normal initialization.

            Uses std = sqrt(2 / (fan_in + fan_out))
            """
            std = jnp.sqrt(2.0 / sum(shape))
            return jax.random.normal(key, shape) * std

        def glorot_uniform(key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
            """Glorot/Xavier uniform initialization.

            Uses limit = sqrt(6 / (fan_in + fan_out))
            """
            limit = jnp.sqrt(6.0 / sum(shape))
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

        def normal(key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
            """Normal initialization with small std."""
            return jax.random.normal(key, shape) * 0.05

        def uniform(key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
            """Uniform initialization in small range."""
            return jax.random.uniform(key, shape, minval=-0.05, maxval=0.05)

        initializers = {
            "glorot_normal": glorot_normal,
            "glorot_uniform": glorot_uniform,
            "normal": normal,
            "uniform": uniform,
        }

        if name not in initializers:
            raise ValueError(
                f"Unknown initializer: {name}. "
                f"Valid options: {list(initializers.keys())}"
            )

        return initializers[name]

    @staticmethod
    def apply_dropout(
        x: jnp.ndarray,
        prob: float,
        key: jax.random.PRNGKey,
        inference: bool = False,
    ) -> jnp.ndarray:
        """Apply dropout to input array.

        During training, randomly sets elements to 0 with probability `prob`
        and scales remaining elements by 1/(1-prob).

        During inference, returns input unchanged.

        Args:
            x: Input array
            prob: Dropout probability (0.0 to 1.0)
            key: JAX PRNG key for randomness
            inference: If True, no dropout is applied

        Returns:
            Array with dropout applied (if not inference mode)

        Example:
            >>> x = jnp.ones((10,))
            >>> key = jax.random.PRNGKey(0)
            >>> x_drop = BaseNetwork.apply_dropout(x, 0.5, key)
            >>> # Roughly half the elements will be 0, others will be 2
        """
        if inference or prob == 0.0:
            return x

        # Generate dropout mask
        keep_prob = 1.0 - prob
        mask = jax.random.bernoulli(key, keep_prob, x.shape)

        # Apply mask and scale
        return jnp.where(mask, x / keep_prob, 0.0)
