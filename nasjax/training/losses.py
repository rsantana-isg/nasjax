"""Loss functions for training networks.

This module provides common loss functions compatible with JAX and Equinox networks.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable


def mse_loss(
    model,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.random.PRNGKey,
    inference: bool = False,
) -> jnp.ndarray:
    """Mean Squared Error loss.

    Args:
        model: Equinox network model
        x: Input data of shape (batch_size, ...)
        y: Target data of shape (batch_size, ...)
        key: JAX PRNG key for dropout (if needed)
        inference: Whether to run in inference mode

    Returns:
        Scalar MSE loss value
    """
    # Vectorize over batch if needed
    if x.ndim > 1:
        predictions = jax.vmap(lambda xi: model(xi, key, inference=inference))(x)
    else:
        predictions = model(x, key, inference=inference)

    return jnp.mean((predictions - y) ** 2)


def cross_entropy_loss(
    model,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.random.PRNGKey,
    inference: bool = False,
) -> jnp.ndarray:
    """Cross-entropy loss for classification.

    Args:
        model: Equinox network model
        x: Input data of shape (batch_size, ...)
        y: Target labels as integers (batch_size,) or one-hot (batch_size, num_classes)
        key: JAX PRNG key for dropout (if needed)
        inference: Whether to run in inference mode

    Returns:
        Scalar cross-entropy loss value
    """
    # Vectorize over batch if needed
    if x.ndim > 1:
        logits = jax.vmap(lambda xi: model(xi, key, inference=inference))(x)
    else:
        logits = model(x, key, inference=inference)

    # Check if y is one-hot or integer labels
    if y.ndim == 1:
        # Integer labels
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    else:
        # One-hot labels
        loss = optax.softmax_cross_entropy(logits, y)

    return jnp.mean(loss)


def accuracy(
    model,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jax.random.PRNGKey,
    inference: bool = True,
) -> jnp.ndarray:
    """Classification accuracy.

    Args:
        model: Equinox network model
        x: Input data of shape (batch_size, ...)
        y: Target labels as integers (batch_size,) or one-hot (batch_size, num_classes)
        key: JAX PRNG key for dropout (if needed)
        inference: Whether to run in inference mode

    Returns:
        Scalar accuracy value (0 to 1)
    """
    # Vectorize over batch if needed
    if x.ndim > 1:
        logits = jax.vmap(lambda xi: model(xi, key, inference=inference))(x)
    else:
        logits = model(x, key, inference=inference)

    predictions = jnp.argmax(logits, axis=-1)

    # Handle one-hot labels
    if y.ndim > 1:
        y = jnp.argmax(y, axis=-1)

    return jnp.mean(predictions == y)


def get_loss_function(name: str) -> Callable:
    """Get loss function by name.

    Args:
        name: Loss function name ('mse', 'cross_entropy', or 'xentropy')

    Returns:
        Loss function

    Raises:
        ValueError: If loss function name is not recognized
    """
    losses = {
        "mse": mse_loss,
        "cross_entropy": cross_entropy_loss,
        "xentropy": cross_entropy_loss,
    }

    if name not in losses:
        raise ValueError(
            f"Unknown loss function: {name}. " f"Valid options: {list(losses.keys())}"
        )

    return losses[name]
