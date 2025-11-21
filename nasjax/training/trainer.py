"""Training loop implementation using Optax.

This module provides a Trainer class and utilities for training Equinox networks.
"""

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from .losses import get_loss_function


class Trainer:
    """Trainer for Equinox networks using Optax optimizers.

    This class handles training of networks with various optimizers and
    loss functions.

    Attributes:
        optimizer_name: Name of the optimizer ('adam', 'sgd', 'rmsprop', 'adamw')
        learning_rate: Learning rate for optimization
        optimizer: Optax optimizer instance
    """

    def __init__(
        self, optimizer: str = "adam", learning_rate: float = 0.001, **optimizer_kwargs
    ):
        """Initialize trainer.

        Args:
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop', 'adamw')
            learning_rate: Learning rate
            **optimizer_kwargs: Additional optimizer-specific arguments
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.optimizer = self._get_optimizer(optimizer, learning_rate, **optimizer_kwargs)

    def _get_optimizer(
        self, name: str, lr: float, **kwargs
    ) -> optax.GradientTransformation:
        """Get Optax optimizer by name.

        Args:
            name: Optimizer name
            lr: Learning rate
            **kwargs: Optimizer-specific arguments

        Returns:
            Optax optimizer

        Raises:
            ValueError: If optimizer name not recognized
        """
        optimizers = {
            "adam": lambda: optax.adam(lr, **kwargs),
            "sgd": lambda: optax.sgd(lr, **kwargs),
            "rmsprop": lambda: optax.rmsprop(lr, **kwargs),
            "adamw": lambda: optax.adamw(lr, **kwargs),
        }

        if name not in optimizers:
            raise ValueError(
                f"Unknown optimizer: {name}. " f"Valid options: {list(optimizers.keys())}"
            )

        return optimizers[name]()

    def train(
        self,
        model: eqx.Module,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        n_epochs: int,
        batch_size: int,
        loss_fn: Callable,
        key: jax.random.PRNGKey,
        x_val: Optional[jnp.ndarray] = None,
        y_val: Optional[jnp.ndarray] = None,
    ) -> Tuple[eqx.Module, dict]:
        """Train the model.

        Args:
            model: Equinox network to train
            x_train: Training inputs
            y_train: Training targets
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            loss_fn: Loss function to minimize
            key: JAX PRNG key
            x_val: Optional validation inputs
            y_val: Optional validation targets

        Returns:
            Tuple of (trained_model, training_history)
            training_history contains 'train_loss' and optionally 'val_loss'
        """
        # Initialize optimizer state
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        # Training history
        history = {"train_loss": [], "val_loss": []}

        # Create training step function
        @eqx.filter_jit
        def train_step(model, opt_state, x_batch, y_batch, key):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(
                model, x_batch, y_batch, key, inference=False
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        # Training loop
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(n_epochs):
            # Shuffle data
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n_samples)
            x_train_shuffled = x_train[perm]
            y_train_shuffled = y_train[perm]

            # Batch training
            epoch_losses = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                key, subkey = jax.random.split(key)
                model, opt_state, batch_loss = train_step(
                    model, opt_state, x_batch, y_batch, subkey
                )
                epoch_losses.append(float(batch_loss))

            # Record training loss
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation loss if provided
            if x_val is not None and y_val is not None:
                key, subkey = jax.random.split(key)
                val_loss = loss_fn(model, x_val, y_val, subkey, inference=True)
                history["val_loss"].append(float(val_loss))

        return model, history


def train_network(
    model: eqx.Module,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    n_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    loss: str = "mse",
    key: Optional[jax.random.PRNGKey] = None,
    x_val: Optional[jnp.ndarray] = None,
    y_val: Optional[jnp.ndarray] = None,
) -> Tuple[eqx.Module, dict]:
    """Convenience function to train a network.

    Args:
        model: Equinox network to train
        x_train: Training inputs
        y_train: Training targets
        n_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        optimizer: Optimizer name
        loss: Loss function name
        key: JAX PRNG key (created if None)
        x_val: Optional validation inputs
        y_val: Optional validation targets

    Returns:
        Tuple of (trained_model, training_history)

    Example:
        >>> model = MLP(descriptor, jax.random.PRNGKey(0))
        >>> trained_model, history = train_network(
        ...     model, x_train, y_train,
        ...     n_epochs=50, batch_size=64,
        ...     loss='cross_entropy'
        ... )
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Get loss function
    loss_fn = get_loss_function(loss)

    # Create trainer
    trainer = Trainer(optimizer=optimizer, learning_rate=learning_rate)

    # Train
    return trainer.train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        n_epochs=n_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        key=key,
        x_val=x_val,
        y_val=y_val,
    )


def create_batches(
    x: jnp.ndarray, y: jnp.ndarray, batch_size: int, key: jax.random.PRNGKey
) -> list:
    """Create randomized batches from data.

    Args:
        x: Input data
        y: Target data
        batch_size: Batch size
        key: JAX PRNG key for shuffling

    Returns:
        List of (x_batch, y_batch) tuples

    Example:
        >>> batches = create_batches(x_train, y_train, 32, jax.random.PRNGKey(0))
        >>> for x_batch, y_batch in batches:
        ...     # Process batch
        ...     pass
    """
    n_samples = len(x)

    # Shuffle data
    perm = jax.random.permutation(key, n_samples)
    x_shuffled = x[perm]
    y_shuffled = y[perm]

    # Create batches
    batches = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batches.append((x_shuffled[i:end_idx], y_shuffled[i:end_idx]))

    return batches
