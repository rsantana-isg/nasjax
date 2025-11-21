"""Fitness evaluator for network descriptors.

This module evaluates the fitness of network descriptors by:
1. Building a network from the descriptor
2. Training the network
3. Evaluating performance on test set
"""

from typing import Optional, Callable
import jax
import jax.numpy as jnp

from ..descriptors.mlp import MLPDescriptor
from ..networks.mlp import MLP
from ..training import train_network
from ..training.losses import get_loss_function, accuracy


class Evaluator:
    """Evaluates fitness of network descriptors.

    The evaluator builds networks from descriptors, trains them, and
    returns a fitness value (lower is better for minimization).

    Attributes:
        loss_fn_name: Name of loss function ('mse', 'cross_entropy')
        optimizer: Name of optimizer ('adam', 'sgd', etc.)
        learning_rate: Learning rate for training
        n_epochs: Number of training epochs
        batch_size: Training batch size
        metric: Evaluation metric ('loss' or 'accuracy')
    """

    def __init__(
        self,
        loss_fn: str = "mse",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        batch_size: int = 32,
        metric: str = "loss",
    ):
        """Initialize evaluator.

        Args:
            loss_fn: Loss function name
            optimizer: Optimizer name
            learning_rate: Learning rate
            n_epochs: Number of epochs to train
            batch_size: Batch size for training
            metric: Metric to return ('loss' or 'accuracy')
                   - 'loss': Return test loss (lower is better)
                   - 'accuracy': Return 1 - accuracy for minimization
        """
        self.loss_fn_name = loss_fn
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.metric = metric

    def evaluate(
        self,
        descriptor: MLPDescriptor,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_test: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> float:
        """Evaluate a descriptor's fitness.

        Args:
            descriptor: MLPDescriptor to evaluate
            x_train: Training inputs
            y_train: Training targets
            x_test: Test inputs
            y_test: Test targets
            key: JAX PRNG key

        Returns:
            Fitness value (lower is better)
            - If metric='loss': returns test loss
            - If metric='accuracy': returns 1 - test_accuracy

        Raises:
            ValueError: If descriptor is invalid or if errors occur during training
        """
        # Validate descriptor
        if not descriptor.validate():
            raise ValueError("Invalid descriptor provided to evaluator")

        try:
            # Split key for network creation and training
            k1, k2 = jax.random.split(key)

            # Build network from descriptor
            network = MLP(descriptor, k1)

            # Train network
            trained_network, history = train_network(
                model=network,
                x_train=x_train,
                y_train=y_train,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                loss=self.loss_fn_name,
                key=k2,
                x_val=x_test,
                y_val=y_test,
            )

            # Evaluate on test set
            k2, k3 = jax.random.split(k2)

            if self.metric == "loss":
                # Return test loss
                loss_fn = get_loss_function(self.loss_fn_name)
                test_loss = loss_fn(trained_network, x_test, y_test, k3, inference=True)
                return float(test_loss)

            elif self.metric == "accuracy":
                # Return 1 - accuracy for minimization
                test_acc = accuracy(trained_network, x_test, y_test, k3, inference=True)
                return float(1.0 - test_acc)

            else:
                raise ValueError(f"Unknown metric: {self.metric}")

        except Exception as e:
            # If evaluation fails, return a very high fitness (bad)
            print(f"Evaluation failed for descriptor: {e}")
            return float('inf')


def evaluate_descriptor(
    descriptor: MLPDescriptor,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_test: jnp.ndarray,
    y_test: jnp.ndarray,
    key: jax.random.PRNGKey,
    loss_fn: str = "mse",
    n_epochs: int = 10,
    batch_size: int = 32,
    metric: str = "loss",
) -> float:
    """Convenience function to evaluate a descriptor.

    Args:
        descriptor: MLPDescriptor to evaluate
        x_train: Training inputs
        y_train: Training targets
        x_test: Test inputs
        y_test: Test targets
        key: JAX PRNG key
        loss_fn: Loss function name
        n_epochs: Number of epochs
        batch_size: Batch size
        metric: Metric to return ('loss' or 'accuracy')

    Returns:
        Fitness value (lower is better)

    Example:
        >>> desc = MLPDescriptor.random_init(784, 10, 5, 128, key)
        >>> fitness = evaluate_descriptor(
        ...     desc, x_train, y_train, x_test, y_test,
        ...     jax.random.PRNGKey(0),
        ...     loss_fn='cross_entropy',
        ...     n_epochs=20
        ... )
    """
    evaluator = Evaluator(
        loss_fn=loss_fn, n_epochs=n_epochs, batch_size=batch_size, metric=metric
    )

    return evaluator.evaluate(descriptor, x_train, y_train, x_test, y_test, key)


def evaluate_population(
    descriptors: list[MLPDescriptor],
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_test: jnp.ndarray,
    y_test: jnp.ndarray,
    key: jax.random.PRNGKey,
    evaluator: Optional[Evaluator] = None,
) -> list[float]:
    """Evaluate fitness of multiple descriptors.

    Args:
        descriptors: List of MLPDescriptors to evaluate
        x_train: Training inputs
        y_train: Training targets
        x_test: Test inputs
        y_test: Test targets
        key: JAX PRNG key
        evaluator: Optional Evaluator instance (created if None)

    Returns:
        List of fitness values

    Example:
        >>> population = [MLPDescriptor.random_init(...) for _ in range(10)]
        >>> fitnesses = evaluate_population(
        ...     population, x_train, y_train, x_test, y_test,
        ...     jax.random.PRNGKey(0)
        ... )
    """
    if evaluator is None:
        evaluator = Evaluator()

    # Split keys for each descriptor
    keys = jax.random.split(key, len(descriptors))

    fitnesses = []
    for desc, k in zip(descriptors, keys):
        fitness = evaluator.evaluate(desc, x_train, y_train, x_test, y_test, k)
        fitnesses.append(fitness)

    return fitnesses
