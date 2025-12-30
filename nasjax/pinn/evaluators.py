"""PINN-specific evaluator for architecture evolution.

This module provides evaluators that assess PINN architectures based on
their ability to solve PDEs.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Optional, Dict

from ..descriptors.mlp import MLPDescriptor
from ..training.trainer import Trainer
from .networks import PINNNetwork
from .benchmarks.base import BasePDEProblem


class PINNEvaluator:
    """Evaluator for PINN architectures.

    This evaluator assesses neural network architectures based on their
    ability to solve PDEs using physics-informed learning.

    The evaluation process:
    1. Build network from descriptor
    2. Generate collocation points for the PDE
    3. Train the network using physics-informed loss
    4. Return final physics loss as fitness

    Attributes:
        problem: PDE problem to solve
        n_train_iters: Number of training iterations per evaluation
        learning_rate: Learning rate for training
        optimizer_name: Name of optimizer to use
        trainer: Optax-based trainer instance

    Example:
        >>> from nasjax.pinn.benchmarks import LinearBurgersProblem
        >>> from nasjax.descriptors import MLPDescriptor
        >>> import jax.numpy as jnp
        >>>
        >>> problem = LinearBurgersProblem(c=1.0, nu=0.02)
        >>> evaluator = PINNEvaluator(problem, n_train_iters=500)
        >>>
        >>> descriptor = MLPDescriptor.random_init(2, 1, 3, 32, key)
        >>> # Empty arrays for x_train, y_train, x_test, y_test (not used by PINN)
        >>> fitness = evaluator.evaluate(
        ...     descriptor,
        ...     jnp.array([]), jnp.array([]),
        ...     jnp.array([]), jnp.array([]),
        ...     key
        ... )
        >>> print(f"Physics loss: {fitness:.6f}")
    """

    def __init__(
        self,
        problem: BasePDEProblem,
        n_train_iters: int = 500,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam',
        batch_size: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize PINN evaluator.

        Args:
            problem: PDE problem to solve
            n_train_iters: Training iterations per evaluation
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop', 'adamw')
            batch_size: Optional batch size for collocation points (None = use all)
            verbose: Whether to print training progress
        """
        self.problem = problem
        self.n_train_iters = n_train_iters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

        # Create trainer
        self.trainer = Trainer(
            optimizer=optimizer,
            learning_rate=learning_rate
        )

    def evaluate(
        self,
        descriptor: MLPDescriptor,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_test: jnp.ndarray,
        key: jax.random.PRNGKey,
        train: bool = True
    ) -> float:
        """Evaluate a PINN architecture.

        Note: For PINNs, x_train, y_train, x_test, y_test are not used
        since we generate collocation points from the PDE problem.
        These parameters are kept for compatibility with the Evolving class.

        Args:
            descriptor: Network architecture descriptor
            x_train: Not used (kept for compatibility)
            y_train: Not used (kept for compatibility)
            x_test: Not used (kept for compatibility)
            y_test: Not used (kept for compatibility)
            key: JAX random key
            train: Whether to train before evaluation (if False, evaluate random init)

        Returns:
            Fitness value (physics loss - lower is better)
        """
        # Validate descriptor
        if not descriptor.validate():
            return float('inf')  # Invalid architecture

        try:
            # Generate collocation points
            key, k1, k2, k3 = jax.random.split(key, 4)

            coords_dict = {
                'collocation': self.problem.generate_collocation_points(k1),
                'ic': self.problem.generate_ic_points(k2),
                'bc': self.problem.generate_bc_points(k3)
            }

            # Build network from descriptor
            key, subkey = jax.random.split(key)
            network = PINNNetwork(descriptor, subkey)

            # Train the network if requested
            if train:
                key, subkey = jax.random.split(key)
                network, history = self._train_pinn(network, coords_dict, subkey)

            # Evaluate final physics loss
            key, subkey = jax.random.split(key)
            loss = self.problem.loss_function(network, coords_dict, subkey)

            return float(loss)

        except Exception as e:
            # If evaluation fails, return very high fitness
            if self.verbose:
                print(f"Evaluation failed: {e}")
            return float('inf')

    def _train_pinn(
        self,
        network: PINNNetwork,
        coords_dict: Dict[str, jnp.ndarray],
        key: jax.random.PRNGKey
    ) -> tuple:
        """Train PINN using physics-informed loss.

        Args:
            network: PINN network to train
            coords_dict: Dictionary with collocation, IC, BC points
            key: JAX random key

        Returns:
            Tuple of (trained_network, training_history)
        """
        # Initialize optimizer
        optimizer = self.trainer.optimizer
        opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

        # Training history
        history = {'train_loss': []}

        # Define loss function that doesn't depend on data
        @eqx.filter_jit
        def compute_loss(model, coords_dict, key):
            """Compute physics-informed loss."""
            return self.problem.loss_function(model, coords_dict, key)

        # Define training step
        @eqx.filter_jit
        def train_step(model, opt_state, coords_dict, key):
            """Single training step."""
            # Compute loss and gradients
            loss, grads = eqx.filter_value_and_grad(compute_loss)(
                model, coords_dict, key
            )

            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            return model, opt_state, loss

        # Training loop
        for iteration in range(self.n_train_iters):
            key, subkey = jax.random.split(key)

            # Optionally regenerate collocation points each iteration
            # (helps with exploration of the domain)
            if iteration % 100 == 0:
                k1, k2, k3 = jax.random.split(key, 3)
                coords_dict = {
                    'collocation': self.problem.generate_collocation_points(k1),
                    'ic': self.problem.generate_ic_points(k2),
                    'bc': self.problem.generate_bc_points(k3)
                }

            network, opt_state, loss = train_step(
                network, opt_state, coords_dict, subkey
            )

            history['train_loss'].append(float(loss))

            # Print progress
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}/{self.n_train_iters}, Loss: {loss:.6f}")

        return network, history

    def evaluate_population(
        self,
        descriptors: list[MLPDescriptor],
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_test: jnp.ndarray,
        key: jax.random.PRNGKey,
        train: bool = True
    ) -> list[float]:
        """Evaluate multiple descriptors.

        Note: For PINNs, x_train, y_train, x_test, y_test are not used.
        They are kept for compatibility with the standard Evaluator interface.

        Args:
            descriptors: List of descriptors to evaluate
            x_train: Not used (kept for compatibility)
            y_train: Not used (kept for compatibility)
            x_test: Not used (kept for compatibility)
            y_test: Not used (kept for compatibility)
            key: JAX random key
            train: Whether to train before evaluation

        Returns:
            List of fitness values

        Example:
            >>> descriptors = [MLPDescriptor.random_init(...) for _ in range(10)]
            >>> fitnesses = evaluator.evaluate_population(
            ...     descriptors, jnp.array([]), jnp.array([]),
            ...     jnp.array([]), jnp.array([]), key
            ... )
        """
        keys = jax.random.split(key, len(descriptors))
        fitnesses = []

        for desc, k in zip(descriptors, keys):
            fitness = self.evaluate(
                desc, x_train, y_train, x_test, y_test, k, train=train
            )
            fitnesses.append(fitness)

        return fitnesses
