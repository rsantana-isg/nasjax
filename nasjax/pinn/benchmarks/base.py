"""Base class for PDE benchmark problems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Dict, Optional


@dataclass
class BasePDEProblem(ABC):
    """Base class for PDE benchmark problems.

    Subclasses should implement:
    - generate_collocation_points()
    - generate_ic_points()
    - initial_condition()
    - loss_function()
    """

    # Domain bounds
    x_min: float = -1.5
    x_max: float = 4.5
    t_min: float = 0.0
    t_max: float = 2.0

    # Number of collocation points
    n_collocation: int = 10000
    n_ic: int = 100
    n_bc: int = 100

    @abstractmethod
    def generate_collocation_points(
        self, key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Generate random collocation points in the domain.

        Args:
            key: JAX random key

        Returns:
            Array of shape (n_collocation, n_dims) with random points
        """
        pass

    @abstractmethod
    def generate_ic_points(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Generate initial condition points.

        Args:
            key: JAX random key

        Returns:
            Array of shape (n_ic, n_dims) with IC points
        """
        pass

    @abstractmethod
    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate initial condition u(x, 0).

        Args:
            x: Spatial coordinates

        Returns:
            Initial values
        """
        pass

    @abstractmethod
    def loss_function(
        self,
        network,
        coords_dict: Dict[str, jnp.ndarray],
        key: jax.random.PRNGKey
    ) -> float:
        """Compute physics-informed loss.

        Args:
            network: PINN network
            coords_dict: Dictionary with 'collocation', 'ic', 'bc' points
            key: JAX random key

        Returns:
            Total loss value
        """
        pass

    def analytical_solution(
        self, coords: jnp.ndarray
    ) -> Optional[jnp.ndarray]:
        """Analytical solution (if available).

        Args:
            coords: Coordinates (x, t)

        Returns:
            Analytical solution values, or None if not available
        """
        return None
