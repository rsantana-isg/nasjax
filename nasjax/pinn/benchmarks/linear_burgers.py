"""Linear Burgers equation benchmark problem.

PDE: ∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0

This is a 1D+time linear convection-diffusion equation.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Dict

from .base import BasePDEProblem
from ..losses import linear_burgers_loss


@dataclass
class LinearBurgersProblem(BasePDEProblem):
    """Linear Burgers equation benchmark.

    PDE: ∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0

    Domain: x ∈ [x_min, x_max], t ∈ [t_min, t_max]
    Initial Condition: u(x, 0) = 10·exp(-(2x)²)

    Attributes:
        c: Convection coefficient
        nu: Diffusion coefficient
        x_min, x_max: Spatial domain bounds
        t_min, t_max: Temporal domain bounds
        n_collocation: Number of collocation points for PDE residual
        n_ic: Number of initial condition points
        n_bc: Number of boundary condition points

    Example:
        >>> problem = LinearBurgersProblem(c=1.0, nu=0.02)
        >>> key = jax.random.PRNGKey(0)
        >>> coords_dict = {
        ...     'collocation': problem.generate_collocation_points(key),
        ...     'ic': problem.generate_ic_points(key),
        ...     'bc': problem.generate_bc_points(key)
        ... }
        >>> loss = problem.loss_function(network, coords_dict, key)
    """

    # PDE parameters
    c: float = 1.0  # convection coefficient
    nu: float = 0.02  # diffusion coefficient

    # Loss weights
    loss_weight_pde: float = 1.0
    loss_weight_ic: float = 1.0
    loss_weight_bc: float = 0.0  # Often set to 0 for periodic BCs

    def generate_collocation_points(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Generate random collocation points in domain.

        Args:
            key: JAX random key

        Returns:
            Array of shape (n_collocation, 2) with (x, t) coordinates
        """
        key_x, key_t = jax.random.split(key)

        x = jax.random.uniform(
            key_x, (self.n_collocation, 1),
            minval=self.x_min, maxval=self.x_max
        )
        t = jax.random.uniform(
            key_t, (self.n_collocation, 1),
            minval=self.t_min, maxval=self.t_max
        )

        return jnp.hstack([x, t])

    def generate_ic_points(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Generate initial condition points at t=0.

        Args:
            key: JAX random key

        Returns:
            Array of shape (n_ic, 2) with (x, 0) coordinates
        """
        x = jax.random.uniform(
            key, (self.n_ic, 1),
            minval=self.x_min, maxval=self.x_max
        )
        t = jnp.zeros((self.n_ic, 1))

        return jnp.hstack([x, t])

    def generate_bc_points(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Generate boundary condition points.

        For periodic boundary conditions, we sample at x_min and x_max.

        Args:
            key: JAX random key

        Returns:
            Array of shape (n_bc, 2) with boundary points
        """
        # Sample time points
        t = jax.random.uniform(
            key, (self.n_bc // 2, 1),
            minval=self.t_min, maxval=self.t_max
        )

        # Left boundary (x = x_min)
        x_left = jnp.full((self.n_bc // 2, 1), self.x_min)
        bc_left = jnp.hstack([x_left, t])

        # Right boundary (x = x_max)
        x_right = jnp.full((self.n_bc // 2, 1), self.x_max)
        bc_right = jnp.hstack([x_right, t])

        return jnp.vstack([bc_left, bc_right])

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Analytical initial condition: u(x, 0) = 10·exp(-(2x)²).

        Args:
            x: Spatial coordinates (can be array)

        Returns:
            Initial condition values
        """
        return 10.0 * jnp.exp(-(2.0 * x) ** 2)

    def boundary_condition(self, t: jnp.ndarray, side: str = 'left') -> jnp.ndarray:
        """Boundary condition (often periodic or zero for this problem).

        Args:
            t: Time coordinates
            side: 'left' or 'right' boundary

        Returns:
            Boundary condition values
        """
        # For simplicity, assume zero boundary conditions
        # or periodic (which means u(x_min, t) = u(x_max, t))
        return jnp.zeros_like(t)

    def loss_function(
        self,
        network,
        coords_dict: Dict[str, jnp.ndarray],
        key: jax.random.PRNGKey
    ) -> float:
        """Compute physics-informed loss for Linear Burgers equation.

        Args:
            network: PINN network
            coords_dict: Dictionary with keys:
                - 'collocation': Collocation points (n_coll, 2)
                - 'ic': Initial condition points (n_ic, 2)
                - 'bc': Boundary condition points (n_bc, 2)
            key: JAX random key

        Returns:
            Total physics-informed loss
        """
        coords = coords_dict['collocation']
        ic_coords = coords_dict['ic']
        bc_coords = coords_dict.get('bc', None)

        # Get IC and BC values
        ic_values = self.initial_condition(ic_coords[:, 0:1])

        if bc_coords is not None and len(bc_coords) > 0:
            bc_values = self.boundary_condition(bc_coords[:, 1:2])
        else:
            bc_values = None

        # Call physics loss function
        return linear_burgers_loss(
            network=network,
            coords=coords,
            ic_coords=ic_coords,
            ic_values=ic_values,
            bc_coords=bc_coords,
            bc_values=bc_values,
            params={'c': self.c, 'nu': self.nu},
            loss_weights={
                'pde': self.loss_weight_pde,
                'ic': self.loss_weight_ic,
                'bc': self.loss_weight_bc
            },
            key=key
        )
