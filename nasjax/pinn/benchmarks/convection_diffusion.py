"""Convection-Diffusion equation benchmark problem.

PDE: v·∂u/∂x = k·∂²u/∂x²

This is a simpler 1D+time steady-state convection-diffusion equation.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Dict

from .base import BasePDEProblem
from ..losses import convection_diffusion_loss


@dataclass
class ConvectionDiffusionProblem(BasePDEProblem):
    """Convection-Diffusion equation benchmark.

    PDE: v·∂u/∂x = k·∂²u/∂x²

    Domain: x ∈ [x_min, x_max], t ∈ [t_min, t_max]

    Attributes:
        v: Velocity (convection coefficient)
        k: Diffusivity coefficient
    """

    # PDE parameters
    v: float = 1.0  # velocity
    k: float = 0.01  # diffusivity

    # Loss weights
    loss_weight_pde: float = 1.0
    loss_weight_ic: float = 1.0
    loss_weight_bc: float = 0.1

    def generate_collocation_points(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Generate random collocation points."""
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
        """Generate initial condition points at t=0."""
        x = jax.random.uniform(
            key, (self.n_ic, 1),
            minval=self.x_min, maxval=self.x_max
        )
        t = jnp.zeros((self.n_ic, 1))

        return jnp.hstack([x, t])

    def generate_bc_points(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Generate boundary condition points."""
        t = jax.random.uniform(
            key, (self.n_bc // 2, 1),
            minval=self.t_min, maxval=self.t_max
        )

        # Left and right boundaries
        x_left = jnp.full((self.n_bc // 2, 1), self.x_min)
        x_right = jnp.full((self.n_bc // 2, 1), self.x_max)

        bc_left = jnp.hstack([x_left, t])
        bc_right = jnp.hstack([x_right, t])

        return jnp.vstack([bc_left, bc_right])

    def initial_condition(self, x: jnp.ndarray) -> jnp.ndarray:
        """Initial condition: Gaussian pulse."""
        return jnp.exp(-((x - 1.0) ** 2) / 0.1)

    def boundary_condition(self, t: jnp.ndarray, side: str = 'left') -> jnp.ndarray:
        """Boundary condition."""
        return jnp.zeros_like(t)

    def loss_function(
        self,
        network,
        coords_dict: Dict[str, jnp.ndarray],
        key: jax.random.PRNGKey
    ) -> float:
        """Compute physics-informed loss."""
        coords = coords_dict['collocation']
        ic_coords = coords_dict['ic']
        bc_coords = coords_dict.get('bc', None)

        ic_values = self.initial_condition(ic_coords[:, 0:1])

        if bc_coords is not None and len(bc_coords) > 0:
            bc_values = self.boundary_condition(bc_coords[:, 1:2])
        else:
            bc_values = None

        return convection_diffusion_loss(
            network=network,
            coords=coords,
            ic_coords=ic_coords,
            ic_values=ic_values,
            bc_coords=bc_coords,
            bc_values=bc_values,
            params={'v': self.v, 'k': self.k},
            loss_weights={
                'pde': self.loss_weight_pde,
                'ic': self.loss_weight_ic,
                'bc': self.loss_weight_bc
            },
            key=key
        )
