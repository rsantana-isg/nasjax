"""Physics-informed loss functions for PDEs.

This module provides loss functions that incorporate physical laws
(PDEs) into the training objective.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Callable

from .networks import PINNNetwork


def linear_burgers_loss(
    network: PINNNetwork,
    coords: jnp.ndarray,
    ic_coords: jnp.ndarray,
    ic_values: jnp.ndarray,
    bc_coords: jnp.ndarray,
    bc_values: jnp.ndarray,
    params: Dict[str, float],
    loss_weights: Dict[str, float],
    key: jax.random.PRNGKey
) -> float:
    """Physics-informed loss for linear Burgers equation.

    PDE: ∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0

    Args:
        network: PINN network
        coords: Collocation points in domain (n_points, 2) - (x, t)
        ic_coords: Initial condition points (n_ic, 2)
        ic_values: Initial condition values (n_ic, 1)
        bc_coords: Boundary condition points (n_bc, 2)
        bc_values: Boundary condition values (n_bc, 1)
        params: PDE parameters {'c': convection, 'nu': diffusion}
        loss_weights: Loss weights {'pde': ..., 'ic': ..., 'bc': ...}
        key: JAX random key

    Returns:
        Total physics-informed loss (scalar)
    """
    c = params['c']  # convection coefficient
    nu = params['nu']  # diffusion coefficient

    # Evaluate network and derivatives at collocation points
    # Use vmap to vectorize over batch
    def eval_derivatives(point):
        return network(point, key=key, inference=True, compute_derivatives=True)

    results = jax.vmap(eval_derivatives)(coords)
    u, u_x, u_t, u_xx, _, _ = results

    # PDE residual: ∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0
    pde_residual = u_t + c * u_x - nu * u_xx
    loss_pde = jnp.mean(pde_residual ** 2)

    # Initial condition loss
    def eval_ic(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_ic = jax.vmap(eval_ic)(ic_coords)
    loss_ic = jnp.mean((u_ic - ic_values) ** 2)

    # Boundary condition loss
    if bc_coords is not None and len(bc_coords) > 0:
        u_bc = jax.vmap(eval_ic)(bc_coords)
        loss_bc = jnp.mean((u_bc - bc_values) ** 2)
    else:
        loss_bc = 0.0

    # Weighted combination
    total_loss = (
        loss_weights.get('pde', 1.0) * loss_pde +
        loss_weights.get('ic', 1.0) * loss_ic +
        loss_weights.get('bc', 0.0) * loss_bc
    )

    return total_loss


def convection_diffusion_loss(
    network: PINNNetwork,
    coords: jnp.ndarray,
    ic_coords: jnp.ndarray,
    ic_values: jnp.ndarray,
    bc_coords: jnp.ndarray,
    bc_values: jnp.ndarray,
    params: Dict[str, float],
    loss_weights: Dict[str, float],
    key: jax.random.PRNGKey
) -> float:
    """Physics-informed loss for convection-diffusion equation.

    PDE: v·∂u/∂x = k·∂²u/∂x²

    Args:
        network: PINN network
        coords: Collocation points
        ic_coords: Initial condition points
        ic_values: Initial condition values
        bc_coords: Boundary condition points
        bc_values: Boundary condition values
        params: PDE parameters {'v': velocity, 'k': diffusivity}
        loss_weights: Loss weights
        key: JAX random key

    Returns:
        Total loss
    """
    v = params['v']  # velocity
    k = params['k']  # diffusivity

    # Evaluate derivatives
    def eval_derivatives(point):
        return network(point, key=key, inference=True, compute_derivatives=True)

    results = jax.vmap(eval_derivatives)(coords)
    u, u_x, _, u_xx, _, _ = results  # Only need u_x and u_xx for this PDE

    # PDE residual: v·∂u/∂x - k·∂²u/∂x² = 0
    pde_residual = v * u_x - k * u_xx
    loss_pde = jnp.mean(pde_residual ** 2)

    # Initial condition loss
    def eval_ic(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_ic = jax.vmap(eval_ic)(ic_coords)
    loss_ic = jnp.mean((u_ic - ic_values) ** 2)

    # Boundary condition loss
    if bc_coords is not None and len(bc_coords) > 0:
        u_bc = jax.vmap(eval_ic)(bc_coords)
        loss_bc = jnp.mean((u_bc - bc_values) ** 2)
    else:
        loss_bc = 0.0

    # Weighted combination
    total_loss = (
        loss_weights.get('pde', 1.0) * loss_pde +
        loss_weights.get('ic', 1.0) * loss_ic +
        loss_weights.get('bc', 0.0) * loss_bc
    )

    return total_loss


def get_physics_loss(pde_name: str) -> Callable:
    """Get physics loss function by PDE name.

    Args:
        pde_name: Name of the PDE ('linear_burgers', 'convection_diffusion', etc.)

    Returns:
        Physics loss function

    Raises:
        ValueError: If PDE name is not recognized
    """
    losses = {
        'linear_burgers': linear_burgers_loss,
        'convection_diffusion': convection_diffusion_loss,
    }

    if pde_name not in losses:
        raise ValueError(
            f"Unknown PDE: {pde_name}. "
            f"Valid options: {list(losses.keys())}"
        )

    return losses[pde_name]
