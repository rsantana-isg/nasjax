"""
PIKAN loss functions.

All residuals are computed using ``jax.grad`` on the network's scalar output
with respect to individual input coordinates, following the approach in
``jaxkan.pikan.pde.get_burgers_res``.

The training optimiser differentiates with respect to the network weights via
``nnx.value_and_grad`` (see ``trainer.py``).  This file only defines the
*forward* loss computation.

Input convention
----------------
Collocation points passed to jaxkan layers are ``(N, 2)`` with column 0 = t,
column 1 = x.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from nasjax.pikans.networks import PIKANNetwork


# ---------------------------------------------------------------------------
# PDE / IC / BC residuals
# ---------------------------------------------------------------------------

def burgers_pde_residual(model: PIKANNetwork, x_pde: jnp.ndarray,
                          t_pde: jnp.ndarray, nu: float) -> jnp.ndarray:
    """Burgers' equation PDE residual for a PIKAN model.

    Uses scalar ``jax.grad`` on the model output w.r.t. individual input
    coordinates (t and x), consistent with the jaxkan PIKAN tutorial.

    Args:
        model:  :class:`~nasjax.pikans.networks.PIKANNetwork`.
        x_pde:  Spatial coordinates, shape ``(N, 1)``.
        t_pde:  Time coordinates, shape ``(N, 1)``.
        nu:     Kinematic viscosity.

    Returns:
        Residual array, shape ``(N, 1)``.
    """
    nu_arr = jnp.array(nu, dtype=jnp.float32)

    def u_fn(t_scalar, x_scalar):
        return model(jnp.array([[t_scalar, x_scalar]]))[0, 0]

    u_t_fn  = jax.grad(u_fn, argnums=0)
    u_x_fn  = jax.grad(u_fn, argnums=1)
    u_xx_fn = jax.grad(u_x_fn, argnums=1)

    def point_res(t_scalar, x_scalar):
        u   = u_fn(t_scalar, x_scalar)
        u_t = u_t_fn(t_scalar, x_scalar)
        u_x = u_x_fn(t_scalar, x_scalar)
        u_xx = u_xx_fn(t_scalar, x_scalar)
        return u_t + u * u_x - nu_arr * u_xx

    return jax.vmap(point_res)(t_pde.ravel(), x_pde.ravel()).reshape(-1, 1)


def ic_residual(model: PIKANNetwork, x_ic: jnp.ndarray,
                t_ic: jnp.ndarray, u_ic: jnp.ndarray) -> jnp.ndarray:
    """IC residual: u_pred(x,0) − u_ic."""
    inp = jnp.concatenate([t_ic, x_ic], axis=1)  # (N, 2): (t=0, x)
    return model(inp) - u_ic


def bc_residuals(model: PIKANNetwork,
                 batch_data: dict) -> dict:
    """Boundary condition residuals (left and right)."""
    inp_l = jnp.concatenate([batch_data["t_bc_l"], batch_data["x_bc_l"]], axis=1)
    inp_r = jnp.concatenate([batch_data["t_bc_r"], batch_data["x_bc_r"]], axis=1)
    return {
        "bc_l": model(inp_l) - batch_data["u_bc_l"],
        "bc_r": model(inp_r) - batch_data["u_bc_r"],
    }


# ---------------------------------------------------------------------------
# Total loss (used with nnx.value_and_grad)
# ---------------------------------------------------------------------------

def compute_loss(
    model: PIKANNetwork,
    batch_data: dict,
    global_lambdas: dict,
    nu: float,
) -> jnp.ndarray:
    """Weighted MSE PIKAN loss.

    This function is intended to be passed to ``nnx.value_and_grad`` so that
    gradients w.r.t. the KAN network parameters are computed automatically.

    Args:
        model:          :class:`~nasjax.pikans.networks.PIKANNetwork`.
        batch_data:     Collocation data dict from
                        :meth:`~nasjax.bpinn.pde.base.PDEProblem.prepare_data`.
        global_lambdas: Dict with ``lambda_pde``, ``lambda_ic``,
                        ``lambda_bc_l``, ``lambda_bc_r`` keys.
        nu:             Kinematic viscosity for the Burgers residual.

    Returns:
        Scalar total loss.
    """
    pde_res = burgers_pde_residual(model, batch_data["x_pde"], batch_data["t_pde"], nu)
    pde_loss = jnp.mean(pde_res ** 2)

    ic_res = ic_residual(model, batch_data["x_ic"], batch_data["t_ic"], batch_data["u_ic"])
    ic_loss = jnp.mean(ic_res ** 2)

    bc_dict = bc_residuals(model, batch_data)
    bc_l_loss = jnp.mean(bc_dict["bc_l"] ** 2)
    bc_r_loss = jnp.mean(bc_dict["bc_r"] ** 2)

    return (
        global_lambdas["lambda_pde"]   * pde_loss
        + global_lambdas["lambda_ic"]  * ic_loss
        + global_lambdas["lambda_bc_l"] * bc_l_loss
        + global_lambdas["lambda_bc_r"] * bc_r_loss
    )
