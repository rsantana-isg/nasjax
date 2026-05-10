"""
PINN loss function: weighted MSE over PDE, IC, and BC residuals.

The PDE residual is computed by calling ``model.out_u_pde`` directly, which
returns (u, du_dx, du_dt, du_dxx).  This avoids the BPINN's ``params_list``
argument that is specific to the Bayesian sampling interface.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp
import equinox as eqx

from .model import PINNModel


def _burgers_pde_residual(u, du_dx, du_dt, du_dxx, nu):
    """Burgers residual: ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x² = 0."""
    return du_dt + u * du_dx - nu * du_dxx


@eqx.filter_jit
def compute_loss(
    model: PINNModel,
    batch_data: Dict[str, jnp.ndarray],
    global_lambdas: Dict[str, float],
    nu: float,
) -> jnp.ndarray:
    """Weighted PINN loss for Burgers' equation.

    Args:
        model:          :class:`~nasjax.pinns.model.PINNModel`.
        batch_data:     Dict with collocation arrays (``x_pde``, ``t_pde``,
                        ``x_ic``, ``t_ic``, ``u_ic``, ``x_bc_l``, …).
        global_lambdas: Dict with ``lambda_pde``, ``lambda_ic``,
                        ``lambda_bc_l``, ``lambda_bc_r`` keys.
        nu:             Kinematic viscosity (e.g. 0.01/π for Burgers).

    Returns:
        Scalar total loss.
    """
    # --- PDE residual -------------------------------------------------------
    u, du_dx, du_dt, du_dxx = model.out_u_pde(
        batch_data["x_pde"], batch_data["t_pde"]
    )
    pde_res = _burgers_pde_residual(u, du_dx, du_dt, du_dxx, nu)
    pde_loss = jnp.mean(pde_res ** 2)

    # --- IC residual --------------------------------------------------------
    u_ic_pred = model.out_u_only(batch_data["x_ic"], batch_data["t_ic"])
    ic_loss = jnp.mean((u_ic_pred - batch_data["u_ic"]) ** 2)

    # --- BC residuals -------------------------------------------------------
    u_bc_l_pred = model.out_u_only(batch_data["x_bc_l"], batch_data["t_bc_l"])
    u_bc_r_pred = model.out_u_only(batch_data["x_bc_r"], batch_data["t_bc_r"])
    bc_l_loss = jnp.mean((u_bc_l_pred - batch_data["u_bc_l"]) ** 2)
    bc_r_loss = jnp.mean((u_bc_r_pred - batch_data["u_bc_r"]) ** 2)

    total = (
        global_lambdas["lambda_pde"]    * pde_loss
        + global_lambdas["lambda_ic"]   * ic_loss
        + global_lambdas["lambda_bc_l"] * bc_l_loss
        + global_lambdas["lambda_bc_r"] * bc_r_loss
    )
    return total
