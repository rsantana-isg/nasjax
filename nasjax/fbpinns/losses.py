"""
FBPINN loss function: weighted MSE over PDE, IC, and BC residuals.

The loss differentiates through the COMBINED output of all subdomains
(weighted sum of windowed local networks), so gradients flow into every
subdomain's weights simultaneously.
"""

from __future__ import annotations

import functools
from typing import Any, Dict

import jax
import jax.numpy as jnp

from .model import FBPINNModel


def compute_loss(
    params: Dict[str, Any],
    model: FBPINNModel,
    problem,
    batch_data: Dict[str, jnp.ndarray],
    global_lambdas: Dict[str, float],
) -> jnp.ndarray:
    """Weighted FBPINN MSE loss.

    Args:
        params:         Subdomain network parameter pytree.
        model:          :class:`~nasjax.fbpinns.model.FBPINNModel` (static geometry).
        problem:        :class:`~nasjax.bpinn.pde.base.PDEProblem` instance.
        batch_data:     Collocation data from
                        :meth:`~nasjax.bpinn.pde.base.PDEProblem.prepare_data`.
        global_lambdas: Dict with ``lambda_pde``, ``lambda_ic``,
                        ``lambda_bc_l``, ``lambda_bc_r`` keys.

    Returns:
        Scalar total loss.
    """
    # --- PDE residual -------------------------------------------------------
    # Wrap model.u_scalar so the problem residual can call out_u_pde / out_u_only
    # using params captured in the closure.
    class _PseudoModel:
        """Thin adapter so BurgersProblem.pde_residual / ic_residual work."""
        def out_u_pde(self, x, t, _params_list=None):
            return model.out_u_pde(params, x, t)

        def out_u_only(self, x, t, _params_list=None):
            return model.out_u_only(params, x, t)

    pseudo = _PseudoModel()
    pde_res = problem.pde_residual(pseudo, batch_data["x_pde"], batch_data["t_pde"], None)
    pde_loss = jnp.mean(pde_res ** 2)

    ic_res = problem.ic_residual(
        pseudo, batch_data["x_ic"], batch_data["t_ic"], batch_data["u_ic"], None
    )
    ic_loss = jnp.mean(ic_res ** 2)

    bc_dict = problem.bc_residuals(pseudo, batch_data, None)
    bc_l_loss = jnp.mean(bc_dict["bc_l"] ** 2)
    bc_r_loss = jnp.mean(bc_dict["bc_r"] ** 2)

    return (
        global_lambdas["lambda_pde"]    * pde_loss
        + global_lambdas["lambda_ic"]   * ic_loss
        + global_lambdas["lambda_bc_l"] * bc_l_loss
        + global_lambdas["lambda_bc_r"] * bc_r_loss
    )
