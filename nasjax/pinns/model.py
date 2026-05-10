"""
PINNModel — top-level Equinox module combining network + autodiff utilities.

The interface mirrors :class:`~nasjax.bpinn.model.BPINNModel` so that the
same PDE problem classes and evaluator pattern can be reused unchanged.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .config import PINNConfig
from .networks import PINNNetwork


class PINNModel(eqx.Module):
    """Standard PINN model with JAX autodiff utilities.

    Args:
        config:  :class:`~nasjax.pinns.config.PINNConfig`.
        key:     JAX PRNG key.
    """
    net: PINNNetwork

    def __init__(self, config: PINNConfig, key: jax.random.PRNGKey) -> None:
        self.net = PINNNetwork(config, key)

    # -----------------------------------------------------------------------
    # Scalar forward pass (required for jax.grad)
    # -----------------------------------------------------------------------

    def u_scalar(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate u(x, t) for **scalar** x and t (0-d arrays).

        Used internally by ``jax.grad`` to compute spatial/temporal derivatives.
        """
        inp = jnp.stack([x, t])
        return jnp.squeeze(self.net(inp))

    # -----------------------------------------------------------------------
    # PDE term computation (scalar; used via vmap for batches)
    # -----------------------------------------------------------------------

    def u_pde_terms(
        self, x: jnp.ndarray, t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute ``(u, ∂u/∂x, ∂u/∂t, ∂²u/∂x²)`` for scalar x, t.

        Uses nested ``jax.grad`` — fully compatible with JIT and vmap.

        Returns:
            ``(u, du_dx, du_dt, du_dxx)`` — all scalar 0-d arrays.
        """
        u     = self.u_scalar(x, t)
        du_dx = jax.grad(lambda xx: self.u_scalar(xx, t))(x)
        du_dt = jax.grad(lambda tt: self.u_scalar(x, tt))(t)
        du_dxx = jax.grad(
            jax.grad(lambda xx: self.u_scalar(xx, t))
        )(x)
        return u, du_dx, du_dt, du_dxx

    # -----------------------------------------------------------------------
    # Batched outputs
    # -----------------------------------------------------------------------

    @eqx.filter_jit
    def out_u_pde(
        self, x: jnp.ndarray, t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Batched (u, ∂u/∂x, ∂u/∂t, ∂²u/∂x²) over N collocation points.

        Args:
            x: Spatial coordinates, shape ``(N, 1)`` or ``(N,)``.
            t: Time coordinates, shape ``(N, 1)`` or ``(N,)``.

        Returns:
            Four arrays each of shape ``(N, 1)``.
        """
        u, du_dx, du_dt, du_dxx = jax.vmap(self.u_pde_terms)(
            x.ravel(), t.ravel()
        )
        reshape = lambda a: a.reshape(-1, 1)
        return reshape(u), reshape(du_dx), reshape(du_dt), reshape(du_dxx)

    @eqx.filter_jit
    def out_u_only(
        self, x: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Batched u(x, t) without derivative computation.

        Returns:
            Array of shape ``(N, 1)``.
        """
        u = jax.vmap(self.u_scalar)(x.ravel(), t.ravel())
        return u.reshape(-1, 1)
