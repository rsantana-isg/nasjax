"""
BPINNModel — the top-level Equinox module that combines:

1. Optional Fourier feature embedding of (x, t) for hard-coded periodic BCs.
2. A Bayesian MLP (Vanilla or Enhanced) that maps embedded input → u(x,t).
3. Scalar autodiff utilities (∂u/∂x, ∂u/∂t, ∂²u/∂x²) via ``jax.grad``.
4. Batched versions that broadcast over collections of (x, t) pairs.

JAX / Equinox design notes
---------------------------
- ``use_fourier``, ``L_fourier``, ``M_fourier`` are ``static=True`` fields so
  Python-level branching on them is resolved at trace time (no traced
  conditional).
- ``u_scalar`` accepts *scalar* ``x`` and ``t`` — this is what ``jax.grad``
  differentiates with respect to.  :meth:`out_u_pde` and
  :meth:`out_u_only` apply ``jax.vmap`` over batches.
- All derivatives flow correctly through the Fourier embedding (``jnp.cos``,
  ``jnp.sin`` are differentiable) as well as through the Bayesian MLP.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .config import BPINNConfig
from .networks import build_bayesian_mlp


# ---------------------------------------------------------------------------
# Fourier feature embedding (static helper, pure function)
# ---------------------------------------------------------------------------

def fourier_feature_embedding(
    x: jnp.ndarray,
    t: jnp.ndarray,
    L: float,
    M: int,
) -> jnp.ndarray:
    """Embed a scalar (x, t) pair using Fourier features.

    Produces: ``[1, cos(ω·x), …, cos(M·ω·x), sin(ω·x), …, sin(M·ω·x), t]``
    where ``ω = 2π / L``.

    Shape of output: ``(2*M + 2,)``.
    """
    omega = 2.0 * jnp.pi / L
    k = jnp.arange(1, M + 1)
    return jnp.concatenate([
        jnp.array([1.0]),
        jnp.cos(k * omega * x),
        jnp.sin(k * omega * x),
        jnp.array([t]),
    ])


# ---------------------------------------------------------------------------
# BPINNModel
# ---------------------------------------------------------------------------

class BPINNModel(eqx.Module):
    """Top-level BPINN module: embedding + Bayesian MLP + autodiff.

    Args:
        mlp:         :class:`BayesianMLPVanilla` or :class:`BayesianMLPEnhanced`.
        use_fourier: Whether Fourier feature embedding is active.
        L_fourier:   Spatial period for Fourier embedding.
        M_fourier:   Number of Fourier modes.
    """

    mlp: eqx.Module

    use_fourier: bool = eqx.field(static=True)
    L_fourier: float = eqx.field(static=True)
    M_fourier: int   = eqx.field(static=True)

    def __init__(self, config: BPINNConfig, key: jax.random.PRNGKey) -> None:
        self.mlp = build_bayesian_mlp(config, key)
        self.use_fourier = config.fourier_embedding.enabled
        self.L_fourier   = config.fourier_embedding.L
        self.M_fourier   = config.fourier_embedding.M

    # -----------------------------------------------------------------------
    # Input embedding
    # -----------------------------------------------------------------------

    def _embed(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Return the (possibly Fourier-embedded) input vector ``z``."""
        if self.use_fourier:
            return fourier_feature_embedding(x, t, self.L_fourier, self.M_fourier)
        return jnp.stack([x, t])  # shape (2,) for scalar x, t

    # -----------------------------------------------------------------------
    # Bayesian sampling (delegates to MLP)
    # -----------------------------------------------------------------------

    def sample_params(
        self,
        key: jax.random.PRNGKey,
        sample: bool = True,
    ) -> Tuple[List[Dict], jnp.ndarray, jnp.ndarray]:
        """Sample network parameters and compute ELBO components.

        Returns:
            ``(params_list, total_log_prior, total_log_var_posterior)``
        """
        return self.mlp.sample_prior_var_posterior(key, sample)

    # -----------------------------------------------------------------------
    # Single-point forward passes (scalar x, t — compatible with jax.grad)
    # -----------------------------------------------------------------------

    def u_scalar(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """Evaluate u(x, t) for **scalar** x and t.

        The scalar requirement is essential for ``jax.grad`` to compute
        ∂u/∂x and ∂u/∂t correctly.

        Returns:
            Scalar (0-d JAX array).
        """
        z = self._embed(x, t)
        return jnp.squeeze(self.mlp(z, params_list))

    def u_pde_terms(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        params_list: List[Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute ``(u, ∂u/∂x, ∂u/∂t, ∂²u/∂x²)`` for scalar x, t.

        Uses nested ``jax.grad`` — fully compatible with JIT and vmap.

        Returns:
            ``(u, du_dx, du_dt, du_dxx)`` — all scalar arrays.
        """
        u     = self.u_scalar(x, t, params_list)
        du_dx = jax.grad(lambda xx: self.u_scalar(xx, t, params_list))(x)
        du_dt = jax.grad(lambda tt: self.u_scalar(x, tt, params_list))(t)
        du_dxx = jax.grad(
            jax.grad(lambda xx: self.u_scalar(xx, t, params_list))
        )(x)
        return u, du_dx, du_dt, du_dxx

    # -----------------------------------------------------------------------
    # Batched forward passes (vmap over N points)
    # -----------------------------------------------------------------------

    @eqx.filter_jit
    def out_u_pde(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        params_list: List[Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Batched  (u, ∂u/∂x, ∂u/∂t, ∂²u/∂x²)  over N collocation points.

        Args:
            x:           Spatial coordinates, shape ``(N,)`` or ``(N, 1)``.
            t:           Time coordinates, shape ``(N,)`` or ``(N, 1)``.
            params_list: Sampled parameters.

        Returns:
            Four arrays each of shape ``(N, 1)``.
        """
        u, du_dx, du_dt, du_dxx = jax.vmap(
            self.u_pde_terms, in_axes=(0, 0, None)
        )(x.ravel(), t.ravel(), params_list)
        reshape = lambda a: a.reshape(-1, 1)
        return reshape(u), reshape(du_dx), reshape(du_dt), reshape(du_dxx)

    @eqx.filter_jit
    def out_u_only(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """Batched u(x, t) — no derivative computation (faster).

        Returns:
            Array of shape ``(N, 1)``.
        """
        u = jax.vmap(
            lambda xi, ti: self.u_scalar(xi, ti, params_list)
        )(x.ravel(), t.ravel())
        return u.reshape(-1, 1)
