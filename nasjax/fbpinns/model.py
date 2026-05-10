"""
FBPINNModel — domain decomposition geometry and combined forward pass.

The model stores the *static* subdomain geometry (centres, half-widths,
bounds) as JAX arrays.  The *dynamic* part (all subdomain network weights) is
passed as a separate pytree ``params`` so that ``jax.value_and_grad`` can
differentiate through it for optimisation.

Combined solution
-----------------
::

    u(x, t) = Σ_m  W_m(x, t)  ·  N_m(norm_m(x, t))

where ``W_m`` is the cosine or sigmoid window function for subdomain ``m`` and
``N_m`` is the subdomain FCN.

Window functions
----------------
``cosine``   ((1 + cos(π(x−μ)/σ))/2)² per dimension, product over dims.
``sigmoid``  sigmoid((x−μ_lo)/s_lo) · sigmoid((μ_hi−x)/s_hi) per dim.

Both are zero outside the subdomain, smooth inside.

JAX design
----------
``u_scalar(params, x, t)`` is a pure function: params and inputs are JAX
arrays, the geometry is a Python-level closure resolved at JIT time.  This
lets us use ``jax.grad`` for PDE derivatives and ``jax.value_and_grad`` for
optimisation.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import FBPINNConfig
from .networks import forward_all, get_activation


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _cosine_window(xmins: jnp.ndarray, xmaxs: jnp.ndarray,
                   pt: jnp.ndarray) -> jnp.ndarray:
    """Cosine POU window for all subdomains at point pt.

    Args:
        xmins: (m, 2) lower bounds.
        xmaxs: (m, 2) upper bounds.
        pt:    (2,) query point.

    Returns:
        (m, 1) window values, 0 outside subdomain.
    """
    mu = (xmins + xmaxs) / 2          # (m, 2)
    sd = (xmaxs - xmins) / 2          # (m, 2)
    raw = ((1.0 + jnp.cos(jnp.pi * (pt[None, :] - mu) / sd)) / 2.0) ** 2  # (m, 2)
    inside = jnp.all(
        (pt[None, :] >= xmins) & (pt[None, :] <= xmaxs), axis=1
    )  # (m,)
    w = jnp.prod(raw, axis=1) * inside    # (m,)
    return w[:, None]                      # (m, 1)


def _sigmoid_window(xmins: jnp.ndarray, xmaxs: jnp.ndarray,
                    pt: jnp.ndarray, tol: float = 1e-3) -> jnp.ndarray:
    """Sigmoid POU window for all subdomains at point pt."""
    mu_lo = xmins                       # (m, 2)
    mu_hi = xmaxs                       # (m, 2)
    sd = (xmaxs - xmins)               # (m, 2)
    # ramp up from xmin, ramp down toward xmax
    s = sd / (2.0 * jnp.log((1.0 - tol) / tol))
    w_lo = jax.nn.sigmoid((pt[None, :] - mu_lo) / (s + 1e-12))   # (m, 2)
    w_hi = jax.nn.sigmoid((mu_hi - pt[None, :]) / (s + 1e-12))   # (m, 2)
    w = jnp.prod(w_lo * w_hi, axis=1)  # (m,)
    return w[:, None]                   # (m, 1)


# ---------------------------------------------------------------------------
# FBPINNModel
# ---------------------------------------------------------------------------

class FBPINNModel:
    """Combines subdomain geometry, window functions and network forward pass.

    This class captures all *static* information at construction time so that
    ``u_scalar`` and ``u_pde_terms`` become pure functions of (params, x, t)
    suitable for ``jax.jit``, ``jax.grad``, and ``jax.value_and_grad``.

    Args:
        config:  :class:`~nasjax.fbpinns.config.FBPINNConfig`.
    """

    def __init__(self, config: FBPINNConfig) -> None:
        dc = config.decomposition
        m_x, m_t = dc.n_subdomains_x, dc.n_subdomains_t
        m = m_x * m_t

        # Build subdomain grid
        xs_centers = np.linspace(dc.x_min, dc.x_max, m_x)
        ts_centers = np.linspace(dc.t_min, dc.t_max, m_t)

        raw_width_x = (dc.x_max - dc.x_min) / max(m_x - 1, 1) if m_x > 1 else (dc.x_max - dc.x_min)
        raw_width_t = (dc.t_max - dc.t_min) / max(m_t - 1, 1) if m_t > 1 else (dc.t_max - dc.t_min)

        # Half-widths including overlap
        half_x = (raw_width_x / 2.0) * (1.0 + dc.overlap)
        half_t = (raw_width_t / 2.0) * (1.0 + dc.overlap)

        # Clamp to domain boundaries
        xmins, xmaxs = [], []
        for xc in xs_centers:
            for tc in ts_centers:
                xmins.append([
                    max(xc - half_x, dc.x_min),
                    max(tc - half_t, dc.t_min),
                ])
                xmaxs.append([
                    min(xc + half_x, dc.x_max),
                    min(tc + half_t, dc.t_max),
                ])

        self.xmins: jnp.ndarray = jnp.array(xmins, dtype=jnp.float32)  # (m, 2)
        self.xmaxs: jnp.ndarray = jnp.array(xmaxs, dtype=jnp.float32)  # (m, 2)
        self.centers: jnp.ndarray = (self.xmins + self.xmaxs) / 2      # (m, 2)
        self.half_widths: jnp.ndarray = (self.xmaxs - self.xmins) / 2  # (m, 2)
        self.n_subdomains: int = m

        self.activation_fn: Callable = get_activation(
            config.layer_specs[0].activation if config.layer_specs else "tanh"
        )
        self.window_name: str = dc.window_fn

    # -----------------------------------------------------------------------
    # Window helper (called inside JIT; geometry is a Python closure)
    # -----------------------------------------------------------------------

    def _window(self, pt: jnp.ndarray) -> jnp.ndarray:
        if self.window_name == "sigmoid":
            return _sigmoid_window(self.xmins, self.xmaxs, pt)
        return _cosine_window(self.xmins, self.xmaxs, pt)

    # -----------------------------------------------------------------------
    # Combined scalar forward pass (pure function — JIT / grad compatible)
    # -----------------------------------------------------------------------

    def u_scalar(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate combined FBPINN output u(x, t) at a **scalar** point.

        All subdomain networks are evaluated simultaneously using stacked
        batched matmuls.  Window functions weight the contributions.

        Args:
            params:  Parameter pytree from
                     :func:`~nasjax.fbpinns.networks.init_params`.
            x:       Scalar spatial coordinate (0-d JAX array).
            t:       Scalar time coordinate (0-d JAX array).

        Returns:
            Scalar u value.
        """
        pt = jnp.stack([x, t])                    # (2,)
        # Normalise to each subdomain's [-1, 1] range
        x_norm_all = (pt[None, :] - self.centers) / (self.half_widths + 1e-12)  # (m, 2)
        # Forward through all subdomains
        u_all = forward_all(params, x_norm_all, self.activation_fn)  # (m, 1)
        # Window weights
        w_all = self._window(pt)                   # (m, 1)
        # Combined solution
        return jnp.sum(w_all * u_all)              # scalar

    # -----------------------------------------------------------------------
    # PDE terms (scalar; vmapped for batches)
    # -----------------------------------------------------------------------

    def u_pde_terms(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple:
        """Compute (u, ∂u/∂x, ∂u/∂t, ∂²u/∂x²) for scalar x, t."""
        u_fn = lambda xx, tt: self.u_scalar(params, xx, tt)
        u     = u_fn(x, t)
        du_dx = jax.grad(u_fn, argnums=0)(x, t)
        du_dt = jax.grad(u_fn, argnums=1)(x, t)
        du_dxx = jax.grad(jax.grad(u_fn, argnums=0), argnums=0)(x, t)
        return u, du_dx, du_dt, du_dxx

    # -----------------------------------------------------------------------
    # Batched outputs
    # -----------------------------------------------------------------------

    def out_u_pde(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple:
        """Batched (u, ∂u/∂x, ∂u/∂t, ∂²u/∂x²), shapes (N,1) each."""
        fn = functools.partial(self.u_pde_terms, params)
        u, du_dx, du_dt, du_dxx = jax.vmap(fn)(x.ravel(), t.ravel())
        reshape = lambda a: a.reshape(-1, 1)
        return reshape(u), reshape(du_dx), reshape(du_dt), reshape(du_dxx)

    def out_u_only(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Batched u(x, t) without derivative computation, shape (N,1)."""
        fn = functools.partial(self.u_scalar, params)
        u = jax.vmap(fn)(x.ravel(), t.ravel())
        return u.reshape(-1, 1)
