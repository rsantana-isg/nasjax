"""
Burgers' equation BPINN problem.

PDE:
    ∂u/∂t + u · ∂u/∂x − (0.01/π) · ∂²u/∂x² = 0
    for  x ∈ [−1, 1],  t ∈ [0, 1].

Initial condition:
    u(x, 0) = −sin(π·x)

Boundary conditions (Dirichlet, u=0):
    u(−1, t) = 0
    u(+1, t) = 0

Reference data is loaded from the MATLAB file ``burgers_shock.mat``
(distributed with the original JAX_BPINN-main code).

All residual methods are pure JAX functions (jit / vmap compatible).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from .base import PDEProblem


class BurgersProblem(PDEProblem):
    """Viscous Burgers' equation on the unit space-time domain.

    Args:
        data_path: Path to ``burgers_shock.mat``.  If ``None`` the default
                   path relative to this file is used.
        nu:        Kinematic viscosity coefficient (default ``0.01 / π``).
    """

    def __init__(
        self,
        data_path: str | None = None,
        nu: float | None = None,
    ) -> None:
        if data_path is None:
            _here = os.path.dirname(__file__)
            # _here is .../nasjax/nasjax/bpinn/pde/
            # ../../../  takes us to .../nasjax/ (the repo root)
            data_path = os.path.join(
                _here,
                "../../../JAX_BPINN-main/Main_BPINN_Functions/Data/burgers_shock.mat",
            )
        self.data_path = os.path.abspath(data_path)
        self.nu = float(nu) if nu is not None else 0.01 / float(np.pi)

    # -----------------------------------------------------------------------
    # Data preparation (Python-level, not JAX-traced)
    # -----------------------------------------------------------------------

    def prepare_data(self, key: Any, config: Any) -> Dict[str, jnp.ndarray]:
        """Load reference data and sample collocation points.

        Collocation point counts come from ``config.collocation``.
        The domain boundaries are taken from the reference data file.

        Returns:
            Dict with keys:
            ``x_pde``, ``t_pde``,
            ``x_ic``,  ``t_ic``,  ``u_ic``,
            ``x_bc_l``, ``t_bc_l``, ``u_bc_l``,
            ``x_bc_r``, ``t_bc_r``, ``u_bc_r``,
            ``x_data``, ``t_data``, ``u_data``  (full reference grid).
        """
        import scipy.io

        mat = scipy.io.loadmat(self.data_path)
        t_vec = mat["t"].flatten()[:, None]   # shape (Nt, 1)
        x_vec = mat["x"].flatten()[:, None]   # shape (Nx, 1)
        u_ref = np.real(mat["usol"]).T        # shape (Nt, Nx)

        X_mesh, T_mesh = np.meshgrid(x_vec, t_vec)
        X_star = np.hstack([X_mesh.flatten()[:, None], T_mesh.flatten()[:, None]])
        u_star = u_ref.flatten()[:, None]

        x_data = jnp.array(X_star[:, 0:1])
        t_data = jnp.array(X_star[:, 1:2])
        u_data = jnp.array(u_star)

        # Domain bounds
        x_min, x_max = float(x_vec.min()), float(x_vec.max())
        t_min, t_max = float(t_vec.min()), float(t_vec.max())

        col = config.collocation
        n_pde, n_ic, n_bc_l, n_bc_r = col.n_pde, col.n_ic, col.n_bc_l, col.n_bc_r

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # PDE interior
        x_pde = jax.random.uniform(k1, (n_pde, 1), minval=x_min + 1e-3, maxval=x_max - 1e-3)
        t_pde = jax.random.uniform(k2, (n_pde, 1), minval=t_min + 1e-3, maxval=t_max - 1e-3)

        # IC:  u(x, 0) = −sin(πx)
        x_ic  = jax.random.uniform(k3, (n_ic, 1), minval=x_min, maxval=x_max)
        t_ic  = jnp.zeros((n_ic, 1))
        u_ic  = -jnp.sin(jnp.pi * x_ic)

        # BC left:  u(−1, t) = 0
        t_bc_l = jax.random.uniform(k4, (n_bc_l, 1), minval=t_min, maxval=t_max)
        x_bc_l = -jnp.ones((n_bc_l, 1))
        u_bc_l = jnp.zeros((n_bc_l, 1))

        # BC right: u(+1, t) = 0
        t_bc_r = jax.random.uniform(k5, (n_bc_r, 1), minval=t_min, maxval=t_max)
        x_bc_r = jnp.ones((n_bc_r, 1))
        u_bc_r = jnp.zeros((n_bc_r, 1))

        return {
            "x_pde":   x_pde,   "t_pde":   t_pde,
            "x_ic":    x_ic,    "t_ic":    t_ic,    "u_ic":    u_ic,
            "x_bc_l":  x_bc_l,  "t_bc_l":  t_bc_l,  "u_bc_l":  u_bc_l,
            "x_bc_r":  x_bc_r,  "t_bc_r":  t_bc_r,  "u_bc_r":  u_bc_r,
            "x_data":  x_data,  "t_data":  t_data,  "u_data":  u_data,
        }

    # -----------------------------------------------------------------------
    # Residuals (JAX-traced, JIT/vmap compatible)
    # -----------------------------------------------------------------------

    def pde_residual(
        self,
        model: Any,
        x: jnp.ndarray,
        t: jnp.ndarray,
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """Burgers residual: ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x² = 0.

        Returns:
            Residual array, shape ``(N, 1)``.
        """
        u, du_dx, du_dt, du_dxx = model.out_u_pde(x, t, params_list)
        residual = du_dt + u * du_dx - self.nu * du_dxx
        return residual

    def ic_residual(
        self,
        model: Any,
        x: jnp.ndarray,
        t: jnp.ndarray,
        u_target: jnp.ndarray,
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """IC residual: u(x, 0) − (−sin(πx)).

        Returns:
            Residual array, shape ``(N, 1)``.
        """
        u_pred = model.out_u_only(x, t, params_list)
        return u_pred - u_target

    def bc_residuals(
        self,
        model: Any,
        batch_data: Dict[str, jnp.ndarray],
        params_list: List[Dict],
    ) -> Dict[str, jnp.ndarray]:
        """Dirichlet BC residuals for left and right boundaries.

        Returns:
            ``{'bc_l': residual_l, 'bc_r': residual_r}``
        """
        u_bc_l_pred = model.out_u_only(
            batch_data["x_bc_l"], batch_data["t_bc_l"], params_list
        )
        u_bc_r_pred = model.out_u_only(
            batch_data["x_bc_r"], batch_data["t_bc_r"], params_list
        )
        return {
            "bc_l": u_bc_l_pred - batch_data["u_bc_l"],
            "bc_r": u_bc_r_pred - batch_data["u_bc_r"],
        }

    # -----------------------------------------------------------------------
    # Reference grid (for visualization)
    # -----------------------------------------------------------------------

    def load_reference_grid(self):
        """Return the reference solution on the full (x, t) mesh.

        Returns:
            x_vec:   (Nx,) numpy array of spatial coordinates.
            t_vec:   (Nt,) numpy array of time coordinates.
            u_exact: (Nt, Nx) numpy array — row i is the solution at t_vec[i].
        """
        import scipy.io
        mat = scipy.io.loadmat(self.data_path)
        x_vec = mat["x"].flatten()          # (Nx,)
        t_vec = mat["t"].flatten()          # (Nt,)
        u_exact = np.real(mat["usol"]).T    # (Nt, Nx)
        return x_vec, t_vec, u_exact

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    @property
    def bc_keys(self) -> List[str]:
        return ["bc_l", "bc_r"]

    @property
    def name(self) -> str:
        return f"Burgers (nu={self.nu:.6f})"
