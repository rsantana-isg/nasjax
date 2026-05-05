"""
Abstract base class for PDE problems used with BPINN.

A :class:`PDEProblem` encapsulates everything that is equation-specific:

1. **Data preparation** — collocation points, boundary/IC targets, and
   reference data for validation.
2. **Residual computation** — equation, initial-condition, and
   boundary-condition residuals.

The BPINN trainer (:mod:`nasjax.bpinn.trainer`) calls these methods without
knowing the underlying equation, making it easy to add new PDEs by subclassing.

To add a new PDE:
    1. Subclass :class:`PDEProblem`.
    2. Implement all abstract methods.
    3. Pass an instance to :func:`~nasjax.bpinn.trainer.train_bpinn`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import jax.numpy as jnp


class PDEProblem(ABC):
    """Interface for a PDE problem in the BPINN framework.

    All methods that will be called inside JAX-traced code (residuals) must
    be pure functions compatible with ``jit`` and ``vmap``.

    Abstract methods
    ----------------
    prepare_data       : build collocation data dict (Python-level, pre-JIT)
    pde_residual       : PDE interior residual
    ic_residual        : initial-condition residual
    bc_residuals       : dict of boundary-condition residuals
    bc_keys            : list of boundary names

    Optional override
    -----------------
    sigma_keys         : match sigma dict keys to bc_keys (default: bc_keys)
    name               : human-readable problem name
    """

    # -----------------------------------------------------------------------
    # Data preparation (Python-level, runs before JAX tracing)
    # -----------------------------------------------------------------------

    @abstractmethod
    def prepare_data(
        self,
        key: Any,
        config: Any,          # BPINNConfig — typed loosely to avoid circular import
    ) -> Dict[str, jnp.ndarray]:
        """Build collocation and reference data.

        Called **once** before training starts.  The returned dict is passed
        to the training loop as ``batch_data``.

        Mandatory keys in the returned dict:
            ``'x_pde'``, ``'t_pde'``      — PDE interior collocation points
            ``'x_ic'``,  ``'t_ic'``, ``'u_ic'`` — IC points + targets
            For each key ``k`` in :attr:`bc_keys`:
                ``'x_{k}'``, ``'t_{k}'``, ``'u_{k}'``

        Optional keys:
            ``'x_data'``, ``'t_data'``, ``'u_data'`` — observed/validation data

        Args:
            key:    JAX PRNG key for random collocation sampling.
            config: :class:`~nasjax.bpinn.config.BPINNConfig` instance.

        Returns:
            Dict of JAX arrays.
        """
        ...

    # -----------------------------------------------------------------------
    # Residuals (JAX-traced — must be pure, JIT- and vmap-compatible)
    # -----------------------------------------------------------------------

    @abstractmethod
    def pde_residual(
        self,
        model: Any,           # BPINNModel
        x: jnp.ndarray,       # shape (N, 1)
        t: jnp.ndarray,       # shape (N, 1)
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """Compute the PDE residual at interior collocation points.

        The residual is zero when the PDE is exactly satisfied.

        Args:
            model:       :class:`~nasjax.bpinn.model.BPINNModel` instance.
            x:           Spatial coordinates, shape ``(N, 1)``.
            t:           Time coordinates, shape ``(N, 1)``.
            params_list: Sampled weight/bias params from the Bayesian MLP.

        Returns:
            Residual array, shape ``(N, 1)``.
        """
        ...

    @abstractmethod
    def ic_residual(
        self,
        model: Any,
        x: jnp.ndarray,       # shape (N, 1)
        t: jnp.ndarray,       # shape (N, 1)  — should be zeros
        u_target: jnp.ndarray, # shape (N, 1)
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """Compute the initial-condition residual  u(x, 0) − u_IC(x).

        Args:
            model:       :class:`~nasjax.bpinn.model.BPINNModel`.
            x:           Spatial coordinates at t=0.
            t:           Time coordinates (all zero).
            u_target:    Target IC values.
            params_list: Sampled parameters.

        Returns:
            Residual array, shape ``(N, 1)``.
        """
        ...

    @abstractmethod
    def bc_residuals(
        self,
        model: Any,
        batch_data: Dict[str, jnp.ndarray],
        params_list: List[Dict],
    ) -> Dict[str, jnp.ndarray]:
        """Compute residuals for all boundary conditions.

        The keys of the returned dict must match :attr:`bc_keys`.

        Args:
            model:       :class:`~nasjax.bpinn.model.BPINNModel`.
            batch_data:  Full data dict from :meth:`prepare_data`.
            params_list: Sampled parameters.

        Returns:
            ``{bc_key: residual_array}`` where each array has shape ``(N, 1)``.
        """
        ...

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    @property
    @abstractmethod
    def bc_keys(self) -> List[str]:
        """List of boundary-condition names, e.g. ``['bc_l', 'bc_r']``.

        These strings are used to:
        - look up collocation data in batch_data (``'x_{key}'``, ``'t_{key}'``,
          ``'u_{key}'``),
        - look up σ values in the sigma dict (via :attr:`sigma_keys`),
        - look up λ values in the lambda dicts.
        """
        ...

    @property
    def sigma_keys(self) -> List[str]:
        """Map bc_keys to sigma dict keys.

        Override this if your sigma dict uses different key names than bc_keys.
        Default: same as :attr:`bc_keys`.
        """
        return list(self.bc_keys)

    @property
    def name(self) -> str:
        """Human-readable problem name."""
        return self.__class__.__name__
