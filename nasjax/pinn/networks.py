"""PINN Network implementation with automatic differentiation.

This module provides a wrapper around MLP networks that computes
derivatives needed for physics-informed loss functions.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional

from ..networks.mlp import MLP
from ..descriptors.mlp import MLPDescriptor


class PINNNetwork(eqx.Module):
    """Physics-Informed Neural Network with automatic differentiation.

    This class wraps an MLP network and provides methods to compute
    derivatives using JAX's automatic differentiation.

    Attributes:
        mlp: Underlying MLP network

    Example:
        >>> descriptor = MLPDescriptor.random_init(2, 1, 3, 32, key)
        >>> pinn = PINNNetwork(descriptor, key)
        >>> coords = jnp.array([[0.5, 1.0]])  # (x, t)
        >>> u, u_x, u_t, u_xx = pinn(coords)  # Get solution and derivatives
    """

    mlp: MLP

    def __init__(self, descriptor: MLPDescriptor, key: jax.random.PRNGKey):
        """Initialize PINN network from descriptor.

        Args:
            descriptor: Network architecture descriptor
            key: JAX random key for initialization
        """
        self.mlp = MLP(descriptor, key)

    def __call__(
        self,
        coords: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None,
        inference: bool = True,
        compute_derivatives: bool = True
    ) -> Tuple[jnp.ndarray, ...]:
        """Forward pass with optional derivative computation.

        Args:
            coords: Input coordinates of shape (n_dims,) for single point
                   or (batch_size, n_dims) for batch
            key: Optional random key for dropout
            inference: Whether to run in inference mode
            compute_derivatives: Whether to compute derivatives

        Returns:
            If compute_derivatives=True:
                For 1D problems (x only): (u, u_x, u_xx)
                For 2D problems (x, t): (u, u_x, u_t, u_xx, u_tt, u_xt)
            If compute_derivatives=False:
                u only

        Example:
            >>> # Single point
            >>> coords = jnp.array([0.5, 1.0])
            >>> u, u_x, u_t, u_xx = pinn(coords)

            >>> # Batch (use vmap)
            >>> coords_batch = jnp.array([[0.5, 1.0], [0.6, 1.1]])
            >>> results = jax.vmap(pinn)(coords_batch)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Simple forward pass
        if not compute_derivatives:
            return self.mlp(coords, key, inference=inference)

        # Determine input dimension
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        n_dims = coords.shape[-1]

        # Forward pass to get solution
        u = self.mlp(coords, key, inference=inference)

        # Compute first derivatives
        if n_dims == 1:
            # 1D problem: only x
            u_x = self._compute_derivative(coords, key, inference, dim=0)
            u_xx = self._compute_second_derivative(coords, key, inference, dim=0)
            return u, u_x, u_xx

        elif n_dims == 2:
            # 2D problem: x and t
            u_x = self._compute_derivative(coords, key, inference, dim=0)
            u_t = self._compute_derivative(coords, key, inference, dim=1)
            u_xx = self._compute_second_derivative(coords, key, inference, dim=0)
            u_tt = self._compute_second_derivative(coords, key, inference, dim=1)
            u_xt = self._compute_mixed_derivative(coords, key, inference, dim1=0, dim2=1)
            return u, u_x, u_t, u_xx, u_tt, u_xt

        elif n_dims == 3:
            # 3D problem: x, y, and t
            u_x = self._compute_derivative(coords, key, inference, dim=0)
            u_y = self._compute_derivative(coords, key, inference, dim=1)
            u_t = self._compute_derivative(coords, key, inference, dim=2)
            u_xx = self._compute_second_derivative(coords, key, inference, dim=0)
            u_yy = self._compute_second_derivative(coords, key, inference, dim=1)
            u_tt = self._compute_second_derivative(coords, key, inference, dim=2)
            return u, u_x, u_y, u_t, u_xx, u_yy, u_tt

        else:
            raise ValueError(f"Unsupported input dimension: {n_dims}")

    def _compute_derivative(
        self,
        coords: jnp.ndarray,
        key: jax.random.PRNGKey,
        inference: bool,
        dim: int
    ) -> jnp.ndarray:
        """Compute first derivative ∂u/∂x_i using autodiff.

        Args:
            coords: Input coordinates
            key: Random key
            inference: Inference mode flag
            dim: Dimension to differentiate w.r.t. (0=x, 1=t, etc.)

        Returns:
            First derivative ∂u/∂x_i
        """
        def forward_fn(coords):
            return self.mlp(coords, key, inference=inference)

        # Compute gradient
        grad_fn = jax.grad(lambda c: jnp.sum(forward_fn(c)))
        gradient = grad_fn(coords)

        return gradient[..., dim:dim+1]

    def _compute_second_derivative(
        self,
        coords: jnp.ndarray,
        key: jax.random.PRNGKey,
        inference: bool,
        dim: int
    ) -> jnp.ndarray:
        """Compute second derivative ∂²u/∂x_i² using autodiff.

        Args:
            coords: Input coordinates
            key: Random key
            inference: Inference mode flag
            dim: Dimension to differentiate w.r.t. (0=x, 1=t, etc.)

        Returns:
            Second derivative ∂²u/∂x_i²
        """
        def forward_fn(coords):
            return self.mlp(coords, key, inference=inference)

        # Compute Hessian
        hess_fn = jax.hessian(lambda c: jnp.sum(forward_fn(c)))
        hessian = hess_fn(coords)

        return hessian[..., dim, dim:dim+1]

    def _compute_mixed_derivative(
        self,
        coords: jnp.ndarray,
        key: jax.random.PRNGKey,
        inference: bool,
        dim1: int,
        dim2: int
    ) -> jnp.ndarray:
        """Compute mixed derivative ∂²u/∂x_i∂x_j using autodiff.

        Args:
            coords: Input coordinates
            key: Random key
            inference: Inference mode flag
            dim1: First dimension
            dim2: Second dimension

        Returns:
            Mixed derivative ∂²u/∂x_i∂x_j
        """
        def forward_fn(coords):
            return self.mlp(coords, key, inference=inference)

        # Compute Hessian
        hess_fn = jax.hessian(lambda c: jnp.sum(forward_fn(c)))
        hessian = hess_fn(coords)

        return hessian[..., dim1, dim2:dim2+1]

    def count_parameters(self) -> int:
        """Count total number of parameters in the network.

        Returns:
            Number of trainable parameters
        """
        return self.mlp.count_parameters()
