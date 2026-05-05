"""
Bayesian linear layer for BPINN.

The :class:`BayesianLinear` layer holds variational parameters (μ, ρ) for its
weights and biases.  During a forward pass the parameters are sampled via the
reparameterisation trick

    w = μ + softplus(ρ) · ε,   ε ~ N(0, I)

so that gradients flow through μ and ρ without differentiating through
sampling.

JAX / Equinox compatibility
----------------------------
- Extends ``eqx.Module`` — fully compatible with ``jit``, ``vmap``, and
  ``filter_value_and_grad``.
- Prior objects are stored as ``eqx.field(static=True)`` — JAX never traces
  through them and they carry no differentiable leaves.
- The activation function is also static — shape inference happens at Python
  level; JAX only traces the actual numeric computation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .priors import build_prior


# ---------------------------------------------------------------------------
# Reparameterisation helpers (pure functions, JIT-friendly)
# ---------------------------------------------------------------------------

def _softplus(rho: jnp.ndarray) -> jnp.ndarray:
    """σ = log(1 + exp(ρ)) — maps ℝ → ℝ⁺."""
    return jnp.log1p(jnp.exp(rho))


def _sample(
    mu: jnp.ndarray,
    rho: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Reparameterisation sample: w = μ + σ·ε,  ε ~ N(0, I)."""
    eps = jax.random.normal(key, shape=mu.shape, dtype=mu.dtype)
    return mu + _softplus(rho) * eps


def gaussian_log_prob_variational(
    samples: jnp.ndarray,
    mu: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Scalar  log q(w | μ, ρ) = Σᵢ log N(wᵢ; μᵢ, σᵢ²)."""
    sigma = _softplus(rho)
    log_prob = (
        -0.5 * jnp.log(2.0 * jnp.pi * jnp.square(sigma))
        - jnp.square(samples - mu) / (2.0 * jnp.square(sigma))
    )
    return jnp.sum(log_prob)


# ---------------------------------------------------------------------------
# Weight / bias initialisation
# ---------------------------------------------------------------------------

def _init_mu_rho(
    in_dim: int,
    out_dim: int,
    key: jax.random.PRNGKey,
    mu_range: Tuple[float, float] = (-0.2, 0.2),
    rho_range: Tuple[float, float] = (-5.0, -4.0),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(weight_mu, weight_rho, bias_mu, bias_rho)`` with uniform init.

    The ρ initialisation in ``rho_range ≈ [-5, -4]`` gives initial σ ≈ 0.007–0.018
    (very small), so the first-epoch behaviour is close to a deterministic
    network with weights from ``mu_range``.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w_shape = (in_dim, out_dim)
    b_shape = (out_dim,)
    w_mu  = jax.random.uniform(k1, w_shape, minval=mu_range[0],  maxval=mu_range[1])
    w_rho = jax.random.uniform(k2, w_shape, minval=rho_range[0], maxval=rho_range[1])
    b_mu  = jax.random.uniform(k3, b_shape, minval=mu_range[0],  maxval=mu_range[1])
    b_rho = jax.random.uniform(k4, b_shape, minval=rho_range[0], maxval=rho_range[1])
    return w_mu, w_rho, b_mu, b_rho


# ---------------------------------------------------------------------------
# BayesianLinear
# ---------------------------------------------------------------------------

class BayesianLinear(eqx.Module):
    """Single Bayesian affine layer with configurable prior and activation.

    Trainable leaves (differentiated by JAX):
        ``weight_mu``, ``weight_rho``, ``bias_mu``, ``bias_rho``

    Static fields (not traced, not differentiated):
        ``input_dim``, ``output_dim``, ``activation``,
        ``weight_prior``, ``bias_prior``

    Args:
        input_dim:       Incoming feature dimension.
        output_dim:      Outgoing feature dimension.
        activation_name: Name of the activation function (e.g. ``'tanh'``),
                         or ``None`` for a linear layer.
        prior_type:      ``'Gaussian'``, ``'Laplace'``, or ``'SnS'``.
        prior_params:    Hyper-parameters for the chosen prior.
        key:             JAX PRNG key for parameter initialisation.
    """

    # --- Trainable variational parameters ---
    weight_mu: jnp.ndarray
    weight_rho: jnp.ndarray
    bias_mu: jnp.ndarray
    bias_rho: jnp.ndarray

    # --- Static metadata (not traced) ---
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    activation: object = eqx.field(static=True)   # callable
    weight_prior: object = eqx.field(static=True)  # prior object
    bias_prior: object = eqx.field(static=True)    # prior object

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_name: Optional[str],
        prior_type: str,
        prior_params: dict,
        key: jax.random.PRNGKey,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Resolve activation
        if activation_name is not None:
            self.activation = getattr(jax.nn, activation_name, lambda x: x)
        else:
            self.activation = lambda x: x  # identity (linear output)

        # Prior objects (shared weight/bias prior per layer)
        self.weight_prior = build_prior(prior_type, prior_params)
        self.bias_prior   = build_prior(prior_type, prior_params)

        # Variational parameters
        self.weight_mu, self.weight_rho, self.bias_mu, self.bias_rho = _init_mu_rho(
            input_dim, output_dim, key
        )

    # -----------------------------------------------------------------------
    # Bayesian computations
    # -----------------------------------------------------------------------

    def sample_params(
        self,
        key: jax.random.PRNGKey,
        sample: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Draw (or return mean) weight/bias samples.

        Args:
            key:    JAX PRNG key.
            sample: ``True`` → stochastic sample; ``False`` → posterior mean.

        Returns:
            Dict ``{'weight': W, 'bias': b}``.
        """
        if sample:
            k_w, k_b = jax.random.split(key)
            weight = _sample(self.weight_mu, self.weight_rho, k_w)
            bias   = _sample(self.bias_mu,   self.bias_rho,   k_b)
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return {"weight": weight, "bias": bias}

    def log_var_posterior(
        self, params: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Scalar  log q(w, b | μ, ρ)  for the given sampled params."""
        return (
            gaussian_log_prob_variational(params["weight"], self.weight_mu, self.weight_rho)
            + gaussian_log_prob_variational(params["bias"],   self.bias_mu,   self.bias_rho)
        )

    def log_prior(
        self, params: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Scalar  log p(w, b)  under the configured prior."""
        return (
            self.weight_prior.log_prob(params["weight"])
            + self.bias_prior.log_prob(params["bias"])
        )

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def __call__(
        self,
        inputs: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Apply affine transform + activation.

        Args:
            inputs: Array of shape ``(..., input_dim)``.
            params: Dict from :meth:`sample_params`.

        Returns:
            Array of shape ``(..., output_dim)``.
        """
        pre_act = inputs @ params["weight"] + params["bias"]
        return self.activation(pre_act)
