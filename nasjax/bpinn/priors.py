"""
Prior distributions for Bayesian neural network weights and biases.

All classes implement a single method:

    log_prob(inputs: jnp.ndarray) -> jnp.ndarray (scalar)

They are plain Python objects — **not** ``eqx.Module`` — because their
hyper-parameters are fixed before JAX tracing.  They are stored as
``eqx.field(static=True)`` inside Bayesian layers so JAX never tries to
differentiate through them.

Available priors
----------------
- :class:`GaussianPrior`           – N(μ, σ²)
- :class:`LaplacePrior`            – Laplace(μ, β)
- :class:`ScaleMixtureGaussianPrior` – Spike-and-Slab: π·N(0,σ₁²) + (1−π)·N(0,σ₂²)

Factory
-------
- :func:`build_prior` – create from ``(prior_type, prior_params)`` strings/dicts.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Gaussian prior
# ---------------------------------------------------------------------------

class GaussianPrior:
    """Isotropic Gaussian prior: p(w) = N(w; μ, σ²).

    Args:
        mean:  Prior mean μ (default 0).
        sigma: Prior standard deviation σ (default 1).
    """

    def __init__(self, mean: float = 0.0, sigma: float = 1.0) -> None:
        self.mean = float(mean)
        self.sigma = float(sigma)

    def log_prob(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """log p(w) = Σ [ -½log(2πσ²) - (w−μ)²/(2σ²) ]."""
        log_prob_per_elem = (
            -0.5 * jnp.log(2.0 * jnp.pi)
            - jnp.log(self.sigma)
            - 0.5 * jnp.square((inputs - self.mean) / self.sigma)
        )
        return jnp.sum(log_prob_per_elem)

    def __repr__(self) -> str:
        return f"GaussianPrior(mean={self.mean}, sigma={self.sigma})"


# ---------------------------------------------------------------------------
# Laplace prior
# ---------------------------------------------------------------------------

class LaplacePrior:
    """Laplace (double-exponential) prior: p(w) = Laplace(w; μ, β).

    Induces sparsity (heavier tails than Gaussian, lighter than SnS slab).

    Args:
        mean: Prior mean μ (default 0).
        beta: Scale parameter β (default 1).
    """

    def __init__(self, mean: float = 0.0, beta: float = 1.0) -> None:
        self.mean = float(mean)
        self.beta = float(beta)

    def log_prob(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """log p(w) = Σ [ -log(2β) - |w−μ|/β ]."""
        log_prob_per_elem = (
            -jnp.log(2.0 * self.beta)
            - jnp.abs(inputs - self.mean) / self.beta
        )
        return jnp.sum(log_prob_per_elem)

    def __repr__(self) -> str:
        return f"LaplacePrior(mean={self.mean}, beta={self.beta})"


# ---------------------------------------------------------------------------
# Spike-and-Slab prior
# ---------------------------------------------------------------------------

class ScaleMixtureGaussianPrior:
    """Spike-and-Slab prior (scale-mixture Gaussian).

    p(w) = π·N(w; 0, σ_spike²) + (1−π)·N(w; 0, σ_slab²)

    The "spike" component (small σ) encourages weights near zero.
    The "slab" component (large σ) allows large weights when needed.

    Args:
        pi:          Mixture weight for the spike component (0 < π < 1).
        sigma_spike: Standard deviation of the spike (small, e.g. exp(-6)).
        sigma_slab:  Standard deviation of the slab  (large, e.g. exp(0)=1).
    """

    def __init__(
        self,
        pi: float = 0.5,
        sigma_spike: float | None = None,
        sigma_slab: float | None = None,
    ) -> None:
        self.pi = float(pi)
        self.sigma_spike = float(sigma_spike) if sigma_spike is not None else math.exp(-6.0)
        self.sigma_slab  = float(sigma_slab)  if sigma_slab  is not None else math.exp(0.0)

    def _log_normal(self, inputs: jnp.ndarray, sigma: float) -> jnp.ndarray:
        """Element-wise log N(w; 0, σ²)."""
        return (
            -0.5 * jnp.log(2.0 * jnp.pi * sigma ** 2)
            - jnp.square(inputs) / (2.0 * sigma ** 2)
        )

    def log_prob(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """log p(w) = Σ log [ π·p_spike(w) + (1−π)·p_slab(w) ]."""
        p_spike = jnp.exp(self._log_normal(inputs, self.sigma_spike))
        p_slab  = jnp.exp(self._log_normal(inputs, self.sigma_slab))
        log_prob_per_elem = jnp.log(self.pi * p_spike + (1.0 - self.pi) * p_slab)
        return jnp.sum(log_prob_per_elem)

    def __repr__(self) -> str:
        return (
            f"ScaleMixtureGaussianPrior("
            f"pi={self.pi}, "
            f"sigma_spike={self.sigma_spike:.4g}, "
            f"sigma_slab={self.sigma_slab:.4g})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_prior(
    prior_type: str,
    prior_params: Dict[str, Any],
) -> GaussianPrior | LaplacePrior | ScaleMixtureGaussianPrior:
    """Build a prior object from a type string and hyper-parameter dict.

    Args:
        prior_type:   ``'Gaussian'``, ``'Laplace'``, or ``'SnS'``.
        prior_params: Dict of hyper-parameters appropriate for *prior_type*:
                      - Gaussian: ``{'mean': float, 'sigma': float}``
                      - Laplace:  ``{'mean': float, 'beta': float}``
                      - SnS:      ``{'pi': float, 'sigma_spike': float,
                                     'sigma_slab': float}``

    Returns:
        A prior instance with a :meth:`log_prob` method.

    Raises:
        ValueError: For unknown prior types.
    """
    if prior_type == "Gaussian":
        return GaussianPrior(
            mean=prior_params.get("mean", 0.0),
            sigma=prior_params.get("sigma", 1.0),
        )
    if prior_type == "Laplace":
        return LaplacePrior(
            mean=prior_params.get("mean", 0.0),
            beta=prior_params.get("beta", 1.0),
        )
    if prior_type == "SnS":
        return ScaleMixtureGaussianPrior(
            pi=prior_params.get("pi", 0.5),
            sigma_spike=prior_params.get("sigma_spike", None),
            sigma_slab=prior_params.get("sigma_slab", None),
        )
    raise ValueError(
        f"Unknown prior type '{prior_type}'. "
        f"Choose from 'Gaussian', 'Laplace', or 'SnS'."
    )
