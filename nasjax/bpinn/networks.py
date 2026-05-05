"""
Bayesian MLP architectures for BPINN.

Two variants are provided:

:class:`BayesianMLPVanilla`
    Standard feed-forward Bayesian MLP.
    input → [hidden layers] → output

:class:`BayesianMLPEnhanced`
    Modified MLP with U/V residual (gating) connections
    (Wang et al., "Understanding and mitigating gradient flow pathologies",
    2021).  Reduces residual noise for stiff problems (e.g. Allen–Cahn).

    Architecture per hidden layer h:
        U  = σ(z @ W₁ + b₁)          # computed once from raw input z
        V  = σ(z @ W₂ + b₂)          # computed once from raw input z
        Hᵢ = σ(Hᵢ₋₁ @ Wᵢ + bᵢ)
        Hᵢ ← (1 − Hᵢ) ⊙ U + Hᵢ ⊙ V

    Requires all hidden layers to have the same width.

Common interface
----------------
Both classes expose:
    ``sample_prior_var_posterior(key, sample) → (params_list, log_prior, log_q)``
    ``__call__(z, params_list) → output``

Factory
-------
:func:`build_bayesian_mlp` returns the appropriate class based on
:attr:`BPINNConfig.enhanced_mlp`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .config import BPINNConfig, BPINNLayerSpec
from .layers import (
    BayesianLinear,
    _init_mu_rho,
    _sample,
    gaussian_log_prob_variational,
)
from .priors import build_prior


# ---------------------------------------------------------------------------
# Helper: build layers from specs
# ---------------------------------------------------------------------------

def _build_layers(
    input_dim: int,
    layer_specs: List[BPINNLayerSpec],
    keys: jnp.ndarray,
) -> Tuple[BayesianLinear, ...]:
    """Build a tuple of :class:`BayesianLinear` from layer specs."""
    layers: List[BayesianLinear] = []
    current_dim = input_dim
    for spec, k in zip(layer_specs, keys):
        layers.append(
            BayesianLinear(
                input_dim=current_dim,
                output_dim=spec.n_neurons,
                activation_name=spec.activation,
                prior_type=spec.prior_type,
                prior_params=spec.prior_params,
                key=k,
            )
        )
        current_dim = spec.n_neurons
    return tuple(layers)


# ---------------------------------------------------------------------------
# Vanilla Bayesian MLP
# ---------------------------------------------------------------------------

class BayesianMLPVanilla(eqx.Module):
    """Standard Bayesian MLP (no residual connections).

    Args:
        input_dim:   Dimension of the combined input vector ``z``.
        layer_specs: Per-layer specifications (see :class:`BPINNLayerSpec`).
        key:         JAX PRNG key.
    """

    layers: Tuple[BayesianLinear, ...]
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        layer_specs: List[BPINNLayerSpec],
        key: jax.random.PRNGKey,
    ) -> None:
        keys = jax.random.split(key, len(layer_specs))
        self.layers = _build_layers(input_dim, layer_specs, keys)
        self.n_layers = len(self.layers)

    # ------------------------------------------------------------------
    # Bayesian sampling
    # ------------------------------------------------------------------

    def sample_prior_var_posterior(
        self,
        key: jax.random.PRNGKey,
        sample: bool = True,
    ) -> Tuple[List[Dict], jnp.ndarray, jnp.ndarray]:
        """Sample all layer parameters and compute ELBO components.

        Args:
            key:    JAX PRNG key.
            sample: ``True`` for stochastic sample; ``False`` for mean.

        Returns:
            ``(params_list, total_log_prior, total_log_var_posterior)``
            where ``params_list[i]`` is ``{'weight': W, 'bias': b}``
            for layer ``i``.
        """
        keys = jax.random.split(key, self.n_layers)
        params_list: List[Dict] = []
        log_prior = jnp.array(0.0)
        log_q     = jnp.array(0.0)

        for layer, k in zip(self.layers, keys):
            p = layer.sample_params(k, sample)
            params_list.append(p)
            log_prior += layer.log_prior(p)
            log_q     += layer.log_var_posterior(p)

        return params_list, log_prior, log_q

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        z: jnp.ndarray,
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            z:           Combined/embedded input vector, shape ``(input_dim,)``.
            params_list: Sampled parameters from :meth:`sample_prior_var_posterior`.

        Returns:
            Output array, shape ``(output_dim,)``.
        """
        out = z
        for layer, params in zip(self.layers, params_list):
            out = layer(out, params)
        return out


# ---------------------------------------------------------------------------
# Enhanced Bayesian MLP (U/V gating)
# ---------------------------------------------------------------------------

class BayesianMLPEnhanced(eqx.Module):
    """Modified Bayesian MLP with U/V residual gating.

    The U and V projection weights also have variational parameters and are
    included in the ELBO computation.

    Requires all **hidden** layers to have the same width so that the gated
    sum ``(1−H)⊙U + H⊙V`` has a consistent shape.

    Args:
        input_dim:   Dimension of the combined input vector ``z``.
        layer_specs: Per-layer specifications.  All hidden layers (all but the
                     last) must have equal ``n_neurons``.
        key:         JAX PRNG key.
    """

    layers: Tuple[BayesianLinear, ...]
    n_layers: int = eqx.field(static=True)

    # U-gate variational parameters (W₁, b₁)
    w1_mu: jnp.ndarray
    w1_rho: jnp.ndarray
    b1_mu: jnp.ndarray
    b1_rho: jnp.ndarray

    # V-gate variational parameters (W₂, b₂)
    w2_mu: jnp.ndarray
    w2_rho: jnp.ndarray
    b2_mu: jnp.ndarray
    b2_rho: jnp.ndarray

    # Priors for gate weights (static)
    u_w_prior: object = eqx.field(static=True)
    u_b_prior: object = eqx.field(static=True)
    v_w_prior: object = eqx.field(static=True)
    v_b_prior: object = eqx.field(static=True)

    # Activation used for U and V (same as first hidden layer)
    gate_act: object = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        layer_specs: List[BPINNLayerSpec],
        key: jax.random.PRNGKey,
    ) -> None:
        # Split keys: n_layers for layers + 2 for U/V
        keys = jax.random.split(key, len(layer_specs) + 2)
        self.layers = _build_layers(input_dim, layer_specs, keys[: len(layer_specs)])
        self.n_layers = len(self.layers)

        # Gate activation mirrors the first hidden layer's activation
        self.gate_act = self.layers[0].activation

        # U/V gate shapes: input_dim → first_hidden_dim
        first_hidden_dim = layer_specs[0].n_neurons
        self.w1_mu, self.w1_rho, self.b1_mu, self.b1_rho = _init_mu_rho(
            input_dim, first_hidden_dim, keys[-2]
        )
        self.w2_mu, self.w2_rho, self.b2_mu, self.b2_rho = _init_mu_rho(
            input_dim, first_hidden_dim, keys[-1]
        )

        # Gate priors: reuse first hidden layer's prior type/params
        gate_prior_type   = layer_specs[0].prior_type
        gate_prior_params = layer_specs[0].prior_params
        self.u_w_prior = build_prior(gate_prior_type, gate_prior_params)
        self.u_b_prior = build_prior(gate_prior_type, gate_prior_params)
        self.v_w_prior = build_prior(gate_prior_type, gate_prior_params)
        self.v_b_prior = build_prior(gate_prior_type, gate_prior_params)

    # ------------------------------------------------------------------
    # Bayesian sampling
    # ------------------------------------------------------------------

    def sample_prior_var_posterior(
        self,
        key: jax.random.PRNGKey,
        sample: bool = True,
    ) -> Tuple[List[Dict], jnp.ndarray, jnp.ndarray]:
        """Sample all parameters (layers + U/V gates) and compute ELBO components.

        Gate samples are stored in ``params_list[0]`` under keys
        ``'w1'``, ``'b1'``, ``'w2'``, ``'b2'`` for the forward pass.

        Returns:
            ``(params_list, total_log_prior, total_log_var_posterior)``
        """
        k_layers, k_uv = jax.random.split(key)
        keys_layers = jax.random.split(k_layers, self.n_layers)
        k1, k2, k3, k4 = jax.random.split(k_uv, 4)

        # --- Layer parameters ---
        params_list: List[Dict] = []
        log_prior = jnp.array(0.0)
        log_q     = jnp.array(0.0)

        for layer, k in zip(self.layers, keys_layers):
            p = layer.sample_params(k, sample)
            params_list.append(p)
            log_prior += layer.log_prior(p)
            log_q     += layer.log_var_posterior(p)

        # --- U/V gate parameters ---
        if sample:
            w1 = _sample(self.w1_mu, self.w1_rho, k1)
            b1 = _sample(self.b1_mu, self.b1_rho, k2)
            w2 = _sample(self.w2_mu, self.w2_rho, k3)
            b2 = _sample(self.b2_mu, self.b2_rho, k4)
        else:
            w1, b1 = self.w1_mu, self.b1_mu
            w2, b2 = self.w2_mu, self.b2_mu

        # Attach gate params to first layer's dict (for forward pass access)
        params_list[0]["w1"] = w1
        params_list[0]["b1"] = b1
        params_list[0]["w2"] = w2
        params_list[0]["b2"] = b2

        # Gate log-prior and log-posterior
        log_prior += (
            self.u_w_prior.log_prob(w1) + self.u_b_prior.log_prob(b1)
            + self.v_w_prior.log_prob(w2) + self.v_b_prior.log_prob(b2)
        )
        log_q += (
            gaussian_log_prob_variational(w1, self.w1_mu, self.w1_rho)
            + gaussian_log_prob_variational(b1, self.b1_mu, self.b1_rho)
            + gaussian_log_prob_variational(w2, self.w2_mu, self.w2_rho)
            + gaussian_log_prob_variational(b2, self.b2_mu, self.b2_rho)
        )

        return params_list, log_prior, log_q

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        z: jnp.ndarray,
        params_list: List[Dict],
    ) -> jnp.ndarray:
        """Forward pass with U/V residual gating.

        Args:
            z:           Combined/embedded input vector.
            params_list: Sampled parameters from :meth:`sample_prior_var_posterior`.
                         ``params_list[0]`` must contain ``'w1'``, ``'b1'``,
                         ``'w2'``, ``'b2'`` (added by the sampler).

        Returns:
            Output array, shape ``(output_dim,)``.
        """
        w1 = params_list[0]["w1"]
        b1 = params_list[0]["b1"]
        w2 = params_list[0]["w2"]
        b2 = params_list[0]["b2"]

        U = self.gate_act(z @ w1 + b1)  # shape: (hidden_dim,)
        V = self.gate_act(z @ w2 + b2)  # shape: (hidden_dim,)

        out = z
        # Hidden layers (all but last)
        for layer, params in zip(self.layers[:-1], params_list[:-1]):
            H = layer(out, params)             # activation(out @ W + b)
            out = (1.0 - H) * U + H * V        # residual gating
        # Output layer (linear, no gating)
        out = self.layers[-1](out, params_list[-1])
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_bayesian_mlp(
    config: BPINNConfig,
    key: jax.random.PRNGKey,
) -> BayesianMLPVanilla | BayesianMLPEnhanced:
    """Build the Bayesian MLP described by *config*.

    Args:
        config: Full BPINN configuration.
        key:    JAX PRNG key for parameter initialisation.

    Returns:
        :class:`BayesianMLPEnhanced` when ``config.enhanced_mlp`` is ``True``,
        :class:`BayesianMLPVanilla` otherwise.

    Raises:
        ValueError: If ``enhanced_mlp=True`` and hidden-layer widths are
                    non-uniform (a requirement of the gating mechanism).
    """
    if config.enhanced_mlp:
        hidden_widths = [s.n_neurons for s in config.hidden_layer_specs]
        if len(set(hidden_widths)) > 1:
            raise ValueError(
                f"Enhanced MLP requires all hidden layers to have the same "
                f"width, but got {hidden_widths}. "
                f"Set enhanced_mlp=False or make hidden widths uniform."
            )
        return BayesianMLPEnhanced(config.effective_input_dim, config.layer_specs, key)
    return BayesianMLPVanilla(config.effective_input_dim, config.layer_specs, key)
