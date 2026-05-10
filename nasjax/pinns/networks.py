"""
Equinox-based MLP networks for standard PINNs.

Supported network types
-----------------------
FCN          Standard fully-connected MLP.
AdaptiveFCN  FCN with per-neuron trainable activation scale (Jagtap et al.).
SIREN        FCN with sinusoidal activations and SIREN weight initialisation.
FourierFCN   FCN with a fixed random Fourier feature embedding at the input.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .config import PINNConfig


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACT_FNS = {
    "tanh":     jnp.tanh,
    "relu":     jax.nn.relu,
    "gelu":     jax.nn.gelu,
    "silu":     jax.nn.silu,
    "sin":      jnp.sin,
    "sigmoid":  jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    None:       lambda x: x,
}


# ---------------------------------------------------------------------------
# Single linear layer with optional adaptive scale
# ---------------------------------------------------------------------------

class PINNLinear(eqx.Module):
    """Linear layer + activation for PINN networks.

    Args:
        input_dim:   Number of input features.
        output_dim:  Number of output features.
        activation:  Activation string or ``None`` (linear).
        adaptive:    If ``True``, adds a per-neuron trainable scale (AdaptiveFCN).
        siren_init:  If ``True``, uses SIREN weight initialisation (√6/n range).
        key:         JAX PRNG key.
    """
    weight: jnp.ndarray
    bias: jnp.ndarray
    activation: Any = eqx.field(static=True)
    adaptive_scale: Optional[jnp.ndarray]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[str],
        adaptive: bool = False,
        siren_init: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        w_key, b_key = jax.random.split(key)

        if siren_init:
            v = jnp.sqrt(6.0 / input_dim)
        else:
            v = jnp.sqrt(1.0 / input_dim)

        self.weight = jax.random.uniform(w_key, (output_dim, input_dim), minval=-v, maxval=v)
        self.bias = jax.random.uniform(b_key, (output_dim,), minval=-v, maxval=v)
        self.activation = _ACT_FNS.get(activation, lambda x: x)
        self.adaptive_scale = jnp.ones(output_dim) if adaptive else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.weight @ x + self.bias
        if self.adaptive_scale is not None:
            # Adaptive activation: a * tanh(h/a)  (generalises to any act)
            h = self.adaptive_scale * self.activation(h / self.adaptive_scale)
        else:
            h = self.activation(h)
        return h


# ---------------------------------------------------------------------------
# Full PINN network
# ---------------------------------------------------------------------------

class PINNNetwork(eqx.Module):
    """PINN MLP with optional Fourier feature input embedding.

    Args:
        config:  :class:`~nasjax.pinns.config.PINNConfig` instance.
        key:     JAX PRNG key.
    """
    layers: Tuple
    fourier_omega: Optional[jnp.ndarray]  # fixed random frequencies (FourierFCN)
    use_fourier: bool = eqx.field(static=True)

    def __init__(self, config: PINNConfig, key: jax.random.PRNGKey):
        n_layers = len(config.layer_specs)
        keys = jax.random.split(key, n_layers + 2)

        self.use_fourier = (
            config.network_type == "FourierFCN" and config.fourier_input.enabled
        )

        if self.use_fourier:
            fi = config.fourier_input
            self.fourier_omega = 2.0 * jnp.pi * (
                fi.mu + fi.sd * jax.random.normal(keys[0], (fi.n_features, config.input_dim))
            )
        else:
            self.fourier_omega = None

        adaptive = config.network_type == "AdaptiveFCN"
        siren = config.network_type == "SIREN"
        dims = config.full_layer_dims  # includes effective_input_dim at index 0

        layer_list = []
        for i, spec in enumerate(config.layer_specs):
            layer_list.append(PINNLinear(
                input_dim=dims[i],
                output_dim=spec.n_neurons,
                activation=spec.activation,
                adaptive=adaptive,
                siren_init=siren,
                key=keys[i + 1],
            ))
        self.layers = tuple(layer_list)

    def _embed(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_fourier:
            z = self.fourier_omega @ x
            return jnp.concatenate([jnp.sin(z), jnp.cos(z)])
        return x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = self._embed(x)
        for layer in self.layers:
            z = layer(z)
        return z
