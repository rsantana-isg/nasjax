"""
Pure-JAX subdomain network parameters and forward pass for FBPINNs.

Parameters are stored as *stacked* pytree arrays along the subdomain axis so
that a single ``jnp.einsum`` applies all ``m`` subdomain networks at once:

    params["layers"][i]["w"] has shape (m, n_out_i, n_in_i)
    params["layers"][i]["b"] has shape (m, n_out_i)

This avoids Python-level loops over subdomains during the forward pass and
makes the whole computation JIT-friendly.

Activation convention
---------------------
The hidden-layer activation is taken from the first hidden layer spec.  The
output layer is always linear (``activation=None``).  For SIREN, all hidden
layers use sinusoidal activations with SIREN initialisation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from .config import FBPINNConfig


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACT_FNS: Dict[Optional[str], Callable] = {
    "tanh":     jnp.tanh,
    "relu":     jax.nn.relu,
    "gelu":     jax.nn.gelu,
    "silu":     jax.nn.silu,
    "sin":      jnp.sin,
    "sigmoid":  jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    None:       lambda x: x,
}


def get_activation(name: Optional[str]) -> Callable:
    return _ACT_FNS.get(name, lambda x: x)


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def init_params(
    config: FBPINNConfig,
    key: jax.random.PRNGKey,
) -> Dict[str, Any]:
    """Initialise parameters for all *m* subdomain networks.

    Args:
        config:  :class:`~nasjax.fbpinns.config.FBPINNConfig`.
        key:     JAX PRNG key.

    Returns:
        Pytree dict ``{"layers": [{"w": (m, n_out, n_in), "b": (m, n_out)}, …]}``.
    """
    m = config.n_subdomains
    dims = config.full_layer_dims        # e.g. [2, 32, 32, 1]
    n_layers = len(dims) - 1
    layer_keys = jax.random.split(key, n_layers)

    layers = []
    for i, lk in enumerate(layer_keys):
        n_in, n_out = dims[i], dims[i + 1]
        sub_keys = jax.random.split(lk, m)

        siren = config.network_type == "SIREN"
        v = jnp.sqrt(6.0 / n_in) if siren else jnp.sqrt(1.0 / n_in)

        W = jax.vmap(
            lambda sk: jax.random.uniform(sk, (n_out, n_in), minval=-v, maxval=v)
        )(sub_keys)   # (m, n_out, n_in)
        b = jnp.zeros((m, n_out))
        layers.append({"w": W, "b": b})

    return {"layers": layers}


# ---------------------------------------------------------------------------
# Batched forward pass (all m subdomains at once)
# ---------------------------------------------------------------------------

def forward_all(
    params: Dict[str, Any],
    x_norm_all: jnp.ndarray,
    activation_fn: Callable,
) -> jnp.ndarray:
    """Forward pass for all m subdomains simultaneously.

    Args:
        params:       Parameter pytree from :func:`init_params`.
        x_norm_all:   Normalised inputs, shape ``(m, input_dim)``.
        activation_fn: Hidden-layer activation function.

    Returns:
        Network outputs, shape ``(m, output_dim)``.
    """
    z = x_norm_all                           # (m, input_dim)
    layers = params["layers"]
    n_hidden = len(layers) - 1

    for i, lp in enumerate(layers):
        W, b = lp["w"], lp["b"]             # (m, n_out, n_in), (m, n_out)
        z = jnp.einsum("moi,mi->mo", W, z) + b   # (m, n_out)
        if i < n_hidden:
            z = activation_fn(z)

    return z                                 # (m, output_dim)
