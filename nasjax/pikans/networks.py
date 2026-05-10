"""
Custom PIKAN networks built from jaxkan layers (flax.nnx).

``PIKANNetwork`` is an ``nnx.Module`` that stacks heterogeneous KAN layers
as specified by ``PIKANConfig.layer_specs``.  Each layer may be a different
KAN type (Chebyshev, Fourier, Legendre, RBF, Sine, Spline).

Convention
----------
Inputs are ordered ``(t, x)`` following the jaxkan convention used in
``jaxkan.pikan.pde.get_burgers_res``.  The PDE collocation format expected by
:meth:`PIKANNetwork.__call__` is therefore ``(N, 2)`` with column 0 = t,
column 1 = x.
"""

from __future__ import annotations

from typing import List

from flax import nnx
from jaxkan.layers import (
    ChebyshevLayer,
    FourierLayer,
    LegendreLayer,
    RBFLayer,
    SineLayer,
    SplineLayer,
)

from .config import PIKANConfig, PIKANLayerSpec


# ---------------------------------------------------------------------------
# Layer type → jaxkan class mapping
# ---------------------------------------------------------------------------

_LAYER_CLS = {
    "chebyshev": ChebyshevLayer,
    "legendre":  LegendreLayer,
    "fourier":   FourierLayer,
    "rbf":       RBFLayer,
    "sine":      SineLayer,
    "spline":    SplineLayer,
}


def _build_layer(n_in: int, spec: PIKANLayerSpec, seed: int) -> nnx.Module:
    """Instantiate a single jaxkan layer from a :class:`PIKANLayerSpec`."""
    cls = _LAYER_CLS.get(spec.layer_type)
    if cls is None:
        raise ValueError(f"Unknown layer_type: '{spec.layer_type}'")
    params = dict(spec.layer_params)
    return cls(n_in=n_in, n_out=spec.n_out, seed=seed, **params)


# ---------------------------------------------------------------------------
# PIKAN network
# ---------------------------------------------------------------------------

class PIKANNetwork(nnx.Module):
    """Heterogeneous KAN network for PDE solving.

    Layers are built from ``config.layer_specs``; each spec may use a
    different layer type and parameter set.  The last layer always produces
    a scalar output (1 neuron) with no nonlinear post-processing.

    Args:
        config:  :class:`~nasjax.pikans.config.PIKANConfig`.
    """

    def __init__(self, config: PIKANConfig) -> None:
        dims = config.full_layer_dims  # [input_dim, n_out_0, n_out_1, …]
        layers = []
        for i, spec in enumerate(config.layer_specs):
            layers.append(_build_layer(dims[i], spec, seed=config.seed + i))
        self.layers = nnx.List(layers)

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input array of shape ``(N, input_dim)`` — column 0 is ``t``,
               column 1 is ``x`` for 2-D problems.

        Returns:
            Output array of shape ``(N, output_dim)``.
        """
        z = x
        for layer in self.layers:
            z = layer(z)
        return z
