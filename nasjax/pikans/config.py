"""
Configuration dataclasses for Physics-Informed Kolmogorov-Arnold Networks (PIKANs).

Each layer is configured independently via a ``PIKANLayerSpec`` that specifies
the output width, the KAN layer type, and the type-specific parameters. This
allows fine-grained mixed architectures (e.g. Chebyshev + Fourier layers).

Supported layer types and their required parameters
---------------------------------------------------
chebyshev   D (degree), flavor ('default'/'modified'/'exact'), add_bias
legendre    Same as chebyshev
fourier     D (order), smooth_init (bool), add_bias
rbf         D (num basis functions), kernel (dict), grid_range, add_bias
sine        D (num basis functions), add_bias
spline      k (spline order), G (grid size), grid_range, add_bias

All layer types also accept:
    residual         (flax.nnx.Module or None)
    external_weights (bool)
    init_scheme      (dict or None)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nasjax.bpinn.config import CollocationConfig


LAYER_TYPES = {"chebyshev", "legendre", "fourier", "rbf", "sine", "spline"}
OPTIMISERS = {"adam", "adamw", "sgd"}


# ---------------------------------------------------------------------------
# Per-layer spec
# ---------------------------------------------------------------------------

@dataclass
class PIKANLayerSpec:
    """Specification for a single KAN layer.

    Args:
        n_out:       Output width (number of neurons in this layer).
        layer_type:  KAN layer type string (see module docstring).
        layer_params: Type-specific parameters passed to the jaxkan constructor.
    """
    n_out: int
    layer_type: str
    layer_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_list(cls, spec) -> "PIKANLayerSpec":
        """Create from ``[n_out, layer_type, layer_params_dict]``."""
        return cls(
            n_out=int(spec[0]),
            layer_type=str(spec[1]),
            layer_params=dict(spec[2]) if len(spec) > 2 else {},
        )

    def to_list(self) -> list:
        return [self.n_out, self.layer_type, self.layer_params]

    def to_dict(self) -> dict:
        return {"n_out": self.n_out, "layer_type": self.layer_type,
                "layer_params": self.layer_params}


def make_layer_specs(
    hidden_widths: List[int],
    output_dim: int = 1,
    layer_type: str = "chebyshev",
    layer_params: Optional[Dict[str, Any]] = None,
) -> List[PIKANLayerSpec]:
    """Convenience constructor for uniform KAN stacks."""
    if layer_params is None:
        layer_params = {"D": 5}
    specs = [
        PIKANLayerSpec(n_out=w, layer_type=layer_type, layer_params=dict(layer_params))
        for w in hidden_widths
    ]
    specs.append(PIKANLayerSpec(n_out=output_dim, layer_type=layer_type,
                                layer_params=dict(layer_params)))
    return specs


# ---------------------------------------------------------------------------
# Global lambda config (shared semantics with PINN/BPINN)
# ---------------------------------------------------------------------------

@dataclass
class GlobalLambdaConfig:
    """Global loss-term weights."""
    lambda_pde: float = 1.0
    lambda_ic: float = 100.0
    lambda_bc_l: float = 1.0
    lambda_bc_r: float = 1.0

    def to_dict(self) -> dict:
        return {"lambda_pde": self.lambda_pde, "lambda_ic": self.lambda_ic,
                "lambda_bc_l": self.lambda_bc_l, "lambda_bc_r": self.lambda_bc_r}

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalLambdaConfig":
        return cls(**d)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class PIKANConfig:
    """Full configuration for a PIKAN.

    The layer_specs use ``[n_out, layer_type, layer_params]`` lists, allowing
    per-layer heterogeneous KAN configurations.  The first field ``n_out``
    plays the role of ``n_neurons`` in PINNLayerSpec, enabling NAS search
    algorithms to treat width as a shared searchable dimension.

    Args:
        input_dim:      Input dimension (2 for Burgers: t, x in jaxkan order).
        layer_specs:    List of ``PIKANLayerSpec`` or raw lists.
        n_epochs:       Number of training epochs.
        learning_rate:  Optimiser learning rate.
        optimiser_name: ``'adam'``, ``'adamw'``, or ``'sgd'``.
        collocation:    Collocation point counts.
        global_lambdas: Loss term weights.
        seed:           PRNG seed.
    """
    input_dim: int = 2
    layer_specs: Optional[List] = None

    n_epochs: int = 1000
    learning_rate: float = 0.001
    optimiser_name: str = "adam"

    collocation: CollocationConfig = field(default_factory=CollocationConfig)
    global_lambdas: GlobalLambdaConfig = field(default_factory=GlobalLambdaConfig)

    seed: int = 1234

    def __post_init__(self):
        if isinstance(self.collocation, dict):
            self.collocation = CollocationConfig(**self.collocation)
        if isinstance(self.global_lambdas, dict):
            self.global_lambdas = GlobalLambdaConfig(**self.global_lambdas)

        if self.layer_specs is None:
            self.layer_specs = make_layer_specs([8, 8])
        else:
            normalized = []
            for s in self.layer_specs:
                if isinstance(s, PIKANLayerSpec):
                    normalized.append(s)
                elif isinstance(s, (list, tuple)):
                    normalized.append(PIKANLayerSpec.from_list(s))
                elif isinstance(s, dict):
                    normalized.append(PIKANLayerSpec(**s))
                else:
                    raise ValueError(f"Invalid layer spec: {s}")
            self.layer_specs = normalized

    @property
    def full_layer_dims(self) -> List[int]:
        return [self.input_dim] + [s.n_out for s in self.layer_specs]

    def to_flat_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"input_dim": self.input_dim}
        for i, spec in enumerate(self.layer_specs):
            d[f"layer_{i}_n_out"] = spec.n_out
            d[f"layer_{i}_type"] = spec.layer_type
            for k, v in spec.layer_params.items():
                d[f"layer_{i}_{k}"] = v
        d["n_layers"] = len(self.layer_specs)
        d["n_epochs"] = self.n_epochs
        d["learning_rate"] = self.learning_rate
        d["optimiser_name"] = self.optimiser_name
        d["collocation_n_pde"] = self.collocation.n_pde
        d["collocation_n_ic"] = self.collocation.n_ic
        d["collocation_n_bc_l"] = self.collocation.n_bc_l
        d["collocation_n_bc_r"] = self.collocation.n_bc_r
        gl = self.global_lambdas
        d["lambda_pde"] = gl.lambda_pde
        d["lambda_ic"] = gl.lambda_ic
        d["lambda_bc_l"] = gl.lambda_bc_l
        d["lambda_bc_r"] = gl.lambda_bc_r
        d["seed"] = self.seed
        return d

    def to_dict(self) -> Dict[str, Any]:
        col = self.collocation
        return {
            "input_dim": self.input_dim,
            "layer_specs": [s.to_dict() for s in self.layer_specs],
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "optimiser_name": self.optimiser_name,
            "collocation": {
                "n_pde": col.n_pde, "n_ic": col.n_ic,
                "n_bc_l": col.n_bc_l, "n_bc_r": col.n_bc_r,
            },
            "global_lambdas": self.global_lambdas.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PIKANConfig":
        return cls(**d)

    def validation_errors(self) -> List[str]:
        errors = []
        for i, spec in enumerate(self.layer_specs):
            if spec.layer_type not in LAYER_TYPES:
                errors.append(f"Layer {i}: unknown layer_type '{spec.layer_type}'")
        if self.optimiser_name not in OPTIMISERS:
            errors.append(f"Unknown optimiser '{self.optimiser_name}'")
        return errors
