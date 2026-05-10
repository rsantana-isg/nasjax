"""
Configuration dataclasses for Finite Basis Physics-Informed Neural Networks (FBPINNs).

FBPINNs decompose the domain into overlapping rectangular subdomains and train
one small network per subdomain.  The final solution is the sum of all
subdomain contributions weighted by smooth partition-of-unity (POU) window
functions.

Architecture representation
---------------------------
Subdomain networks use ``[n_neurons, activation]`` layer specs (same format as
PINNConfig), enabling shared NAS search over width and activation parameters.
All subdomains share the same architecture.

Domain decomposition
--------------------
For 2-D problems (x, t):
  - ``n_subdomains_x`` subdomains cover the spatial dimension.
  - ``n_subdomains_t`` subdomains cover the time dimension.
  - Adjacent subdomains overlap by a fraction ``overlap`` of their total width.

Window functions
----------------
``'cosine'``   ((1 + cos(π(x-μ)/σ))/2)² — smooth, zero outside subdomain.
``'sigmoid'``  Product of sigmoid ramp-ups/ramp-downs.

Reference: Moseley et al. (2023), "Finite basis physics-informed neural networks".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nasjax.bpinn.config import CollocationConfig


ACTIVATIONS = {"tanh", "relu", "gelu", "silu", "sin", "sigmoid", "softplus", None}
NETWORK_TYPES = {"FCN", "AdaptiveFCN", "SIREN"}
WINDOW_FNS = {"cosine", "sigmoid"}
OPTIMISERS = {"adam", "adamw", "sgd"}


# ---------------------------------------------------------------------------
# Layer spec (same format as PINNLayerSpec for NAS compatibility)
# ---------------------------------------------------------------------------

@dataclass
class FBPINNLayerSpec:
    """Specification for one layer of a subdomain network."""
    n_neurons: int
    activation: Optional[str]

    @classmethod
    def from_list(cls, spec) -> "FBPINNLayerSpec":
        return cls(n_neurons=int(spec[0]), activation=spec[1])

    def to_list(self) -> list:
        return [self.n_neurons, self.activation]

    def to_dict(self) -> dict:
        return {"n_neurons": self.n_neurons, "activation": self.activation}


def make_layer_specs(
    hidden_widths: List[int],
    output_dim: int = 1,
    activation: str = "tanh",
) -> List[FBPINNLayerSpec]:
    """Convenience constructor for uniform hidden stacks."""
    specs = [FBPINNLayerSpec(n_neurons=w, activation=activation) for w in hidden_widths]
    specs.append(FBPINNLayerSpec(n_neurons=output_dim, activation=None))
    return specs


# ---------------------------------------------------------------------------
# Decomposition config
# ---------------------------------------------------------------------------

@dataclass
class DecompositionConfig:
    """Domain decomposition parameters.

    Args:
        n_subdomains_x: Number of subdomains in the spatial (x) direction.
        n_subdomains_t: Number of subdomains in the time (t) direction.
        overlap:        Fractional overlap between adjacent subdomains.
                        E.g. ``0.5`` means 50% of the subdomain width overlaps.
        x_min:          Left boundary of the spatial domain.
        x_max:          Right boundary of the spatial domain.
        t_min:          Start of the time domain.
        t_max:          End of the time domain.
        window_fn:      Window function type (``'cosine'`` or ``'sigmoid'``).
    """
    n_subdomains_x: int = 4
    n_subdomains_t: int = 4
    overlap: float = 0.5
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    window_fn: str = "cosine"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "DecompositionConfig":
        return cls(**d)


# ---------------------------------------------------------------------------
# Global lambda config
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
class FBPINNConfig:
    """Full configuration for a FBPINN.

    Subdomain networks use the same ``[n_neurons, activation]`` format as
    PINNConfig, so NAS search routines can optimise both simultaneously.

    Args:
        input_dim:      Input dimension (2 for Burgers: x, t).
        layer_specs:    List of ``FBPINNLayerSpec`` or ``[n, act]`` lists.
                        Shared across all subdomains.
        network_type:   One of ``'FCN'``, ``'AdaptiveFCN'``, ``'SIREN'``.
        decomposition:  :class:`DecompositionConfig`.
        n_epochs:       Number of training epochs.
        learning_rate:  Optimiser learning rate.
        optimiser_name: One of ``'adam'``, ``'adamw'``, ``'sgd'``.
        collocation:    Collocation point counts.
        global_lambdas: Loss term weights.
        seed:           PRNG seed.
    """
    input_dim: int = 2
    layer_specs: Optional[List] = None
    network_type: str = "FCN"

    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)

    n_epochs: int = 2000
    learning_rate: float = 0.001
    optimiser_name: str = "adam"

    collocation: CollocationConfig = field(default_factory=CollocationConfig)
    global_lambdas: GlobalLambdaConfig = field(default_factory=GlobalLambdaConfig)

    seed: int = 1234

    def __post_init__(self):
        if isinstance(self.decomposition, dict):
            self.decomposition = DecompositionConfig(**self.decomposition)
        if isinstance(self.collocation, dict):
            self.collocation = CollocationConfig(**self.collocation)
        if isinstance(self.global_lambdas, dict):
            self.global_lambdas = GlobalLambdaConfig(**self.global_lambdas)

        if self.layer_specs is None:
            self.layer_specs = make_layer_specs([32, 32])
        else:
            normalized = []
            for s in self.layer_specs:
                if isinstance(s, FBPINNLayerSpec):
                    normalized.append(s)
                elif isinstance(s, (list, tuple)):
                    normalized.append(FBPINNLayerSpec.from_list(s))
                elif isinstance(s, dict):
                    normalized.append(FBPINNLayerSpec(**s))
                else:
                    raise ValueError(f"Invalid layer spec: {s}")
            self.layer_specs = normalized

    @property
    def n_subdomains(self) -> int:
        d = self.decomposition
        return d.n_subdomains_x * d.n_subdomains_t

    @property
    def full_layer_dims(self) -> List[int]:
        return [self.input_dim] + [s.n_neurons for s in self.layer_specs]

    def to_flat_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "input_dim": self.input_dim,
            "network_type": self.network_type,
        }
        for i, spec in enumerate(self.layer_specs):
            d[f"layer_{i}_n_neurons"] = spec.n_neurons
            d[f"layer_{i}_activation"] = spec.activation
        d["n_layers"] = len(self.layer_specs)
        dc = self.decomposition
        d["n_subdomains_x"] = dc.n_subdomains_x
        d["n_subdomains_t"] = dc.n_subdomains_t
        d["overlap"] = dc.overlap
        d["window_fn"] = dc.window_fn
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
            "network_type": self.network_type,
            "layer_specs": [s.to_dict() for s in self.layer_specs],
            "decomposition": self.decomposition.to_dict(),
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
    def from_dict(cls, d: dict) -> "FBPINNConfig":
        return cls(**d)

    def validation_errors(self) -> List[str]:
        errors = []
        for i, spec in enumerate(self.layer_specs):
            if spec.activation not in ACTIVATIONS:
                errors.append(f"Layer {i}: unknown activation '{spec.activation}'")
        if self.network_type not in NETWORK_TYPES:
            errors.append(f"Unknown network_type '{self.network_type}'")
        if self.decomposition.window_fn not in WINDOW_FNS:
            errors.append(f"Unknown window_fn '{self.decomposition.window_fn}'")
        if self.optimiser_name not in OPTIMISERS:
            errors.append(f"Unknown optimiser '{self.optimiser_name}'")
        return errors
