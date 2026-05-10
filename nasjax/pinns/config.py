"""
Configuration dataclasses for standard Physics-Informed Neural Networks (PINNs).

Architecture is represented as a list of [n_neurons, activation] lists, keeping
the first two fields compatible with BPINNConfig for shared NAS search routines.

Supported network types
-----------------------
FCN          Fully-connected tanh/relu/… MLP (standard PINN)
AdaptiveFCN  FCN with per-neuron trainable activation scales
SIREN        FCN with sinusoidal activations (SIREN init)
FourierFCN   FCN preceded by a random Fourier feature embedding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nasjax.bpinn.config import CollocationConfig  # shared collocation config


ACTIVATIONS = {"tanh", "relu", "gelu", "silu", "sin", "sigmoid", "softplus", None}
NETWORK_TYPES = {"FCN", "AdaptiveFCN", "SIREN", "FourierFCN"}
OPTIMISERS = {"adam", "adamw", "sgd"}


# ---------------------------------------------------------------------------
# Layer spec
# ---------------------------------------------------------------------------

@dataclass
class PINNLayerSpec:
    """Specification for a single PINN layer: [n_neurons, activation]."""
    n_neurons: int
    activation: Optional[str]

    @classmethod
    def from_list(cls, spec) -> "PINNLayerSpec":
        return cls(n_neurons=int(spec[0]), activation=spec[1])

    def to_list(self) -> list:
        return [self.n_neurons, self.activation]

    def to_dict(self) -> dict:
        return {"n_neurons": self.n_neurons, "activation": self.activation}


def make_layer_specs(
    hidden_widths: List[int],
    output_dim: int = 1,
    activation: str = "tanh",
) -> List[PINNLayerSpec]:
    """Convenience constructor for uniform hidden stacks."""
    specs = [PINNLayerSpec(n_neurons=w, activation=activation) for w in hidden_widths]
    specs.append(PINNLayerSpec(n_neurons=output_dim, activation=None))
    return specs


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FourierInputConfig:
    """Random Fourier feature input embedding (for FourierFCN)."""
    enabled: bool = False
    n_features: int = 64    # number of random Fourier features
    mu: float = 0.0         # mean of frequency distribution
    sd: float = 1.0         # std dev of frequency distribution

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "n_features": self.n_features,
                "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_dict(cls, d: dict) -> "FourierInputConfig":
        return cls(**d)


@dataclass
class GlobalLambdaConfig:
    """Global loss-term weights for PINN training."""
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
class PINNConfig:
    """Full configuration for a standard PINN.

    The layer_specs follow the same ``[n_neurons, activation]`` convention as
    the first two fields of :class:`~nasjax.bpinn.config.BPINNLayerSpec`, so
    the same neuro-evolutionary search algorithms can operate on both.

    Args:
        input_dim:      Input dimension (2 for Burgers: x, t).
        layer_specs:    List of ``PINNLayerSpec`` or ``[n, act]`` lists.
        network_type:   One of ``'FCN'``, ``'AdaptiveFCN'``, ``'SIREN'``,
                        ``'FourierFCN'``.
        fourier_input:  Config for random Fourier feature embedding.
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

    fourier_input: FourierInputConfig = field(default_factory=FourierInputConfig)

    n_epochs: int = 5000
    learning_rate: float = 0.001
    optimiser_name: str = "adam"

    collocation: CollocationConfig = field(default_factory=CollocationConfig)
    global_lambdas: GlobalLambdaConfig = field(default_factory=GlobalLambdaConfig)

    seed: int = 1234

    def __post_init__(self):
        if isinstance(self.fourier_input, dict):
            self.fourier_input = FourierInputConfig(**self.fourier_input)
        if isinstance(self.collocation, dict):
            self.collocation = CollocationConfig(**self.collocation)
        if isinstance(self.global_lambdas, dict):
            self.global_lambdas = GlobalLambdaConfig(**self.global_lambdas)

        if self.layer_specs is None:
            self.layer_specs = make_layer_specs([50, 50, 50, 50])
        else:
            normalized = []
            for s in self.layer_specs:
                if isinstance(s, PINNLayerSpec):
                    normalized.append(s)
                elif isinstance(s, (list, tuple)):
                    normalized.append(PINNLayerSpec.from_list(s))
                elif isinstance(s, dict):
                    normalized.append(PINNLayerSpec(**s))
                else:
                    raise ValueError(f"Invalid layer spec: {s}")
            self.layer_specs = normalized

    @property
    def effective_input_dim(self) -> int:
        if self.network_type == "FourierFCN" and self.fourier_input.enabled:
            return 2 * self.fourier_input.n_features
        return self.input_dim

    @property
    def full_layer_dims(self) -> List[int]:
        return [self.effective_input_dim] + [s.n_neurons for s in self.layer_specs]

    def to_flat_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "input_dim": self.input_dim,
            "network_type": self.network_type,
        }
        for i, spec in enumerate(self.layer_specs):
            d[f"layer_{i}_n_neurons"] = spec.n_neurons
            d[f"layer_{i}_activation"] = spec.activation
        d["n_layers"] = len(self.layer_specs)
        fi = self.fourier_input
        d["fourier_input_enabled"] = fi.enabled
        d["fourier_input_n_features"] = fi.n_features
        d["fourier_input_mu"] = fi.mu
        d["fourier_input_sd"] = fi.sd
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
            "fourier_input": self.fourier_input.to_dict(),
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
    def from_dict(cls, d: dict) -> "PINNConfig":
        return cls(**d)

    def validation_errors(self) -> List[str]:
        errors = []
        for i, spec in enumerate(self.layer_specs):
            if spec.activation not in ACTIVATIONS:
                errors.append(f"Layer {i}: unknown activation '{spec.activation}'")
        if self.network_type not in NETWORK_TYPES:
            errors.append(f"Unknown network_type '{self.network_type}'")
        if self.optimiser_name not in OPTIMISERS:
            errors.append(f"Unknown optimiser '{self.optimiser_name}'")
        return errors
