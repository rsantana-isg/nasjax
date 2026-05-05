"""
BPINN Configuration Classes.

This module defines all data structures that specify a Bayesian Physics-Informed
Neural Network (BPINN).  Everything here is plain Python — no JAX arrays are
created.  These objects are built *before* entering JAX-traced code and are
safe to pickle, hash, and serialise for evolutionary search.

Architecture is expressed as a list of BPINNLayerSpec objects (or raw lists):

    layer_specs = [
        [50, 'tanh', 'Gaussian', {'mean': 0.0, 'sigma': 1.0}],  # hidden 1
        [50, 'tanh', 'Gaussian', {'mean': 0.0, 'sigma': 1.0}],  # hidden 2
        [1,  None,  'Gaussian', {'mean': 0.0, 'sigma': 1.0}],  # output (linear)
    ]

Parameters covered from docs/BPINN_docs/parameters_which_affect_BPINNs.tex:
  - Prior type + hyperparameters  (per layer)
  - Activation function           (per layer)
  - Width and depth               (layer_specs list length / n_neurons)
  - Number of epochs              (n_epochs)
  - Number of collocation points  (CollocationConfig)
  - Likelihood sigmas             (SigmaConfig)
  - Local lambdas (RBA)           (local_lambda_tech + RBAConfig)
  - Global lambdas                (GlobalLambdaConfig)
  - Modified/Enhanced MLP         (enhanced_mlp)
  - Hard-coded BCs (Fourier emb.) (FourierEmbeddingConfig)
  - Optimiser                     (optimiser_name + learning_rate)
  - MC samples per epoch          (k_samples_per_epoch)
  - Mini-batch / resampling       (resample_each_epoch)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Valid choices  (used for validation & evolutionary mutation boundaries)
# ---------------------------------------------------------------------------

VALID_ACTIVATIONS: Tuple[Optional[str], ...] = (
    "tanh", "relu", "elu", "sigmoid", "softplus", "softsign", None
)
VALID_PRIOR_TYPES: Tuple[str, ...] = ("Gaussian", "Laplace", "SnS")
VALID_LOCAL_LAMBDA_TECHS: Tuple[str, ...] = ("None", "Residual_RBA", "Likelihood_RBA")
VALID_OPTIMISERS: Tuple[str, ...] = ("adam", "sgd", "rmsprop")


# ---------------------------------------------------------------------------
# Per-layer specification
# ---------------------------------------------------------------------------

@dataclass
class BPINNLayerSpec:
    """Specification for a single Bayesian MLP layer.

    Each spec controls one layer's width, non-linearity, and Bayesian prior.

    Attributes:
        n_neurons:    Number of neurons (output dimension of this layer).
        activation:   Name of the activation function, or ``None`` for linear.
                      Must be one of :data:`VALID_ACTIVATIONS`.
        prior_type:   Prior distribution: ``'Gaussian'``, ``'Laplace'``, or
                      ``'SnS'`` (Spike-and-Slab).
        prior_params: Dict of prior hyper-parameters:
                        - Gaussian → ``{'mean': float, 'sigma': float}``
                        - Laplace  → ``{'mean': float, 'beta': float}``
                        - SnS      → ``{'pi': float, 'sigma_spike': float,
                                        'sigma_slab': float}``

    List representation (index-compatible with the original code style):

        spec = BPINNLayerSpec.from_list([50, 'tanh', 'Gaussian', {'mean':0, 'sigma':1}])
    """

    n_neurons: int
    activation: Optional[str]
    prior_type: str
    prior_params: Dict[str, float]

    def __post_init__(self) -> None:
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Invalid activation '{self.activation}'. "
                f"Choose from {VALID_ACTIVATIONS}."
            )
        if self.prior_type not in VALID_PRIOR_TYPES:
            raise ValueError(
                f"Invalid prior_type '{self.prior_type}'. "
                f"Choose from {VALID_PRIOR_TYPES}."
            )
        if self.n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {self.n_neurons}.")

    # --- Conversion helpers ------------------------------------------------

    @classmethod
    def from_list(cls, spec: list) -> "BPINNLayerSpec":
        """Build from ``[n_neurons, activation, prior_type, prior_params]`` list."""
        if len(spec) != 4:
            raise ValueError(
                "Layer spec list must have exactly 4 elements: "
                "[n_neurons, activation, prior_type, prior_params]. "
                f"Got {len(spec)} elements."
            )
        n_neurons, activation, prior_type, prior_params = spec
        return cls(
            n_neurons=int(n_neurons),
            activation=activation,
            prior_type=prior_type,
            prior_params=dict(prior_params),
        )

    def to_list(self) -> list:
        """Return ``[n_neurons, activation, prior_type, prior_params]``."""
        return [self.n_neurons, self.activation, self.prior_type, dict(self.prior_params)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "activation": self.activation,
            "prior_type": self.prior_type,
            "prior_params": dict(self.prior_params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BPINNLayerSpec":
        return cls(
            n_neurons=d["n_neurons"],
            activation=d["activation"],
            prior_type=d["prior_type"],
            prior_params=dict(d["prior_params"]),
        )


# ---------------------------------------------------------------------------
# Convenience constructors for layer specs
# ---------------------------------------------------------------------------

def default_gaussian_layer(
    n_neurons: int,
    activation: Optional[str] = "tanh",
    mean: float = 0.0,
    sigma: float = 1.0,
) -> BPINNLayerSpec:
    """Create a hidden layer with an isotropic Gaussian prior."""
    return BPINNLayerSpec(
        n_neurons=n_neurons,
        activation=activation,
        prior_type="Gaussian",
        prior_params={"mean": mean, "sigma": sigma},
    )


def make_layer_specs(
    hidden_widths: List[int],
    output_dim: int = 1,
    activation: Optional[str] = "tanh",
    prior_type: str = "Gaussian",
    prior_params: Optional[Dict[str, float]] = None,
) -> List[BPINNLayerSpec]:
    """Build a uniform hidden stack followed by a linear output layer.

    Args:
        hidden_widths: Neuron counts for each hidden layer.
        output_dim:    Neurons in the (linear) output layer.
        activation:    Activation for all hidden layers.
        prior_type:    Prior for all layers.
        prior_params:  Prior parameters.  Defaults are applied if ``None``.

    Returns:
        List of :class:`BPINNLayerSpec` (hidden layers + linear output).
    """
    if prior_params is None:
        if prior_type == "Gaussian":
            prior_params = {"mean": 0.0, "sigma": 1.0}
        elif prior_type == "Laplace":
            prior_params = {"mean": 0.0, "beta": 1.0}
        elif prior_type == "SnS":
            prior_params = {
                "pi": 0.5,
                "sigma_spike": math.exp(-6.0),
                "sigma_slab": math.exp(0.0),
            }
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")

    specs = [
        BPINNLayerSpec(n, activation, prior_type, dict(prior_params))
        for n in hidden_widths
    ]
    # Output layer: linear (no activation)
    specs.append(BPINNLayerSpec(output_dim, None, prior_type, dict(prior_params)))
    return specs


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class CollocationConfig:
    """Number of collocation points for each loss term.

    Larger counts give more likelihood terms and increase the relative weight
    of fitting accuracy vs. the prior.
    """

    n_pde: int = 10_000  #: PDE residual interior points
    n_ic: int = 400      #: Initial-condition points
    n_bc_l: int = 400    #: Left boundary points
    n_bc_r: int = 400    #: Right boundary points


@dataclass
class SigmaConfig:
    """Gaussian likelihood standard deviations for each loss term.

    Smaller σ → stricter fitting (more weight on residuals).
    Used in:  log p(D|θ) ∝ -Σ residual² / (2σ²)
    """

    sigma_pde: float = 0.001  #: PDE interior residual σ
    sigma_ic: float = 0.005   #: Initial-condition σ
    sigma_bc_l: float = 0.005 #: Left-boundary σ
    sigma_bc_r: float = 0.005 #: Right-boundary σ


@dataclass
class GlobalLambdaConfig:
    """Global multipliers weighting each ELBO term.

    -ELBO = λ_vp·log q(θ|φ) − λ_pr·log p(θ) − (λ_pde·log p_pde + ...)

    ``lambda_lik_ic`` is often set > 1 (e.g., 100) to enforce IC strongly.
    """

    lambda_lik_pde: float = 1.0      #: Weight on PDE likelihood term
    lambda_lik_ic: float = 100.0     #: Weight on IC likelihood term (often >> 1)
    lambda_lik_bc_l: float = 1.0     #: Weight on left-BC likelihood
    lambda_lik_bc_r: float = 1.0     #: Weight on right-BC likelihood
    lambda_prior: float = 1.0        #: Weight on log prior
    lambda_var_posterior: float = 1.0 #: Weight on variational posterior


@dataclass
class RBAConfig:
    """Residual-Based Attention (RBA) hyperparameters.

    Controls how quickly local per-point weights are adapted.
    """

    gamma: float = 0.999  #: EMA decay for lambda update
    lr: float = 0.001     #: Learning rate for lambda update
    eps: float = 1e-12    #: Small constant for numerical stability


@dataclass
class FourierEmbeddingConfig:
    """Fourier feature embedding for hard-coded periodic boundary conditions.

    Transforms the 1-D spatial coordinate x as:
        [1, cos(ω·x), cos(2ω·x), ..., cos(M·ω·x),
            sin(ω·x), sin(2ω·x), ..., sin(M·ω·x), t]
    where ω = 2π / L.

    This makes the network output automatically periodic in x without needing
    explicit BC likelihood terms.
    """

    enabled: bool = False  #: Toggle Fourier embedding on/off
    L: float = 2.0         #: Spatial period (domain length)
    M: int = 10            #: Number of Fourier modes

    @property
    def output_dim(self) -> int:
        """Dimension of embedded input: 2*M + 2 (cos + sin modes + 1 bias + t)."""
        return 2 * self.M + 2


# ---------------------------------------------------------------------------
# Master BPINNConfig
# ---------------------------------------------------------------------------

@dataclass
class BPINNConfig:
    """Complete specification of a Bayesian PINN (BPINN) training run.

    This is the **single entry point** that parametrises every aspect of the
    BPINN as described in *Parameters which affect BPINN* (Iturralde, 2026):

    =====================  ==============================================
    LaTeX section          BPINNConfig field(s)
    =====================  ==============================================
    Prior type             ``layer_specs[i].prior_type + prior_params``
    Activation function    ``layer_specs[i].activation``
    Width and depth        ``layer_specs`` (number and width of layers)
    Number of epochs       ``n_epochs``
    Collocation points     ``collocation`` (CollocationConfig)
    Likelihood sigmas      ``sigma`` (SigmaConfig)
    Local lambdas (RBA)    ``local_lambda_tech``, ``rba``
    Global lambdas         ``global_lambdas`` (GlobalLambdaConfig)
    Modified MLP           ``enhanced_mlp``
    Hard-coded BCs         ``fourier_embedding`` (FourierEmbeddingConfig)
    Optimisation           ``optimiser_name``, ``learning_rate``
    MC samples / epoch     ``k_samples_per_epoch``
    Resampling             ``resample_each_epoch``
    =====================  ==============================================

    Architecture representation
    ---------------------------
    ``layer_specs`` is a list of :class:`BPINNLayerSpec` objects (or raw
    ``[n_neurons, activation, prior_type, prior_params]`` lists that are
    automatically normalised).  Each element specifies one layer — hidden
    layers followed by the output layer.  Example::

        layer_specs = [
            [50, 'tanh', 'Gaussian', {'mean': 0.0, 'sigma': 1.0}],
            [50, 'tanh', 'Gaussian', {'mean': 0.0, 'sigma': 1.0}],
            [1,  None,  'Gaussian', {'mean': 0.0, 'sigma': 1.0}],
        ]

    Evolutionary search compatibility
    ----------------------------------
    ``to_flat_dict()`` / ``from_flat_dict()`` expose all numeric
    hyperparameters as a single-level dict so that search algorithms can
    treat them as a continuous / discrete vector.

    Attributes:
        input_dim:            Raw input dimension (2 for (x,t) in Burgers).
        layer_specs:          Per-layer specs (see above).
        enhanced_mlp:         Use Modified MLP with U/V residual connections.
        fourier_embedding:    Fourier feature embedding configuration.
        n_epochs:             Training iterations.
        learning_rate:        Optimiser learning rate.
        optimiser_name:       Optimiser algorithm (``'adam'``, ``'sgd'``, …).
        k_samples_per_epoch:  Number of MC weight samples per ELBO estimate.
        collocation:          Collocation point counts.
        sigma:                Likelihood standard deviations.
        local_lambda_tech:    ``'None'`` | ``'Residual_RBA'`` | ``'Likelihood_RBA'``.
        global_lambdas:       ELBO term global weights.
        rba:                  RBA adaptation hyper-parameters.
        resample_each_epoch:  Re-draw collocation points each epoch.
        seed:                 JAX PRNG seed.
    """

    # --- Input/output dimensions -------------------------------------------
    input_dim: int = 2  # (x, t) for 1-D+time PDEs like Burgers

    # --- Architecture -------------------------------------------------------
    # Accepts list[list] or list[BPINNLayerSpec]; normalised in __post_init__
    layer_specs: Optional[List] = None

    enhanced_mlp: bool = True  # Modified MLP (U/V gates)

    fourier_embedding: FourierEmbeddingConfig = field(
        default_factory=FourierEmbeddingConfig
    )

    # --- Training -----------------------------------------------------------
    n_epochs: int = 500
    learning_rate: float = 0.001
    optimiser_name: str = "adam"
    k_samples_per_epoch: int = 1  # K MC samples per ELBO estimate

    # --- Collocation --------------------------------------------------------
    collocation: CollocationConfig = field(default_factory=CollocationConfig)

    # --- Likelihood ---------------------------------------------------------
    sigma: SigmaConfig = field(default_factory=SigmaConfig)

    # --- Lambda weighting ---------------------------------------------------
    local_lambda_tech: str = "None"
    global_lambdas: GlobalLambdaConfig = field(default_factory=GlobalLambdaConfig)
    rba: RBAConfig = field(default_factory=RBAConfig)

    # --- Resampling ---------------------------------------------------------
    resample_each_epoch: bool = False

    # --- Reproducibility ----------------------------------------------------
    seed: int = 1234

    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Default architecture: two hidden layers of width 50, Gaussian prior
        if self.layer_specs is None:
            self.layer_specs = make_layer_specs([50, 50])

        # Normalise layer_specs elements to BPINNLayerSpec
        normalised: List[BPINNLayerSpec] = []
        for s in self.layer_specs:
            if isinstance(s, list):
                normalised.append(BPINNLayerSpec.from_list(s))
            elif isinstance(s, BPINNLayerSpec):
                normalised.append(s)
            else:
                raise TypeError(
                    f"layer_specs elements must be list or BPINNLayerSpec, "
                    f"got {type(s)}"
                )
        self.layer_specs = normalised

        # Normalise sub-configs passed as plain dicts
        if isinstance(self.fourier_embedding, dict):
            self.fourier_embedding = FourierEmbeddingConfig(**self.fourier_embedding)
        if isinstance(self.collocation, dict):
            self.collocation = CollocationConfig(**self.collocation)
        if isinstance(self.sigma, dict):
            self.sigma = SigmaConfig(**self.sigma)
        if isinstance(self.global_lambdas, dict):
            self.global_lambdas = GlobalLambdaConfig(**self.global_lambdas)
        if isinstance(self.rba, dict):
            self.rba = RBAConfig(**self.rba)

        if self.local_lambda_tech not in VALID_LOCAL_LAMBDA_TECHS:
            raise ValueError(
                f"Invalid local_lambda_tech '{self.local_lambda_tech}'. "
                f"Choose from {VALID_LOCAL_LAMBDA_TECHS}."
            )
        if self.optimiser_name not in VALID_OPTIMISERS:
            raise ValueError(
                f"Invalid optimiser '{self.optimiser_name}'. "
                f"Choose from {VALID_OPTIMISERS}."
            )

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def effective_input_dim(self) -> int:
        """Input dimension *after* optional Fourier embedding."""
        if self.fourier_embedding.enabled:
            return self.fourier_embedding.output_dim
        return self.input_dim

    @property
    def output_dim(self) -> int:
        """Output neuron count (last layer)."""
        return self.layer_specs[-1].n_neurons

    @property
    def hidden_layer_specs(self) -> List[BPINNLayerSpec]:
        """All specs except the final (output) layer."""
        return self.layer_specs[:-1]

    @property
    def full_layer_dims(self) -> List[int]:
        """``[input_dim, h1, h2, ..., output_dim]`` including I/O."""
        return [self.effective_input_dim] + [s.n_neurons for s in self.layer_specs]

    @property
    def n_hidden_layers(self) -> int:
        """Number of hidden layers."""
        return len(self.layer_specs) - 1

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(self) -> bool:
        """Return ``True`` if the configuration is self-consistent."""
        if not self.layer_specs:
            return False

        # Enhanced MLP requires uniform hidden-layer width
        if self.enhanced_mlp and self.n_hidden_layers > 0:
            hidden_widths = [s.n_neurons for s in self.hidden_layer_specs]
            if len(set(hidden_widths)) > 1:
                return False

        if self.k_samples_per_epoch < 1:
            return False

        # Resampling conflicts with RBA (RBA requires fixed collocation points)
        if self.resample_each_epoch and self.local_lambda_tech != "None":
            return False

        return True

    def validation_errors(self) -> List[str]:
        """Return a list of human-readable validation error strings."""
        errors: List[str] = []
        if not self.layer_specs:
            errors.append("layer_specs is empty.")
        if self.enhanced_mlp and self.n_hidden_layers > 0:
            hidden_widths = [s.n_neurons for s in self.hidden_layer_specs]
            if len(set(hidden_widths)) > 1:
                errors.append(
                    f"Enhanced MLP requires uniform hidden-layer width, "
                    f"but widths are {hidden_widths}."
                )
        if self.k_samples_per_epoch < 1:
            errors.append("k_samples_per_epoch must be >= 1.")
        if self.resample_each_epoch and self.local_lambda_tech != "None":
            errors.append(
                "resample_each_epoch=True conflicts with RBA "
                "(RBA requires fixed collocation points)."
            )
        return errors

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten the entire config to a one-level dict for evolutionary search.

        All parameters that could be search targets are included.
        Nested sub-configs use ``'_'`` as a separator.
        Layer specs are encoded with ``'layer_{i}_'`` prefixes.

        Returns:
            Flat dictionary suitable for logging, mutation, or NAS search.
        """
        flat: Dict[str, Any] = {
            "input_dim": self.input_dim,
            "enhanced_mlp": self.enhanced_mlp,
            "fourier_embedding_enabled": self.fourier_embedding.enabled,
            "fourier_embedding_L": self.fourier_embedding.L,
            "fourier_embedding_M": self.fourier_embedding.M,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "optimiser_name": self.optimiser_name,
            "k_samples_per_epoch": self.k_samples_per_epoch,
            "collocation_n_pde": self.collocation.n_pde,
            "collocation_n_ic": self.collocation.n_ic,
            "collocation_n_bc_l": self.collocation.n_bc_l,
            "collocation_n_bc_r": self.collocation.n_bc_r,
            "sigma_pde": self.sigma.sigma_pde,
            "sigma_ic": self.sigma.sigma_ic,
            "sigma_bc_l": self.sigma.sigma_bc_l,
            "sigma_bc_r": self.sigma.sigma_bc_r,
            "local_lambda_tech": self.local_lambda_tech,
            "global_lambda_lik_pde": self.global_lambdas.lambda_lik_pde,
            "global_lambda_lik_ic": self.global_lambdas.lambda_lik_ic,
            "global_lambda_lik_bc_l": self.global_lambdas.lambda_lik_bc_l,
            "global_lambda_lik_bc_r": self.global_lambdas.lambda_lik_bc_r,
            "global_lambda_prior": self.global_lambdas.lambda_prior,
            "global_lambda_var_posterior": self.global_lambdas.lambda_var_posterior,
            "rba_gamma": self.rba.gamma,
            "rba_lr": self.rba.lr,
            "rba_eps": self.rba.eps,
            "resample_each_epoch": self.resample_each_epoch,
            "seed": self.seed,
            "n_layers": len(self.layer_specs),
        }
        for i, spec in enumerate(self.layer_specs):
            flat[f"layer_{i}_n_neurons"] = spec.n_neurons
            flat[f"layer_{i}_activation"] = spec.activation
            flat[f"layer_{i}_prior_type"] = spec.prior_type
            for k, v in spec.prior_params.items():
                flat[f"layer_{i}_prior_{k}"] = v
        return flat

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a nested dict (JSON-compatible)."""
        return {
            "input_dim": self.input_dim,
            "layer_specs": [s.to_dict() for s in self.layer_specs],
            "enhanced_mlp": self.enhanced_mlp,
            "fourier_embedding": asdict(self.fourier_embedding),
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "optimiser_name": self.optimiser_name,
            "k_samples_per_epoch": self.k_samples_per_epoch,
            "collocation": asdict(self.collocation),
            "sigma": asdict(self.sigma),
            "local_lambda_tech": self.local_lambda_tech,
            "global_lambdas": asdict(self.global_lambdas),
            "rba": asdict(self.rba),
            "resample_each_epoch": self.resample_each_epoch,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BPINNConfig":
        """Reconstruct from a nested dict produced by :meth:`to_dict`."""
        d = dict(d)
        d["layer_specs"] = [BPINNLayerSpec.from_dict(s) for s in d["layer_specs"]]
        d["fourier_embedding"] = FourierEmbeddingConfig(**d["fourier_embedding"])
        d["collocation"] = CollocationConfig(**d["collocation"])
        d["sigma"] = SigmaConfig(**d["sigma"])
        d["global_lambdas"] = GlobalLambdaConfig(**d["global_lambdas"])
        d["rba"] = RBAConfig(**d["rba"])
        return cls(**d)
