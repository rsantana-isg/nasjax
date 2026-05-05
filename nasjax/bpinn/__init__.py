"""
nasjax.bpinn — Bayesian Physics-Informed Neural Networks (BPINNs) in JAX.

This sub-package provides a fully parameterised BPINN implementation built on
top of JAX + Equinox.  Every aspect of the BPINN described in

    *Parameters which affect BPINN* (Iturralde, 2026)

is exposed as a Python-level configuration parameter.

Quick start
-----------
::

    import jax
    from nasjax.bpinn import BPINNConfig, train_bpinn
    from nasjax.bpinn.pde import BurgersProblem

    config = BPINNConfig(
        input_dim=2,
        layer_specs=[
            [50, 'tanh', 'Gaussian', {'mean': 0.0, 'sigma': 1.0}],
            [50, 'tanh', 'Gaussian', {'mean': 0.0, 'sigma': 1.0}],
            [1,  None,  'Gaussian', {'mean': 0.0, 'sigma': 1.0}],
        ],
        enhanced_mlp=True,
        n_epochs=500,
        learning_rate=0.001,
        local_lambda_tech='None',
    )

    problem = BurgersProblem()
    result  = train_bpinn(config, problem)
    print(f"Final -ELBO: {result.final_neg_elbo:.4e}")

Module layout
-------------
config     — BPINNConfig, BPINNLayerSpec, sub-configs
priors     — GaussianPrior, LaplacePrior, ScaleMixtureGaussianPrior
layers     — BayesianLinear
networks   — BayesianMLPVanilla, BayesianMLPEnhanced, build_bayesian_mlp
model      — BPINNModel (embedding + MLP + autodiff utilities)
losses     — compute_loss, init helpers
trainer    — make_step, train_bpinn, TrainingResult
evaluator  — BPINNEvaluator (for NAS integration)
pde/       — PDEProblem (abstract), BurgersProblem
"""

from .config import (
    BPINNConfig,
    BPINNLayerSpec,
    CollocationConfig,
    FourierEmbeddingConfig,
    GlobalLambdaConfig,
    RBAConfig,
    SigmaConfig,
    make_layer_specs,
)
from .model import BPINNModel
from .trainer import TrainingResult, make_step, train_bpinn
from .evaluator import BPINNEvaluator
from .pde import BurgersProblem, PDEProblem

__all__ = [
    # Config
    "BPINNConfig",
    "BPINNLayerSpec",
    "CollocationConfig",
    "FourierEmbeddingConfig",
    "GlobalLambdaConfig",
    "RBAConfig",
    "SigmaConfig",
    "make_layer_specs",
    # Model
    "BPINNModel",
    # Training
    "TrainingResult",
    "make_step",
    "train_bpinn",
    # Evaluation / NAS
    "BPINNEvaluator",
    # PDE problems
    "PDEProblem",
    "BurgersProblem",
]
