"""
nasjax.fbpinns — Finite Basis Physics-Informed Neural Networks (FBPINNs) in JAX.

Decomposes the domain into overlapping rectangular subdomains, each with its
own small FCN.  The solution is the partition-of-unity weighted sum of all
subdomain contributions.  Implemented in pure JAX (no equinox / flax.nnx).

Quick start
-----------
::

    from nasjax.fbpinns import FBPINNConfig, DecompositionConfig, train_fbpinn
    from nasjax.fbpinns.pde import BurgersProblem

    config = FBPINNConfig(
        layer_specs=[[32, 'tanh'], [32, 'tanh'], [1, None]],
        decomposition=DecompositionConfig(n_subdomains_x=4, n_subdomains_t=4, overlap=0.5),
        n_epochs=2000,
    )
    problem = BurgersProblem()
    result = train_fbpinn(config, problem)
    print(f"Final loss: {result.final_loss:.4e}")

Module layout
-------------
config     — FBPINNConfig, FBPINNLayerSpec, DecompositionConfig, sub-configs
networks   — init_params, forward_all (pure JAX)
model      — FBPINNModel (domain geometry + combined forward)
losses     — compute_loss
trainer    — make_step, train_fbpinn, TrainingResult
evaluator  — FBPINNEvaluator (NAS integration)
pde/       — PDEProblem, BurgersProblem (re-exported from nasjax.bpinn.pde)
"""

from .config import (
    FBPINNConfig,
    FBPINNLayerSpec,
    DecompositionConfig,
    GlobalLambdaConfig,
    make_layer_specs,
)
from .model import FBPINNModel
from .networks import init_params
from .trainer import TrainingResult, make_step, train_fbpinn
from .evaluator import FBPINNEvaluator, relative_l2_error
from .pde import PDEProblem, BurgersProblem

__all__ = [
    # Config
    "FBPINNConfig",
    "FBPINNLayerSpec",
    "DecompositionConfig",
    "GlobalLambdaConfig",
    "make_layer_specs",
    # Model
    "FBPINNModel",
    "init_params",
    # Training
    "TrainingResult",
    "make_step",
    "train_fbpinn",
    # Evaluation / NAS
    "FBPINNEvaluator",
    "relative_l2_error",
    # PDE problems
    "PDEProblem",
    "BurgersProblem",
]
