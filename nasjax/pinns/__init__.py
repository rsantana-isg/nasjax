"""
nasjax.pinns — Standard Physics-Informed Neural Networks (PINNs) in JAX.

Supports FCN, AdaptiveFCN, SIREN, and FourierFCN architectures.
Architecture is represented as ``[n_neurons, activation]`` lists, compatible
with BPINNConfig for unified NAS search.

Quick start
-----------
::

    from nasjax.pinns import PINNConfig, train_pinn
    from nasjax.pinns.pde import BurgersProblem

    config = PINNConfig(
        layer_specs=[[50,'tanh'],[50,'tanh'],[50,'tanh'],[1,None]],
        n_epochs=5000,
    )
    problem = BurgersProblem()
    result = train_pinn(config, problem)
    print(f"Final loss: {result.final_loss:.4e}")

Module layout
-------------
config     — PINNConfig, PINNLayerSpec, sub-configs
networks   — PINNNetwork (eqx.Module)
model      — PINNModel (autodiff utilities)
losses     — compute_loss
trainer    — make_step, train_pinn, TrainingResult
evaluator  — PINNEvaluator (NAS integration)
pde/       — PDEProblem, BurgersProblem (re-exported from nasjax.bpinn.pde)
"""

from .config import (
    PINNConfig,
    PINNLayerSpec,
    FourierInputConfig,
    GlobalLambdaConfig,
    make_layer_specs,
)
from .model import PINNModel
from .trainer import TrainingResult, make_step, train_pinn
from .evaluator import PINNEvaluator, relative_l2_error
from .pde import PDEProblem, BurgersProblem

__all__ = [
    # Config
    "PINNConfig",
    "PINNLayerSpec",
    "FourierInputConfig",
    "GlobalLambdaConfig",
    "make_layer_specs",
    # Model
    "PINNModel",
    # Training
    "TrainingResult",
    "make_step",
    "train_pinn",
    # Evaluation / NAS
    "PINNEvaluator",
    "relative_l2_error",
    # PDE problems
    "PDEProblem",
    "BurgersProblem",
]
