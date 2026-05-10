"""
nasjax.pikans — Physics-Informed Kolmogorov-Arnold Networks (PIKANs) in JAX.

Built on jaxkan (flax.nnx-based) KAN layers.  Supports per-layer
heterogeneous configurations (Chebyshev, Fourier, Legendre, RBF, Sine, Spline).

Quick start
-----------
::

    from nasjax.pikans import PIKANConfig, train_pikan
    from nasjax.pikans.pde import BurgersProblem

    config = PIKANConfig(
        layer_specs=[
            [8, 'chebyshev', {'D': 5}],
            [8, 'chebyshev', {'D': 5}],
            [1, 'chebyshev', {'D': 5}],
        ],
        n_epochs=1000,
    )
    problem = BurgersProblem()
    result = train_pikan(config, problem)
    print(f"Final loss: {result.final_loss:.4e}")

Module layout
-------------
config     — PIKANConfig, PIKANLayerSpec, sub-configs
networks   — PIKANNetwork (nnx.Module)
losses     — compute_loss, residual helpers
trainer    — train_pikan, TrainingResult
evaluator  — PIKANEvaluator (NAS integration)
pde/       — PDEProblem, BurgersProblem (re-exported from nasjax.bpinn.pde)
"""

from .config import (
    PIKANConfig,
    PIKANLayerSpec,
    GlobalLambdaConfig,
    make_layer_specs,
)
from .networks import PIKANNetwork
from .trainer import TrainingResult, train_pikan
from .evaluator import PIKANEvaluator, relative_l2_error
from .pde import PDEProblem, BurgersProblem

__all__ = [
    # Config
    "PIKANConfig",
    "PIKANLayerSpec",
    "GlobalLambdaConfig",
    "make_layer_specs",
    # Model
    "PIKANNetwork",
    # Training
    "TrainingResult",
    "train_pikan",
    # Evaluation / NAS
    "PIKANEvaluator",
    "relative_l2_error",
    # PDE problems
    "PDEProblem",
    "BurgersProblem",
]
