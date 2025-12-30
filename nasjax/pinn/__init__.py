"""Physics-Informed Neural Networks (PINNs) for NASJAX.

This module provides components for evolving PINN architectures:
- PINNNetwork: Neural network with automatic differentiation
- Physics losses: PDE-specific loss functions
- Benchmarks: Standard PDE problems
- Evaluators: Fitness evaluation for PINN architectures
"""

from .networks import PINNNetwork
from .evaluators import PINNEvaluator
from .benchmarks import LinearBurgersProblem, ConvectionDiffusionProblem

__all__ = [
    "PINNNetwork",
    "PINNEvaluator",
    "LinearBurgersProblem",
    "ConvectionDiffusionProblem",
]
