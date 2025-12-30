"""PDE benchmark problems for PINN evolution.

This module provides standard PDE problems for testing and
evaluating PINN architectures.
"""

from .linear_burgers import LinearBurgersProblem
from .convection_diffusion import ConvectionDiffusionProblem

__all__ = [
    "LinearBurgersProblem",
    "ConvectionDiffusionProblem",
]
