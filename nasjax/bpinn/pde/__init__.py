"""PDE problem definitions for BPINN."""

from .base import PDEProblem
from .burgers import BurgersProblem

__all__ = ["PDEProblem", "BurgersProblem"]
