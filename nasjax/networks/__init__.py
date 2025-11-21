"""Neural network implementations in Equinox - phenotypes built from descriptors"""

from .base import BaseNetwork
from .mlp import MLP

__all__ = [
    "BaseNetwork",
    "MLP",
]
