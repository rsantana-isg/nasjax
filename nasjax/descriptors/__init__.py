"""Network descriptors - PyTree-based architecture specifications (genotypes)"""

from .base import BaseDescriptor
from .mlp import MLPDescriptor

__all__ = [
    "BaseDescriptor",
    "MLPDescriptor",
]
