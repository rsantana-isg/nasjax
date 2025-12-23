"""Network descriptors - PyTree-based architecture specifications (genotypes)"""

from .base import BaseDescriptor
from .cnn import CNNDescriptor
from .mlp import MLPDescriptor
from .rnn import RNNDescriptor
from .tcnn import TCNNDescriptor

__all__ = [
    "BaseDescriptor",
    "CNNDescriptor",
    "MLPDescriptor",
    "RNNDescriptor",
    "TCNNDescriptor",
]
