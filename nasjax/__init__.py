"""NASJAX - Neural Architecture Search with JAX

A neuroevolutionary framework for neural architecture search built on JAX and
Equinox.  Includes a fully-parameterised Bayesian PINN (BPINN) sub-package for
solving and optimising physics-informed neural networks.
"""

__version__ = "0.1.0"

# Main components will be imported here as they're implemented
# from .evolution import Evolving
# from .descriptors import MLPDescriptor, CNNDescriptor, RNNDescriptor
# from .networks import MLP, CNN, RNN

# BPINN sub-package (Bayesian Physics-Informed Neural Networks)
from . import bpinn

__all__ = [
    "__version__",
    "bpinn",
]
