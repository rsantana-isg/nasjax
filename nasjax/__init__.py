"""NASJAX - Neural Architecture Search with JAX

A neuroevolutionary framework for neural architecture search built on JAX and
Equinox.  Includes fully-parameterised physics-informed neural network
sub-packages for BPINNs, PINNs, PIKANs, and FBPINNs.
"""

__version__ = "0.1.0"

# Main components will be imported here as they're implemented
# from .evolution import Evolving
# from .descriptors import MLPDescriptor, CNNDescriptor, RNNDescriptor
# from .networks import MLP, CNN, RNN

# Physics-informed NN sub-packages
from . import bpinn    # Bayesian PINNs (equinox)
from . import pinns    # Standard PINNs (equinox)
from . import pikans   # Kolmogorov-Arnold PINNs (jaxkan/flax.nnx)
from . import fbpinns  # Finite Basis PINNs (pure JAX)

__all__ = [
    "__version__",
    "bpinn",
    "pinns",
    "pikans",
    "fbpinns",
]
