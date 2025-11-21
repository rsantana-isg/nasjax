"""Base descriptor class for all network architectures.

This module defines the abstract base class that all network descriptors inherit from.
Descriptors are immutable PyTree structures that represent the genotype (architecture
specification) of a neural network.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDescriptor(ABC):
    """Abstract base class for all network descriptors.

    A descriptor is an immutable PyTree structure that specifies the architecture
    of a neural network. It represents the genotype in the neuroevolutionary algorithm.

    All concrete descriptor classes (MLP, CNN, RNN, etc.) should inherit from this
    class and implement the required abstract methods.

    Attributes:
        input_dim: Input dimension(s) of the network
        output_dim: Output dimension(s) of the network
        max_num_layers: Maximum number of layers allowed (for mutation constraints)
        max_num_neurons: Maximum neurons per layer (for mutation constraints)
    """

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the descriptor satisfies all constraints.

        This method checks that:
        - Layer counts are within bounds
        - Neuron/filter counts are within bounds
        - All lists have consistent lengths
        - Architecture is structurally valid

        Returns:
            True if descriptor is valid, False otherwise
        """
        pass

    @staticmethod
    @abstractmethod
    def random_init(*args, **kwargs):
        """Create a randomly initialized descriptor.

        This factory method creates a new descriptor with random architecture
        parameters. All random operations should use the JAX PRNG key for
        reproducibility.

        Args:
            *args: Positional arguments (descriptor-specific)
            **kwargs: Keyword arguments including:
                - key: JAX PRNG key for randomness (required)
                - Other descriptor-specific parameters

        Returns:
            A new randomly initialized descriptor instance
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary for serialization.

        This method converts the descriptor to a plain dictionary that can
        be serialized to JSON or other formats.

        Returns:
            Dictionary representation of the descriptor
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Reconstruct descriptor from dictionary.

        This method reconstructs a descriptor from its dictionary representation.

        Args:
            data: Dictionary containing descriptor data

        Returns:
            Reconstructed descriptor instance
        """
        pass

    def __repr__(self) -> str:
        """String representation of the descriptor."""
        return f"{self.__class__.__name__}(...)"
