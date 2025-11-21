"""Tests for MLP network class."""

import jax
import jax.numpy as jnp
import pytest

from nasjax.descriptors.mlp import MLPDescriptor
from nasjax.networks.mlp import MLP


class TestMLPCreation:
    """Test MLP network creation from descriptors."""

    def test_create_from_descriptor(self):
        """Test basic MLP creation from descriptor."""
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        # Create descriptor
        desc = MLPDescriptor.random_init(784, 10, 3, 64, k1)

        # Build network
        network = MLP(desc, k2)

        # Check network structure
        assert len(network.layers) == len(desc.dims) + 1  # +1 for output layer
        assert network.input_dim == 784
        assert network.output_dim == 10

    def test_create_with_specific_architecture(self):
        """Test MLP with specific architecture."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=5,
            dims=(32, 16),
            act_functions=("relu", "sigmoid"),
            init_functions=("glorot_normal", "normal"),
            dropout_probs=(0.1, 0.2),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=64,
        )

        key = jax.random.PRNGKey(0)
        network = MLP(desc, key)

        assert len(network.layers) == 3  # 2 hidden + 1 output
        assert len(network.activations) == 3

    def test_invalid_descriptor_raises_error(self):
        """Test that invalid descriptor raises ValueError."""
        # Create invalid descriptor (too many neurons)
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(200,),  # Exceeds max
            act_functions=("relu",),
            init_functions=("normal",),
            dropout_probs=(0.1,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )

        with pytest.raises(ValueError):
            MLP(desc, jax.random.PRNGKey(0))


class TestMLPForwardPass:
    """Test MLP forward pass computation."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        # Create simple network
        desc = MLPDescriptor.random_init(28 * 28, 10, 2, 32, k1, dropout=False)
        network = MLP(desc, k2)

        # Create input
        x = jax.random.normal(k3, (28 * 28,))

        # Forward pass
        output = network(x, inference=True)

        # Check output shape
        assert output.shape == (10,)

    def test_forward_pass_inference_mode(self):
        """Test forward pass in inference mode."""
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1, dropout=True)
        network = MLP(desc, k2)
        x = jax.random.normal(k3, (64,))

        # Inference mode (no key needed)
        output = network(x, inference=True)
        assert output.shape == (10,)

    def test_forward_pass_training_mode_with_dropout(self):
        """Test forward pass in training mode with dropout."""
        key = jax.random.PRNGKey(2)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1, dropout=True)
        network = MLP(desc, k2)
        x = jax.random.normal(k3, (64,))

        # Training mode (requires key)
        output = network(x, key=k4, inference=False)
        assert output.shape == (10,)

    def test_forward_pass_training_without_key_raises_error(self):
        """Test that training mode without key raises error."""
        key = jax.random.PRNGKey(3)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1, dropout=True)
        network = MLP(desc, k2)
        x = jax.random.normal(k3, (64,))

        # Should raise error (no key provided for dropout)
        with pytest.raises(ValueError):
            network(x, inference=False)

    def test_forward_pass_no_dropout(self):
        """Test forward pass without dropout enabled."""
        key = jax.random.PRNGKey(4)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1, dropout=False)
        network = MLP(desc, k2)
        x = jax.random.normal(k3, (64,))

        # No key needed even in training mode (no dropout)
        output = network(x, inference=False)
        assert output.shape == (10,)

    def test_forward_pass_wrong_input_dim_raises_error(self):
        """Test that wrong input dimension raises error."""
        key = jax.random.PRNGKey(5)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(100, 10, 2, 32, k1)
        network = MLP(desc, k2)

        # Wrong input dimension
        x = jax.random.normal(k3, (50,))  # Expected 100

        with pytest.raises(ValueError):
            network(x, inference=True)

    def test_forward_pass_multidimensional_input(self):
        """Test forward pass with multidimensional input (flattened)."""
        key = jax.random.PRNGKey(6)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(28 * 28, 10, 2, 32, k1)
        network = MLP(desc, k2)

        # Multidimensional input (will be flattened)
        x = jax.random.normal(k3, (28, 28))

        output = network(x, inference=True)
        assert output.shape == (10,)


class TestMLPDeterminism:
    """Test deterministic behavior of MLP."""

    def test_same_key_same_output(self):
        """Test that same key produces same output."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1)
        network = MLP(desc, k2)
        x = jax.random.normal(k3, (64,))

        # Multiple forward passes should give same result
        output1 = network(x, inference=True)
        output2 = network(x, inference=True)

        assert jnp.allclose(output1, output2)

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1)
        network = MLP(desc, k2)

        x1 = jax.random.normal(k3, (64,))
        x2 = jax.random.normal(k4, (64,))

        output1 = network(x1, inference=True)
        output2 = network(x2, inference=True)

        # Outputs should be different (very high probability)
        assert not jnp.allclose(output1, output2)


class TestMLPUtilities:
    """Test MLP utility methods."""

    def test_count_parameters(self):
        """Test parameter counting."""
        desc = MLPDescriptor(
            input_dim=100,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )

        network = MLP(desc, jax.random.PRNGKey(0))
        n_params = network.count_parameters()

        # Calculate expected:
        # Layer 1: 100 -> 64: (100 * 64) + 64 = 6464
        # Layer 2: 64 -> 32: (64 * 32) + 32 = 2080
        # Layer 3: 32 -> 10: (32 * 10) + 10 = 330
        # Total: 8874
        expected = (100 * 64 + 64) + (64 * 32 + 32) + (32 * 10 + 10)
        assert n_params == expected

    def test_repr(self):
        """Test string representation."""
        desc = MLPDescriptor(
            input_dim=784,
            output_dim=10,
            dims=(64, 32),
            act_functions=("relu", "relu"),
            init_functions=("normal", "normal"),
            dropout_probs=(0.1, 0.1),
            use_dropout=True,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=128,
        )

        network = MLP(desc, jax.random.PRNGKey(0))
        repr_str = repr(network)

        assert "MLP" in repr_str
        assert "layers=3" in repr_str  # 2 hidden + 1 output
        assert "params=" in repr_str


class TestMLPJAXCompatibility:
    """Test JAX transformations with MLP."""

    def test_jit_compilation(self):
        """Test that MLP works with JIT compilation."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1, dropout=False)
        network = MLP(desc, k2)

        # JIT compile the forward pass
        @jax.jit
        def forward(x):
            return network(x, inference=True)

        x = jax.random.normal(k3, (64,))
        output = forward(x)

        assert output.shape == (10,)

    def test_vmap_over_batch(self):
        """Test that MLP works with vmap over batches."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1, dropout=False)
        network = MLP(desc, k2)

        # Create batch of inputs
        x_batch = jax.random.normal(k3, (16, 64))

        # Vectorize over batch
        outputs = jax.vmap(lambda x: network(x, inference=True))(x_batch)

        assert outputs.shape == (16, 10)

    def test_grad_computation(self):
        """Test that gradients can be computed."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        desc = MLPDescriptor.random_init(64, 10, 2, 32, k1, dropout=False)
        network = MLP(desc, k2)
        x = jax.random.normal(k3, (64,))
        y_true = jnp.zeros(10)

        # Define loss function
        def loss_fn(net, x, y):
            pred = net(x, inference=True)
            return jnp.mean((pred - y) ** 2)

        # Compute gradients
        grads = jax.grad(loss_fn)(network, x, y_true)

        # Gradients should exist for all parameters
        assert grads is not None


class TestMLPActivations:
    """Test different activation functions."""

    def test_relu_activation(self):
        """Test MLP with ReLU activation."""
        desc = MLPDescriptor(
            input_dim=10,
            output_dim=5,
            dims=(8,),
            act_functions=("relu",),
            init_functions=("normal",),
            dropout_probs=(0.0,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=32,
        )

        network = MLP(desc, jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (10,))
        output = network(x, inference=True)

        assert output.shape == (5,)

    def test_no_activation(self):
        """Test MLP with no activation (None)."""
        desc = MLPDescriptor(
            input_dim=10,
            output_dim=5,
            dims=(8,),
            act_functions=(None,),
            init_functions=("normal",),
            dropout_probs=(0.0,),
            use_dropout=False,
            use_batch_norm=False,
            max_num_layers=5,
            max_num_neurons=32,
        )

        network = MLP(desc, jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (10,))
        output = network(x, inference=True)

        assert output.shape == (5,)
