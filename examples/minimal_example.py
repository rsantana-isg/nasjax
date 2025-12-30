"""Minimal example demonstrating NASJAX foundational components.

This script shows how to:
1. Create a random MLP descriptor (genotype)
2. Build an Equinox network from the descriptor (phenotype)
3. Perform forward passes
4. Use JAX transformations (jit, vmap)
"""

import sys
from pathlib import Path

# Add parent directory to path for development (no install needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import equinox as eqx

from nasjax.descriptors import MLPDescriptor
from nasjax.networks import MLP


def main():
    """Run minimal example."""
    print("=" * 60)
    print("NASJAX Minimal Example - Foundational Components")
    print("=" * 60)

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Step 1: Create a random MLP descriptor
    print("\n1. Creating random MLP descriptor...")
    descriptor = MLPDescriptor.random_init(
        input_dim=784,  # MNIST image size (28x28 flattened)
        output_dim=10,  # 10 classes
        max_num_layers=5,
        max_num_neurons=128,
        key=k1,
        dropout=True,
        batch_norm=False,
    )

    print(f"   {descriptor}")
    print(f"   Valid: {descriptor.validate()}")
    print(f"   Architecture: {[784] + list(descriptor.dims) + [10]}")
    print(f"   Activations: {descriptor.act_functions}")

    # Step 2: Build Equinox network from descriptor
    print("\n2. Building Equinox network from descriptor...")
    network = MLP(descriptor, k2)
    print(f"   {network}")
    print(f"   Total parameters: {network.count_parameters():,}")

    # Step 3: Single forward pass (inference mode)
    print("\n3. Single forward pass (inference mode)...")
    x_single = jax.random.normal(k3, (784,))
    output_single = network(x_single, inference=True)
    print(f"   Input shape: {x_single.shape}")
    print(f"   Output shape: {output_single.shape}")
    print(f"   Output (first 5): {output_single[:5]}")

    # Step 4: Batch forward pass with vmap
    print("\n4. Batch forward pass with vmap...")
    k4 = jax.random.PRNGKey(100)
    x_batch = jax.random.normal(k4, (32, 784))  # Batch of 32

    # Vectorize over batch dimension
    outputs_batch = jax.vmap(lambda x: network(x, inference=True))(x_batch)
    print(f"   Batch input shape: {x_batch.shape}")
    print(f"   Batch output shape: {outputs_batch.shape}")

    # Step 5: JIT compilation
    print("\n5. JIT compilation for speed...")

    @jax.jit
    def forward_jit(x):
        return network(x, inference=True)

    output_jit = forward_jit(x_single)
    print(f"   JIT output shape: {output_jit.shape}")
    print(f"   Outputs match: {jnp.allclose(output_single, output_jit)}")

    # Step 6: Gradient computation
    print("\n6. Computing gradients...")

    def loss_fn(net, x, y):
        pred = net(x, k3, inference=True)
        return jnp.mean((pred - y) ** 2)

    y_target = jnp.zeros(10)
    # Use Equinox's filter_grad to handle the module correctly
    grads = eqx.filter_grad(loss_fn)(network, x_single, y_target)
    print(f"   Gradients computed successfully!")
    print(f"   First layer weight gradient shape: {grads.layers[0].weight.shape}")

    # Step 7: Descriptor serialization
    print("\n7. Descriptor serialization...")
    desc_dict = descriptor.to_dict()
    descriptor_reloaded = MLPDescriptor.from_dict(desc_dict)
    print(f"   Serialization successful: {descriptor == descriptor_reloaded}")

    # Step 8: PyTree operations
    print("\n8. PyTree operations...")
    leaves, treedef = jax.tree_util.tree_flatten(descriptor)
    print(f"   Descriptor is a valid PyTree!")
    print(f"   Number of leaves: {len(leaves)}")

    print("\n" + "=" * 60)
    print("All foundational components working correctly! ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Implement mutation operators")
    print("  - Add training loop with Optax")
    print("  - Implement population and evolution")
    print("  - Add CNN and RNN support")


if __name__ == "__main__":
    main()
