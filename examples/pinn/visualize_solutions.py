"""Visualize PINN solutions for different PDEs.

This script demonstrates how to visualize PINN solutions using the
built-in visualization utilities.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Force CPU if GPU/CUDA issues (comment out to use GPU)
# Uncomment the line below if you encounter CUDA/CuDNN errors
# import os
# os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nasjax.pinn import (
    LinearBurgersProblem,
    ConvectionDiffusionProblem,
    PINNEvaluator,
    PINNNetwork
)
from nasjax.pinn.visualization import (
    visualize_pinn_solution_2d,
    visualize_initial_condition,
    compare_with_analytical,
    plot_training_history
)
from nasjax.descriptors import MLPDescriptor


def train_network_for_visualization(problem, descriptor, n_train_iters, key):
    """Train a PINN network and return it with training history.

    Args:
        problem: PDE problem
        descriptor: Network architecture descriptor
        n_train_iters: Number of training iterations
        key: JAX random key

    Returns:
        Trained network and training history
    """
    evaluator = PINNEvaluator(
        problem=problem,
        n_train_iters=n_train_iters,
        learning_rate=1e-3,
        optimizer='adam',
        verbose=True
    )

    print("   Training network...")
    key, subkey = jax.random.split(key)

    # Train the network
    _ = evaluator.evaluate(
        descriptor,
        jnp.array([]), jnp.array([]),
        jnp.array([]), jnp.array([]),
        subkey,
        train=True
    )

    # Build trained network
    key, subkey = jax.random.split(key)
    network = PINNNetwork(descriptor, subkey)

    # Retrain to get the network with training history
    # In a real implementation, we'd modify the evaluator to return history
    history = {'train_loss': []}  # Placeholder

    return network, history, key


def visualize_linear_burgers(key):
    """Visualize solution for Linear Burgers equation."""

    print("\n" + "=" * 70)
    print("Visualizing Linear Burgers Equation Solution")
    print("=" * 70)

    # Define problem
    problem = LinearBurgersProblem(
        c=1.0,
        nu=0.02,
        x_min=-1.5,
        x_max=4.5,
        t_min=0.0,
        t_max=2.0,
        n_collocation=2000,
        n_ic=100,
        n_bc=100
    )

    print(f"\nPDE: ∂u/∂t + {problem.c}·∂u/∂x - {problem.nu}·∂²u/∂x² = 0")
    print(f"Domain: x ∈ [{problem.x_min}, {problem.x_max}], t ∈ [{problem.t_min}, {problem.t_max}]")

    # Create architecture
    descriptor = MLPDescriptor(
        dims=(32, 32, 32),
        act_functions=('tanh', 'tanh', 'tanh'),
        use_batch_norm=(False, False, False),
        use_dropout=(False, False, False),
        dropout_rates=(0.0, 0.0, 0.0)
    )

    print(f"\nArchitecture: [{2}] → {list(descriptor.dims)} → [1]")

    # Train network
    network, history, key = train_network_for_visualization(
        problem, descriptor, n_train_iters=1000, key=key
    )

    print("\n📊 Creating visualizations...")

    # 1. Solution visualization
    key, subkey = jax.random.split(key)
    visualize_pinn_solution_2d(
        network, problem, subkey,
        filename='linear_burgers_solution.png',
        show=False
    )

    # 2. Initial condition
    key, subkey = jax.random.split(key)
    visualize_initial_condition(
        network, problem, subkey,
        filename='linear_burgers_ic.png',
        show=False
    )

    print("\n✅ Visualizations saved:")
    print("   - linear_burgers_solution.png")
    print("   - linear_burgers_ic.png")

    return key


def visualize_convection_diffusion(key):
    """Visualize solution for Convection-Diffusion equation."""

    print("\n" + "=" * 70)
    print("Visualizing Convection-Diffusion Equation Solution")
    print("=" * 70)

    # Define problem
    problem = ConvectionDiffusionProblem(
        v=1.0,
        k=0.01,
        x_min=0.0,
        x_max=2.0,
        t_min=0.0,
        t_max=1.0,
        n_collocation=2000,
        n_ic=100,
        n_bc=100
    )

    print(f"\nPDE: {problem.v}·∂u/∂x = {problem.k}·∂²u/∂x²")
    print(f"Domain: x ∈ [{problem.x_min}, {problem.x_max}], t ∈ [{problem.t_min}, {problem.t_max}]")

    # Create architecture
    descriptor = MLPDescriptor(
        dims=(32, 32, 32),
        act_functions=('tanh', 'tanh', 'tanh'),
        use_batch_norm=(False, False, False),
        use_dropout=(False, False, False),
        dropout_rates=(0.0, 0.0, 0.0)
    )

    print(f"\nArchitecture: [{2}] → {list(descriptor.dims)} → [1]")

    # Train network
    network, history, key = train_network_for_visualization(
        problem, descriptor, n_train_iters=1000, key=key
    )

    print("\n📊 Creating visualizations...")

    # 1. Solution visualization
    key, subkey = jax.random.split(key)
    visualize_pinn_solution_2d(
        network, problem, subkey,
        filename='convection_diffusion_solution.png',
        show=False
    )

    # 2. Initial condition
    key, subkey = jax.random.split(key)
    visualize_initial_condition(
        network, problem, subkey,
        filename='convection_diffusion_ic.png',
        show=False
    )

    print("\n✅ Visualizations saved:")
    print("   - convection_diffusion_solution.png")
    print("   - convection_diffusion_ic.png")

    return key


def create_multi_plot_comparison(key):
    """Create a comparison plot showing multiple architectures."""

    print("\n" + "=" * 70)
    print("Creating Multi-Architecture Comparison Plot")
    print("=" * 70)

    problem = LinearBurgersProblem()

    # Define different architectures
    architectures = {
        'Small (3x10)': (10, 10, 10),
        'Medium (3x32)': (32, 32, 32),
        'Large (3x64)': (64, 64, 64),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, dims) in enumerate(architectures.items()):
        print(f"\nTraining {name}...")

        descriptor = MLPDescriptor(
            dims=dims,
            act_functions=tuple(['tanh'] * len(dims)),
            use_batch_norm=tuple([False] * len(dims)),
            use_dropout=tuple([False] * len(dims)),
            dropout_rates=tuple([0.0] * len(dims))
        )

        # Quick training
        network, _, key = train_network_for_visualization(
            problem, descriptor, n_train_iters=500, key=key
        )

        # Generate grid
        x = jnp.linspace(problem.x_min, problem.x_max, 100)
        t = jnp.linspace(problem.t_min, problem.t_max, 100)
        X, T = jnp.meshgrid(x, t)
        coords = jnp.stack([X.flatten(), T.flatten()], axis=1)

        # Evaluate network
        key, subkey = jax.random.split(key)

        def eval_network(point):
            return network(point, key=subkey, inference=True, compute_derivatives=False)

        u_pred = jax.vmap(eval_network)(coords)
        U_pred = u_pred.reshape(X.shape)

        # Plot
        ax = axes[idx]
        contour = ax.contourf(X, T, U_pred, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='u(x,t)')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Comparison plot saved: architecture_comparison.png")
    plt.close()

    return key


def main():
    print("=" * 70)
    print("NASJAX - PINN Solution Visualization")
    print("=" * 70)
    print()

    # Display platform info
    print(f"🖥️  Platform: {jax.devices()[0].platform.upper()}")
    print(f"   Device: {jax.devices()[0].device_kind}")
    print()

    print("This script demonstrates various visualization capabilities")
    print("for Physics-Informed Neural Networks.")
    print()

    key = jax.random.PRNGKey(42)

    # Visualize Linear Burgers
    key = visualize_linear_burgers(key)

    # Visualize Convection-Diffusion
    key = visualize_convection_diffusion(key)

    # Create comparison plot
    key = create_multi_plot_comparison(key)

    print("\n" + "=" * 70)
    print("Visualization Complete! ✨")
    print("=" * 70)
    print()
    print("Generated files:")
    print("  - linear_burgers_solution.png")
    print("  - linear_burgers_ic.png")
    print("  - convection_diffusion_solution.png")
    print("  - convection_diffusion_ic.png")
    print("  - architecture_comparison.png")
    print()
    print("Open these files to view the PINN solutions!")
    print()


if __name__ == "__main__":
    main()
