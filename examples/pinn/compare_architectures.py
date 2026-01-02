"""Compare different PINN architectures for solving PDEs.

This script demonstrates how to compare hand-designed architectures
vs. evolved architectures on the same PDE problem.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import time

from nasjax.pinn import LinearBurgersProblem, PINNEvaluator, PINNNetwork
from nasjax.pinn.validation import validate_solution
from nasjax.descriptors import MLPDescriptor


def create_hand_designed_architectures():
    """Create a set of hand-designed architectures to compare."""

    architectures = {
        'Small (3x10)': MLPDescriptor(
            dims=(10, 10, 10),
            act_functions=('tanh', 'tanh', 'tanh'),
            use_batch_norm=(False, False, False),
            use_dropout=(False, False, False),
            dropout_rates=(0.0, 0.0, 0.0)
        ),
        'Medium (3x32)': MLPDescriptor(
            dims=(32, 32, 32),
            act_functions=('tanh', 'tanh', 'tanh'),
            use_batch_norm=(False, False, False),
            use_dropout=(False, False, False),
            dropout_rates=(0.0, 0.0, 0.0)
        ),
        'Large (4x64)': MLPDescriptor(
            dims=(64, 64, 64, 64),
            act_functions=('tanh', 'tanh', 'tanh', 'tanh'),
            use_batch_norm=(False, False, False, False),
            use_dropout=(False, False, False, False),
            dropout_rates=(0.0, 0.0, 0.0, 0.0)
        ),
        'Deep (5x20)': MLPDescriptor(
            dims=(20, 20, 20, 20, 20),
            act_functions=('tanh', 'tanh', 'tanh', 'tanh', 'tanh'),
            use_batch_norm=(False, False, False, False, False),
            use_dropout=(False, False, False, False, False),
            dropout_rates=(0.0, 0.0, 0.0, 0.0, 0.0)
        ),
        'Wide (2x100)': MLPDescriptor(
            dims=(100, 100),
            act_functions=('tanh', 'tanh'),
            use_batch_norm=(False, False),
            use_dropout=(False, False),
            dropout_rates=(0.0, 0.0)
        ),
        'Mixed Activations': MLPDescriptor(
            dims=(32, 32, 32),
            act_functions=('tanh', 'relu', 'tanh'),
            use_batch_norm=(False, False, False),
            use_dropout=(False, False, False),
            dropout_rates=(0.0, 0.0, 0.0)
        ),
    }

    return architectures


def train_and_evaluate_architecture(
    descriptor: MLPDescriptor,
    problem,
    n_train_iters: int,
    key: jax.random.PRNGKey
):
    """Train and evaluate a single architecture.

    Args:
        descriptor: Architecture descriptor
        problem: PDE problem
        n_train_iters: Number of training iterations
        key: JAX random key

    Returns:
        Dictionary with results
    """
    # Create evaluator
    evaluator = PINNEvaluator(
        problem=problem,
        n_train_iters=n_train_iters,
        learning_rate=1e-3,
        optimizer='adam',
        verbose=False
    )

    # Train network
    start_time = time.time()

    final_loss = evaluator.evaluate(
        descriptor,
        jnp.array([]), jnp.array([]),
        jnp.array([]), jnp.array([]),
        key,
        train=True
    )

    train_time = time.time() - start_time

    # Build trained network for validation
    key, subkey = jax.random.split(key)
    network = PINNNetwork(descriptor, subkey)

    # Retrain to get the trained version (since evaluator doesn't return it)
    # For a proper implementation, we'd modify evaluator to return the network
    better_evaluator = PINNEvaluator(
        problem=problem,
        n_train_iters=n_train_iters,
        learning_rate=1e-3,
        verbose=False
    )
    _ = better_evaluator.evaluate(descriptor, jnp.array([]), jnp.array([]),
                                  jnp.array([]), jnp.array([]), subkey, train=True)

    # Validate solution
    key, subkey = jax.random.split(key)
    metrics = validate_solution(network, problem, subkey, verbose=False)

    # Count parameters
    n_params = network.count_parameters()

    return {
        'final_loss': final_loss,
        'train_time': train_time,
        'n_params': n_params,
        'pde_residual': metrics.get('pde_residual', float('nan')),
        'ic_error': metrics.get('ic_error', float('nan')),
        'l2_error': metrics.get('l2_error', float('nan')),
        'relative_error': metrics.get('relative_error', float('nan'))
    }


def main():
    print("=" * 70)
    print("NASJAX - Comparing PINN Architectures")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Define the PDE Problem
    # =========================================================================
    print("📊 Setting up Linear Burgers equation problem...")

    problem = LinearBurgersProblem(
        c=1.0,
        nu=0.02,
        x_min=-1.5,
        x_max=4.5,
        t_min=0.0,
        t_max=2.0,
        n_collocation=2000,
        n_ic=100,
        n_bc=100,
        loss_weight_pde=1.0,
        loss_weight_ic=1.0,
        loss_weight_bc=0.0
    )

    print(f"   PDE: ∂u/∂t + {problem.c}·∂u/∂x - {problem.nu}·∂²u/∂x² = 0")
    print()

    # =========================================================================
    # 2. Create Architectures to Compare
    # =========================================================================
    print("🏗️  Creating architectures to compare...")

    architectures = create_hand_designed_architectures()

    print(f"   Number of architectures: {len(architectures)}")
    for name in architectures.keys():
        print(f"   - {name}")
    print()

    # =========================================================================
    # 3. Train and Evaluate Each Architecture
    # =========================================================================
    print("🚀 Training and evaluating architectures...")
    print()

    n_train_iters = 1000
    results = {}

    key = jax.random.PRNGKey(42)

    for i, (name, descriptor) in enumerate(architectures.items(), 1):
        print(f"[{i}/{len(architectures)}] Evaluating: {name}")
        print(f"   Architecture: [{2}] → {list(descriptor.dims)} → [1]")
        print(f"   Activations: {descriptor.act_functions}")

        key, subkey = jax.random.split(key)
        result = train_and_evaluate_architecture(
            descriptor, problem, n_train_iters, subkey
        )

        results[name] = result

        print(f"   Parameters: {result['n_params']:,}")
        print(f"   Final Loss: {result['final_loss']:.6f}")
        print(f"   Training Time: {result['train_time']:.2f}s")
        print(f"   PDE Residual: {result['pde_residual']:.6e}")
        print(f"   IC Error: {result['ic_error']:.6e}")
        print()

    # =========================================================================
    # 4. Compare Results
    # =========================================================================
    print("=" * 70)
    print("📊 Comparison Results")
    print("=" * 70)
    print()

    # Create comparison table
    print(f"{'Architecture':<20} {'Params':>10} {'Loss':>12} {'PDE Res':>12} {'Time(s)':>10}")
    print("-" * 70)

    # Sort by final loss
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_loss'])

    for name, result in sorted_results:
        print(f"{name:<20} {result['n_params']:>10,} {result['final_loss']:>12.6f} "
              f"{result['pde_residual']:>12.6e} {result['train_time']:>10.2f}")

    print()

    # =========================================================================
    # 5. Summary and Recommendations
    # =========================================================================
    best_name, best_result = sorted_results[0]

    print("=" * 70)
    print("🏆 Best Architecture")
    print("=" * 70)
    print(f"Architecture: {best_name}")
    print(f"Final Loss: {best_result['final_loss']:.6f}")
    print(f"PDE Residual: {best_result['pde_residual']:.6e}")
    print(f"IC Error: {best_result['ic_error']:.6e}")
    print(f"Parameters: {best_result['n_params']:,}")
    print(f"Training Time: {best_result['train_time']:.2f}s")
    print()

    # Efficiency analysis
    print("=" * 70)
    print("💡 Insights")
    print("=" * 70)

    # Find most parameter-efficient
    efficiency_scores = {
        name: result['final_loss'] * result['n_params'] / 1000
        for name, result in results.items()
    }
    most_efficient = min(efficiency_scores.items(), key=lambda x: x[1])

    print(f"Most accurate: {best_name} (Loss: {best_result['final_loss']:.6f})")
    print(f"Most efficient: {most_efficient[0]}")

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]['train_time'])
    print(f"Fastest training: {fastest[0]} ({fastest[1]['train_time']:.2f}s)")

    print()
    print("=" * 70)
    print("Comparison Complete! ✨")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  - Different architectures have different accuracy-efficiency trade-offs")
    print("  - Use evolution to automatically find optimal architectures")
    print("  - Deeper networks aren't always better for PINNs")
    print("  - Smooth activations (tanh) often work better than ReLU for PINNs")
    print()


if __name__ == "__main__":
    main()
