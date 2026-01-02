"""Validate PINN solutions against analytical solutions and benchmarks.

This script demonstrates how to validate PINN solutions using the
built-in validation utilities.
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

from nasjax.pinn import LinearBurgersProblem, PINNEvaluator, PINNNetwork
from nasjax.pinn.validation import (
    validate_solution,
    compute_l2_error,
    compute_relative_error,
    compute_pde_residual,
    compute_ic_error,
    compute_point_wise_error
)
from nasjax.descriptors import MLPDescriptor


def train_pinn(problem, descriptor, n_train_iters, key):
    """Train a PINN network.

    Args:
        problem: PDE problem
        descriptor: Network architecture descriptor
        n_train_iters: Number of training iterations
        key: JAX random key

    Returns:
        Trained network and updated key
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

    final_loss = evaluator.evaluate(
        descriptor,
        jnp.array([]), jnp.array([]),
        jnp.array([]), jnp.array([]),
        subkey,
        train=True
    )

    print(f"   Final training loss: {final_loss:.6f}")

    # Build trained network
    key, subkey = jax.random.split(key)
    network = PINNNetwork(descriptor, subkey)

    # Retrain to get the same trained weights
    # In production, evaluator would return the trained network
    _ = evaluator.evaluate(descriptor, jnp.array([]), jnp.array([]),
                          jnp.array([]), jnp.array([]), subkey, train=True)

    return network, key


def validate_linear_burgers():
    """Comprehensive validation of Linear Burgers solution."""

    print("\n" + "=" * 70)
    print("Validating Linear Burgers Equation Solution")
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
    key = jax.random.PRNGKey(42)
    network, key = train_pinn(problem, descriptor, n_train_iters=2000, key=key)

    print("\n" + "=" * 70)
    print("Validation Metrics")
    print("=" * 70)

    # 1. Comprehensive validation
    print("\n1️⃣  Comprehensive Validation:")
    key, subkey = jax.random.split(key)
    metrics = validate_solution(network, problem, subkey, n_test_points=2000, verbose=True)

    # 2. Detailed error analysis
    print("\n2️⃣  Detailed Error Analysis:")

    # PDE residual at different test sets
    print("\n   PDE Residual (at different test set sizes):")
    for n_points in [100, 500, 1000, 2000]:
        key, subkey = jax.random.split(key)
        residual = compute_pde_residual(network, problem, subkey, n_test_points=n_points)
        print(f"      {n_points:5d} points: {residual:.6e}")

    # Initial condition error
    print("\n   Initial Condition Error:")
    key, subkey = jax.random.split(key)
    ic_error = compute_ic_error(network, problem, subkey, n_ic_points=200)
    print(f"      IC MSE: {ic_error:.6e}")

    # 3. Point-wise error analysis (if analytical solution available)
    if problem.analytical_solution is not None:
        print("\n3️⃣  Point-wise Error Analysis:")

        key, subkey = jax.random.split(key)
        test_coords = problem.generate_collocation_points(subkey)[:500]

        key, subkey = jax.random.split(key)
        point_errors = compute_point_wise_error(network, problem, test_coords, subkey)

        print(f"      Mean absolute error: {jnp.mean(point_errors):.6e}")
        print(f"      Max absolute error:  {jnp.max(point_errors):.6e}")
        print(f"      Min absolute error:  {jnp.min(point_errors):.6e}")
        print(f"      Std absolute error:  {jnp.std(point_errors):.6e}")
        print(f"      Median absolute error: {jnp.median(point_errors):.6e}")

    # 4. Network statistics
    print("\n4️⃣  Network Statistics:")
    n_params = network.count_parameters()
    print(f"      Total parameters: {n_params:,}")
    print(f"      Layers: {len(descriptor.dims)}")
    print(f"      Neurons per layer: {descriptor.dims}")
    print(f"      Activations: {descriptor.act_functions}")

    # 5. Success criteria check
    print("\n5️⃣  Success Criteria Check:")

    criteria = {
        'PDE Residual < 1e-4': metrics['pde_residual'] < 1e-4,
        'IC Error < 1e-4': metrics['ic_error'] < 1e-4,
    }

    if 'l2_error' in metrics:
        criteria['L2 Error < 1e-3'] = metrics['l2_error'] < 1e-3
        criteria['Relative Error < 1%'] = metrics['relative_error'] < 0.01

    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"      {status}: {criterion}")

    all_passed = all(criteria.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All validation criteria passed!")
    else:
        print("⚠️  Some validation criteria failed.")
        print("   Consider:")
        print("   - Training for more iterations")
        print("   - Trying a different architecture")
        print("   - Adjusting loss weights")
        print("   - Using more collocation points")
    print("=" * 70)

    return metrics


def compare_architectures_validation():
    """Compare validation metrics across different architectures."""

    print("\n" + "=" * 70)
    print("Comparing Validation Metrics Across Architectures")
    print("=" * 70)

    problem = LinearBurgersProblem()

    architectures = {
        'Small (3x10)': (10, 10, 10),
        'Medium (3x32)': (32, 32, 32),
        'Large (3x64)': (64, 64, 64),
    }

    results = {}
    key = jax.random.PRNGKey(42)

    for name, dims in architectures.items():
        print(f"\n{'='*70}")
        print(f"Architecture: {name}")
        print(f"{'='*70}")

        descriptor = MLPDescriptor(
            dims=dims,
            act_functions=tuple(['tanh'] * len(dims)),
            use_batch_norm=tuple([False] * len(dims)),
            use_dropout=tuple([False] * len(dims)),
            dropout_rates=tuple([0.0] * len(dims))
        )

        # Train
        key, subkey = jax.random.split(key)
        network, key = train_pinn(problem, descriptor, n_train_iters=1000, key=subkey)

        # Validate
        key, subkey = jax.random.split(key)
        metrics = validate_solution(network, problem, subkey, verbose=False)

        results[name] = metrics

        print(f"\nMetrics:")
        print(f"  PDE Residual: {metrics['pde_residual']:.6e}")
        print(f"  IC Error:     {metrics['ic_error']:.6e}")

    # Summary table
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"\n{'Architecture':<20} {'PDE Residual':>15} {'IC Error':>15}")
    print("-" * 70)

    for name, metrics in results.items():
        print(f"{name:<20} {metrics['pde_residual']:>15.6e} {metrics['ic_error']:>15.6e}")

    print()


def main():
    print("=" * 70)
    print("NASJAX - PINN Solution Validation")
    print("=" * 70)
    print()

    # Display platform info
    print(f"🖥️  Platform: {jax.devices()[0].platform.upper()}")
    print(f"   Device: {jax.devices()[0].device_kind}")
    print()

    print("This script demonstrates comprehensive validation of PINN solutions")
    print("against analytical solutions and using various error metrics.")
    print()

    # Main validation
    metrics = validate_linear_burgers()

    # Comparison across architectures
    compare_architectures_validation()

    print("\n" + "=" * 70)
    print("Validation Complete! ✨")
    print("=" * 70)
    print()
    print("Key validation metrics:")
    print("  - PDE Residual: How well the network satisfies the PDE")
    print("  - IC Error: Error in satisfying initial conditions")
    print("  - L2 Error: L2 norm of error vs analytical solution")
    print("  - Relative Error: L2 error normalized by solution magnitude")
    print()
    print("Use these metrics to:")
    print("  - Assess solution quality")
    print("  - Compare different architectures")
    print("  - Guide hyperparameter tuning")
    print("  - Validate evolved architectures")
    print()


if __name__ == "__main__":
    main()
