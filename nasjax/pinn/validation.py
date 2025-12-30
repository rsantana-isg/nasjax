"""Validation tools for PINN solutions.

This module provides utilities for validating PINN solutions against
analytical solutions and computing error metrics.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional

from .networks import PINNNetwork
from .benchmarks.base import BasePDEProblem


def compute_l2_error(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_test_points: int = 1000
) -> float:
    """Compute L2 error against analytical solution.

    L2 error = sqrt(mean((u_pred - u_true)²))

    Args:
        network: PINN network
        problem: PDE problem with analytical_solution method
        key: JAX random key
        n_test_points: Number of test points

    Returns:
        L2 error value

    Raises:
        ValueError: If analytical solution is not available
    """
    if problem.analytical_solution is None:
        raise ValueError("Analytical solution not available for this problem")

    # Generate test points
    test_coords = problem.generate_collocation_points(key)
    if len(test_coords) > n_test_points:
        test_coords = test_coords[:n_test_points]

    # PINN predictions
    def eval_network(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_pred = jax.vmap(eval_network)(test_coords)

    # Analytical solution
    u_true = problem.analytical_solution(test_coords)

    # Compute L2 error
    l2_error = jnp.sqrt(jnp.mean((u_pred - u_true) ** 2))

    return float(l2_error)


def compute_relative_error(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_test_points: int = 1000
) -> float:
    """Compute relative L2 error.

    Relative error = L2_error / sqrt(mean(u_true²))

    Args:
        network: PINN network
        problem: PDE problem
        key: JAX random key
        n_test_points: Number of test points

    Returns:
        Relative error (percentage if multiplied by 100)
    """
    if problem.analytical_solution is None:
        raise ValueError("Analytical solution not available")

    # Generate test points
    test_coords = problem.generate_collocation_points(key)
    if len(test_coords) > n_test_points:
        test_coords = test_coords[:n_test_points]

    # PINN predictions
    def eval_network(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_pred = jax.vmap(eval_network)(test_coords)

    # Analytical solution
    u_true = problem.analytical_solution(test_coords)

    # Compute relative error
    l2_error = jnp.sqrt(jnp.mean((u_pred - u_true) ** 2))
    l2_norm = jnp.sqrt(jnp.mean(u_true ** 2))

    relative_error = l2_error / (l2_norm + 1e-10)  # Add small epsilon for stability

    return float(relative_error)


def compute_pde_residual(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_test_points: int = 1000
) -> float:
    """Compute PDE residual (how well the solution satisfies the PDE).

    This is the physics loss evaluated on test points.

    Args:
        network: PINN network
        problem: PDE problem
        key: JAX random key
        n_test_points: Number of test points

    Returns:
        Mean squared PDE residual
    """
    # Generate test points
    k1, k2, k3 = jax.random.split(key, 3)

    coords_dict = {
        'collocation': problem.generate_collocation_points(k1)[:n_test_points],
        'ic': problem.generate_ic_points(k2),
        'bc': problem.generate_bc_points(k3)
    }

    # Compute physics loss
    residual = problem.loss_function(network, coords_dict, key)

    return float(residual)


def validate_solution(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_test_points: int = 1000,
    verbose: bool = True
) -> Dict[str, float]:
    """Comprehensive validation of PINN solution.

    Computes multiple error metrics and returns them in a dictionary.

    Args:
        network: PINN network
        problem: PDE problem
        key: JAX random key
        n_test_points: Number of test points
        verbose: Whether to print results

    Returns:
        Dictionary with error metrics:
        - 'pde_residual': PDE residual
        - 'l2_error': L2 error (if analytical solution available)
        - 'relative_error': Relative error (if analytical solution available)
        - 'ic_error': Initial condition error

    Example:
        >>> metrics = validate_solution(network, problem, key)
        >>> print(f"PDE Residual: {metrics['pde_residual']:.6f}")
        >>> print(f"L2 Error: {metrics['l2_error']:.6f}")
    """
    metrics = {}

    # PDE residual
    pde_residual = compute_pde_residual(network, problem, key, n_test_points)
    metrics['pde_residual'] = pde_residual

    # L2 error (if analytical solution available)
    if problem.analytical_solution is not None:
        l2_error = compute_l2_error(network, problem, key, n_test_points)
        metrics['l2_error'] = l2_error

        relative_error = compute_relative_error(network, problem, key, n_test_points)
        metrics['relative_error'] = relative_error

    # Initial condition error
    ic_error = compute_ic_error(network, problem, key)
    metrics['ic_error'] = ic_error

    # Print results if verbose
    if verbose:
        print("=" * 60)
        print("PINN Solution Validation")
        print("=" * 60)
        print(f"PDE Residual:       {pde_residual:.6e}")
        print(f"IC Error:           {ic_error:.6e}")

        if 'l2_error' in metrics:
            print(f"L2 Error:           {metrics['l2_error']:.6e}")
            print(f"Relative Error:     {metrics['relative_error']:.2%}")

        print("=" * 60)

    return metrics


def compute_ic_error(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_ic_points: int = 200
) -> float:
    """Compute error in satisfying initial condition.

    Args:
        network: PINN network
        problem: PDE problem
        key: JAX random key
        n_ic_points: Number of IC points to test

    Returns:
        Mean squared error in initial condition
    """
    # Generate IC points
    ic_coords = problem.generate_ic_points(key)
    if len(ic_coords) > n_ic_points:
        ic_coords = ic_coords[:n_ic_points]

    # Network prediction at t=0
    def eval_network(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_pred = jax.vmap(eval_network)(ic_coords)

    # Analytical IC
    u_true = problem.initial_condition(ic_coords[:, 0:1])

    # Compute MSE
    ic_error = jnp.mean((u_pred - u_true) ** 2)

    return float(ic_error)


def compute_point_wise_error(
    network: PINNNetwork,
    problem: BasePDEProblem,
    coords: jnp.ndarray,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Compute point-wise error at specific coordinates.

    Args:
        network: PINN network
        problem: PDE problem
        coords: Coordinates to evaluate (n_points, n_dims)
        key: JAX random key

    Returns:
        Point-wise absolute errors (n_points,)
    """
    if problem.analytical_solution is None:
        raise ValueError("Analytical solution not available")

    # PINN predictions
    def eval_network(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_pred = jax.vmap(eval_network)(coords)

    # Analytical solution
    u_true = problem.analytical_solution(coords)

    # Point-wise error
    errors = jnp.abs(u_pred - u_true).flatten()

    return errors
