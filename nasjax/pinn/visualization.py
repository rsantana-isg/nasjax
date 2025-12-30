"""Visualization tools for PINN solutions.

This module provides utilities for visualizing PINN solutions and
comparing them with analytical solutions.
"""

import jax
import jax.numpy as jnp
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import cm

from .networks import PINNNetwork
from .benchmarks.base import BasePDEProblem


def visualize_pinn_solution_2d(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_points: int = 100,
    filename: Optional[str] = 'pinn_solution.png',
    show: bool = False
):
    """Visualize 2D PINN solution (x, t) → u.

    Creates a contour plot of the solution u(x,t) over the domain.

    Args:
        network: Trained PINN network
        problem: PDE problem (defines domain)
        key: JAX random key
        n_points: Number of points in each dimension for visualization
        filename: Output filename (None to skip saving)
        show: Whether to display the plot

    Example:
        >>> visualize_pinn_solution_2d(network, problem, key, filename='solution.png')
    """
    # Generate grid
    x = jnp.linspace(problem.x_min, problem.x_max, n_points)
    t = jnp.linspace(problem.t_min, problem.t_max, n_points)
    X, T = jnp.meshgrid(x, t)

    # Evaluate network on grid
    coords = jnp.stack([X.flatten(), T.flatten()], axis=1)

    def eval_network(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_pred = jax.vmap(eval_network)(coords)
    U_pred = u_pred.reshape(X.shape)

    # Create plot
    plt.figure(figsize=(12, 5))

    # Contour plot
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, T, U_pred, levels=50, cmap='viridis')
    plt.colorbar(contour, label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PINN Solution: u(x,t)')
    plt.grid(True, alpha=0.3)

    # 3D surface plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, T, U_pred, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title('PINN Solution (3D)')
    plt.colorbar(surf, ax=ax, shrink=0.5)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {filename}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_initial_condition(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_points: int = 100,
    filename: Optional[str] = 'initial_condition.png',
    show: bool = False
):
    """Visualize initial condition: u(x, 0) vs analytical IC.

    Args:
        network: PINN network
        problem: PDE problem
        key: JAX random key
        n_points: Number of points to sample
        filename: Output filename
        show: Whether to display
    """
    x = jnp.linspace(problem.x_min, problem.x_max, n_points)
    t = jnp.zeros_like(x)
    coords = jnp.stack([x, t], axis=1)

    # PINN prediction
    def eval_network(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_pred = jax.vmap(eval_network)(coords).flatten()

    # Analytical IC
    u_true = problem.initial_condition(x.reshape(-1, 1)).flatten()

    # Plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, u_true, 'b-', label='Analytical IC', linewidth=2)
    plt.plot(x, u_pred, 'r--', label='PINN IC', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x, 0)')
    plt.title('Initial Condition Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    error = jnp.abs(u_pred - u_true)
    plt.plot(x, error, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('|u_pred - u_true|')
    plt.title(f'Absolute Error (Mean: {jnp.mean(error):.6f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved IC visualization to {filename}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: dict,
    filename: Optional[str] = 'training_history.png',
    show: bool = False
):
    """Plot training loss history.

    Args:
        history: Dictionary with 'train_loss' key
        filename: Output filename
        show: Whether to display
    """
    train_loss = history.get('train_loss', [])

    if len(train_loss) == 0:
        print("No training history to plot")
        return

    plt.figure(figsize=(10, 4))

    # Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Physics Loss')
    plt.title('Training Progress (Linear Scale)')
    plt.grid(True, alpha=0.3)

    # Log scale
    plt.subplot(1, 2, 2)
    plt.semilogy(train_loss, 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Physics Loss')
    plt.title('Training Progress (Log Scale)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {filename}")

    if show:
        plt.show()
    else:
        plt.close()


def compare_with_analytical(
    network: PINNNetwork,
    problem: BasePDEProblem,
    key: jax.random.PRNGKey,
    n_points: int = 100,
    filename: Optional[str] = 'comparison.png',
    show: bool = False
):
    """Compare PINN solution with analytical solution (if available).

    Args:
        network: PINN network
        problem: PDE problem
        key: JAX random key
        n_points: Number of points for comparison
        filename: Output filename
        show: Whether to display
    """
    if problem.analytical_solution is None:
        print("No analytical solution available for this problem")
        return

    # Generate test points
    x = jnp.linspace(problem.x_min, problem.x_max, n_points)
    t = jnp.linspace(problem.t_min, problem.t_max, n_points)
    X, T = jnp.meshgrid(x, t)
    coords = jnp.stack([X.flatten(), T.flatten()], axis=1)

    # PINN prediction
    def eval_network(point):
        return network(point, key=key, inference=True, compute_derivatives=False)

    u_pred = jax.vmap(eval_network)(coords)
    U_pred = u_pred.reshape(X.shape)

    # Analytical solution
    u_true = problem.analytical_solution(coords)
    U_true = u_true.reshape(X.shape)

    # Error
    error = jnp.abs(U_pred - U_true)
    Error = error.reshape(X.shape)

    # Plot comparison
    fig = plt.figure(figsize=(15, 4))

    # PINN solution
    ax1 = plt.subplot(1, 3, 1)
    contour1 = plt.contourf(X, T, U_pred, levels=50, cmap='viridis')
    plt.colorbar(contour1, label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PINN Solution')

    # Analytical solution
    ax2 = plt.subplot(1, 3, 2)
    contour2 = plt.contourf(X, T, U_true, levels=50, cmap='viridis')
    plt.colorbar(contour2, label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Analytical Solution')

    # Error
    ax3 = plt.subplot(1, 3, 3)
    contour3 = plt.contourf(X, T, Error, levels=50, cmap='hot')
    plt.colorbar(contour3, label='|Error|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Absolute Error (Max: {jnp.max(error):.6f})')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {filename}")

    if show:
        plt.show()
    else:
        plt.close()
