"""Evolve PINN architectures for Convection-Diffusion equation.

This example demonstrates evolution for the Convection-Diffusion equation:
    v·∂u/∂x = k·∂²u/∂x²
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Force CPU if GPU/CUDA issues (comment out to use GPU)
# Uncomment the line below if you encounter CUDA/CuDNN errors
# os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp

from nasjax.pinn import ConvectionDiffusionProblem, PINNEvaluator
from nasjax.evolution import Evolving, EvolvingConfig


def main():
    print("=" * 70)
    print("NASJAX - Evolving PINN for Convection-Diffusion Equation")
    print("=" * 70)
    print()

    # Display platform info
    print(f"🖥️  Platform: {jax.devices()[0].platform.upper()}")
    print(f"   Device: {jax.devices()[0].device_kind}")
    print()

    # =========================================================================
    # 1. Define the PDE Problem
    # =========================================================================
    print("📊 Step 1: Defining Convection-Diffusion equation problem...")

    problem = ConvectionDiffusionProblem(
        v=1.0,                    # Velocity (convection)
        k=0.01,                   # Diffusivity
        x_min=0.0,                # Spatial domain
        x_max=2.0,
        t_min=0.0,                # Temporal domain
        t_max=1.0,
        n_collocation=2000,       # Number of PDE residual points
        n_ic=100,                 # Initial condition points
        n_bc=100,                 # Boundary condition points
        loss_weight_pde=1.0,      # PDE loss weight
        loss_weight_ic=1.0,       # IC loss weight
        loss_weight_bc=0.1,       # BC loss weight
    )

    print(f"   PDE: {problem.v}·∂u/∂x = {problem.k}·∂²u/∂x²")
    print(f"   Domain: x ∈ [{problem.x_min}, {problem.x_max}], t ∈ [{problem.t_min}, {problem.t_max}]")
    print(f"   Initial condition: u(x,0) = exp(-((x-1)²)/0.1)")
    print()

    # =========================================================================
    # 2. Create PINN Evaluator
    # =========================================================================
    print("⚙️  Step 2: Configuring PINN evaluator...")

    evaluator = PINNEvaluator(
        problem=problem,
        n_train_iters=200,        # Training iterations per network
        learning_rate=1e-3,       # Adam learning rate
        optimizer='adam',
        verbose=False
    )

    print(f"   Training iterations: {evaluator.n_train_iters}")
    print(f"   Learning rate: {evaluator.learning_rate}")
    print(f"   Optimizer: {evaluator.optimizer_name}")
    print()

    # =========================================================================
    # 3. Configure Evolution
    # =========================================================================
    print("🧬 Step 3: Configuring evolution parameters...")

    config = EvolvingConfig(
        pop_size=8,
        n_generations=5,
        mutation_prob=0.9,
        crossover_prob=0.0,
        elitism=2,
        selection_method='tournament',
        tournament_size=3
    )

    print(f"   Population size: {config.pop_size}")
    print(f"   Generations: {config.n_generations}")
    print(f"   Mutation probability: {config.mutation_prob}")
    print(f"   Elitism: {config.elitism}")
    print()

    # =========================================================================
    # 4. Run Evolution
    # =========================================================================
    print("🚀 Step 4: Running evolutionary algorithm...")
    print()

    evolving = Evolving(
        input_dim=2,              # (x, t) coordinates
        output_dim=1,             # u(x,t) solution
        max_num_layers=4,
        max_num_neurons=32,
        config=config,
        evaluator=evaluator,
        use_crossover=False
    )

    # Run evolution
    key = jax.random.PRNGKey(42)
    population, log = evolving.evolve(
        x_train=jnp.array([]),
        y_train=jnp.array([]),
        x_test=jnp.array([]),
        y_test=jnp.array([]),
        key=key,
        verbose=True
    )

    print()

    # =========================================================================
    # 5. Analyze Results
    # =========================================================================
    print("📈 Step 5: Analyzing results...")
    print()

    # Get best individual
    best_individual = population.get_best(1)[0]
    best_descriptor = best_individual.descriptor

    print("🏆 Best Architecture Found:")
    print(f"   Layers: {len(best_descriptor.dims)}")
    print(f"   Architecture: [{2}] → {list(best_descriptor.dims)} → [1]")
    print(f"   Activations: {best_descriptor.act_functions}")
    print(f"   Best Fitness (Physics Loss): {best_individual.fitness:.6f}")
    print()

    # Evolution statistics
    print("📊 Evolution Statistics:")
    print(f"   Initial best fitness: {log['best_fitness'][0]:.6f}")
    print(f"   Final best fitness: {log['best_fitness'][-1]:.6f}")
    improvement = (log['best_fitness'][0] - log['best_fitness'][-1]) / log['best_fitness'][0] * 100
    print(f"   Improvement: {improvement:.1f}%")
    print()

    # Print generation-by-generation progress
    print("📉 Generation Progress:")
    print(f"   {'Gen':<5} {'Best':>12} {'Mean':>12} {'Std':>12}")
    print(f"   {'-'*5} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(len(log['generation'])):
        gen = log['generation'][i]
        best = log['best_fitness'][i]
        mean = log['mean_fitness'][i]
        std = log['std_fitness'][i]
        print(f"   {gen:<5} {best:>12.6f} {mean:>12.6f} {std:>12.6f}")
    print()

    # =========================================================================
    # 6. Test Best Network
    # =========================================================================
    print("🔬 Step 6: Testing best network...")

    from nasjax.pinn import PINNNetwork

    # Build best network
    key, subkey = jax.random.split(key)
    best_network = PINNNetwork(best_descriptor, subkey)

    # Count parameters
    n_params = best_network.count_parameters()
    print(f"   Total parameters: {n_params:,}")
    print()

    print("=" * 70)
    print("Evolution Complete! ✨")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - Run visualize_solutions.py to see the PINN solution")
    print("  - Run compare_architectures.py to compare different architectures")
    print()


if __name__ == "__main__":
    main()
