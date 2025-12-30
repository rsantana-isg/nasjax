"""Complete Neural Architecture Search Example using NASJAX.

This example demonstrates the full evolution pipeline:
1. Loading and preparing data
2. Configuring evolution parameters
3. Running the evolutionary algorithm
4. Analyzing results
5. Using the best network
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for development (no install needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force CPU if GPU has issues (you can remove this to use GPU)
# os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nasjax.evolution import Evolving, EvolvingConfig
from nasjax.evaluation import Evaluator
from nasjax.networks import MLP


def main():
    print("=" * 70)
    print("NASJAX - Neural Architecture Search with JAX")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Prepare Dataset
    # =========================================================================
    print("📊 Step 1: Preparing dataset...")

    # Create synthetic binary classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to JAX arrays
    X_train = jnp.array(X_train, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.int32)
    y_test = jnp.array(y_test, dtype=jnp.int32)

    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")
    print(f"   Features:  {X_train.shape[1]}")
    print(f"   Classes:   {len(jnp.unique(y_train))}")
    print()

    # =========================================================================
    # 2. Configure Evolution
    # =========================================================================
    print("⚙️  Step 2: Configuring evolution parameters...")

    # Evolution configuration
    config = EvolvingConfig(
        pop_size=10,              # Population size
        n_generations=5,          # Number of generations
        mutation_prob=0.8,        # Probability of mutation
        crossover_prob=0.2,       # Probability of crossover
        elitism=2,                # Number of elite individuals to preserve
        selection_method="tournament",  # Selection method
        tournament_size=3,        # Tournament size
    )

    # Fitness evaluator configuration
    evaluator = Evaluator(
        loss_fn="cross_entropy",  # Loss function for classification
        optimizer="adam",         # Optimizer
        learning_rate=0.01,       # Learning rate
        n_epochs=20,              # Training epochs per evaluation
        batch_size=32,            # Batch size
        metric="loss",            # Metric to minimize
    )

    print(f"   Population size:    {config.pop_size}")
    print(f"   Generations:        {config.n_generations}")
    print(f"   Mutation prob:      {config.mutation_prob}")
    print(f"   Crossover prob:     {config.crossover_prob}")
    print(f"   Elitism:            {config.elitism}")
    print(f"   Training epochs:    {evaluator.n_epochs}")
    print()

    # =========================================================================
    # 3. Run Evolution
    # =========================================================================
    print("🧬 Step 3: Running evolutionary algorithm...")
    print()

    # Create evolution instance
    evolving = Evolving(
        input_dim=X_train.shape[1],
        output_dim=2,              # Binary classification
        max_num_layers=4,          # Maximum hidden layers
        max_num_neurons=64,        # Maximum neurons per layer
        config=config,
        evaluator=evaluator,
        use_crossover=True,        # Enable crossover
    )

    # Run evolution
    key = jax.random.PRNGKey(42)
    population, log = evolving.evolve(
        X_train, y_train,
        X_test, y_test,
        key=key,
        verbose=True,
    )

    print()

    # =========================================================================
    # 4. Analyze Results
    # =========================================================================
    print("📈 Step 4: Analyzing results...")
    print()

    # Get best individual
    best_individual = population.get_best(1)[0]
    best_descriptor = best_individual.descriptor

    print("🏆 Best Architecture Found:")
    print(f"   Layers:        {len(best_descriptor.dims)}")
    print(f"   Architecture:  {best_descriptor.dims}")
    print(f"   Activations:   {best_descriptor.act_functions}")
    print(f"   Best Fitness:  {best_individual.fitness:.6f}")
    print()

    # Evolution statistics
    print("📊 Evolution Statistics:")
    print(f"   Initial best:  {log['best_fitness'][0]:.6f}")
    print(f"   Final best:    {log['best_fitness'][-1]:.6f}")
    print(f"   Improvement:   {log['best_fitness'][0] - log['best_fitness'][-1]:.6f}")
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
    # 5. Use Best Network
    # =========================================================================
    print("🚀 Step 5: Testing best network...")

    # Build network from best descriptor
    key, subkey = jax.random.split(key)
    best_network = MLP(best_descriptor, subkey)

    # Train the best network for longer
    from nasjax.training import train_network
    from nasjax.training.losses import accuracy

    key, subkey = jax.random.split(key)
    trained_network, history = train_network(
        model=best_network,
        x_train=X_train,
        y_train=y_train,
        n_epochs=50,
        batch_size=32,
        learning_rate=0.01,
        optimizer="adam",
        loss="cross_entropy",
        key=subkey,
        x_val=X_test,
        y_val=y_test,
    )

    # Evaluate accuracy
    key, subkey = jax.random.split(key)
    train_acc = accuracy(trained_network, X_train, y_train, subkey, inference=True)
    test_acc = accuracy(trained_network, X_test, y_test, subkey, inference=True)

    print(f"   Training accuracy:   {train_acc:.4f}")
    print(f"   Test accuracy:       {test_acc:.4f}")
    print()

    # Count parameters
    n_params = best_network.count_parameters()
    print(f"   Total parameters:    {n_params:,}")
    print()

    # =========================================================================
    # 6. Save Results (Optional)
    # =========================================================================
    print("💾 Step 6: Saving results...")

    # Save best descriptor to dict
    descriptor_dict = best_descriptor.to_dict()
    print(f"   Best descriptor saved (serializable)")
    print()

    print("=" * 70)
    print("Evolution Complete! ✨")
    print("=" * 70)


if __name__ == "__main__":
    main()
