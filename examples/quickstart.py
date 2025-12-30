"""Quickstart Example for NASJAX.

This is a minimal example showing how to evolve a neural network architecture
for a simple classification task.
"""

import sys
from pathlib import Path

# Add parent directory to path for development (no install needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nasjax.evolution import evolve_architecture


def main():
    # Generate data
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=2,
        random_state=42
    )

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = jnp.array(scaler.fit_transform(X_train), dtype=jnp.float32)
    X_test = jnp.array(scaler.transform(X_test), dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.int32)
    y_test = jnp.array(y_test, dtype=jnp.int32)

    print("Running Neural Architecture Search...")
    print(f"Dataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print()

    # Run evolution (one-liner!)
    best_descriptor, log = evolve_architecture(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        input_dim=10,
        output_dim=2,
        key=jax.random.PRNGKey(42),
        pop_size=8,
        n_generations=3,
        max_num_layers=3,
        max_num_neurons=32,
        n_epochs=10,
        loss_fn="cross_entropy",
        use_crossover=True,
        verbose=True,
    )

    print()
    print("Best Architecture:")
    print(f"  Layers: {best_descriptor.dims}")
    print(f"  Activations: {best_descriptor.act_functions}")
    print(f"  Best Fitness: {log['best_fitness'][-1]:.6f}")
    print()
    print("Done! ✨")


if __name__ == "__main__":
    main()
