"""Example: Evolving MLPs for Classification

This script demonstrates how to use NASJAX to evolve Multi-Layer Perceptron
architectures for a classification problem. The example uses the Iris dataset
from scikit-learn, a classic small-scale classification problem.

The evolutionary algorithm will:
1. Initialize a population of random MLP architectures
2. Evaluate each architecture by training on the Iris dataset
3. Select the best performing architectures
4. Apply mutations to create new architectures
5. Repeat for multiple generations

The goal is to evolve networks of relatively low complexity that maximize
classification accuracy.
"""

import jax
import jax.numpy as jnp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from nasjax.descriptors import MLPDescriptor
from nasjax.networks import MLP
from nasjax.training import train_network
from nasjax.training.losses import accuracy
from nasjax.evolution.mutation import apply_random_mutation


def load_and_prepare_data(test_size=0.3, random_state=42):
    """Load and prepare the Iris dataset.
    
    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
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
    
    return X_train, X_test, y_train, y_test


def evaluate_individual(descriptor, X_train, y_train, X_test, y_test, key):
    """Evaluate a single MLP descriptor's fitness.
    
    Args:
        descriptor: MLPDescriptor to evaluate
        X_train: Training inputs
        y_train: Training labels
        X_test: Test inputs
        y_test: Test labels
        key: JAX PRNG key
        
    Returns:
        Fitness value (1 - accuracy, lower is better)
    """
    try:
        # Validate descriptor
        if not descriptor.validate():
            return 1.0  # Worst possible fitness
        
        # Split keys
        k1, k2 = jax.random.split(key)
        
        # Build network from descriptor
        network = MLP(descriptor, k1)
        
        # Train network
        trained_network, _ = train_network(
            network,
            X_train,
            y_train,
            n_epochs=50,
            batch_size=16,
            learning_rate=0.001,
            optimizer="adam",
            loss="cross_entropy",
            key=k2,
        )
        
        # Evaluate on test set
        k3 = jax.random.split(key)[0]
        test_accuracy = accuracy(trained_network, X_test, y_test, k3, inference=True)
        
        # Return 1 - accuracy for minimization (lower is better)
        fitness = 1.0 - float(test_accuracy)
        
        return fitness
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return 1.0  # Worst possible fitness on error


def evolve_mlp(
    X_train,
    y_train,
    X_test,
    y_test,
    population_size=10,
    n_generations=5,
    max_num_layers=4,
    max_num_neurons=50,
    mutation_prob=0.8,
    seed=42,
):
    """Evolve MLP architectures for classification.
    
    Args:
        X_train: Training inputs
        y_train: Training labels
        X_test: Test inputs
        y_test: Test labels
        population_size: Number of individuals in population
        n_generations: Number of evolutionary generations
        max_num_layers: Maximum number of hidden layers
        max_num_neurons: Maximum neurons per layer
        mutation_prob: Probability of mutation
        seed: Random seed
        
    Returns:
        Tuple of (best_individual, best_fitness, history)
    """
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    
    # Determine input and output dimensions
    input_dim = X_train.shape[1]
    output_dim = len(jnp.unique(y_train))
    
    print("=" * 70)
    print("NASJAX: Evolving MLPs for Iris Classification")
    print("=" * 70)
    print(f"Dataset: Iris ({len(X_train)} train, {len(X_test)} test)")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Population size: {population_size}")
    print(f"Generations: {n_generations}")
    print(f"Max layers: {max_num_layers}, Max neurons: {max_num_neurons}")
    print("=" * 70)
    
    # Initialize random population
    population = []
    fitness_values = []
    
    print("\nInitializing population...")
    for i in range(population_size):
        key, subkey = jax.random.split(key)
        descriptor = MLPDescriptor.random_init(
            input_dim=input_dim,
            output_dim=output_dim,
            max_num_layers=max_num_layers,
            max_num_neurons=max_num_neurons,
            key=subkey,
            dropout=False,
            batch_norm=False,
        )
        population.append(descriptor)
        fitness_values.append(float('inf'))  # Not evaluated yet
    
    # Evolution loop
    history = {
        'best_fitness': [],
        'avg_fitness': [],
        'best_architecture': []
    }
    
    for generation in range(n_generations):
        print(f"\n{'='*70}")
        print(f"Generation {generation + 1}/{n_generations}")
        print(f"{'='*70}")
        
        # Evaluate population
        print("Evaluating population...")
        for i, descriptor in enumerate(population):
            key, subkey = jax.random.split(key)
            fitness = evaluate_individual(
                descriptor, X_train, y_train, X_test, y_test, subkey
            )
            fitness_values[i] = fitness
            
            # Convert to accuracy for display (fitness is 1 - accuracy)
            accuracy_pct = (1 - fitness) * 100
            print(f"  Individual {i+1}: Accuracy = {accuracy_pct:.2f}%, "
                  f"Architecture = {list(descriptor.dims)}")
        
        # Find best individual
        best_idx = int(jnp.argmin(jnp.array(fitness_values)))
        best_fitness = fitness_values[best_idx]
        best_descriptor = population[best_idx]
        best_accuracy = (1 - best_fitness) * 100
        
        # Calculate statistics
        avg_fitness = float(jnp.mean(jnp.array(fitness_values)))
        avg_accuracy = (1 - avg_fitness) * 100
        
        # Update history
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(avg_fitness)
        history['best_architecture'].append(list(best_descriptor.dims))
        
        print(f"\nGeneration {generation + 1} Results:")
        print(f"  Best Accuracy: {best_accuracy:.2f}%")
        print(f"  Avg Accuracy:  {avg_accuracy:.2f}%")
        print(f"  Best Architecture: {list(best_descriptor.dims)}")
        
        # Early stopping if we achieve very high accuracy
        if best_accuracy >= 98.0:
            print(f"\n🎉 Achieved {best_accuracy:.2f}% accuracy! Stopping early.")
            break
        
        # Create next generation (if not the last generation)
        if generation < n_generations - 1:
            print("\nCreating next generation...")
            new_population = []
            
            # Elitism: Keep best individual
            new_population.append(best_descriptor)
            
            # Fill rest of population with mutated individuals
            for i in range(population_size - 1):
                key, subkey = jax.random.split(key)
                
                # Select parent (tournament selection)
                key, k1, k2 = jax.random.split(key, 3)
                idx1 = int(jax.random.randint(k1, (), 0, population_size))
                idx2 = int(jax.random.randint(k2, (), 0, population_size))
                parent = population[idx1] if fitness_values[idx1] < fitness_values[idx2] else population[idx2]
                
                # Apply mutation
                if jax.random.uniform(subkey) < mutation_prob:
                    key, subkey = jax.random.split(key)
                    child = apply_random_mutation(parent, subkey)
                else:
                    child = parent
                
                new_population.append(child)
            
            population = new_population
            fitness_values = [float('inf')] * population_size
    
    # Final evaluation of best individual
    print("\n" + "=" * 70)
    print("Evolution Complete!")
    print("=" * 70)
    print(f"Best Architecture: {list(best_descriptor.dims)}")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Activations: {best_descriptor.act_functions}")
    print(f"Initializers: {best_descriptor.init_functions}")
    
    return best_descriptor, best_fitness, history


def main():
    """Run the MLP evolution example."""
    # Load and prepare data
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Run evolution
    best_descriptor, best_fitness, history = evolve_mlp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        population_size=8,  # Small population for quick execution
        n_generations=5,     # Few generations for demonstration
        max_num_layers=3,    # Relatively low complexity
        max_num_neurons=30,  # Small networks
        mutation_prob=0.8,
        seed=42,
    )
    
    # Display evolution progress
    print("\n" + "=" * 70)
    print("Evolution History:")
    print("=" * 70)
    for i, (best_fit, avg_fit, arch) in enumerate(zip(
        history['best_fitness'],
        history['avg_fitness'],
        history['best_architecture']
    )):
        best_acc = (1 - best_fit) * 100
        avg_acc = (1 - avg_fit) * 100
        print(f"Gen {i+1}: Best={best_acc:.2f}%, Avg={avg_acc:.2f}%, Arch={arch}")
    
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
