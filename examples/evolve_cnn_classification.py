"""Example: Evolving MLPs for Image Classification

This script demonstrates how to use NASJAX to evolve Multi-Layer Perceptron
architectures for image classification. The example uses the digits dataset
from scikit-learn, which contains 8x8 grayscale images of handwritten digits (0-9).

While this example uses MLPs (with flattened images), it demonstrates the
principles that would apply to CNN evolution for image classification tasks.
Future versions of NASJAX will include full CNN support.

The evolutionary algorithm will:
1. Initialize a population of random MLP architectures
2. Evaluate each architecture by training on the digits dataset
3. Select the best performing architectures
4. Apply mutations to create new architectures
5. Repeat for multiple generations

The goal is to evolve networks of relatively low complexity that maximize
classification accuracy on handwritten digit recognition.
"""

import jax
import jax.numpy as jnp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from nasjax.descriptors import MLPDescriptor
from nasjax.networks import MLP
from nasjax.training import train_network
from nasjax.training.losses import accuracy
from nasjax.evolution.mutation import apply_random_mutation


def load_and_prepare_data(test_size=0.25, random_state=42):
    """Load and prepare the digits dataset.
    
    The digits dataset contains 1797 8x8 grayscale images of handwritten digits.
    Images are flattened to 64-dimensional vectors for use with MLPs.
    
    Args:
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Load digits dataset (8x8 images of digits 0-9)
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset: {len(X)} images, shape: 8x8 = {X.shape[1]} features")
    print(f"Classes: {len(np.unique(y))} digits (0-9)")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features (important for neural networks)
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
    
    Builds a network from the descriptor, trains it on the training data,
    and evaluates its accuracy on the test data.
    
    Args:
        descriptor: MLPDescriptor to evaluate
        X_train: Training inputs (flattened images)
        y_train: Training labels (digit classes)
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
        
        # Split keys for network creation and training
        k1, k2 = jax.random.split(key)
        
        # Build network from descriptor
        network = MLP(descriptor, k1)
        
        # Train network with appropriate hyperparameters for digit recognition
        trained_network, history = train_network(
            network,
            X_train,
            y_train,
            n_epochs=30,         # Moderate training
            batch_size=32,       # Small batches for better convergence
            learning_rate=0.001, # Standard learning rate
            optimizer="adam",    # Adam optimizer for stable training
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
        print(f"  Error evaluating individual: {e}")
        return 1.0  # Worst possible fitness on error


def evolve_mlp_for_images(
    X_train,
    y_train,
    X_test,
    y_test,
    population_size=10,
    n_generations=5,
    max_num_layers=4,
    max_num_neurons=100,
    mutation_prob=0.8,
    seed=42,
):
    """Evolve MLP architectures for image classification.
    
    This function implements a simple evolutionary algorithm:
    - Tournament selection for parent selection
    - Elitism (best individual always survives)
    - Mutation-only evolution (no crossover)
    
    Args:
        X_train: Training inputs (flattened images)
        y_train: Training labels
        X_test: Test inputs
        y_test: Test labels
        population_size: Number of individuals in population
        n_generations: Number of evolutionary generations
        max_num_layers: Maximum number of hidden layers
        max_num_neurons: Maximum neurons per layer
        mutation_prob: Probability of mutation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (best_descriptor, best_fitness, history)
    """
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    
    # Determine input and output dimensions
    input_dim = X_train.shape[1]  # 64 for digits (8x8 images)
    output_dim = len(jnp.unique(y_train))  # 10 for digits (0-9)
    
    print("=" * 70)
    print("NASJAX: Evolving MLPs for Handwritten Digit Recognition")
    print("=" * 70)
    print(f"Dataset: Digits ({len(X_train)} train, {len(X_test)} test)")
    print(f"Input dim: {input_dim} (8x8 images flattened)")
    print(f"Output dim: {output_dim} classes")
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
            dropout=False,       # Disable dropout for simplicity
            batch_norm=False,    # Disable batch norm for simplicity
        )
        population.append(descriptor)
        fitness_values.append(float('inf'))  # Not evaluated yet
    
    print(f"  Created {len(population)} random architectures")
    
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
            
            # Convert fitness to accuracy for display
            accuracy_pct = (1 - fitness) * 100
            
            print(f"  Individual {i+1}: Accuracy = {accuracy_pct:.2f}%, "
                  f"Layers = {len(descriptor.dims)}, "
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
        
        print(f"\nGeneration {generation + 1} Summary:")
        print(f"  Best Accuracy:  {best_accuracy:.2f}%")
        print(f"  Avg Accuracy:   {avg_accuracy:.2f}%")
        print(f"  Best Architecture: {list(best_descriptor.dims)}")
        print(f"  Best Activations:  {best_descriptor.act_functions}")
        
        # Early stopping if we achieve very high accuracy
        if best_accuracy >= 97.0:
            print(f"\n🎉 Achieved {best_accuracy:.2f}% accuracy! Stopping early.")
            break
        
        # Create next generation (if not the last generation)
        if generation < n_generations - 1:
            print("\nCreating next generation through mutation...")
            new_population = []
            
            # Elitism: Keep best individual unchanged
            new_population.append(best_descriptor)
            print(f"  Elite: Preserved best individual")
            
            # Fill rest of population with mutated individuals
            for i in range(population_size - 1):
                key, subkey = jax.random.split(key)
                
                # Tournament selection: pick better of two random individuals
                key, k1, k2 = jax.random.split(key, 3)
                idx1 = int(jax.random.randint(k1, (), 0, population_size))
                idx2 = int(jax.random.randint(k2, (), 0, population_size))
                parent = population[idx1] if fitness_values[idx1] < fitness_values[idx2] else population[idx2]
                
                # Apply mutation with given probability
                if jax.random.uniform(subkey) < mutation_prob:
                    key, subkey = jax.random.split(key)
                    child = apply_random_mutation(parent, subkey)
                else:
                    child = parent
                
                new_population.append(child)
            
            population = new_population
            fitness_values = [float('inf')] * population_size
    
    # Final summary
    print("\n" + "=" * 70)
    print("Evolution Complete!")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Best Architecture:  {list(best_descriptor.dims)}")
    print(f"Number of Layers:   {len(best_descriptor.dims)}")
    print(f"Activations:        {best_descriptor.act_functions}")
    print(f"Initializers:       {best_descriptor.init_functions}")
    
    return best_descriptor, best_fitness, history


def main():
    """Run the image classification evolution example."""
    # Load and prepare data
    print("Loading handwritten digits dataset...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Run evolution with relatively low complexity networks
    best_descriptor, best_fitness, history = evolve_mlp_for_images(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        population_size=6,   # Small population for faster execution
        n_generations=4,     # Few generations for demonstration
        max_num_layers=3,    # Low complexity: max 3 hidden layers
        max_num_neurons=64,  # Low complexity: max 64 neurons per layer
        mutation_prob=0.8,   # High mutation rate for exploration
        seed=42,
    )
    
    # Display evolution progress
    print("\n" + "=" * 70)
    print("Evolution History:")
    print("=" * 70)
    print(f"{'Gen':<4} {'Best Acc':<10} {'Avg Acc':<10} {'Architecture'}")
    print("-" * 70)
    for i, (best_fit, avg_fit, arch) in enumerate(zip(
        history['best_fitness'],
        history['avg_fitness'],
        history['best_architecture']
    )):
        best_acc = (1 - best_fit) * 100
        avg_acc = (1 - avg_fit) * 100
        print(f"{i+1:<4} {best_acc:>6.2f}%    {avg_acc:>6.2f}%    {arch}")
    
    print("\n" + "=" * 70)
    print("Note: This example uses MLPs with flattened images.")
    print("Future NASJAX versions will include full CNN evolution support")
    print("for more efficient image processing with convolutional layers.")
    print("=" * 70)
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
