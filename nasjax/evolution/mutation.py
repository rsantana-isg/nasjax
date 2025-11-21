"""Mutation operators for network descriptors.

This module implements functional mutation operators that transform descriptors
into new descriptors. All mutations are immutable - they return new objects
rather than modifying in place.
"""

from typing import Optional, Callable, List
import jax
import jax.numpy as jnp
from ..descriptors.mlp import MLPDescriptor, ACTIVATIONS, INITIALIZATIONS


def mutate_add_layer(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Add a random hidden layer to the MLP.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key for randomness

    Returns:
        Tuple of (new_descriptor, success_flag)
        If mutation cannot be applied (too many layers), returns (original, False)
    """
    # Check if we can add more layers
    if len(descriptor.dims) >= descriptor.max_num_layers:
        return descriptor, False

    # Split keys for random choices
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Random position to insert new layer
    position = int(jax.random.randint(k1, (), 0, len(descriptor.dims) + 1))

    # Random properties for new layer
    new_dim = int(jax.random.randint(k2, (), 2, descriptor.max_num_neurons + 1))

    act_idx = int(jax.random.randint(k3, (), 0, len(ACTIVATIONS)))
    new_activation = ACTIVATIONS[act_idx]

    init_idx = int(jax.random.randint(k4, (), 0, len(INITIALIZATIONS)))
    new_initializer = INITIALIZATIONS[init_idx]

    new_dropout_prob = float(jax.random.uniform(k4) * 0.5)

    # Create new lists with inserted layer
    new_dims = list(descriptor.dims)
    new_dims.insert(position, new_dim)

    new_acts = list(descriptor.act_functions)
    new_acts.insert(position, new_activation)

    new_inits = list(descriptor.init_functions)
    new_inits.insert(position, new_initializer)

    new_dropout_probs = list(descriptor.dropout_probs)
    new_dropout_probs.insert(position, new_dropout_prob)

    # Create new descriptor
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=tuple(new_dims),
        act_functions=tuple(new_acts),
        init_functions=tuple(new_inits),
        dropout_probs=tuple(new_dropout_probs),
        use_dropout=descriptor.use_dropout,
        use_batch_norm=descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


def mutate_remove_layer(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Remove a random hidden layer from the MLP.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key for randomness

    Returns:
        Tuple of (new_descriptor, success_flag)
        If mutation cannot be applied (only 1 layer), returns (original, False)
    """
    # Check if we can remove layers (need at least 1)
    if len(descriptor.dims) <= 1:
        return descriptor, False

    # Random position to remove
    position = int(jax.random.randint(key, (), 0, len(descriptor.dims)))

    # Create new lists without the removed layer
    new_dims = list(descriptor.dims)
    new_dims.pop(position)

    new_acts = list(descriptor.act_functions)
    new_acts.pop(position)

    new_inits = list(descriptor.init_functions)
    new_inits.pop(position)

    new_dropout_probs = list(descriptor.dropout_probs)
    new_dropout_probs.pop(position)

    # Create new descriptor
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=tuple(new_dims),
        act_functions=tuple(new_acts),
        init_functions=tuple(new_inits),
        dropout_probs=tuple(new_dropout_probs),
        use_dropout=descriptor.use_dropout,
        use_batch_norm=descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


def mutate_layer_size(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Change the size (number of neurons) of a random layer.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key for randomness

    Returns:
        Tuple of (new_descriptor, success_flag)
    """
    k1, k2 = jax.random.split(key)

    # Random layer to modify
    layer_idx = int(jax.random.randint(k1, (), 0, len(descriptor.dims)))

    # Random new size (different from current)
    current_size = descriptor.dims[layer_idx]
    new_size = int(jax.random.randint(k2, (), 2, descriptor.max_num_neurons + 1))

    # If by chance we get the same size, try one more time
    if new_size == current_size:
        new_size = int(jax.random.randint(k2, (), 2, descriptor.max_num_neurons + 1))

    # Create new dims tuple
    new_dims = list(descriptor.dims)
    new_dims[layer_idx] = new_size

    # Create new descriptor
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=tuple(new_dims),
        act_functions=descriptor.act_functions,
        init_functions=descriptor.init_functions,
        dropout_probs=descriptor.dropout_probs,
        use_dropout=descriptor.use_dropout,
        use_batch_norm=descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


def mutate_activation(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Change the activation function of a random layer.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key for randomness

    Returns:
        Tuple of (new_descriptor, success_flag)
    """
    k1, k2 = jax.random.split(key)

    # Random layer to modify
    layer_idx = int(jax.random.randint(k1, (), 0, len(descriptor.act_functions)))

    # Random new activation (different from current if possible)
    current_act = descriptor.act_functions[layer_idx]
    available_acts = [a for a in ACTIVATIONS if a != current_act]

    if not available_acts:
        available_acts = ACTIVATIONS

    act_idx = int(jax.random.randint(k2, (), 0, len(available_acts)))
    new_activation = available_acts[act_idx]

    # Create new activations tuple
    new_acts = list(descriptor.act_functions)
    new_acts[layer_idx] = new_activation

    # Create new descriptor
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=descriptor.dims,
        act_functions=tuple(new_acts),
        init_functions=descriptor.init_functions,
        dropout_probs=descriptor.dropout_probs,
        use_dropout=descriptor.use_dropout,
        use_batch_norm=descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


def mutate_initializer(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Change the weight initializer of a random layer.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key for randomness

    Returns:
        Tuple of (new_descriptor, success_flag)
    """
    k1, k2 = jax.random.split(key)

    # Random layer to modify
    layer_idx = int(jax.random.randint(k1, (), 0, len(descriptor.init_functions)))

    # Random new initializer (different from current if possible)
    current_init = descriptor.init_functions[layer_idx]
    available_inits = [i for i in INITIALIZATIONS if i != current_init]

    if not available_inits:
        available_inits = INITIALIZATIONS

    init_idx = int(jax.random.randint(k2, (), 0, len(available_inits)))
    new_initializer = available_inits[init_idx]

    # Create new initializers tuple
    new_inits = list(descriptor.init_functions)
    new_inits[layer_idx] = new_initializer

    # Create new descriptor
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=descriptor.dims,
        act_functions=descriptor.act_functions,
        init_functions=tuple(new_inits),
        dropout_probs=descriptor.dropout_probs,
        use_dropout=descriptor.use_dropout,
        use_batch_norm=descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


def mutate_dropout_toggle(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Toggle dropout on/off for the entire network.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key (unused but kept for consistency)

    Returns:
        Tuple of (new_descriptor, success_flag)
    """
    # Create new descriptor with toggled dropout
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=descriptor.dims,
        act_functions=descriptor.act_functions,
        init_functions=descriptor.init_functions,
        dropout_probs=descriptor.dropout_probs,
        use_dropout=not descriptor.use_dropout,
        use_batch_norm=descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


def mutate_dropout_probs(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Randomize dropout probabilities for all layers.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key for randomness

    Returns:
        Tuple of (new_descriptor, success_flag)
    """
    # Generate new random dropout probabilities (0 to 0.5)
    new_dropout_probs = tuple(
        float(p) for p in jax.random.uniform(key, (len(descriptor.dims),)) * 0.5
    )

    # Create new descriptor
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=descriptor.dims,
        act_functions=descriptor.act_functions,
        init_functions=descriptor.init_functions,
        dropout_probs=new_dropout_probs,
        use_dropout=descriptor.use_dropout,
        use_batch_norm=descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


def mutate_batch_norm_toggle(
    descriptor: MLPDescriptor, key: jax.random.PRNGKey
) -> tuple[MLPDescriptor, bool]:
    """Toggle batch normalization on/off for the entire network.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key (unused but kept for consistency)

    Returns:
        Tuple of (new_descriptor, success_flag)
    """
    # Create new descriptor with toggled batch normalization
    new_descriptor = MLPDescriptor(
        input_dim=descriptor.input_dim,
        output_dim=descriptor.output_dim,
        dims=descriptor.dims,
        act_functions=descriptor.act_functions,
        init_functions=descriptor.init_functions,
        dropout_probs=descriptor.dropout_probs,
        use_dropout=descriptor.use_dropout,
        use_batch_norm=not descriptor.use_batch_norm,
        max_num_layers=descriptor.max_num_layers,
        max_num_neurons=descriptor.max_num_neurons,
    )

    return new_descriptor, True


# List of all available mutation operators
MUTATION_OPERATORS = [
    mutate_add_layer,
    mutate_remove_layer,
    mutate_layer_size,
    mutate_activation,
    mutate_initializer,
    mutate_dropout_toggle,
    mutate_dropout_probs,
    mutate_batch_norm_toggle,
]


def apply_random_mutation(
    descriptor: MLPDescriptor,
    key: jax.random.PRNGKey,
    mutation_operators: Optional[List[Callable]] = None,
) -> MLPDescriptor:
    """Apply a random mutation to the descriptor.

    Tries mutations until one succeeds or all have been attempted.

    Args:
        descriptor: MLPDescriptor to mutate
        key: JAX PRNG key for randomness
        mutation_operators: Optional list of mutation operators to use.
                          If None, uses all available operators.

    Returns:
        Mutated descriptor (or original if no mutation succeeded)
    """
    if mutation_operators is None:
        mutation_operators = MUTATION_OPERATORS

    # Shuffle mutation operators
    k1, k2 = jax.random.split(key)
    indices = jax.random.permutation(k1, len(mutation_operators))

    # Try mutations in random order until one succeeds
    for idx in indices:
        mutation_fn = mutation_operators[int(idx)]
        k2, subkey = jax.random.split(k2)
        new_descriptor, success = mutation_fn(descriptor, subkey)

        if success:
            return new_descriptor

    # If no mutation succeeded, return original
    return descriptor
