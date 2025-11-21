"""Crossover operators for MLPDescriptor.

This module implements protected crossover operators that combine two parent
descriptors to create valid offspring descriptors. "Protected" means the
offspring is guaranteed to satisfy all architectural constraints.
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from ..descriptors.mlp import MLPDescriptor


def uniform_crossover(
    parent1: MLPDescriptor,
    parent2: MLPDescriptor,
    key: jax.random.PRNGKey,
) -> MLPDescriptor:
    """Uniform crossover: randomly select each property from a parent.

    This is a protected crossover that ensures the offspring has valid architecture.
    For layer-based properties, it selects the number of layers from one parent,
    then copies that many layers from each parent with 50% probability.

    Args:
        parent1: First parent descriptor
        parent2: Second parent descriptor
        key: JAX PRNG key

    Returns:
        Offspring descriptor

    Example:
        >>> p1 = MLPDescriptor(input_dim=10, output_dim=2, dims=(64, 32), ...)
        >>> p2 = MLPDescriptor(input_dim=10, output_dim=2, dims=(128, 64, 32), ...)
        >>> child = uniform_crossover(p1, p2, jax.random.PRNGKey(0))
        >>> child.validate()
        True
    """
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # Select number of layers from one parent
    n_layers = len(parent1.dims) if jax.random.uniform(k1) < 0.5 else len(parent2.dims)

    # Ensure it respects max_num_layers
    max_layers = min(parent1.max_num_layers, parent2.max_num_layers)
    max_neurons = min(parent1.max_num_neurons, parent2.max_num_neurons)
    n_layers = min(n_layers, max_layers)

    # Build layer-wise properties by randomly selecting from parents
    dims = []
    act_functions = []
    init_functions = []
    dropout_probs = []

    for i in range(n_layers):
        key, subkey = jax.random.split(key)

        # Choose which parent to take this layer from
        if jax.random.uniform(subkey) < 0.5:
            # Take from parent1 (if has this layer)
            if i < len(parent1.dims):
                dims.append(min(parent1.dims[i], max_neurons))
                act_functions.append(parent1.act_functions[i])
                init_functions.append(parent1.init_functions[i])
                dropout_probs.append(parent1.dropout_probs[i])
            else:
                # Fallback to parent2
                idx = min(i, len(parent2.dims) - 1)
                dims.append(min(parent2.dims[idx], max_neurons))
                act_functions.append(parent2.act_functions[idx])
                init_functions.append(parent2.init_functions[idx])
                dropout_probs.append(parent2.dropout_probs[idx])
        else:
            # Take from parent2 (if has this layer)
            if i < len(parent2.dims):
                dims.append(min(parent2.dims[i], max_neurons))
                act_functions.append(parent2.act_functions[i])
                init_functions.append(parent2.init_functions[i])
                dropout_probs.append(parent2.dropout_probs[i])
            else:
                # Fallback to parent1
                idx = min(i, len(parent1.dims) - 1)
                dims.append(min(parent1.dims[idx], max_neurons))
                act_functions.append(parent1.act_functions[idx])
                init_functions.append(parent1.init_functions[idx])
                dropout_probs.append(parent1.dropout_probs[idx])

    # Select boolean flags from parents
    use_dropout = parent1.use_dropout if jax.random.uniform(k2) < 0.5 else parent2.use_dropout
    use_batch_norm = (
        parent1.use_batch_norm if jax.random.uniform(k3) < 0.5 else parent2.use_batch_norm
    )

    # Create offspring descriptor
    offspring = MLPDescriptor(
        input_dim=parent1.input_dim,  # Must match both parents
        output_dim=parent1.output_dim,  # Must match both parents
        dims=tuple(dims),
        act_functions=tuple(act_functions),
        init_functions=tuple(init_functions),
        dropout_probs=tuple(dropout_probs),
        use_dropout=use_dropout,
        use_batch_norm=use_batch_norm,
        max_num_layers=max_layers,
        max_num_neurons=min(parent1.max_num_neurons, parent2.max_num_neurons),
    )

    return offspring


def one_point_crossover(
    parent1: MLPDescriptor,
    parent2: MLPDescriptor,
    key: jax.random.PRNGKey,
) -> MLPDescriptor:
    """One-point crossover: split at a random layer index.

    Takes layers 0 to cut_point from parent1, and remaining layers from parent2.
    This is a protected crossover that ensures valid architecture.

    Args:
        parent1: First parent descriptor
        parent2: Second parent descriptor
        key: JAX PRNG key

    Returns:
        Offspring descriptor

    Example:
        >>> p1 = MLPDescriptor(..., dims=(64, 32, 16), ...)
        >>> p2 = MLPDescriptor(..., dims=(128, 64, 32, 16), ...)
        >>> child = one_point_crossover(p1, p2, jax.random.PRNGKey(0))
    """
    k1, k2 = jax.random.split(key)

    # Determine cut point (layers before this come from parent1)
    min_len = min(len(parent1.dims), len(parent2.dims))
    max_len = max(len(parent1.dims), len(parent2.dims))

    # Ensure total length doesn't exceed constraints
    max_layers = min(parent1.max_num_layers, parent2.max_num_layers)
    max_neurons = min(parent1.max_num_neurons, parent2.max_num_neurons)

    # Random cut point
    if min_len > 1:
        cut_point = int(jax.random.randint(k1, (), 1, min_len))
    else:
        cut_point = 1

    # Take layers from parent1 up to cut_point (clamping neuron counts)
    dims = [min(dim, max_neurons) for dim in parent1.dims[:cut_point]]
    act_functions = list(parent1.act_functions[:cut_point])
    init_functions = list(parent1.init_functions[:cut_point])
    dropout_probs = list(parent1.dropout_probs[:cut_point])

    # Add layers from parent2 after cut_point
    remaining = max_layers - cut_point
    if remaining > 0:
        # Take remaining layers from parent2
        p2_start = min(cut_point, len(parent2.dims))
        for i in range(remaining):
            p2_idx = p2_start + i
            if p2_idx < len(parent2.dims):
                dims.append(min(parent2.dims[p2_idx], max_neurons))
                act_functions.append(parent2.act_functions[p2_idx])
                init_functions.append(parent2.init_functions[p2_idx])
                dropout_probs.append(parent2.dropout_probs[p2_idx])
            else:
                break

    # Select boolean flags from a random parent
    use_dropout = parent1.use_dropout if jax.random.uniform(k2) < 0.5 else parent2.use_dropout
    use_batch_norm = parent1.use_batch_norm if jax.random.uniform(k2) < 0.5 else parent2.use_batch_norm

    offspring = MLPDescriptor(
        input_dim=parent1.input_dim,
        output_dim=parent1.output_dim,
        dims=tuple(dims),
        act_functions=tuple(act_functions),
        init_functions=tuple(init_functions),
        dropout_probs=tuple(dropout_probs),
        use_dropout=use_dropout,
        use_batch_norm=use_batch_norm,
        max_num_layers=max_layers,
        max_num_neurons=min(parent1.max_num_neurons, parent2.max_num_neurons),
    )

    return offspring


def layer_wise_crossover(
    parent1: MLPDescriptor,
    parent2: MLPDescriptor,
    key: jax.random.PRNGKey,
) -> MLPDescriptor:
    """Layer-wise crossover: for each layer, mix properties from both parents.

    For each layer, randomly select layer size, activation, initializer, and
    dropout probability from either parent. This creates more diverse offspring
    than uniform crossover.

    Args:
        parent1: First parent descriptor
        parent2: Second parent descriptor
        key: JAX PRNG key

    Returns:
        Offspring descriptor

    Example:
        >>> child = layer_wise_crossover(p1, p2, jax.random.PRNGKey(0))
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Select number of layers
    n_layers = len(parent1.dims) if jax.random.uniform(k1) < 0.5 else len(parent2.dims)
    max_layers = min(parent1.max_num_layers, parent2.max_num_layers)
    max_neurons = min(parent1.max_num_neurons, parent2.max_num_neurons)
    n_layers = min(n_layers, max_layers)

    dims = []
    act_functions = []
    init_functions = []
    dropout_probs = []

    for i in range(n_layers):
        k_dim, k_act, k_init, k_drop, key = jax.random.split(key, 5)

        # For each property, randomly choose from available parent layers
        # Layer size (clamped to max_neurons)
        if i < len(parent1.dims) and i < len(parent2.dims):
            selected_dim = parent1.dims[i] if jax.random.uniform(k_dim) < 0.5 else parent2.dims[i]
            dims.append(min(selected_dim, max_neurons))
        elif i < len(parent1.dims):
            dims.append(min(parent1.dims[i], max_neurons))
        else:
            dims.append(min(parent2.dims[min(i, len(parent2.dims) - 1)], max_neurons))

        # Activation function
        if i < len(parent1.act_functions) and i < len(parent2.act_functions):
            act_functions.append(
                parent1.act_functions[i]
                if jax.random.uniform(k_act) < 0.5
                else parent2.act_functions[i]
            )
        elif i < len(parent1.act_functions):
            act_functions.append(parent1.act_functions[i])
        else:
            act_functions.append(parent2.act_functions[min(i, len(parent2.act_functions) - 1)])

        # Initializer
        if i < len(parent1.init_functions) and i < len(parent2.init_functions):
            init_functions.append(
                parent1.init_functions[i]
                if jax.random.uniform(k_init) < 0.5
                else parent2.init_functions[i]
            )
        elif i < len(parent1.init_functions):
            init_functions.append(parent1.init_functions[i])
        else:
            init_functions.append(parent2.init_functions[min(i, len(parent2.init_functions) - 1)])

        # Dropout probability
        if i < len(parent1.dropout_probs) and i < len(parent2.dropout_probs):
            dropout_probs.append(
                parent1.dropout_probs[i]
                if jax.random.uniform(k_drop) < 0.5
                else parent2.dropout_probs[i]
            )
        elif i < len(parent1.dropout_probs):
            dropout_probs.append(parent1.dropout_probs[i])
        else:
            dropout_probs.append(parent2.dropout_probs[min(i, len(parent2.dropout_probs) - 1)])

    # Boolean flags
    use_dropout = parent1.use_dropout if jax.random.uniform(k2) < 0.5 else parent2.use_dropout
    use_batch_norm = parent1.use_batch_norm if jax.random.uniform(k3) < 0.5 else parent2.use_batch_norm

    offspring = MLPDescriptor(
        input_dim=parent1.input_dim,
        output_dim=parent1.output_dim,
        dims=tuple(dims),
        act_functions=tuple(act_functions),
        init_functions=tuple(init_functions),
        dropout_probs=tuple(dropout_probs),
        use_dropout=use_dropout,
        use_batch_norm=use_batch_norm,
        max_num_layers=max_layers,
        max_num_neurons=min(parent1.max_num_neurons, parent2.max_num_neurons),
    )

    return offspring


def averaged_crossover(
    parent1: MLPDescriptor,
    parent2: MLPDescriptor,
    key: jax.random.PRNGKey,
) -> MLPDescriptor:
    """Averaged crossover: average numeric properties from both parents.

    For layer sizes and dropout probabilities, takes the average of both parents.
    For categorical properties (activation, initializer), randomly selects from parents.
    This can create intermediate architectures.

    Args:
        parent1: First parent descriptor
        parent2: Second parent descriptor
        key: JAX PRNG key

    Returns:
        Offspring descriptor

    Example:
        >>> child = averaged_crossover(p1, p2, jax.random.PRNGKey(0))
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Select number of layers (average, rounded)
    n_layers = int((len(parent1.dims) + len(parent2.dims)) / 2)
    max_layers = min(parent1.max_num_layers, parent2.max_num_layers)
    max_neurons = min(parent1.max_num_neurons, parent2.max_num_neurons)
    n_layers = min(max(n_layers, 1), max_layers)

    dims = []
    act_functions = []
    init_functions = []
    dropout_probs = []

    for i in range(n_layers):
        k_act, k_init, key = jax.random.split(key, 3)

        # Average layer sizes (when both parents have this layer)
        if i < len(parent1.dims) and i < len(parent2.dims):
            avg_dim = int((parent1.dims[i] + parent2.dims[i]) / 2)
            # Clamp to valid range [2, max_neurons]
            dims.append(max(2, min(avg_dim, max_neurons)))
        elif i < len(parent1.dims):
            dims.append(min(parent1.dims[i], max_neurons))
        else:
            dims.append(min(parent2.dims[min(i, len(parent2.dims) - 1)], max_neurons))

        # Randomly select activation
        if i < len(parent1.act_functions) and i < len(parent2.act_functions):
            act_functions.append(
                parent1.act_functions[i]
                if jax.random.uniform(k_act) < 0.5
                else parent2.act_functions[i]
            )
        elif i < len(parent1.act_functions):
            act_functions.append(parent1.act_functions[i])
        else:
            act_functions.append(parent2.act_functions[min(i, len(parent2.act_functions) - 1)])

        # Randomly select initializer
        if i < len(parent1.init_functions) and i < len(parent2.init_functions):
            init_functions.append(
                parent1.init_functions[i]
                if jax.random.uniform(k_init) < 0.5
                else parent2.init_functions[i]
            )
        elif i < len(parent1.init_functions):
            init_functions.append(parent1.init_functions[i])
        else:
            init_functions.append(parent2.init_functions[min(i, len(parent2.init_functions) - 1)])

        # Average dropout probabilities
        if i < len(parent1.dropout_probs) and i < len(parent2.dropout_probs):
            avg_dropout = (parent1.dropout_probs[i] + parent2.dropout_probs[i]) / 2
            dropout_probs.append(avg_dropout)
        elif i < len(parent1.dropout_probs):
            dropout_probs.append(parent1.dropout_probs[i])
        else:
            dropout_probs.append(parent2.dropout_probs[min(i, len(parent2.dropout_probs) - 1)])

    # Boolean flags (random selection)
    use_dropout = parent1.use_dropout if jax.random.uniform(k2) < 0.5 else parent2.use_dropout
    use_batch_norm = parent1.use_batch_norm if jax.random.uniform(k3) < 0.5 else parent2.use_batch_norm

    offspring = MLPDescriptor(
        input_dim=parent1.input_dim,
        output_dim=parent1.output_dim,
        dims=tuple(dims),
        act_functions=tuple(act_functions),
        init_functions=tuple(init_functions),
        dropout_probs=tuple(dropout_probs),
        use_dropout=use_dropout,
        use_batch_norm=use_batch_norm,
        max_num_layers=max_layers,
        max_num_neurons=min(parent1.max_num_neurons, parent2.max_num_neurons),
    )

    return offspring


# Default crossover operators available
CROSSOVER_OPERATORS = [
    uniform_crossover,
    one_point_crossover,
    layer_wise_crossover,
    averaged_crossover,
]


def apply_random_crossover(
    parent1: MLPDescriptor,
    parent2: MLPDescriptor,
    key: jax.random.PRNGKey,
    crossover_operators=None,
) -> MLPDescriptor:
    """Apply a randomly selected crossover operator.

    Args:
        parent1: First parent descriptor
        parent2: Second parent descriptor
        key: JAX PRNG key
        crossover_operators: Optional list of crossover functions to choose from.
            If None, uses all available operators.

    Returns:
        Offspring descriptor created by random crossover

    Example:
        >>> p1 = MLPDescriptor.random_init(10, 2, 3, 64, jax.random.PRNGKey(0))
        >>> p2 = MLPDescriptor.random_init(10, 2, 3, 64, jax.random.PRNGKey(1))
        >>> child = apply_random_crossover(p1, p2, jax.random.PRNGKey(2))
        >>> child.validate()
        True
    """
    if crossover_operators is None:
        crossover_operators = CROSSOVER_OPERATORS

    # Randomly select a crossover operator
    k1, k2 = jax.random.split(key)
    idx = int(jax.random.randint(k1, (), 0, len(crossover_operators)))
    crossover_fn = crossover_operators[idx]

    # Apply crossover
    offspring = crossover_fn(parent1, parent2, k2)

    return offspring
