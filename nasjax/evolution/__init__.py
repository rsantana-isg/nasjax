"""Evolution engine for NASJAX - genetic operators and population management"""

from .mutation import (
    mutate_add_layer,
    mutate_remove_layer,
    mutate_layer_size,
    mutate_activation,
    mutate_initializer,
    mutate_dropout_toggle,
    mutate_dropout_probs,
    mutate_batch_norm_toggle,
    apply_random_mutation,
    MUTATION_OPERATORS,
)

from .crossover import (
    uniform_crossover,
    one_point_crossover,
    layer_wise_crossover,
    averaged_crossover,
    apply_random_crossover,
    CROSSOVER_OPERATORS,
)

from .population import (
    Population,
    Individual,
    initialize_population,
    select_parents,
    tournament_selection,
)

from .evolving import (
    Evolving,
    EvolvingConfig,
    evolve_architecture,
)

__all__ = [
    # Mutation operators
    "mutate_add_layer",
    "mutate_remove_layer",
    "mutate_layer_size",
    "mutate_activation",
    "mutate_initializer",
    "mutate_dropout_toggle",
    "mutate_dropout_probs",
    "mutate_batch_norm_toggle",
    "apply_random_mutation",
    "MUTATION_OPERATORS",
    # Crossover operators
    "uniform_crossover",
    "one_point_crossover",
    "layer_wise_crossover",
    "averaged_crossover",
    "apply_random_crossover",
    "CROSSOVER_OPERATORS",
    # Population management
    "Population",
    "Individual",
    "initialize_population",
    "select_parents",
    "tournament_selection",
    # Main evolution loop
    "Evolving",
    "EvolvingConfig",
    "evolve_architecture",
]
