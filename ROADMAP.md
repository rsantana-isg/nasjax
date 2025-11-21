# DEATF to NASJAX Migration Roadmap

## Executive Summary

This roadmap outlines the step-by-step process for migrating DEATF (Distributed Evolutionary Algorithms in TensorFlow) to NASJAX, a modern JAX-based neuroevolution library using Equinox for neural network definitions and Optax for optimization.

**Timeline Estimate:** 8-12 weeks for full migration
**Difficulty Level:** Intermediate to Advanced
**Key Technologies:** JAX, Equinox, Optax, PyTrees

---

## Project Structure

```
nasjax/
├── nasjax/                          # Main package
│   ├── __init__.py
│   ├── descriptors/                 # Network descriptor PyTrees
│   │   ├── __init__.py
│   │   ├── base.py                 # Base descriptor classes
│   │   ├── mlp.py                  # MLP descriptor
│   │   ├── cnn.py                  # CNN descriptor
│   │   ├── rnn.py                  # RNN descriptor
│   │   └── tcnn.py                 # Transposed CNN descriptor
│   ├── networks/                    # Equinox network implementations
│   │   ├── __init__.py
│   │   ├── mlp.py                  # MLP network
│   │   ├── cnn.py                  # CNN network
│   │   ├── rnn.py                  # RNN network
│   │   └── tcnn.py                 # TCNN network
│   ├── evolution/                   # Evolution engine
│   │   ├── __init__.py
│   │   ├── population.py           # Population management
│   │   ├── selection.py            # Selection operators
│   │   ├── crossover.py            # Crossover operators
│   │   └── mutation.py             # Mutation operators
│   ├── training/                    # Training utilities
│   │   ├── __init__.py
│   │   ├── optimizers.py           # Optimizer wrappers
│   │   ├── losses.py               # Loss functions
│   │   └── trainer.py              # Training loops
│   ├── evaluation/                  # Fitness evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py            # Base evaluator
│   │   └── metrics.py              # Evaluation metrics
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── random.py               # PRNG utilities
│       ├── serialization.py        # Save/load utilities
│       └── validation.py           # Input validation
├── examples/                        # Usage examples
│   ├── simple_mlp.py
│   ├── cnn_classification.py
│   ├── rnn_sequence.py
│   └── multi_network.py
├── tests/                           # Unit tests
│   ├── test_descriptors.py
│   ├── test_networks.py
│   ├── test_evolution.py
│   └── test_training.py
├── benchmarks/                      # Performance benchmarks
├── docs/                            # Documentation
├── setup.py
├── pyproject.toml
├── requirements.txt
├── TRANSFORMATIONS.md              # This document
├── ROADMAP.md                      # This roadmap
└── README.md
```

---

## Phase 1: Foundation and Setup (Week 1)

### 1.1 Project Initialization

**Tasks:**
- [x] Create project structure
- [ ] Setup `pyproject.toml` with JAX dependencies
- [ ] Create basic `setup.py`
- [ ] Initialize git repository
- [ ] Setup pre-commit hooks (black, isort, mypy)

**Dependencies to Add:**
```toml
[project]
name = "nasjax"
version = "0.1.0"
dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "equinox>=0.11.0",
    "optax>=0.1.7",
    "chex>=0.1.8",
    "jaxtyping>=0.2.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "sphinx>=7.1.0",
]
```

**Deliverables:**
- ✅ Working package structure
- ✅ All dependencies installable
- ✅ Basic CI/CD pipeline (GitHub Actions)

---

## Phase 2: Core Descriptor Implementation (Week 2-3)

### 2.1 Base Descriptor Classes

**File:** `nasjax/descriptors/base.py`

**Tasks:**
1. Implement base descriptor with PyTree support
2. Define common attributes (input_dim, output_dim, etc.)
3. Create validation utilities
4. Implement serialization support

**Implementation Checklist:**
- [ ] `BaseDescriptor` abstract class
- [ ] PyTree registration (`tree_flatten`, `tree_unflatten`)
- [ ] Immutable dataclass structure (using `chex.dataclass` or `NamedTuple`)
- [ ] Common utility methods (`validate()`, `copy()`, `replace()`)
- [ ] Type annotations with `jaxtyping`

**Code Template:**
```python
import chex
from typing import List, Optional
from jaxtyping import Array, Float

@chex.dataclass
class BaseDescriptor:
    """Base class for all network descriptors"""
    input_dim: int
    output_dim: int
    max_num_layers: int
    max_num_neurons: int

    def validate(self) -> bool:
        """Validate descriptor constraints"""
        raise NotImplementedError

    def replace(self, **changes):
        """Functional update - returns new descriptor"""
        return dataclasses.replace(self, **changes)
```

### 2.2 MLP Descriptor

**File:** `nasjax/descriptors/mlp.py`

**Tasks:**
1. Port `MLPDescriptor` from DEATF
2. Convert lists to tuples (immutable)
3. Replace numpy with JAX arrays where needed
4. Implement random initialization (functional style)

**Implementation Checklist:**
- [ ] `MLPDescriptor` dataclass
- [ ] `random_init()` function (takes PRNG key)
- [ ] Validation logic
- [ ] Helper functions for shape calculations
- [ ] Unit tests

**Code Template:**
```python
@chex.dataclass
class MLPDescriptor(BaseDescriptor):
    dims: Tuple[int, ...]  # Immutable tuple
    act_functions: Tuple[str, ...]
    init_functions: Tuple[str, ...]
    dropout_probs: Tuple[float, ...]
    batch_norm: bool
    dropout: bool

    @staticmethod
    def random_init(input_dim: int,
                   output_dim: int,
                   max_num_layers: int,
                   max_num_neurons: int,
                   key: jax.random.PRNGKey,
                   dropout: bool = False,
                   batch_norm: bool = False) -> 'MLPDescriptor':
        """Create random MLP descriptor"""
        # Implementation
        pass
```

### 2.3 CNN, RNN, TCNN Descriptors

**Files:**
- `nasjax/descriptors/cnn.py`
- `nasjax/descriptors/rnn.py`
- `nasjax/descriptors/tcnn.py`

**Tasks:**
1. Port each descriptor type
2. Implement shape calculation helpers
3. Add validation specific to each type
4. Create comprehensive unit tests

**Priority Order:**
1. MLP (simplest, core functionality)
2. CNN (moderately complex)
3. RNN (complex, recurrent state)
4. TCNN (similar to CNN, lower priority)

**Deliverables:**
- ✅ All descriptor classes implemented
- ✅ Full test coverage (>90%)
- ✅ Documentation with examples

---

## Phase 3: Network Implementations (Week 3-4)

### 3.1 Base Network Class

**File:** `nasjax/networks/base.py`

**Tasks:**
1. Define `eqx.Module` base class
2. Implement common activation/initialization mappings
3. Create utilities for building layers from descriptors

**Code Template:**
```python
import equinox as eqx
import jax
import jax.numpy as jnp

class BaseNetwork(eqx.Module):
    """Base class for all networks"""

    @staticmethod
    def get_activation(name: str):
        """Map activation name to JAX function"""
        activations = {
            'relu': jax.nn.relu,
            'elu': jax.nn.elu,
            'sigmoid': jax.nn.sigmoid,
            'tanh': jax.nn.tanh,
            'softplus': jax.nn.softplus,
            'softsign': jax.nn.softsign,
            None: lambda x: x,
        }
        return activations[name]

    @staticmethod
    def get_initializer(name: str):
        """Map initializer name to function"""
        # Implementation
        pass
```

### 3.2 MLP Network

**File:** `nasjax/networks/mlp.py`

**Tasks:**
1. Implement `MLP` as `eqx.Module`
2. Build layers from `MLPDescriptor`
3. Implement `__call__` with dropout support
4. Add inference mode

**Implementation Checklist:**
- [ ] `MLP` class inheriting from `eqx.Module`
- [ ] Layer construction from descriptor
- [ ] Forward pass with dropout/batch norm
- [ ] Inference mode (no dropout)
- [ ] PRNG key threading
- [ ] Unit tests with known outputs

**Code Template:**
```python
class MLP(BaseNetwork):
    layers: List[eqx.nn.Linear]
    act_functions: List
    dropout_probs: Tuple[float, ...]
    use_dropout: bool
    use_batch_norm: bool
    batch_norm_layers: Optional[List]

    def __init__(self, descriptor: MLPDescriptor, key: jax.random.PRNGKey):
        """Build MLP from descriptor"""
        keys = jax.random.split(key, len(descriptor.dims) + 1)

        self.layers = []
        in_dim = descriptor.input_dim

        for i, out_dim in enumerate(descriptor.dims):
            layer = eqx.nn.Linear(in_dim, out_dim, key=keys[i])
            self.layers.append(layer)
            in_dim = out_dim

        # Final output layer
        self.layers.append(eqx.nn.Linear(in_dim, descriptor.output_dim, key=keys[-1]))

        self.act_functions = [self.get_activation(f) for f in descriptor.act_functions]
        self.dropout_probs = descriptor.dropout_probs
        self.use_dropout = descriptor.dropout
        self.use_batch_norm = descriptor.batch_norm

    def __call__(self, x, key=None, *, inference=False):
        """Forward pass"""
        for i, (layer, act_fn, drop_prob) in enumerate(zip(
            self.layers[:-1], self.act_functions, self.dropout_probs
        )):
            x = layer(x)
            x = act_fn(x)

            if self.use_dropout and not inference:
                key, subkey = jax.random.split(key)
                mask = jax.random.bernoulli(subkey, 1 - drop_prob, x.shape)
                x = jnp.where(mask, x / (1 - drop_prob), 0)

        # Final layer
        x = self.layers[-1](x)
        return x
```

### 3.3 CNN, RNN, TCNN Networks

**Files:**
- `nasjax/networks/cnn.py`
- `nasjax/networks/rnn.py`
- `nasjax/networks/tcnn.py`

**Tasks:**
1. Implement each network type using Equinox
2. Handle special cases (pooling, recurrence, transposed convs)
3. Extensive testing

**Special Considerations:**

**CNN:**
- Mixed layer types (Conv, MaxPool, AvgPool)
- Custom pooling implementation needed
- Shape tracking

**RNN:**
- Recurrent cells (LSTM, GRU, SimpleRNN)
- Bidirectional support
- Sequence processing with `jax.lax.scan`

**TCNN:**
- Transposed convolutions
- Output shape calculations
- Generative tasks

**Deliverables:**
- ✅ All network implementations
- ✅ Validated against TensorFlow outputs
- ✅ Performance benchmarks

---

## Phase 4: Evolution Engine (Week 5-6)

### 4.1 Mutation Operators

**File:** `nasjax/evolution/mutation.py`

**Tasks:**
1. Port all mutation types to functional style
2. Ensure all mutations return new descriptors
3. Implement mutation probability logic

**Implementation Pattern:**
```python
def mutate_add_layer(descriptor: MLPDescriptor,
                     key: jax.random.PRNGKey) -> MLPDescriptor:
    """Add layer - returns new descriptor"""
    if len(descriptor.dims) >= descriptor.max_num_layers:
        return descriptor  # Cannot add, return unchanged

    key, k1, k2, k3 = jax.random.split(key, 4)

    # Random position
    pos = jax.random.randint(k1, (), 0, len(descriptor.dims))

    # Random layer properties
    new_dim = jax.random.randint(k2, (), 2, descriptor.max_num_neurons)
    new_act = jax.random.choice(k3, jnp.array([0, 1, 2, 3, 4, 5, 6]))

    # Create new descriptor with added layer
    new_dims = tuple(descriptor.dims[:pos]) + (new_dim,) + tuple(descriptor.dims[pos:])
    # ... similar for other attributes

    return descriptor.replace(dims=new_dims, ...)
```

**Mutation Types to Implement:**

**MLP:**
- [ ] `mutate_add_layer`
- [ ] `mutate_remove_layer`
- [ ] `mutate_change_dimension`
- [ ] `mutate_activation`
- [ ] `mutate_weight_init`
- [ ] `mutate_dropout`
- [ ] `mutate_batch_norm`

**CNN:**
- [ ] `mutate_add_conv_layer`
- [ ] `mutate_remove_conv_layer`
- [ ] `mutate_change_filters`
- [ ] `mutate_change_strides`
- [ ] `mutate_layer_type`

**RNN:**
- [ ] `mutate_add_rnn_layer`
- [ ] `mutate_remove_rnn_layer`
- [ ] `mutate_change_rnn_type`
- [ ] `mutate_change_units`
- [ ] `mutate_change_bidirectional`

**Master Mutation Function:**
```python
def apply_mutation(descriptor: BaseDescriptor,
                  key: jax.random.PRNGKey,
                  mutation_probs: Dict[str, float]) -> BaseDescriptor:
    """Apply random mutation to descriptor"""

    # Get available mutations for descriptor type
    mutations = get_mutations_for_type(type(descriptor))

    # Select mutation
    key, subkey = jax.random.split(key)
    mutation_fn = jax.random.choice(subkey, mutations)

    # Apply mutation
    return mutation_fn(descriptor, key)
```

### 4.2 Crossover Operators

**File:** `nasjax/evolution/crossover.py`

**Tasks:**
1. Implement component exchange
2. Support multi-network individuals
3. Ensure valid offspring

**Code Template:**
```python
def crossover(parent1: Individual,
             parent2: Individual,
             key: jax.random.PRNGKey) -> Tuple[Individual, Individual]:
    """Single-point or uniform crossover"""

    # Select crossover points
    key, subkey = jax.random.split(key)
    n_components = len(parent1.descriptors)
    cx_mask = jax.random.bernoulli(subkey, 0.5, (n_components,))

    # Exchange components
    offspring1_descs = {}
    offspring2_descs = {}

    for i, (key, desc1, desc2) in enumerate(zip(
        parent1.descriptors.keys(),
        parent1.descriptors.values(),
        parent2.descriptors.values()
    )):
        if cx_mask[i]:
            offspring1_descs[key] = desc1
            offspring2_descs[key] = desc2
        else:
            offspring1_descs[key] = desc2
            offspring2_descs[key] = desc1

    return (
        Individual(descriptors=offspring1_descs, hyperparameters=parent1.hyperparameters),
        Individual(descriptors=offspring2_descs, hyperparameters=parent2.hyperparameters)
    )
```

### 4.3 Selection Operators

**File:** `nasjax/evolution/selection.py`

**Tasks:**
1. Implement tournament selection
2. Implement roulette selection
3. Implement NSGA-II (multi-objective)
4. Implement elitism

**Implementation Checklist:**
- [ ] `select_tournament`
- [ ] `select_roulette`
- [ ] `select_best` (elitism)
- [ ] `select_nsga2` (multi-objective)
- [ ] `select_random`

**Code Template:**
```python
def select_tournament(population: List[Individual],
                     fitness_values: jnp.ndarray,
                     n_select: int,
                     tournament_size: int,
                     key: jax.random.PRNGKey) -> List[Individual]:
    """Tournament selection"""

    selected = []
    for i in range(n_select):
        key, subkey = jax.random.split(key)

        # Random tournament
        contestants = jax.random.choice(
            subkey, len(population), (tournament_size,), replace=False
        )

        # Best in tournament
        winner_idx = contestants[jnp.argmin(fitness_values[contestants])]
        selected.append(population[winner_idx])

    return selected
```

### 4.4 Population Management

**File:** `nasjax/evolution/population.py`

**Tasks:**
1. Implement population initialization
2. Create population update logic
3. Add statistics tracking

**Code Template:**
```python
@chex.dataclass
class Population:
    individuals: List[Individual]
    fitness_values: jnp.ndarray
    generation: int
    statistics: Dict[str, float]

    def update_fitness(self, new_fitness: jnp.ndarray) -> 'Population':
        """Functional update of fitness"""
        return self.replace(fitness_values=new_fitness)

    def get_best(self, n: int = 1) -> List[Individual]:
        """Get n best individuals"""
        indices = jnp.argsort(self.fitness_values)[:n]
        return [self.individuals[i] for i in indices]

def initialize_population(descriptor_classes: List,
                         n_inputs: List,
                         n_outputs: List,
                         pop_size: int,
                         key: jax.random.PRNGKey,
                         **kwargs) -> Population:
    """Create random initial population"""

    keys = jax.random.split(key, pop_size)
    individuals = []

    for k in keys:
        # Create random descriptors
        descriptors = {}
        for i, desc_class in enumerate(descriptor_classes):
            k, subkey = jax.random.split(k)
            descriptors[f"n{i}"] = desc_class.random_init(
                n_inputs[i], n_outputs[i], key=subkey, **kwargs
            )

        individuals.append(Individual(descriptors=descriptors, hyperparameters={}))

    return Population(
        individuals=individuals,
        fitness_values=jnp.full(pop_size, jnp.inf),
        generation=0,
        statistics={}
    )
```

**Deliverables:**
- ✅ All evolution operators implemented
- ✅ Population management working
- ✅ Unit tests for each operator

---

## Phase 5: Training and Evaluation (Week 7-8)

### 5.1 Training Loop

**File:** `nasjax/training/trainer.py`

**Tasks:**
1. Implement JAX training loop
2. Support different optimizers (Optax)
3. Add batching support
4. JIT compilation

**Code Template:**
```python
import optax

class Trainer:
    """Handles training of networks"""

    def __init__(self,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.01):
        self.optimizer = self._get_optimizer(optimizer, learning_rate)

    def train(self,
             model: eqx.Module,
             x_train: jnp.ndarray,
             y_train: jnp.ndarray,
             batch_size: int,
             n_iters: int,
             key: jax.random.PRNGKey,
             loss_fn=None) -> eqx.Module:
        """Train model for n iterations"""

        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def train_step(model, opt_state, x_batch, y_batch, key):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(
                model, x_batch, y_batch, key
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        # Training loop
        for epoch in range(n_iters):
            # Create batches
            for batch_x, batch_y in get_batches(x_train, y_train, batch_size, key):
                key, subkey = jax.random.split(key)
                model, opt_state, loss = train_step(
                    model, opt_state, batch_x, batch_y, subkey
                )

        return model

    @staticmethod
    def _get_optimizer(name: str, lr: float):
        optimizers = {
            'adam': optax.adam(lr),
            'sgd': optax.sgd(lr),
            'rmsprop': optax.rmsprop(lr),
            'adamw': optax.adamw(lr),
        }
        return optimizers[name]
```

### 5.2 Loss Functions

**File:** `nasjax/training/losses.py`

**Tasks:**
1. Implement MSE loss
2. Implement cross-entropy loss
3. Support custom losses

**Code Template:**
```python
def mse_loss(model, x, y, key, inference=False):
    """Mean squared error loss"""
    pred = jax.vmap(lambda x_i: model(x_i, key, inference=inference))(x)
    return jnp.mean((pred - y) ** 2)

def cross_entropy_loss(model, x, y, key, inference=False):
    """Cross-entropy loss for classification"""
    logits = jax.vmap(lambda x_i: model(x_i, key, inference=inference))(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

def get_loss_function(name: str):
    """Get loss function by name"""
    losses = {
        'mse': mse_loss,
        'cross_entropy': cross_entropy_loss,
        'xentropy': cross_entropy_loss,
    }
    return losses[name]
```

### 5.3 Fitness Evaluator

**File:** `nasjax/evaluation/evaluator.py`

**Tasks:**
1. Build network from descriptor
2. Train network
3. Evaluate on test set
4. Return fitness value

**Code Template:**
```python
class Evaluator:
    """Evaluates fitness of individuals"""

    def __init__(self,
                 loss_fn: str = 'mse',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 n_iters: int = 50):
        self.loss_fn = get_loss_function(loss_fn)
        self.trainer = Trainer(optimizer, learning_rate)
        self.batch_size = batch_size
        self.n_iters = n_iters

    def evaluate(self,
                descriptor: BaseDescriptor,
                network_class,
                x_train: jnp.ndarray,
                y_train: jnp.ndarray,
                x_test: jnp.ndarray,
                y_test: jnp.ndarray,
                key: jax.random.PRNGKey) -> float:
        """Evaluate single descriptor"""

        # Build network
        key, subkey = jax.random.split(key)
        model = network_class(descriptor, subkey)

        # Train
        key, subkey = jax.random.split(key)
        model = self.trainer.train(
            model, x_train, y_train,
            self.batch_size, self.n_iters,
            subkey, self.loss_fn
        )

        # Evaluate
        key, subkey = jax.random.split(key)
        test_loss = self.loss_fn(model, x_test, y_test, subkey, inference=True)

        return float(test_loss)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_batch(self,
                      descriptors: List[BaseDescriptor],
                      network_class,
                      data,
                      keys: jax.random.PRNGKey) -> jnp.ndarray:
        """Vectorized evaluation of multiple descriptors"""
        return jax.vmap(
            lambda desc, key: self.evaluate(desc, network_class, *data, key)
        )(descriptors, keys)
```

**Deliverables:**
- ✅ Training pipeline working
- ✅ Evaluation pipeline working
- ✅ JIT compilation enabled
- ✅ Benchmarks vs TensorFlow

---

## Phase 6: Main Evolution Loop (Week 9)

### 6.1 Evolving Class

**File:** `nasjax/evolution/evolving.py`

**Tasks:**
1. Port main `Evolving` class API
2. Integrate all components
3. Maintain similar user interface to DEATF

**Code Template:**
```python
class Evolving:
    """Main class for neuroevolution"""

    def __init__(self,
                 desc_list: List,
                 x_trains: List[jnp.ndarray],
                 y_trains: List[jnp.ndarray],
                 x_tests: List[jnp.ndarray],
                 y_tests: List[jnp.ndarray],
                 n_inputs: List,
                 n_outputs: List,
                 evaluation: str = 'mse',
                 population: int = 20,
                 generations: int = 10,
                 batch_size: int = 32,
                 iters: int = 50,
                 learning_rate: float = 0.01,
                 max_num_layers: int = 10,
                 max_num_neurons: int = 100,
                 selection: str = 'tournament',
                 mutation_prob: float = 0.8,
                 crossover_prob: float = 0.2,
                 seed: int = 0,
                 **kwargs):

        self.key = jax.random.PRNGKey(seed)

        # Store configuration
        self.desc_list = desc_list
        self.population_size = population
        self.generations = generations
        self.evaluation = evaluation

        # Initialize evaluator
        self.evaluator = Evaluator(
            loss_fn=evaluation,
            batch_size=batch_size,
            n_iters=iters,
            learning_rate=learning_rate
        )

        # Initialize population
        self.key, subkey = jax.random.split(self.key)
        self.population = initialize_population(
            desc_list, n_inputs, n_outputs,
            population, subkey,
            max_num_layers=max_num_layers,
            max_num_neurons=max_num_neurons,
            **kwargs
        )

        # Store data
        self.data = {
            'x_trains': x_trains,
            'y_trains': y_trains,
            'x_tests': x_tests,
            'y_tests': y_tests
        }

        # Evolution parameters
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.selection_fn = self._get_selection_fn(selection)

    def evolve(self) -> Tuple[Population, Dict, List[Individual]]:
        """Run evolution for specified generations"""

        logbook = []
        hall_of_fame = []

        for gen in range(self.generations):
            print(f"Generation {gen + 1}/{self.generations}")

            # Evaluate population
            self.key, subkey = jax.random.split(self.key)
            fitness_values = self._evaluate_population(subkey)
            self.population = self.population.update_fitness(fitness_values)

            # Update hall of fame
            best = self.population.get_best(n=1)[0]
            hall_of_fame.append(best)

            # Statistics
            stats = {
                'gen': gen,
                'min': float(jnp.min(fitness_values)),
                'max': float(jnp.max(fitness_values)),
                'avg': float(jnp.mean(fitness_values)),
                'std': float(jnp.std(fitness_values))
            }
            logbook.append(stats)
            print(stats)

            # Selection
            self.key, subkey = jax.random.split(self.key)
            selected = self.selection_fn(
                self.population.individuals,
                fitness_values,
                self.population_size,
                subkey
            )

            # Crossover
            self.key, subkey = jax.random.split(self.key)
            offspring = self._apply_crossover(selected, subkey)

            # Mutation
            self.key, subkey = jax.random.split(self.key)
            offspring = self._apply_mutation(offspring, subkey)

            # Update population
            self.population = self.population.replace(individuals=offspring)

        return self.population, logbook, hall_of_fame

    def _evaluate_population(self, key):
        """Evaluate all individuals in population"""
        keys = jax.random.split(key, len(self.population.individuals))
        fitness_values = []

        for ind, k in zip(self.population.individuals, keys):
            # Get descriptor and network class
            desc = ind.descriptors['n0']  # Simplified for now
            network_class = self._get_network_class(type(desc))

            # Evaluate
            fitness = self.evaluator.evaluate(
                desc, network_class,
                self.data['x_trains'][0],
                self.data['y_trains'][0],
                self.data['x_tests'][0],
                self.data['y_tests'][0],
                k
            )
            fitness_values.append(fitness)

        return jnp.array(fitness_values)

    def _apply_mutation(self, individuals, key):
        """Apply mutation to population"""
        keys = jax.random.split(key, len(individuals))
        mutated = []

        for ind, k in zip(individuals, keys):
            if jax.random.uniform(k) < self.mutation_prob:
                # Mutate
                k, subkey = jax.random.split(k)
                mutated_desc = apply_mutation(ind.descriptors['n0'], subkey)
                ind = ind.replace(descriptors={'n0': mutated_desc})
            mutated.append(ind)

        return mutated

    def _apply_crossover(self, individuals, key):
        """Apply crossover to population"""
        offspring = []
        keys = jax.random.split(key, len(individuals) // 2)

        for i in range(0, len(individuals) - 1, 2):
            parent1, parent2 = individuals[i], individuals[i + 1]

            if jax.random.uniform(keys[i // 2]) < self.crossover_prob:
                child1, child2 = crossover(parent1, parent2, keys[i // 2])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        return offspring
```

**Deliverables:**
- ✅ Complete evolution loop
- ✅ API compatible with DEATF examples
- ✅ Integration tests

---

## Phase 7: Examples and Testing (Week 10)

### 7.1 Port Examples

**Tasks:**
1. Port `simple.py` - Basic MLP evolution
2. Port `cnn_class.py` - CNN classification
3. Port `rnn.py` - Sequence modeling
4. Create new JAX-specific examples

**Example: Simple MLP Evolution**
```python
# examples/simple_mlp.py
import jax
import jax.numpy as jnp
from nasjax import Evolving
from nasjax.descriptors import MLPDescriptor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load data
X, y = load_digits(return_X_y=True)
X = X.astype(jnp.float32) / 16.0
y = jax.nn.one_hot(y, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Run evolution
evolution = Evolving(
    desc_list=[MLPDescriptor],
    x_trains=[X_train],
    y_trains=[y_train],
    x_tests=[X_test],
    y_tests=[y_test],
    n_inputs=[[64]],
    n_outputs=[[10]],
    evaluation='cross_entropy',
    population=20,
    generations=15,
    batch_size=64,
    iters=50,
    max_num_layers=5,
    max_num_neurons=100,
    seed=42
)

# Evolve
population, logbook, hall_of_fame = evolution.evolve()

# Get best
best_individual = hall_of_fame[-1]
print(f"Best fitness: {best_individual.fitness}")
print(f"Best descriptor: {best_individual.descriptors['n0']}")
```

### 7.2 Unit Tests

**Test Coverage Goals:**
- Descriptors: 95%
- Networks: 90%
- Evolution operators: 90%
- Training: 85%

**Key Test Files:**
```
tests/
├── test_descriptors.py
│   ├── test_mlp_descriptor_creation
│   ├── test_mlp_descriptor_random_init
│   ├── test_mlp_descriptor_validation
│   ├── test_cnn_descriptor_creation
│   └── ...
├── test_networks.py
│   ├── test_mlp_forward_pass
│   ├── test_mlp_output_shape
│   ├── test_cnn_forward_pass
│   └── ...
├── test_evolution.py
│   ├── test_mutation_add_layer
│   ├── test_crossover
│   ├── test_selection_tournament
│   └── ...
└── test_training.py
    ├── test_trainer_convergence
    ├── test_loss_functions
    └── ...
```

### 7.3 Integration Tests

**Tasks:**
1. End-to-end evolution runs
2. Compare results with TensorFlow version
3. Validate numerical stability

**Deliverables:**
- ✅ All examples working
- ✅ >90% test coverage
- ✅ CI/CD passing

---

## Phase 8: Optimization and Performance (Week 11)

### 8.1 JIT Compilation

**Tasks:**
1. Add `@jax.jit` to critical paths
2. Profile bottlenecks
3. Optimize hot loops

**Target Functions for JIT:**
- Training steps
- Loss computation
- Network forward pass
- Mutation operators (where possible)

### 8.2 Vectorization

**Tasks:**
1. Vectorize population evaluation with `vmap`
2. Batch network training
3. Parallel mutation

**Code Example:**
```python
@partial(jax.vmap, in_axes=(0, None, None, 0))
def evaluate_individual_vectorized(descriptor, data, network_class, key):
    """Vectorized evaluation across population"""
    return evaluate_individual(descriptor, data, network_class, key)

# Use it
fitness_values = evaluate_individual_vectorized(
    descriptors,  # Batched
    data,
    MLP,
    keys  # Batched
)
```

### 8.3 Multi-Device Support

**Tasks:**
1. Add `pmap` for multi-GPU training
2. Distributed fitness evaluation
3. Sharded populations

**Code Example:**
```python
@partial(jax.pmap, axis_name='devices')
def parallel_evaluate(population_shard, data_shard, keys_shard):
    """Evaluate population shard on each device"""
    return jax.vmap(evaluate_individual)(
        population_shard, data_shard, keys_shard
    )
```

**Deliverables:**
- ✅ 10-100x speedup over DEATF
- ✅ GPU/TPU support
- ✅ Benchmarks documented

---

## Phase 9: Documentation (Week 12)

### 9.1 API Documentation

**Tasks:**
1. Sphinx documentation
2. Docstrings for all public APIs
3. Tutorial notebooks

**Structure:**
```
docs/
├── index.md
├── quickstart.md
├── tutorials/
│   ├── basic_mlp.md
│   ├── cnn_classification.md
│   ├── custom_evaluation.md
│   └── multi_network.md
├── api/
│   ├── descriptors.md
│   ├── networks.md
│   ├── evolution.md
│   └── training.md
├── migration_guide.md  # From DEATF
└── examples/
```

### 9.2 Migration Guide

**Tasks:**
1. Side-by-side code comparisons
2. Breaking changes documentation
3. FAQ section

### 9.3 Performance Benchmarks

**Tasks:**
1. Document speedups
2. Compare with DEATF
3. Scaling experiments

**Deliverables:**
- ✅ Complete documentation
- ✅ Migration guide
- ✅ Published benchmarks

---

## Phase 10: Release and Maintenance

### 10.1 Release Preparation

**Tasks:**
- [ ] Version 0.1.0 release
- [ ] PyPI package
- [ ] GitHub release
- [ ] Announcement

### 10.2 Community

**Tasks:**
- [ ] Contribution guidelines
- [ ] Code of conduct
- [ ] Issue templates
- [ ] Example gallery

---

## Key Milestones

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| M1: Foundation | 1 | Project setup complete |
| M2: Descriptors | 3 | All descriptors implemented |
| M3: Networks | 4 | All networks implemented |
| M4: Evolution | 6 | Evolution engine complete |
| M5: Training | 8 | Training pipeline complete |
| M6: Integration | 9 | Main API complete |
| M7: Examples | 10 | All examples ported |
| M8: Performance | 11 | Optimizations complete |
| M9: Documentation | 12 | Docs published |
| M10: Release | 12 | v0.1.0 released |

---

## Success Criteria

### Functional
- ✅ All DEATF examples work in NASJAX
- ✅ Results match TensorFlow version (within numerical precision)
- ✅ API is user-friendly and well-documented

### Performance
- ✅ 10x faster than DEATF on CPU
- ✅ 100x faster on GPU
- ✅ Scales to large populations (1000+)

### Quality
- ✅ >90% test coverage
- ✅ CI/CD passing
- ✅ No critical bugs
- ✅ Documentation complete

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| JAX learning curve | High | Start simple, gradual complexity |
| Numerical instability | Medium | Extensive testing vs TensorFlow |
| Performance issues | Medium | Profile early, optimize iteratively |
| Equinox limitations | Low | Fallback to Flax if needed |

### Project Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope creep | High | Stick to roadmap, defer nice-to-haves |
| Time overrun | Medium | Prioritize core features |
| Breaking changes in JAX | Low | Pin versions, monitor releases |

---

## Future Enhancements (Post v0.1.0)

### v0.2.0
- Multi-objective optimization (NSGA-III)
- Distributed evolution (multi-node)
- Advanced architectures (Transformers, ResNets)

### v0.3.0
- Neural architecture search (DARTS, ENAS)
- Meta-learning integration
- Hyperparameter optimization

### v1.0.0
- Production-ready
- Stable API
- Comprehensive benchmarks
- Research publication

---

## Conclusion

This roadmap provides a structured 12-week plan to migrate DEATF to NASJAX. By following this phased approach, we ensure:

1. **Solid foundation** - Core abstractions correct from the start
2. **Incremental progress** - Each phase builds on previous
3. **Early validation** - Testing at every stage
4. **Performance focus** - Optimization built-in, not bolted-on
5. **User-friendly** - API design prioritized

The migration leverages JAX's strengths (JIT, vmap, pmap) while maintaining the conceptual simplicity of DEATF's neuroevolution approach. The result will be a modern, high-performance neuroevolution library ready for research and production use.
