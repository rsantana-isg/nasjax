# DEATF to NASJAX Transformation Guide

## Overview

This document outlines the technical transformations required to port the DEATF (Distributed Evolutionary Algorithms in TensorFlow) library to NASJAX, a JAX-based implementation using Equinox for neural network definitions.

## 1. Core Conceptual Changes

### 1.1 From List-Based to PyTree-Based Representations

**Current (DEATF - TensorFlow):**
- Neural networks represented as lists/arrays of parameters
- Example: `MLPDescriptor.dims = [64, 128, 32]`
- Mutations modify these lists in-place

**Target (NASJAX - JAX):**
- Neural networks represented as PyTrees (nested dictionaries/dataclasses)
- PyTrees are immutable and composable
- Mutations return new PyTrees (functional programming paradigm)

**Key Transformation:**
```python
# OLD (DEATF):
class MLPDescriptor:
    def __init__(self):
        self.dims = [64, 128, 32]
        self.act_functions = [relu, relu, sigmoid]
        self.init_functions = [GlorotNormal, GlorotNormal, GlorotNormal]

# NEW (NASJAX):
from typing import NamedTuple, List
import jax.tree_util as jtu

@jtu.register_pytree_node_class
class MLPDescriptor(NamedTuple):
    dims: List[int]
    act_functions: List[str]  # Store as strings/enums
    init_functions: List[str]
    dropout_probs: List[float]
    batch_norm: bool
    dropout: bool

    # PyTree methods for JAX compatibility
    def tree_flatten(self):
        leaves = (self.dims, self.dropout_probs)
        aux_data = {
            'act_functions': self.act_functions,
            'init_functions': self.init_functions,
            'batch_norm': self.batch_norm,
            'dropout': self.dropout
        }
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        dims, dropout_probs = leaves
        return cls(dims=dims, dropout_probs=dropout_probs, **aux_data)
```

---

## 2. Neural Network Implementation Changes

### 2.1 From TensorFlow Keras to Equinox

**Current (TensorFlow):**
```python
class MLP(Network):
    def building(self, x):
        for i in range(self.descriptor.number_hidden_layers-1):
            x = Dense(self.descriptor.dims[i],
                     activation=self.descriptor.act_functions[i],
                     kernel_initializer=self.descriptor.init_functions[i])(x)
            if self.descriptor.dropout:
                x = Dropout(self.descriptor.dropout_probs[i])(x)
        return x
```

**Target (Equinox):**
```python
import equinox as eqx
import jax.nn as jnn

class MLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    act_functions: List
    dropout_probs: List[float]
    use_dropout: bool
    use_batch_norm: bool
    batch_norm_layers: Optional[List[eqx.nn.BatchNorm]]

    def __init__(self, descriptor: MLPDescriptor, key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(descriptor.dims))

        self.layers = []
        in_dim = descriptor.input_dim
        for i, (out_dim, k) in enumerate(zip(descriptor.dims, keys)):
            self.layers.append(
                eqx.nn.Linear(in_dim, out_dim, key=k)
            )
            in_dim = out_dim

        self.act_functions = [self._get_activation(f) for f in descriptor.act_functions]
        self.dropout_probs = descriptor.dropout_probs
        self.use_dropout = descriptor.dropout
        self.use_batch_norm = descriptor.batch_norm

    def __call__(self, x, key=None, *, inference=False):
        for i, (layer, act_fn) in enumerate(zip(self.layers, self.act_functions)):
            x = layer(x)
            if act_fn is not None:
                x = act_fn(x)
            if self.use_dropout and not inference:
                key, subkey = jax.random.split(key)
                x = eqx.nn.Dropout(self.dropout_probs[i])(x, key=subkey, inference=inference)
        return x

    @staticmethod
    def _get_activation(name):
        activations = {
            'relu': jnn.relu,
            'elu': jnn.elu,
            'sigmoid': jnn.sigmoid,
            'tanh': jnn.tanh,
            'softplus': jnn.softplus,
            'softsign': jnn.softsign,
            None: None
        }
        return activations[name]
```

### 2.2 Activation and Initialization Functions Mapping

**TensorFlow → JAX Mapping:**

| TensorFlow | JAX |
|------------|-----|
| `tf.nn.relu` | `jax.nn.relu` |
| `tf.nn.elu` | `jax.nn.elu` |
| `tf.nn.sigmoid` | `jax.nn.sigmoid` |
| `tf.nn.tanh` | `jax.nn.tanh` |
| `tf.nn.softplus` | `jax.nn.softplus` |
| `tf.nn.softsign` | `jax.nn.softsign` |

**Weight Initializers:**

| TensorFlow Keras | JAX/Equinox |
|------------------|-------------|
| `RandomNormal()` | `jax.random.normal(key, shape) * std` |
| `RandomUniform()` | `jax.random.uniform(key, shape, minval, maxval)` |
| `GlorotNormal()` | `eqx.nn.Linear` default (Xavier) |
| `GlorotUniform()` | Custom initialization with Glorot formula |

**Implementation Strategy:**
```python
def get_initializer(init_name: str):
    """Map initialization function names to JAX/Equinox initializers"""
    initializers = {
        'RandomNormal': lambda key, shape: jax.random.normal(key, shape) * 0.05,
        'RandomUniform': lambda key, shape: jax.random.uniform(key, shape, -0.05, 0.05),
        'GlorotNormal': lambda key, shape: jax.random.normal(key, shape) * jnp.sqrt(2.0 / sum(shape)),
        'GlorotUniform': lambda key, shape: jax.random.uniform(key, shape,
            -jnp.sqrt(6.0 / sum(shape)), jnp.sqrt(6.0 / sum(shape)))
    }
    return initializers[init_name]
```

---

## 3. Evolution Algorithm Changes

### 3.1 From DEAP to Custom JAX-Compatible Evolution

**Current (DEAP):**
- Uses DEAP's `creator`, `toolbox`, `algorithms`
- Mutation/crossover operators modify objects in-place
- Population is a list of mutable `DescriptorContainer` objects

**Target (JAX-Compatible):**
- Custom evolutionary operators using functional programming
- All operations return new immutable structures
- Can use existing libraries like `evosax` or `evox` for JAX-native evolution

**Options for Evolution in JAX:**

1. **Option A: Custom Implementation**
   - Pure Python/JAX implementation
   - Full control over operators
   - Can leverage `jax.vmap` and `jax.pmap` for parallelization

2. **Option B: EvoJAX (Recommended)**
   - [github.com/google/evojax](https://github.com/google/evojax)
   - JAX-native evolutionary algorithms
   - GPU/TPU acceleration
   - Supports custom fitness functions

3. **Option C: Evosax**
   - [github.com/RobertTLange/evosax](https://github.com/RobertTLange/evosax)
   - Scalable evolution strategies in JAX
   - Good for neuroevolution

4. **Option D: Hybrid Approach**
   - Keep DEAP for evolutionary logic
   - Use JAX/Equinox only for network evaluation
   - Easier initial port but less performance gains

**Recommended: Option A + B Hybrid**
- Start with custom implementation for descriptors (immutable PyTrees)
- Use EvoJAX for population management and selection
- Leverage JAX's functional programming for mutations

### 3.2 Mutation Operators Transformation

**Current (In-Place Mutation):**
```python
def mut_add_layer(self):
    """Modifies self.network in place"""
    self.dims = np.insert(self.dims, layer_pos, lay_dims)
    self.number_hidden_layers += 1
    return True
```

**Target (Functional Mutation):**
```python
def mut_add_layer(descriptor: MLPDescriptor, key: jax.random.PRNGKey) -> MLPDescriptor:
    """Returns new descriptor with added layer"""
    layer_pos = jax.random.randint(key, (), 0, len(descriptor.dims))
    new_dim = jax.random.randint(key, (), MIN_NEURONS, MAX_NEURONS)

    new_dims = descriptor.dims[:layer_pos] + [new_dim] + descriptor.dims[layer_pos:]
    new_act_fns = descriptor.act_functions[:layer_pos] + ['relu'] + descriptor.act_functions[layer_pos:]

    return descriptor._replace(
        dims=new_dims,
        act_functions=new_act_fns
    )
```

### 3.3 Fitness Evaluation Transformation

**Current (TensorFlow):**
```python
def simple_eval(self, individual):
    net = MLP(individual.desc_list["n0"])
    inp = Input(shape=self.n_inputs[0])
    out = net.building(inp)
    model = Model(inputs=inp, outputs=out)

    opt = tf.keras.optimizers.Adam(learning_rate=self.lrate)
    model.compile(loss=self.evaluation, optimizer=opt)
    model.fit(x_train, y_train, epochs=self.iters, batch_size=self.batch_size)

    return model.evaluate(x_test, y_test),
```

**Target (JAX/Equinox):**
```python
import optax

def simple_eval(descriptor: MLPDescriptor,
                x_train, y_train,
                x_test, y_test,
                key: jax.random.PRNGKey) -> float:
    """Evaluate network using JAX/Optax"""

    # Build network from descriptor
    key, subkey = jax.random.split(key)
    model = MLP(descriptor, subkey)

    # Setup optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Training loop
    @eqx.filter_jit
    def train_step(model, opt_state, x, y, key):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Train for N iterations
    for epoch in range(n_iters):
        for batch_x, batch_y in get_batches(x_train, y_train, batch_size):
            key, subkey = jax.random.split(key)
            model, opt_state, loss = train_step(model, opt_state, batch_x, batch_y, subkey)

    # Evaluate on test set
    test_loss = compute_loss(model, x_test, y_test, key, inference=True)
    return float(test_loss)

def compute_loss(model, x, y, key, inference=False):
    """Compute loss (MSE or CrossEntropy)"""
    pred = jax.vmap(lambda x_i: model(x_i, key, inference=inference))(x)
    return jnp.mean((pred - y) ** 2)  # or cross_entropy
```

---

## 4. Data Structure Transformations

### 4.1 Descriptor Container

**Current (DEAP):**
```python
class DescriptorContainer(object):
    def __init__(self, desc_list):
        self.desc_list = desc_list  # {"n0": MLPDescriptor, "hypers": {...}}
```

**Target (JAX PyTree):**
```python
from typing import Dict, Any
import chex

@chex.dataclass
class Individual:
    """Immutable container for network descriptors"""
    descriptors: Dict[str, Any]  # {"n0": MLPDescriptor, "n1": CNNDescriptor}
    hyperparameters: Dict[str, float]  # {"learning_rate": 0.01, "optimizer": "adam"}
    fitness: Optional[float] = None

    def replace(self, **changes):
        """Functional update - returns new Individual"""
        return dataclasses.replace(self, **changes)
```

### 4.2 Population Representation

**Current (DEAP):**
```python
# List of mutable containers
population = [DescriptorContainer(...) for _ in range(pop_size)]
```

**Target (JAX):**
```python
# PyTree with stacked arrays for vectorization
Population = List[Individual]

# For even better JAX performance, stack parameters:
@chex.dataclass
class VectorizedPopulation:
    """All individuals in vectorized form for fast JAX operations"""
    # Shape: (pop_size, max_layers, ...)
    all_dims: jnp.ndarray
    all_act_functions: jnp.ndarray  # Encoded as integers
    all_dropout_probs: jnp.ndarray
    fitness_values: jnp.ndarray
```

---

## 5. Specific Network Type Transformations

### 5.1 CNN Descriptor → Equinox CNN

**Current (TensorFlow):**
```python
class CNNDescriptor:
    layers: List[int]  # 0=AvgPool, 1=MaxPool, 2=Conv
    filters: List[np.ndarray]  # [[h, w, c], ...]
    strides: List[np.ndarray]

class CNN:
    def building(self, x):
        for i in range(num_layers):
            if self.descriptor.layers[i] == 2:
                x = Conv2D(...)(x)
            elif self.descriptor.layers[i] == 1:
                x = MaxPooling2D(...)(x)
```

**Target (Equinox):**
```python
@chex.dataclass
class CNNDescriptor:
    layer_types: List[str]  # ['conv', 'maxpool', 'conv']
    filters: List[Tuple[int, int, int]]  # [(h, w, c), ...]
    strides: List[Tuple[int, int]]
    act_functions: List[str]
    init_functions: List[str]
    batch_norm: bool

class CNN(eqx.Module):
    conv_layers: List[eqx.nn.Conv2d]
    pool_layers: List
    layer_order: List[Tuple[str, int]]  # [('conv', 0), ('pool', 0), ...]

    def __init__(self, descriptor: CNNDescriptor, key):
        keys = jax.random.split(key, sum(1 for t in descriptor.layer_types if t == 'conv'))

        self.conv_layers = []
        self.pool_layers = []
        self.layer_order = []

        conv_idx = 0
        pool_idx = 0

        for i, layer_type in enumerate(descriptor.layer_types):
            if layer_type == 'conv':
                h, w, out_c = descriptor.filters[i]
                self.conv_layers.append(
                    eqx.nn.Conv2d(in_c, out_c, kernel_size=(h, w),
                                 stride=descriptor.strides[i], key=keys[conv_idx])
                )
                self.layer_order.append(('conv', conv_idx))
                conv_idx += 1
            elif layer_type == 'maxpool':
                self.layer_order.append(('maxpool', pool_idx))
                pool_idx += 1

    def __call__(self, x, key=None, inference=False):
        for layer_type, idx in self.layer_order:
            if layer_type == 'conv':
                x = self.conv_layers[idx](x)
                x = jnn.relu(x)  # Apply activation
            elif layer_type == 'maxpool':
                x = max_pool_2d(x, window_shape=(2, 2), strides=(2, 2))
        return x

def max_pool_2d(x, window_shape, strides):
    """Custom max pooling using jax.lax"""
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max,
        window_dimensions=(1,) + window_shape + (1,),
        window_strides=(1,) + strides + (1,),
        padding='VALID'
    )
```

### 5.2 RNN Descriptor → Equinox RNN

**Current (TensorFlow):**
```python
class RNNDescriptor:
    rnn_layers: List  # [SimpleRNN, LSTM, GRU]
    units_in_layer: List[int]
    bidirectional: List[bool]
```

**Target (Equinox):**
```python
import equinox as eqx

@chex.dataclass
class RNNDescriptor:
    layer_types: List[str]  # ['lstm', 'gru', 'lstm']
    units: List[int]
    bidirectional: List[bool]
    act_functions: List[str]
    init_functions: List[str]

class RNN(eqx.Module):
    rnn_cells: List
    bidirectional_flags: List[bool]

    def __init__(self, descriptor: RNNDescriptor, key):
        keys = jax.random.split(key, len(descriptor.layer_types))

        self.rnn_cells = []
        in_size = descriptor.input_dim

        for layer_type, units, bidir, k in zip(
            descriptor.layer_types, descriptor.units,
            descriptor.bidirectional, keys
        ):
            if layer_type == 'lstm':
                cell = eqx.nn.LSTMCell(in_size, units, key=k)
            elif layer_type == 'gru':
                cell = eqx.nn.GRUCell(in_size, units, key=k)
            else:  # 'vanilla'
                cell = CustomRNNCell(in_size, units, key=k)

            if bidir:
                k1, k2 = jax.random.split(k)
                cell = BidirectionalRNN(cell, units, keys=(k1, k2))

            self.rnn_cells.append(cell)
            in_size = units * (2 if bidir else 1)

    def __call__(self, x_sequence, key=None):
        """Process sequence through RNN layers"""
        hidden = x_sequence
        for cell in self.rnn_cells:
            hidden = scan_rnn(cell, hidden)
        return hidden[-1]  # Return last timestep

def scan_rnn(cell, inputs):
    """Scan RNN cell over sequence"""
    def step(carry, x):
        new_carry = cell(x, carry)
        return new_carry, new_carry

    init_carry = cell.init_state()
    _, outputs = jax.lax.scan(step, init_carry, inputs)
    return outputs
```

---

## 6. Training and Optimization

### 6.1 Optimizer Transformation

**TensorFlow Optimizers → Optax:**

| TensorFlow | Optax |
|------------|-------|
| `tf.keras.optimizers.Adam()` | `optax.adam()` |
| `tf.keras.optimizers.SGD()` | `optax.sgd()` |
| `tf.keras.optimizers.RMSprop()` | `optax.rmsprop()` |

**Implementation:**
```python
import optax

def get_optimizer(name: str, learning_rate: float):
    optimizers = {
        'adam': optax.adam(learning_rate),
        'sgd': optax.sgd(learning_rate),
        'rmsprop': optax.rmsprop(learning_rate),
        'adamw': optax.adamw(learning_rate),
    }
    return optimizers[name]
```

### 6.2 Loss Functions

**TensorFlow → JAX:**

| TensorFlow | JAX |
|------------|-----|
| `tf.losses.mean_squared_error` | `jnp.mean((pred - y)**2)` |
| `tf.nn.softmax_cross_entropy_with_logits` | `optax.softmax_cross_entropy()` |

```python
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
```

---

## 7. Serialization and Checkpointing

### 7.1 Saving Descriptors

**Current (DEAP/Pickle):**
```python
import pickle
with open('best_individual.pkl', 'wb') as f:
    pickle.dump(hall_of_fame[0], f)
```

**Target (JAX/Equinox):**
```python
import equinox as eqx
import pickle

# Save descriptor (lightweight)
with open('best_descriptor.pkl', 'wb') as f:
    pickle.dump(descriptor, f)

# Save full trained model (with weights)
eqx.tree_serialise_leaves('best_model.eqx', model)

# Load model
model = eqx.tree_deserialise_leaves('best_model.eqx', model_template)
```

---

## 8. Performance Optimizations Enabled by JAX

### 8.1 JIT Compilation

```python
@eqx.filter_jit
def train_step(model, opt_state, x, y, key):
    """JIT-compiled training step for massive speedup"""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
```

### 8.2 Vectorized Population Evaluation

```python
@jax.jit
def evaluate_population(population_descriptors, data, key):
    """Evaluate entire population in parallel on GPU/TPU"""
    keys = jax.random.split(key, len(population_descriptors))

    # vmap over population
    fitness_values = jax.vmap(
        lambda desc, k: evaluate_individual(desc, data, k)
    )(population_descriptors, keys)

    return fitness_values
```

### 8.3 Multi-Device Parallelization

```python
@partial(jax.pmap, axis_name='devices')
def parallel_evolution_step(population_shard, data_shard, key):
    """Distribute evolution across multiple GPUs/TPUs"""
    # Each device processes a shard of the population
    fitness = evaluate_population(population_shard, data_shard, key)
    return fitness
```

---

## 9. Key Libraries and Dependencies

### 9.1 Core Dependencies

**Replace:**
- `tensorflow >= 2.5` → `jax >= 0.4.20`, `jaxlib`
- `deap >= 1.0` → Custom or `evojax` / `evosax`

**Add:**
- `equinox >= 0.11.0` - Neural network library for JAX
- `optax >= 0.1.7` - Optimization library
- `chex >= 0.1.8` - Testing and utilities
- `jaxtyping >= 0.2.0` - Type annotations

**Keep:**
- `numpy` - Still useful for data preprocessing
- `scikit-learn` - Metrics and data utilities

### 9.2 Optional Enhancements

- `evojax` - JAX-native evolutionary algorithms
- `evosax` - Evolution strategies
- `jraph` - For graph neural networks (future)
- `orbax` - Checkpointing
- `flax` - Alternative to Equinox (more Keras-like)

---

## 10. Migration Strategy Summary

### Phase 1: Core Transformations
1. Convert descriptors to immutable PyTree structures
2. Implement Equinox neural network classes
3. Replace TensorFlow layers with Equinox equivalents

### Phase 2: Evolution Engine
1. Implement functional mutation operators
2. Create custom or adapt EvoJAX for evolution
3. Implement JAX-based fitness evaluation

### Phase 3: Training Pipeline
1. Replace TensorFlow training with Optax
2. Implement JIT-compiled training loops
3. Add vectorized population evaluation

### Phase 4: Optimization
1. Add JIT compilation to critical paths
2. Implement vmap for population-level parallelism
3. Add pmap for multi-device training

### Phase 5: Testing and Validation
1. Port examples to JAX
2. Validate results match TensorFlow version
3. Benchmark performance improvements

---

## 11. Expected Benefits

### 11.1 Performance
- **10-100x speedup** from JIT compilation
- **GPU/TPU acceleration** for evolution
- **Parallelized fitness evaluation** across population

### 11.2 Flexibility
- **Functional programming** - easier to reason about
- **Composable transformations** - vmap, pmap, jit
- **Immutable structures** - safer concurrent evolution

### 11.3 Modern Ecosystem
- **Growing JAX ecosystem** - Equinox, Optax, EvoJAX
- **Better numerical stability** - JAX's numerics
- **Easier distributed training** - pmap, pjit

---

## 12. Challenges and Considerations

### 12.1 Technical Challenges
1. **Learning curve** - Functional programming paradigm
2. **No in-place operations** - All mutations create new objects
3. **PRNG key management** - Must thread keys through all functions
4. **Shape constraints** - Some operations require static shapes

### 12.2 Migration Considerations
1. **Breaking changes** - API will differ from DEATF
2. **Backwards compatibility** - May need adapters for old checkpoints
3. **Testing complexity** - Need to validate against TensorFlow results
4. **Documentation** - Extensive examples needed for new API

---

## Conclusion

This transformation from TensorFlow/DEAP to JAX/Equinox represents a significant architectural shift towards functional programming and modern ML frameworks. While challenging, the benefits in performance, composability, and maintainability make this an excellent investment for the future of neuroevolution research.

The key insight is replacing mutable list-based representations with immutable PyTree-based structures, enabling JAX's powerful transformations (jit, vmap, pmap) to dramatically accelerate the evolutionary process.
