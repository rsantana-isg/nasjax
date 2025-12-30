# NASJAX - Neural Architecture Search with JAX

![Python](https://img.shields.io/badge/Python-3.9+-blue)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green)](https://jax.readthedocs.io/)
[![Equinox](https://img.shields.io/badge/Equinox-0.11+-brightgreen)](https://docs.kidger.site/equinox/)

**NASJAX** is a modern neuroevolutionary framework for neural architecture search, built on JAX and Equinox. It is a from-scratch reimplementation of [DEATF](https://github.com/IvanHCenalmor/deatf) (Distributed Evolutionary Algorithms in TensorFlow), designed to leverage JAX's high-performance automatic differentiation, JIT compilation, and hardware acceleration.

## 🎯 Project Status

**Current Phase:** Initial Planning and Architecture Design

This project is being actively developed to port DEATF from TensorFlow/DEAP to JAX/Equinox. See the roadmap below for details.

---

## 🚀 Key Features (Planned)

- **JAX-Native Architecture**: Functional programming paradigm with immutable PyTree-based representations
- **High Performance**: 10-100x speedup over TensorFlow through JIT compilation and vectorization
- **Hardware Acceleration**: Native GPU/TPU support via JAX
- **Flexible Evolution**: Support for MLPs, CNNs, RNNs, and custom architectures
- **Modern ML Stack**: Built on Equinox (networks) and Optax (optimization)
- **Composable Transformations**: Leverage JAX's `jit`, `vmap`, and `pmap` for parallelization
- **Type-Safe**: Full type annotations with runtime checking

---

## 📋 Documentation

### Core Documents

1. **[TRANSFORMATIONS.md](./TRANSFORMATIONS.md)** - Detailed technical guide mapping DEATF concepts to JAX/Equinox
   - List-based to PyTree-based representations
   - TensorFlow to Equinox network implementations
   - DEAP to JAX-native evolution operators
   - Performance optimization strategies

2. **[ROADMAP.md](./ROADMAP.md)** - Complete 12-week implementation plan
   - Phase-by-phase breakdown
   - Milestones and deliverables
   - Success criteria
   - Risk mitigation

3. **[ROADMAP_PINNs.md](./ROADMAP_PINNs.md)** - Physics-Informed Neural Networks (PINNs) Evolution
   - Comprehensive guide for evolving PINN architectures
   - Integration strategy with NASJAX framework
   - PDE benchmark selection and implementation phases
   - API design and usage examples

### Quick Links

- **Migration from DEATF**: See [TRANSFORMATIONS.md](./TRANSFORMATIONS.md)
- **Implementation Plan**: See [ROADMAP.md](./ROADMAP.md)
- **PINN Evolution**: See [ROADMAP_PINNs.md](./ROADMAP_PINNs.md)
- **Original DEATF**: [github.com/IvanHCenalmor/deatf](https://github.com/IvanHCenalmor/deatf)

---

## 🔧 Installation (Coming Soon)

```bash
# Once released
pip install nasjax

# Or from source (development)
git clone https://github.com/your-org/nasjax.git
cd nasjax
pip install -e ".[dev]"
```

### Dependencies

**Core:**
- `jax >= 0.4.20`
- `jaxlib >= 0.4.20`
- `equinox >= 0.11.0`
- `optax >= 0.1.7`
- `chex >= 0.1.8`
- `jaxtyping >= 0.2.0`

**Utilities:**
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`

---

## 📚 Quick Start (Preview)

### Basic MLP Evolution

```python
import jax
import jax.numpy as jnp
from nasjax import Evolving
from nasjax.descriptors import MLPDescriptor

# Load your data
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data

# Configure evolution
evolution = Evolving(
    desc_list=[MLPDescriptor],
    x_trains=[X_train],
    y_trains=[y_train],
    x_tests=[X_test],
    y_tests=[y_test],
    n_inputs=[[784]],
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

# Run evolution
population, logbook, hall_of_fame = evolution.evolve()

# Get best architecture
best_individual = hall_of_fame[-1]
print(f"Best fitness: {best_individual.fitness}")
print(f"Architecture: {best_individual.descriptors['n0']}")
```

### Custom Evaluation Function

```python
def custom_eval(model, x_train, y_train, x_test, y_test, key):
    """Custom evaluation with your own training logic"""

    # Train model
    trained_model = train_with_custom_logic(model, x_train, y_train, key)

    # Evaluate
    predictions = jax.vmap(lambda x: trained_model(x, inference=True))(x_test)
    accuracy = compute_accuracy(predictions, y_test)

    return 1.0 - accuracy  # Return loss (lower is better)

# Use custom evaluation
evolution = Evolving(
    evaluation=custom_eval,
    # ... other parameters
)
```

---

## 🏗️ Architecture Overview

### Conceptual Shift: DEATF → NASJAX

| Aspect | DEATF (TensorFlow) | NASJAX (JAX) |
|--------|-------------------|--------------|
| **Representation** | Mutable lists/arrays | Immutable PyTrees |
| **Networks** | TensorFlow/Keras | Equinox modules |
| **Evolution** | DEAP library | JAX-native operators |
| **Training** | Keras `.fit()` | Optax + custom loops |
| **Mutations** | In-place modifications | Functional transformations |
| **Parallelization** | Limited | `vmap`, `pmap` for GPU/TPU |

### Key Components

```
nasjax/
├── descriptors/     # PyTree-based network descriptors (genotype)
│   ├── mlp.py      # Multi-layer perceptron
│   ├── cnn.py      # Convolutional neural network
│   ├── rnn.py      # Recurrent neural network
│   └── tcnn.py     # Transposed CNN (generative)
├── networks/        # Equinox network implementations (phenotype)
│   ├── mlp.py
│   ├── cnn.py
│   ├── rnn.py
│   └── tcnn.py
├── evolution/       # JAX-native evolutionary operators
│   ├── mutation.py
│   ├── crossover.py
│   ├── selection.py
│   └── population.py
├── training/        # Optax-based training utilities
│   ├── trainer.py
│   ├── losses.py
│   └── optimizers.py
└── evaluation/      # Fitness evaluation
    ├── evaluator.py
    └── metrics.py
```

---

## 🎨 Design Philosophy

### Immutability and Functional Programming

Unlike DEATF which uses mutable objects, NASJAX embraces JAX's functional paradigm:

```python
# DEATF (mutable)
descriptor.add_layer(position, neurons)  # Modifies in-place

# NASJAX (immutable)
new_descriptor = add_layer(descriptor, position, neurons)  # Returns new object
```

### PyTree-Based Representations

Networks are represented as PyTrees (nested structures that JAX can transform):

```python
@chex.dataclass
class MLPDescriptor:
    dims: Tuple[int, ...]              # (64, 128, 32)
    act_functions: Tuple[str, ...]     # ('relu', 'relu', 'sigmoid')
    init_functions: Tuple[str, ...]    # ('glorot_normal', ...)
    dropout_probs: Tuple[float, ...]   # (0.1, 0.1, 0.0)
    batch_norm: bool
    dropout: bool
```

This enables:
- **JIT compilation** of mutation operators
- **Vectorization** (`vmap`) over populations
- **Parallelization** (`pmap`) across devices
- **Gradient-based** architecture search (future)

---

## 🚀 Performance Expectations

Based on JAX's capabilities, we expect significant speedups:

| Operation | DEATF (TF) | NASJAX (JAX CPU) | NASJAX (JAX GPU) |
|-----------|------------|------------------|------------------|
| Single network eval | 1x | 5-10x | 50-100x |
| Population eval | 1x | 10-20x | 100-200x |
| Evolution iteration | 1x | 10x | 100x |

*Note: Actual performance will be benchmarked during development.*

---

## 📈 Roadmap

### Phases (12 weeks)

1. **Week 1**: Foundation and setup
2. **Weeks 2-3**: Core descriptor implementations
3. **Weeks 3-4**: Network implementations (Equinox)
4. **Weeks 5-6**: Evolution engine (mutations, crossover, selection)
5. **Weeks 7-8**: Training and evaluation pipelines
6. **Week 9**: Main evolution loop and integration
7. **Week 10**: Examples and comprehensive testing
8. **Week 11**: Performance optimization (JIT, vmap, pmap)
9. **Week 12**: Documentation and release

See [ROADMAP.md](./ROADMAP.md) for detailed breakdown.

---

## 🔬 Research Applications

NASJAX is designed for:

- **Neural Architecture Search (NAS)**: Automated architecture discovery
- **Hyperparameter Optimization**: Joint evolution of architecture + hyperparameters
- **Multi-Objective Optimization**: Balancing accuracy, efficiency, size
- **Meta-Learning**: Evolving learning algorithms
- **Generative Models**: Evolving GANs, VAEs, autoencoders
- **Recurrent Architectures**: Custom RNN/LSTM designs for sequences
- **Physics-Informed Neural Networks (PINNs)**: Evolving architectures for solving PDEs
  - See [ROADMAP_PINNs.md](./ROADMAP_PINNs.md) for comprehensive guide

---

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **Core Implementation**: Descriptor/network implementations
- **Evolution Operators**: Advanced selection/mutation strategies
- **Benchmarks**: Comparing with other NAS methods
- **Examples**: Domain-specific use cases
- **Documentation**: Tutorials, guides, API docs

See `CONTRIBUTING.md` (coming soon) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DEATF**: Original TensorFlow implementation by [Iván Hidalgo-Cenalmor](https://github.com/IvanHCenalmor)
- **EvoFlow**: Foundational work by Unai Garciarena
- **JAX Team**: For the amazing JAX ecosystem
- **Equinox**: Patrick Kidger's elegant neural network library

---

## 📖 References

### Original Papers

1. **DEATF/EvoFlow**:
   - Garciarena, U., Santana, R., & Mendiburu, A. (2018). "Evolved GANs for generating Pareto set approximations." *GECCO 2018*.

2. **JAX**:
   - Bradbury, J., et al. (2018). "JAX: composable transformations of Python+NumPy programs."

3. **Evolutionary Algorithms**:
   - Fortin, F. A., et al. (2012). "DEAP: Evolutionary algorithms made easy." *JMLR*.

### Related Work

- **DARTS**: Differentiable Architecture Search
- **ENAS**: Efficient Neural Architecture Search
- **NAS-Bench**: Benchmarks for architecture search
- **AutoML-Zero**: Evolving ML algorithms from scratch

---

## 📧 Contact

- **Project Lead**: [Your Name/Organization]
- **Issues**: [GitHub Issues](https://github.com/your-org/nasjax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/nasjax/discussions)

---

## 🌟 Star History

If you find NASJAX useful, please consider starring the repository!

---

**Note**: This project is under active development. The API is subject to change until v1.0.0. See [ROADMAP.md](./ROADMAP.md) for current status.
