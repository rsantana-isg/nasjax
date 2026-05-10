# Roadmap: Hyperparameter Optimisation of Physics-Informed Models via Neuro-Evolution

## Overview

This document describes the steps required to optimise the hyperparameters of four families
of Physics-Informed Neural Networks using a **neuro-evolutionary approach** that exploits
the evaluator functions defined in the `nasjax` sub-packages.

| Sub-package | Model type | Backend | Evaluator class |
|---|---|---|---|
| `nasjax.bpinn` | Bayesian PINN (BPINN) | equinox | `BPINNEvaluator` |
| `nasjax.pinns` | Standard PINN | equinox | `PINNEvaluator` |
| `nasjax.pikans` | Physics-Informed KAN (PIKAN) | jaxkan/flax.nnx | `PIKANEvaluator` |
| `nasjax.fbpinns` | Finite Basis PINN (FBPINN) | pure JAX | `FBPINNEvaluator` |

All evaluators share the same interface:

```python
fitness = evaluator.evaluate(config, key)         # from a full config object
fitness = evaluator.evaluate_config_dict(d, key)  # from a nested dict
fitness = evaluator.evaluate_layer_specs(specs, key, **kwargs)  # architecture only
```

Smaller fitness values are better (relative L2 error or final loss).

---

## Phase 1 — Configuration Space Definition

**Goal**: Formally define the hyperparameter spaces for each model family.

### 1.1 Shared parameters (BPINN, PINN, FBPINN)

| Parameter | Type | Range / Options |
|---|---|---|
| Hidden layer widths | integer list | [16, 32, 64, 128] per layer |
| Number of hidden layers | integer | 1–6 |
| Activation function | categorical | tanh, relu, gelu, silu, sin |
| Optimiser | categorical | adam, adamw |
| Learning rate | continuous (log) | 1e-4 – 1e-2 |
| Number of epochs | integer | 500 – 10 000 |
| λ_pde | continuous (log) | 1e-2 – 1e2 |
| λ_ic | continuous (log) | 1e0 – 1e4 |
| n_pde (collocation points) | integer | 1 000 – 20 000 |

### 1.2 BPINN-specific parameters

| Parameter | Type | Range / Options |
|---|---|---|
| Prior type (per layer) | categorical | Gaussian, Laplace, SnS |
| Prior sigma | continuous (log) | 1e-2 – 1e1 |
| Local lambda technique | categorical | None, Residual_RBA, Likelihood_RBA |
| Global λ_prior | continuous (log) | 1e-2 – 1e1 |
| MC samples per epoch (K) | integer | 1, 3, 5, 10 |
| Enhanced MLP | binary | True, False |
| Resample each epoch | binary | True, False |

### 1.3 PIKAN-specific parameters

| Parameter | Type | Range / Options |
|---|---|---|
| Layer type (per layer) | categorical | chebyshev, legendre, fourier, rbf, sine, spline |
| Basis degree D | integer | 3 – 10 |
| KAN flavor | categorical | default, modified, exact (for chebyshev/legendre) |
| Heterogeneous layers | binary | allow mixing layer types across layers |

### 1.4 FBPINN-specific parameters

| Parameter | Type | Range / Options |
|---|---|---|
| n_subdomains_x | integer | 2 – 16 |
| n_subdomains_t | integer | 2 – 16 |
| Overlap fraction | continuous | 0.1 – 0.9 |
| Window function | categorical | cosine, sigmoid |
| Subdomain network width | integer list | same as shared |

**Deliverables**: Python dicts / dataclasses encoding the search space boundaries; a space
sampling function `sample_config(space, rng)` returning a valid config.

---

## Phase 2 — Evaluator Integration and Fitness Computation

**Goal**: Wrap each evaluator so that any population-based search algorithm can call it
with a decoded genome (chromosome) and receive a scalar fitness.

### 2.1 Genome ↔ Config conversion

Each search algorithm represents a candidate as a **flat numeric vector** (genome).  A
codec layer translates between the genome and a typed config object:

```python
class BPINNCodec:
    def decode(self, genome: np.ndarray) -> BPINNConfig: ...
    def encode(self, config: BPINNConfig) -> np.ndarray: ...

class PINNCodec:
    def decode(self, genome: np.ndarray) -> PINNConfig: ...

class PIKANCodec:
    def decode(self, genome: np.ndarray) -> PIKANConfig: ...

class FBPINNCodec:
    def decode(self, genome: np.ndarray) -> FBPINNConfig: ...
```

Continuous variables are encoded directly; categorical variables are encoded as one-hot or
integer indices; list variables (layer widths) are encoded as fixed-length padded arrays.

### 2.2 Fitness function signature

```python
def fitness_fn(genome: np.ndarray, evaluator, codec, key) -> float:
    config = codec.decode(genome)
    errors = config.validation_errors()
    if errors:
        return float("inf")  # penalise invalid configs
    return evaluator.evaluate(config, key)
```

### 2.3 Parallelisation

JAX is already JIT-compiled for each evaluator call.  At the population level, parallel
evaluation is achieved via:
- **Multi-process**: `multiprocessing.Pool` with each worker receiving a different JAX key
  and a disjoint subset of the population.
- **SLURM**: one job per individual (see `slurm/` pattern in CLAUDE.md); each job writes
  its fitness to a `.dat` file; the launcher aggregates results.

**Deliverables**: `Codec` classes per model type; `fitness_fn` wrapper; SLURM launcher
`slurm/launch_nas.py` and `slurm/slurm_nas.sh`.

---

## Phase 3 — Neuro-Evolutionary Search Algorithms

**Goal**: Implement (or integrate) evolutionary algorithms that operate on the config spaces
defined in Phase 1 using the fitness functions from Phase 2.

### 3.1 Algorithm options (ordered by complexity)

| Algorithm | Suited for | Notes |
|---|---|---|
| Random Search | Baseline | No structure assumed; embarrassingly parallel |
| Evolutionary Strategy (CMA-ES) | Continuous params only | covariance-matrix adaptation |
| Genetic Algorithm (GA) | Mixed discrete + continuous | selection + crossover + mutation |
| Neuroevolution of Augmenting Topologies (NEAT) | Architecture topology | fixed-topology variant for PINN |
| Estimation of Distribution Algorithm (EDA) | Any | UMDA / TreeEDA from `pateda` |
| Differential Evolution (DE) | Continuous | scipy-based, easy baseline |

For the first experimental cycle, **Random Search + GA** are recommended as they require
no gradient information and work with mixed discrete/continuous spaces.

### 3.2 Common evolutionary loop

```python
population = [sample_config(space, rng) for _ in range(pop_size)]
for generation in range(n_generations):
    fitnesses = [fitness_fn(g, evaluator, codec, key) for g in population]
    parents = select(population, fitnesses, n_parents)
    offspring = crossover_and_mutate(parents)
    population = elitism(parents, offspring, fitnesses)
    log_best(generation, population, fitnesses)
```

### 3.3 Coordinating multiple model families

To **jointly** search across BPINN, PINN, PIKAN, and FBPINN:
1. Add a discrete `model_type` gene to the genome.
2. The codec dispatches to the appropriate sub-codec based on `model_type`.
3. The evaluator dispatcher selects the right evaluator.
4. The evolutionary algorithm treats `model_type` as any other categorical variable.

This enables **cross-model** comparisons within a single evolutionary run, finding the
best-performing model-type–hyperparameter combination for a given PDE.

**Deliverables**: `scripts/run_nas.py` (seed as first positional arg), `slurm/launch_nas.py`,
`slurm/slurm_nas.sh`.

---

## Phase 4 — Experimental Design and Benchmarking

**Goal**: Run systematic experiments comparing model families and hyperparameter
optimisation strategies.

### 4.1 PDE benchmark suite

Start with Burgers' equation (already implemented); extend to:
- **Poisson equation** (steady-state): simple benchmark for architecture comparison.
- **Allen-Cahn equation**: periodic BCs — tests Fourier embedding in BPINN/PINN.
- **Navier-Stokes (lid-driven cavity)**: 2-D, tests FBPINN domain decomposition.
- **Wave equation**: time-dependent, 2-D spatial.

Each PDE requires a new `PDEProblem` subclass in the `pde/` subfolder of each
sub-package.

### 4.2 Experimental protocol

```
seeds: 10 independent runs per configuration (seeds 111–120)
pop_size: 20 individuals
n_generations: 50
fitness metric: relative L2 error (lower is better)
evaluation budget: 5000 training epochs per candidate
```

### 4.3 Metrics to report

| Metric | Description |
|---|---|
| Best relative L2 error | Accuracy of the best discovered config |
| Mean relative L2 error | Robustness across seeds |
| Number of fitness evaluations to reach ε | Convergence speed |
| Distribution of discovered hyperparameters | Interpretability |

Statistical comparison: Kruskal-Wallis + post-hoc Dunn test (scipy) across model families.

**Deliverables**: `scripts/run_nas.py`, `scripts/analyze_nas.py`, `scripts/compare_models.py`;
LaTeX tables and `.eps` figures.

---

## Phase 5 — Analysis and Reporting

**Goal**: Extract interpretable insights from search results.

### 5.1 Hyperparameter importance analysis

- **fANOVA** (functional ANOVA): partition variance in fitness attributable to each
  hyperparameter.  Use `pyrfr` or `ConfigSpace` library.
- **Ablation study**: fix all hyperparameters to best-found values except one; vary that
  one across its range to measure sensitivity.

### 5.2 Transfer of optimal configs across PDEs

Test whether configs optimised on Burgers' equation transfer to Poisson or Allen-Cahn:
1. Take the best config found on Burgers'.
2. Evaluate it (without retraining the search) on the other PDE.
3. Compare to the config found by optimising directly on that PDE.

### 5.3 Model-specific findings to document

- **BPINN**: does Laplace prior outperform Gaussian? Does RBA improve convergence?
- **PINN**: does SIREN outperform FCN for high-frequency solutions?
- **PIKAN**: which KAN layer type yields the best accuracy/cost trade-off?
- **FBPINN**: what subdomain density and overlap fraction minimise error?

**Deliverables**: Analysis scripts; paper sections on experimental results.

---

## Implementation Notes

### Evaluator call pattern

All evaluators follow this contract (lower is better):

```python
# BPINNEvaluator
from nasjax.bpinn import BPINNEvaluator, BPINNConfig, BurgersProblem
evaluator = BPINNEvaluator(problem=BurgersProblem(), metric="rel_l2")
fitness = evaluator.evaluate(BPINNConfig(...), key)

# PINNEvaluator
from nasjax.pinns import PINNEvaluator, PINNConfig
evaluator = PINNEvaluator(problem=BurgersProblem(), metric="rel_l2")
fitness = evaluator.evaluate(PINNConfig(...), key)

# PIKANEvaluator
from nasjax.pikans import PIKANEvaluator, PIKANConfig
evaluator = PIKANEvaluator(problem=BurgersProblem(), metric="rel_l2")
fitness = evaluator.evaluate(PIKANConfig(...), key)

# FBPINNEvaluator
from nasjax.fbpinns import FBPINNEvaluator, FBPINNConfig
evaluator = FBPINNEvaluator(problem=BurgersProblem(), metric="rel_l2")
fitness = evaluator.evaluate(FBPINNConfig(...), key)
```

### Config serialisation

All configs support round-trip serialisation via `to_dict()` / `from_dict(d)` and
`to_flat_dict()`.  The flat dict is convenient for logging to CSV:

```python
row = config.to_flat_dict()
row["fitness"] = fitness
row["seed"] = seed
```

### Shared architecture representation for cross-model NAS

For BPINN, PINN, and FBPINN, the architecture is encoded as `[n_neurons, activation]`
per layer.  For PIKAN, it is `[n_out, layer_type, layer_params]`.  A unified NAS search
over BPINN + PINN could share the first two fields directly, enabling the same crossover
and mutation operators on the width and activation dimensions.

---

## File Structure After Full Implementation

```
nasjax/
├── bpinn/           # Bayesian PINN (complete)
├── pinns/           # Standard PINN (complete)
├── pikans/          # PIKAN (complete)
├── fbpinns/         # FBPINN (complete)
scripts/
├── run_nas.py            # seed  model_type  pop_size  n_gen  [extras]
├── analyze_nas.py        # parse results/*.dat → results.csv
├── compare_models.py     # tables + figures across model types
slurm/
├── launch_nas.py         # generate sbatch commands
├── slurm_nas.sh          # SLURM batch script
results/
├── nas_bpinn_burgers_*.dat
├── nas_pinn_burgers_*.dat
├── nas_pikan_burgers_*.dat
├── nas_fbpinn_burgers_*.dat
examples/
├── bpinn_burgers.py      # complete BPINN demo
├── pinn_burgers.py       # complete PINN demo
├── pikan_burgers.py      # complete PIKAN demo
├── fbpinn_burgers.py     # complete FBPINN demo
```
