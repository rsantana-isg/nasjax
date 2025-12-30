# ROADMAP: Physics-Informed Neural Networks (PINNs) Architecture Evolution in NASJAX

## Executive Summary

This roadmap outlines the integration of Physics-Informed Neural Networks (PINNs) architecture evolution into the NASJAX framework. PINNs are neural networks trained to solve differential equations by incorporating physical laws directly into the loss function. This document provides a comprehensive plan to leverage NASJAX's neuroevolutionary capabilities to automatically discover optimal PINN architectures for solving partial differential equations (PDEs).

**Timeline Estimate:** 8-10 weeks for full implementation  
**Difficulty Level:** Advanced  
**Key Technologies:** JAX, Equinox, NASJAX, Automatic Differentiation, PDEs

---

## Table of Contents

1. [Background on PINNs](#1-background-on-pinns)
2. [Current PINN Implementation Analysis](#2-current-pinn-implementation-analysis)
3. [NASJAX Architecture Overview](#3-nasjax-architecture-overview)
4. [PDE Benchmark Selection](#4-pde-benchmark-selection)
5. [PINN Integration Strategy](#5-pinn-integration-strategy)
6. [Progressive Evolution Approach](#6-progressive-evolution-approach)
7. [Implementation Phases](#7-implementation-phases)
8. [API Design and Examples](#8-api-design-and-examples)
9. [Success Criteria and Validation](#9-success-criteria-and-validation)
10. [Future Enhancements](#10-future-enhancements)

---

## 1. Background on PINNs

### 1.1 What are Physics-Informed Neural Networks?

Physics-Informed Neural Networks (PINNs) are a class of neural networks that encode physical laws (described by PDEs) as soft constraints in the loss function. Unlike traditional data-driven neural networks, PINNs:

- **Solve differential equations** without requiring large amounts of simulation data
- **Incorporate physics knowledge** directly into the training process
- **Respect conservation laws** and boundary/initial conditions
- **Provide continuous solutions** over the entire domain
- **Handle inverse problems** naturally by treating PDE parameters as learnable

### 1.2 PINN Architecture Components

A typical PINN consists of:

1. **Input Layer**: Spatial and/or temporal coordinates (x, t)
2. **Hidden Layers**: Standard MLP layers (fully connected)
3. **Output Layer**: Solution field u(x, t)
4. **Derivative Computation**: Automatic differentiation to compute ∂u/∂x, ∂u/∂t, etc.
5. **Physics Loss**: Residual of the PDE evaluated at collocation points

### 1.3 PINN Loss Function Structure

```
Total Loss = λ_PDE × L_PDE + λ_IC × L_IC + λ_BC × L_BC + λ_data × L_data
```

Where:
- **L_PDE**: Residual of the differential equation in the domain
- **L_IC**: Mismatch at initial conditions
- **L_BC**: Mismatch at boundary conditions  
- **L_data**: Mismatch with observational data (if available)
- **λ**: Weighting coefficients for loss balancing

### 1.4 Why Evolve PINN Architectures?

Traditional PINN training faces several challenges:

1. **Architecture Sensitivity**: PINN performance is highly sensitive to network depth and width
2. **Optimization Challenges**: Physics loss landscapes are non-convex with many local minima
3. **Hyperparameter Tuning**: Manual tuning of architecture and loss weights is tedious
4. **Stiff PDEs**: Some equations require specific architectural properties
5. **Multi-scale Problems**: Complex phenomena may need adaptive architectures

**Neuroevolution can address these challenges by:**
- Automatically discovering optimal architectures for specific PDE types
- Exploring the architecture space in parallel
- Evolving loss weight schedules and activation functions
- Finding architectures that avoid poor local minima
- Co-evolving architecture and training hyperparameters

---

## 2. Current PINN Implementation Analysis

### 2.1 Existing Implementation Structure

The repository includes reference PINN implementations in `Neuroevolution-of-PINNs-main/codes/` with two variants:

1. **CMA-ES based** (`codes/CMAES/`)
2. **xNES+NAG based** (`codes/xNES+NAG/`)

Each implementation includes 5 benchmark problems:
- `linear.py` - Linear convection-diffusion equation
- `nonlinear.py` - Nonlinear Burgers' equation
- `kdv.py` - Korteweg-de Vries equation
- `convection_diffusion.py` - 1D convection-diffusion
- `projectile.py` - Projectile motion

### 2.2 Current PINN Architecture Pattern

```python
# From linear.py (simplified)
class PINNs(nn.Module):
    def setup(self):
        # Fixed architecture: 3 hidden layers with 10 nodes each
        self.layers = [
            nn.Dense(10, kernel_init=jax.nn.initializers.glorot_uniform()),
            nn.tanh,
            nn.Dense(10, kernel_init=jax.nn.initializers.glorot_uniform()),
            nn.tanh,
            nn.Dense(10, kernel_init=jax.nn.initializers.glorot_uniform()),
            nn.tanh,
            nn.Dense(1, kernel_init=jax.nn.initializers.glorot_uniform(), use_bias=False)
        ]

    @nn.compact
    def __call__(self, inputs):
        # Extract coordinates
        x, t = inputs[:, 0:1], inputs[:, 1:2]
        
        # Forward pass to get u(x,t)
        u = get_u(x, t)
        
        # Automatic differentiation for derivatives
        u_x = jacfwd(get_u, argnums=0)(x, t)
        u_t = jacfwd(get_u, argnums=1)(x, t)
        u_xx = hessian(get_u, argnums=0)(x, t)
        
        # Return all quantities needed for physics loss
        return jnp.hstack([u, u_x, u_xx, u_t])
```

### 2.3 Key Observations

**Strengths:**
- ✓ Uses JAX for automatic differentiation
- ✓ Modular structure separating network definition and loss
- ✓ JIT compilation for performance
- ✓ Vectorized operations with `vmap`

**Limitations:**
- ✗ **Fixed architecture**: Hardcoded 3 layers with 10 nodes each
- ✗ **No architecture search**: Manual architecture design required
- ✗ **Limited flexibility**: Cannot easily explore different depths/widths
- ✗ **Single activation**: Only uses `tanh` activation
- ✗ **No normalization**: No batch norm or other regularization

**Opportunity:** The modular loss function design makes it ideal for integration with NASJAX's evolutionary framework.

---

## 3. NASJAX Architecture Overview

### 3.1 Core Design Principles

NASJAX follows a clean separation of concerns:

1. **Descriptors** (Genotype): Immutable PyTree representations of architecture
2. **Networks** (Phenotype): Equinox modules built from descriptors
3. **Evolution**: JAX-native genetic operators (mutation, crossover, selection)
4. **Evaluation**: Fitness evaluation with training and testing
5. **Training**: Optax-based optimization loops

### 3.2 Descriptor → Network Pipeline

```
MLPDescriptor (Genotype)
    ↓
    [dims=(64, 128, 32), activations=('relu', 'tanh', 'sigmoid')]
    ↓
MLP Network (Phenotype)
    ↓
    [Linear(in=784, out=64) → ReLU → Linear(64, 128) → Tanh → ...]
    ↓
Forward Pass / Training
```

### 3.3 Evolution Loop

```
1. Initialize random population of descriptors
2. For each generation:
   a. Build networks from descriptors
   b. Train networks (short training)
   c. Evaluate fitness on validation set
   d. Select parents based on fitness
   e. Apply crossover (optional)
   f. Apply mutations to offspring
   g. Form new population
3. Return best architecture
```

### 3.4 Mutation Operators for MLPs

Current MLP mutations in NASJAX:
- `mutate_add_layer`: Add a new hidden layer
- `mutate_remove_layer`: Remove a hidden layer
- `mutate_change_dimension`: Change neurons in a layer
- `mutate_activation`: Change activation function
- `mutate_weight_init`: Change weight initialization
- `mutate_dropout`: Modify dropout rates
- `mutate_batch_norm`: Toggle batch normalization

---

## 4. PDE Benchmark Selection

### 4.1 Recommended Starting Benchmark: Linear Convection-Diffusion

**Equation:**
```
∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0
```

**Why this benchmark?**

✓ **Simple Physics**: Linear PDE with clear physical interpretation  
✓ **Known Solutions**: Analytical solutions available for validation  
✓ **Fast Evaluation**: 1D spatial + 1D temporal domain  
✓ **Existing Data**: CSV data already provided in repository  
✓ **Documented**: Reference implementation available  
✓ **Scalable**: Can be extended to 2D later  

**Problem Setup:**
- **Domain**: x ∈ [-1.5, 4.5], t ∈ [0, 2.0]
- **Initial Condition**: u(x, 0) = 10·exp(-(2x)²)
- **Parameters**: c = 1 (convection), ν = 0.02 (diffusion)
- **Input Dimension**: 2 (x, t)
- **Output Dimension**: 1 (u)

### 4.2 Benchmark Suite (Progressive Complexity)

| Level | Problem | Equation | Derivatives | Difficulty |
|-------|---------|----------|-------------|-----------|
| 1 | Linear Burgers | ∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0 | u, u_x, u_t, u_xx | Easy |
| 2 | Convection-Diffusion | v·∂u/∂x = k·∂²u/∂x² | u, u_x, u_xx | Easy |
| 3 | Nonlinear Burgers | ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0 | u, u_x, u_t, u_xx | Medium |
| 4 | Korteweg-de Vries | ∂u/∂t + u·∂u/∂x + ∂³u/∂x³ = 0 | u, u_x, u_t, u_xxx | Hard |
| 5 | 2D Poisson | ∇²u = f | u, u_xx, u_yy | Medium |

**Recommendation:** Start with Level 1 (Linear Burgers), then progress to others.

---

## 5. PINN Integration Strategy

### 5.1 Integration Philosophy

**Key Principle:** Leverage existing NASJAX infrastructure while adding PINN-specific components.

**What to reuse:**
- ✓ MLPDescriptor (can represent PINN architectures)
- ✓ MLP network class (with modifications)
- ✓ Evolution operators (mutation, crossover, selection)
- ✓ Population management
- ✓ Training utilities

**What to add:**
- ➕ PINNDescriptor (extends MLPDescriptor with PINN-specific parameters)
- ➕ PINNNetwork (extends MLP with automatic differentiation)
- ➕ Physics-informed loss functions
- ➕ PDE-specific data generators
- ➕ PINN-specific evaluator

### 5.2 Architecture Decision: Extension vs Specialization

**Option A: Pure Extension** (RECOMMENDED)
```python
# Use existing MLPDescriptor
# Add PINN-specific evaluation/training
PINNEvaluator(
    descriptor_class=MLPDescriptor,
    loss_fn=physics_loss_linear_burgers,
    ...
)
```

**Advantages:**
- ✓ Minimal code duplication
- ✓ Reuse all existing mutations
- ✓ Simpler maintenance
- ✓ PINNs are just MLPs with different loss functions

**Option B: Specialized Descriptor**
```python
@jtu.register_pytree_node_class
class PINNDescriptor(MLPDescriptor):
    # Additional PINN-specific fields
    pde_type: str
    derivative_order: int
    loss_weights: Tuple[float, ...]
```

**Advantages:**
- ✓ More explicit PINN semantics
- ✓ Can add PINN-specific mutations
- ✓ Clearer API for users

**Decision:** Start with **Option A** for MVP, consider Option B for future enhancements.

### 5.3 Component Mapping

| NASJAX Component | PINN Adaptation | Changes Required |
|------------------|-----------------|------------------|
| MLPDescriptor | Use as-is | None |
| MLP Network | Add derivative computation | Extend with AD wrapper |
| Evaluator | Replace loss function | New PINNEvaluator |
| Trainer | Optionally customize | Use existing or modify |
| Evolution | Use as-is | None |
| Mutation | Use as-is | None (future: add PINN-specific) |

---

## 6. Progressive Evolution Approach

### 6.1 Phase 1: Architecture-Only Evolution (Fixed Weights)

**Approach:** Evolve network architecture while using a fixed weight initialization.

**Rationale:**
- Fastest evaluation (no training)
- Tests if architecture alone impacts PINN performance
- Baseline for comparison with trained networks

**Fitness:** Evaluate physics loss on initial random weights

**Limitations:** May not find truly optimal architectures without training

### 6.2 Phase 2: Architecture + Short Training

**Approach:** Evolve architecture, train each candidate for a fixed number of steps.

**Rationale:**
- More realistic fitness evaluation
- Still computationally feasible
- Balances exploration and exploitation

**Fitness:** Evaluate physics loss after short training (e.g., 100-500 iterations)

**Recommended:** This is the **primary approach** for PINN evolution in NASJAX.

### 6.3 Phase 3: Co-evolution of Architecture and Hyperparameters

**Approach:** Evolve architecture along with training hyperparameters.

**Evolvable Hyperparameters:**
- Learning rate
- Optimizer choice (Adam, SGD, RMSprop)
- Loss weights (λ_PDE, λ_IC, λ_BC)
- Batch size for collocation points
- Number of training iterations

**Rationale:**
- Architecture and training interact
- Optimal hyperparameters vary by architecture
- More comprehensive search

**Fitness:** Performance after adaptive training

**Future Enhancement:** Implement after Phase 2 is validated.

### 6.4 Phase 4: Multi-Objective Evolution

**Approach:** Optimize multiple objectives simultaneously.

**Objectives:**
1. **Accuracy**: Physics loss / prediction error
2. **Efficiency**: Number of parameters / FLOPs
3. **Training Speed**: Convergence rate
4. **Robustness**: Performance across different initial conditions

**Algorithm:** NSGA-II or NSGA-III for Pareto front

**Rationale:**
- Real applications need accuracy AND efficiency
- Discover diverse architectures for different use cases
- Understand trade-offs

**Future Enhancement:** Implement after Phase 2 success.

---

## 7. Implementation Phases

### Phase 1: Foundation and Infrastructure (Week 1-2)

#### Tasks

1. **Create PINN Module Structure**
   ```
   nasjax/
   ├── pinn/
   │   ├── __init__.py
   │   ├── networks.py          # PINNNetwork class
   │   ├── losses.py            # Physics loss functions
   │   ├── evaluators.py        # PINNEvaluator
   │   ├── benchmarks/          # PDE benchmark problems
   │   │   ├── __init__.py
   │   │   ├── linear_burgers.py
   │   │   ├── convection_diffusion.py
   │   │   └── ...
   │   └── utils.py             # Derivative computation helpers
   ```

2. **Implement Derivative Computation Wrapper**
   ```python
   # nasjax/pinn/networks.py
   class PINNNetwork(eqx.Module):
       """Wrapper around MLP for PINN functionality."""
       
       mlp: MLP
       
       def __init__(self, descriptor: MLPDescriptor, key: jax.random.PRNGKey):
           self.mlp = MLP(descriptor, key)
       
       def __call__(self, coords, compute_derivatives=True):
           """Forward pass with optional derivative computation.
           
           Args:
               coords: Input coordinates (x, t) of shape (n_points, 2)
               compute_derivatives: Whether to compute derivatives
               
           Returns:
               If compute_derivatives=True: (u, u_x, u_t, u_xx, ...)
               If compute_derivatives=False: u
           """
           if not compute_derivatives:
               return self.mlp(coords, inference=True)
           
           # Compute derivatives using JAX autodiff
           def forward_fn(coords):
               return self.mlp(coords, inference=True)
           
           u = forward_fn(coords)
           u_x = jax.jacfwd(forward_fn, argnums=0)(coords)  # ∂u/∂x
           u_t = jax.jacfwd(forward_fn, argnums=1)(coords)  # ∂u/∂t
           u_xx = jax.hessian(forward_fn, argnums=0)(coords)  # ∂²u/∂x²
           
           return u, u_x, u_t, u_xx
   ```

3. **Implement Physics Loss Function for Linear Burgers**
   ```python
   # nasjax/pinn/losses.py
   def linear_burgers_loss(
       network: PINNNetwork,
       coords: jnp.ndarray,
       ic_coords: jnp.ndarray,
       bc_coords: jnp.ndarray,
       params: Dict[str, float],
       loss_weights: Dict[str, float]
   ) -> float:
       """Physics-informed loss for linear Burgers equation.
       
       PDE: ∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0
       """
       c = params['c']  # convection
       nu = params['nu']  # diffusion
       
       # Evaluate network and derivatives at collocation points
       u, u_x, u_t, u_xx = jax.vmap(lambda p: network(p))(coords)
       
       # PDE residual
       pde_residual = u_t + c * u_x - nu * u_xx
       loss_pde = jnp.mean(pde_residual ** 2)
       
       # Initial condition loss
       u_ic = jax.vmap(lambda p: network(p, compute_derivatives=False))(ic_coords)
       u_ic_target = initial_condition(ic_coords[:, 0])  # u(x, 0)
       loss_ic = jnp.mean((u_ic - u_ic_target) ** 2)
       
       # Boundary condition loss (if needed)
       # ...
       
       # Weighted combination
       total_loss = (
           loss_weights['pde'] * loss_pde +
           loss_weights['ic'] * loss_ic
       )
       
       return total_loss
   ```

4. **Create Benchmark Problem Class**
   ```python
   # nasjax/pinn/benchmarks/linear_burgers.py
   from dataclasses import dataclass
   import jax.numpy as jnp
   
   @dataclass
   class LinearBurgersProblem:
       """Linear Burgers equation benchmark."""
       
       # PDE parameters
       c: float = 1.0  # convection coefficient
       nu: float = 0.02  # diffusion coefficient
       
       # Domain
       x_min: float = -1.5
       x_max: float = 4.5
       t_min: float = 0.0
       t_max: float = 2.0
       
       # Collocation points
       n_collocation: int = 10000
       n_ic: int = 100
       n_bc: int = 100
       
       def generate_collocation_points(self, key):
           """Generate random collocation points in domain."""
           key_x, key_t = jax.random.split(key)
           x = jax.random.uniform(key_x, (self.n_collocation, 1), 
                                  minval=self.x_min, maxval=self.x_max)
           t = jax.random.uniform(key_t, (self.n_collocation, 1),
                                  minval=self.t_min, maxval=self.t_max)
           return jnp.hstack([x, t])
       
       def initial_condition(self, x):
           """Analytical initial condition: u(x, 0) = 10·exp(-(2x)²)."""
           return 10.0 * jnp.exp(-(2.0 * x) ** 2)
       
       def loss_function(self, network, coords_dict):
           """Compute physics-informed loss."""
           return linear_burgers_loss(
               network,
               coords_dict['collocation'],
               coords_dict['ic'],
               coords_dict['bc'],
               params={'c': self.c, 'nu': self.nu},
               loss_weights={'pde': 1.0, 'ic': 1.0, 'bc': 0.0}
           )
   ```

#### Deliverables
- ✅ PINN module structure created
- ✅ PINNNetwork class implemented
- ✅ Linear Burgers loss function working
- ✅ Benchmark problem class defined
- ✅ Unit tests for derivative computation

---

### Phase 2: PINN-Specific Evaluator (Week 2-3)

#### Tasks

1. **Implement PINNEvaluator**
   ```python
   # nasjax/pinn/evaluators.py
   from nasjax.evaluation.evaluator import Evaluator
   from nasjax.training.trainer import Trainer
   
   class PINNEvaluator:
       """Evaluator for PINN architectures."""
       
       def __init__(
           self,
           problem: LinearBurgersProblem,
           trainer: Optional[Trainer] = None,
           n_train_iters: int = 500,
           learning_rate: float = 1e-3,
       ):
           """Initialize PINN evaluator.
           
           Args:
               problem: PDE problem to solve
               trainer: Optional custom trainer
               n_train_iters: Training iterations per evaluation
               learning_rate: Learning rate for training
           """
           self.problem = problem
           self.n_train_iters = n_train_iters
           
           if trainer is None:
               self.trainer = Trainer(
                   optimizer='adam',
                   learning_rate=learning_rate
               )
           else:
               self.trainer = trainer
       
       def evaluate(
           self,
           descriptor: MLPDescriptor,
           key: jax.random.PRNGKey,
           train: bool = True
       ) -> float:
           """Evaluate a PINN architecture.
           
           Args:
               descriptor: Network architecture descriptor
               key: Random key
               train: Whether to train before evaluation
               
           Returns:
               Fitness value (lower is better)
           """
           # Generate collocation points
           key, subkey = jax.random.split(key)
           coords_dict = {
               'collocation': self.problem.generate_collocation_points(subkey),
               'ic': self.problem.generate_ic_points(subkey),
               'bc': self.problem.generate_bc_points(subkey)
           }
           
           # Build network from descriptor
           key, subkey = jax.random.split(key)
           network = PINNNetwork(descriptor, subkey)
           
           # Optionally train the network
           if train:
               key, subkey = jax.random.split(key)
               network = self.train_pinn(network, coords_dict, subkey)
           
           # Evaluate physics loss
           loss = self.problem.loss_function(network, coords_dict)
           
           return float(loss)
       
       def train_pinn(
           self,
           network: PINNNetwork,
           coords_dict: Dict[str, jnp.ndarray],
           key: jax.random.PRNGKey
       ) -> PINNNetwork:
           """Train PINN using physics-informed loss."""
           
           # Define loss function for this specific problem
           def loss_fn(net):
               return self.problem.loss_function(net, coords_dict)
           
           # Train using Optax
           network = self.trainer.train(
               model=network,
               loss_fn=loss_fn,
               n_iters=self.n_train_iters,
               key=key
           )
           
           return network
   ```

2. **Integrate with Evolving Class**
   ```python
   # Usage example
   from nasjax.evolution.evolving import Evolving
   from nasjax.pinn.benchmarks.linear_burgers import LinearBurgersProblem
   from nasjax.pinn.evaluators import PINNEvaluator
   from nasjax.descriptors import MLPDescriptor
   
   # Setup PINN problem
   problem = LinearBurgersProblem()
   evaluator = PINNEvaluator(problem, n_train_iters=500)
   
   # Run evolution
   evolution = Evolving(
       input_dim=2,  # (x, t)
       output_dim=1,  # u
       max_num_layers=5,
       max_num_neurons=64,
       evaluator=evaluator,  # Use PINN evaluator
       config=EvolvingConfig(
           pop_size=20,
           n_generations=30,
           mutation_prob=0.8,
           elitism=2
       )
   )
   
   # Evolve PINN architectures
   population, logbook, best = evolution.evolve()
   ```

#### Deliverables
- ✅ PINNEvaluator implemented
- ✅ Integration with Evolving class
- ✅ Training loop for PINNs
- ✅ End-to-end test with linear Burgers

---

### Phase 3: Examples and Validation (Week 3-4)

#### Tasks

1. **Create Example Scripts**
   ```
   examples/pinn/
   ├── evolve_linear_burgers.py
   ├── evolve_convection_diffusion.py
   ├── compare_architectures.py
   └── visualize_solutions.py
   ```

2. **Implement Visualization Tools**
   ```python
   # examples/pinn/visualize_solutions.py
   import matplotlib.pyplot as plt
   
   def visualize_pinn_solution(network, problem, filename='pinn_solution.png'):
       """Visualize PINN solution."""
       # Generate grid
       x = jnp.linspace(problem.x_min, problem.x_max, 100)
       t = jnp.linspace(problem.t_min, problem.t_max, 100)
       X, T = jnp.meshgrid(x, t)
       
       # Evaluate network
       coords = jnp.stack([X.flatten(), T.flatten()], axis=1)
       u_pred = jax.vmap(lambda p: network(p, compute_derivatives=False))(coords)
       U_pred = u_pred.reshape(X.shape)
       
       # Plot
       plt.figure(figsize=(10, 4))
       plt.contourf(X, T, U_pred, levels=50, cmap='viridis')
       plt.colorbar(label='u(x,t)')
       plt.xlabel('x')
       plt.ylabel('t')
       plt.title('PINN Solution')
       plt.savefig(filename)
   ```

3. **Validation Against Analytical Solutions**
   ```python
   def validate_solution(network, problem, n_test_points=1000):
       """Compare PINN solution with analytical solution."""
       # Generate test points
       key = jax.random.PRNGKey(42)
       test_coords = problem.generate_collocation_points(key)
       
       # PINN predictions
       u_pred = jax.vmap(lambda p: network(p, compute_derivatives=False))(test_coords)
       
       # Analytical solution (if available)
       u_true = problem.analytical_solution(test_coords)
       
       # Compute errors
       l2_error = jnp.sqrt(jnp.mean((u_pred - u_true) ** 2))
       relative_error = l2_error / jnp.sqrt(jnp.mean(u_true ** 2))
       
       print(f"L2 Error: {l2_error:.6f}")
       print(f"Relative Error: {relative_error:.6%}")
       
       return l2_error, relative_error
   ```

#### Deliverables
- ✅ Complete example scripts
- ✅ Visualization utilities
- ✅ Validation against analytical solutions
- ✅ Documentation and tutorial

---

### Phase 4: Advanced Features (Week 5-6)

#### Tasks

1. **PINN-Specific Mutations** (Optional Enhancement)
   ```python
   # nasjax/pinn/mutations.py
   def mutate_activation_for_pde(descriptor: MLPDescriptor, key) -> MLPDescriptor:
       """Mutate activation with PDE-appropriate choices.
       
       PINNs often work best with smooth activations like tanh, sin.
       """
       pinn_activations = ['tanh', 'sin', 'softplus', 'swish']
       # Implementation...
   
   def mutate_add_residual_connection(descriptor, key):
       """Add skip connection (helps with gradient flow in PINNs)."""
       # Implementation...
   ```

2. **Adaptive Loss Weighting**
   ```python
   def evolve_loss_weights(descriptor, key):
       """Evolve loss weights (λ_PDE, λ_IC, λ_BC) along with architecture."""
       # Could be added to descriptor or trained adaptively
   ```

3. **Multi-Objective Evolution for PINNs**
   ```python
   # Objectives: accuracy, network size, training speed
   from nasjax.evolution.selection import nsga2_selection
   
   def evaluate_multi_objective(descriptor):
       accuracy = evaluate_accuracy(descriptor)
       size = count_parameters(descriptor)
       speed = measure_training_time(descriptor)
       return [accuracy, size, speed]
   ```

4. **Additional Benchmarks**
   - Implement remaining benchmarks (nonlinear Burgers, KdV, etc.)
   - Add 2D problems (Poisson, heat equation)

#### Deliverables
- ✅ PINN-specific mutations (optional)
- ✅ Adaptive loss weighting (optional)
- ✅ Multi-objective support (optional)
- ✅ Extended benchmark suite

---

## 8. API Design and Examples

### 8.1 Basic Usage

```python
import jax
from nasjax.evolution.evolving import Evolving, EvolvingConfig
from nasjax.pinn.benchmarks import LinearBurgersProblem
from nasjax.pinn.evaluators import PINNEvaluator
from nasjax.descriptors import MLPDescriptor

# 1. Define the PDE problem
problem = LinearBurgersProblem(
    c=1.0,          # convection coefficient
    nu=0.02,        # diffusion coefficient
    n_collocation=10000,
    n_ic=100
)

# 2. Create PINN evaluator
evaluator = PINNEvaluator(
    problem=problem,
    n_train_iters=500,
    learning_rate=1e-3
)

# 3. Configure evolution
config = EvolvingConfig(
    pop_size=20,
    n_generations=30,
    mutation_prob=0.8,
    crossover_prob=0.0,  # Start without crossover
    elitism=2,
    selection_method='tournament',
    tournament_size=3
)

# 4. Run evolution
evolution = Evolving(
    input_dim=2,           # (x, t)
    output_dim=1,          # u
    max_num_layers=5,
    max_num_neurons=64,
    config=config,
    evaluator=evaluator
)

# 5. Evolve!
population, logbook, hall_of_fame = evolution.evolve()

# 6. Get best architecture
best_descriptor = hall_of_fame[-1]
print(f"Best architecture: {best_descriptor}")
print(f"Best fitness: {best_descriptor.fitness}")
print(f"Architecture: {[2] + list(best_descriptor.dims) + [1]}")

# 7. Build and evaluate best network
key = jax.random.PRNGKey(42)
best_network = PINNNetwork(best_descriptor, key)

# 8. Visualize solution
from nasjax.pinn.visualization import visualize_pinn_solution
visualize_pinn_solution(best_network, problem, filename='best_pinn.png')
```

### 8.2 Advanced Usage: Custom PDE

```python
from nasjax.pinn.benchmarks.base import BasePDEProblem
from dataclasses import dataclass

@dataclass
class CustomPDE(BasePDEProblem):
    """Custom PDE: ∂u/∂t - α·∂²u/∂x² = f(x,t)"""
    
    alpha: float = 0.1
    
    def loss_function(self, network, coords_dict):
        """Physics-informed loss for custom PDE."""
        coords = coords_dict['collocation']
        
        # Network predictions
        u, u_x, u_t, u_xx = jax.vmap(lambda p: network(p))(coords)
        
        # PDE residual
        source_term = self.compute_source_term(coords)
        pde_residual = u_t - self.alpha * u_xx - source_term
        loss_pde = jnp.mean(pde_residual ** 2)
        
        # Add IC/BC losses...
        
        return loss_pde
    
    def compute_source_term(self, coords):
        """Source term f(x,t)."""
        x, t = coords[:, 0], coords[:, 1]
        return jnp.sin(jnp.pi * x) * jnp.exp(-t)

# Use custom PDE with evolution
custom_problem = CustomPDE()
evaluator = PINNEvaluator(custom_problem)
evolution = Evolving(..., evaluator=evaluator)
```

### 8.3 Batch Evolution with Multiple PDEs

```python
from nasjax.pinn.benchmarks import (
    LinearBurgersProblem,
    ConvectionDiffusionProblem,
    NonlinearBurgersProblem
)

# Evolve on multiple PDEs simultaneously
problems = [
    LinearBurgersProblem(),
    ConvectionDiffusionProblem(),
    NonlinearBurgersProblem()
]

results = {}
for problem in problems:
    print(f"\n=== Evolving for {problem.__class__.__name__} ===")
    evaluator = PINNEvaluator(problem)
    evolution = Evolving(..., evaluator=evaluator)
    _, _, best = evolution.evolve()
    results[problem.__class__.__name__] = best

# Compare architectures across PDEs
for name, descriptor in results.items():
    print(f"{name}: dims={descriptor.dims}, fitness={descriptor.fitness}")
```

### 8.4 Architecture Analysis

```python
from nasjax.pinn.analysis import analyze_architecture

# Analyze evolved architecture
analysis = analyze_architecture(
    descriptor=best_descriptor,
    problem=problem,
    n_test_points=5000
)

print(f"Number of parameters: {analysis['n_params']}")
print(f"L2 error: {analysis['l2_error']:.6f}")
print(f"Relative error: {analysis['relative_error']:.2%}")
print(f"Training time: {analysis['train_time']:.2f}s")
print(f"Inference time: {analysis['inference_time']:.4f}s")
```

---

## 9. Success Criteria and Validation

### 9.1 Functional Success Criteria

✅ **Core Functionality**
- [ ] PINNNetwork successfully computes derivatives via autodiff
- [ ] Physics loss correctly implements PDE residuals
- [ ] Evolution loop runs without errors
- [ ] Best architecture improves over generations
- [ ] Solution satisfies PDE within acceptable tolerance

✅ **API Usability**
- [ ] Simple 10-line example works out of box
- [ ] Custom PDE can be defined in <50 lines
- [ ] Clear documentation and tutorials available
- [ ] Error messages are helpful

### 9.2 Performance Criteria

✅ **Accuracy**
- [ ] L2 error < 1e-3 for linear Burgers (with training)
- [ ] Relative error < 1% compared to analytical solution
- [ ] PDE residual < 1e-4 in the domain

✅ **Evolution Quality**
- [ ] Best fitness improves by >50% over 30 generations
- [ ] Diversity maintained in population
- [ ] No premature convergence

✅ **Computational Efficiency**
- [ ] Single evaluation completes in <10 seconds (with 500 training iters)
- [ ] Full evolution (30 gen, pop=20) completes in <2 hours on GPU
- [ ] 10x faster than naive non-vectorized implementation

### 9.3 Validation Approach

**Step 1: Unit Tests**
```python
def test_derivative_computation():
    """Test automatic differentiation for PINN."""
    # Create simple network
    descriptor = MLPDescriptor.random_init(2, 1, 3, 32, key)
    network = PINNNetwork(descriptor, key)
    
    # Evaluate at test point
    coords = jnp.array([[0.5, 1.0]])
    u, u_x, u_t, u_xx = network(coords)
    
    # Check shapes
    assert u.shape == (1, 1)
    assert u_x.shape == (1, 1)
    assert u_t.shape == (1, 1)
    assert u_xx.shape == (1, 1)
    
    # Check derivatives are non-zero
    assert not jnp.all(u_x == 0)

def test_physics_loss():
    """Test physics loss computation."""
    # Test that loss decreases with training
    # ...
```

**Step 2: Integration Tests**
```python
def test_end_to_end_evolution():
    """Test complete evolution pipeline."""
    problem = LinearBurgersProblem()
    evaluator = PINNEvaluator(problem, n_train_iters=10)  # Fast test
    evolution = Evolving(2, 1, 3, 32, evaluator=evaluator)
    
    config = EvolvingConfig(pop_size=5, n_generations=3)
    population, logbook, _ = evolution.evolve()
    
    # Check evolution ran
    assert len(logbook) == 3
    assert len(population.individuals) == 5
    
    # Check fitness improved
    assert logbook[-1]['min'] < logbook[0]['min']
```

**Step 3: Benchmark Validation**
```python
def test_linear_burgers_accuracy():
    """Validate against known solution."""
    # Train best architecture
    best_network = train_best_architecture()
    
    # Compare with analytical solution
    l2_error, rel_error = validate_solution(best_network, problem)
    
    # Check accuracy
    assert l2_error < 1e-3
    assert rel_error < 0.01
```

**Step 4: Comparison Study**
- Compare evolved architecture vs. hand-designed architecture
- Compare evolution vs. random search
- Compare NASJAX-PINN vs. original implementation

### 9.4 Metrics to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| L2 Error | < 1e-3 | Compare with analytical solution |
| Relative Error | < 1% | Normalized by solution magnitude |
| PDE Residual | < 1e-4 | Evaluate residual in domain |
| Evolution Time | < 2h | Wall-clock time for 30 generations |
| Memory Usage | < 8GB | Peak GPU memory |
| Best Fitness Improvement | > 50% | First vs. last generation |
| Architecture Diversity | Shannon entropy > 2.0 | Measure descriptor variation |

---

## 10. Future Enhancements

### 10.1 Short-term (Next 3-6 months)

1. **Extended Benchmark Suite**
   - Nonlinear Burgers equation
   - Korteweg-de Vries (KdV) equation
   - 2D Poisson equation
   - Allen-Cahn equation
   - Navier-Stokes (2D)

2. **PINN-Specific Mutations**
   - Activation function evolution (sin, tanh, swish)
   - Fourier feature mapping
   - Adaptive weighting for loss terms
   - Skip connections for deeper networks

3. **Improved Training Strategies**
   - Curriculum learning (start with easy regions)
   - Adaptive sampling of collocation points
   - Sequential training (IC → PDE → refinement)
   - Learning rate scheduling

### 10.2 Medium-term (6-12 months)

1. **Multi-Objective Optimization**
   - Accuracy vs. efficiency trade-off
   - Pareto front exploration
   - NSGA-II/III implementation

2. **Inverse Problems**
   - Learn PDE parameters (c, ν, etc.) from data
   - Co-evolve architecture and PDE parameters
   - Uncertainty quantification

3. **Advanced Architectures**
   - Convolutional PINNs for spatial problems
   - Recurrent PINNs for temporal problems
   - Graph neural networks for irregular domains
   - Fourier Neural Operators (FNO) integration

4. **Scalability**
   - Domain decomposition for large problems
   - Multi-GPU parallelization with `pmap`
   - Efficient batching strategies

### 10.3 Long-term (1-2 years)

1. **Differentiable Physics Engine**
   - Integrate with JAX-based physics simulators
   - End-to-end differentiable PDE solvers
   - Hybrid neural-symbolic approaches

2. **AutoML for PINNs**
   - Automatic PDE detection from data
   - Meta-learning across PDE families
   - Transfer learning between related PDEs

3. **Production Deployment**
   - Model serving for real-time inference
   - Uncertainty estimation
   - Robustness certification
   - Edge deployment (embedded systems)

4. **Scientific Applications**
   - Fluid dynamics (CFD)
   - Structural mechanics (FEM)
   - Quantum mechanics (Schrödinger equation)
   - Climate modeling
   - Molecular dynamics

---

## Conclusion

This roadmap provides a comprehensive plan for integrating PINN architecture evolution into NASJAX. The key design decisions are:

1. **Leverage Existing Infrastructure**: Use MLPDescriptor and evolution operators as-is
2. **Add PINN-Specific Components**: PINNNetwork, physics losses, PINN evaluator
3. **Start Simple**: Begin with linear PDEs and basic evolution
4. **Progressive Complexity**: Add advanced features incrementally
5. **Modular Design**: Each component is independently testable

**Expected Timeline:**
- **Weeks 1-2**: Foundation (PINN module structure, derivatives, losses)
- **Weeks 2-3**: Evaluator and integration with evolution
- **Weeks 3-4**: Examples, validation, documentation
- **Weeks 5-6**: Advanced features (optional)

**Success Metrics:**
- L2 error < 1e-3 for linear Burgers
- Evolution completes in < 2 hours
- 50%+ fitness improvement over generations

By following this roadmap, NASJAX will become a powerful platform for automatically discovering optimal PINN architectures, making physics-informed machine learning more accessible and effective.

---

## References

1. **PINN Papers in Repository:**
   - `docs/pinn_references/2212.07624v3.pdf`: "Accelerated Neuroevolution of Physics-informed Neural Networks"
   - `docs/pinn_references/Neuroevolving_monotonic_PINNs_for_particle_breakage_analysis.pdf`

2. **Original PINN Work:**
   - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

3. **Neuroevolution:**
   - Stanley, K. O., & Miikkulainen, R. (2002). "Evolving neural networks through augmenting topologies." *Evolutionary computation*, 10(2), 99-127.

4. **NASJAX:**
   - See `README.md` and `ROADMAP.md` in this repository for NASJAX design philosophy

5. **JAX:**
   - Bradbury, J., et al. (2018). "JAX: composable transformations of Python+NumPy programs." http://github.com/google/jax

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** NASJAX Development Team  
**Status:** Living Document - Will be updated as implementation progresses
