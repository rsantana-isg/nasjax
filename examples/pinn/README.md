# PINN Examples - Physics-Informed Neural Networks

This directory contains examples demonstrating how to use NASJAX to evolve Physics-Informed Neural Network (PINN) architectures for solving partial differential equations (PDEs).

## What are PINNs?

Physics-Informed Neural Networks (PINNs) are neural networks that solve differential equations by:
1. Encoding physical laws (PDEs) directly in the loss function
2. Using automatic differentiation to compute derivatives
3. Training on collocation points without requiring simulation data

## Examples

### `evolve_linear_burgers.py` - Main Evolution Example

Demonstrates the complete workflow for evolving PINN architectures to solve the Linear Burgers equation:

```
∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0
```

**What it does:**
- Defines the PDE problem (Linear Burgers equation)
- Configures PINN evaluator with physics-informed loss
- Runs evolutionary algorithm to discover optimal architectures
- Analyzes and reports results

**Run it:**
```bash
cd examples/pinn
python evolve_linear_burgers.py
```

**Expected output:**
- Evolution progress over 5 generations
- Best architecture found
- Physics loss improvement
- Performance statistics

**Typical runtime:** ~2-5 minutes (CPU), faster on GPU

## How It Works

### 1. Define the PDE Problem

```python
from nasjax.pinn import LinearBurgersProblem

problem = LinearBurgersProblem(
    c=1.0,                # Convection coefficient
    nu=0.02,              # Diffusion coefficient
    n_collocation=2000,   # Number of PDE residual points
    n_ic=100,             # Initial condition points
)
```

### 2. Create PINN Evaluator

```python
from nasjax.pinn import PINNEvaluator

evaluator = PINNEvaluator(
    problem=problem,
    n_train_iters=200,    # Train each network for 200 iterations
    learning_rate=1e-3,
    optimizer='adam'
)
```

### 3. Run Evolution

```python
from nasjax.evolution import Evolving, EvolvingConfig

evolving = Evolving(
    input_dim=2,          # (x, t) coordinates
    output_dim=1,         # u(x,t) solution
    max_num_layers=4,
    max_num_neurons=32,
    evaluator=evaluator   # Use PINN evaluator
)

population, log = evolving.evolve(...)
```

### 4. Get Best Architecture

```python
best = population.get_best(1)[0]
print(f"Best architecture: {best.descriptor.dims}")
print(f"Physics loss: {best.fitness:.6f}")
```

## Available PDE Problems

### Linear Burgers Equation
```python
from nasjax.pinn import LinearBurgersProblem
problem = LinearBurgersProblem(c=1.0, nu=0.02)
```
- **PDE:** ∂u/∂t + c·∂u/∂x - ν·∂²u/∂x² = 0
- **Difficulty:** Easy
- **Good for:** Testing and getting started

### Convection-Diffusion Equation
```python
from nasjax.pinn import ConvectionDiffusionProblem
problem = ConvectionDiffusionProblem(v=1.0, k=0.01)
```
- **PDE:** v·∂u/∂x = k·∂²u/∂x²
- **Difficulty:** Easy
- **Good for:** Steady-state problems

## Customization

### Adjust Evolution Parameters

```python
config = EvolvingConfig(
    pop_size=20,           # Larger population = better exploration
    n_generations=10,      # More generations = better convergence
    mutation_prob=0.9,     # High mutation for architecture search
    elitism=2,             # Preserve best individuals
)
```

### Adjust Training Parameters

```python
evaluator = PINNEvaluator(
    problem=problem,
    n_train_iters=500,     # More iterations = better trained networks
    learning_rate=1e-3,    # Lower LR for stability
    optimizer='adam',      # Try 'sgd', 'rmsprop', 'adamw'
)
```

### Adjust Architecture Search Space

```python
evolving = Evolving(
    input_dim=2,
    output_dim=1,
    max_num_layers=5,      # Allow deeper networks
    max_num_neurons=64,    # Allow wider layers
    ...
)
```

## Visualization and Validation

The PINN module includes tools for visualization and validation:

```python
from nasjax.pinn.visualization import (
    visualize_pinn_solution_2d,
    visualize_initial_condition,
    compare_with_analytical
)

from nasjax.pinn.validation import (
    validate_solution,
    compute_l2_error,
    compute_pde_residual
)

# Visualize solution
visualize_pinn_solution_2d(network, problem, key, filename='solution.png')

# Validate against analytical solution
metrics = validate_solution(network, problem, key)
print(f"L2 Error: {metrics['l2_error']:.6f}")
print(f"PDE Residual: {metrics['pde_residual']:.6f}")
```

## Tips for Best Results

1. **Start Simple:** Begin with Linear Burgers before trying harder PDEs
2. **Tune Training:** More training iterations per network = better fitness evaluation
3. **Population Size:** Larger populations explore more architectures
4. **Activation Functions:** PINNs often work best with smooth activations (tanh, sin)
5. **Loss Weights:** Adjust `loss_weight_pde`, `loss_weight_ic`, `loss_weight_bc` if needed
6. **Collocation Points:** More points = better PDE satisfaction but slower training

## Common Issues

### GPU/CUDA Errors

**Issue:** "DNN library initialization failed" or CuDNN version mismatch
- **Quick Fix:** Run with CPU mode: `JAX_PLATFORMS=cpu python evolve_linear_burgers.py`
- **Details:** See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for complete GPU setup instructions

### Performance Issues

**Issue:** Evolution is slow
- **Solution:** Reduce `n_train_iters`, `pop_size`, or `n_collocation`
- **Or:** Use GPU if available (10-100x faster than CPU)

**Issue:** High physics loss
- **Solution:** Increase `n_train_iters`, try different learning rates, or allow larger architectures

**Issue:** Networks not improving
- **Solution:** Increase `mutation_prob`, reduce `elitism`, or increase `pop_size`

### For More Help

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for:
- Complete GPU/CUDA troubleshooting
- Memory optimization tips
- Platform-specific notes (Linux/macOS/Windows)
- Verification and testing procedures

## Next Steps

After running the examples:
1. Try different PDE problems
2. Experiment with evolution parameters
3. Visualize the evolved solutions
4. Create custom PDE problems (see ROADMAP_PINNs.md)
5. Extend to 2D/3D problems

## References

- See `ROADMAP_PINNs.md` in the repository root for full implementation details
- Original PINN paper: Raissi et al. (2019) "Physics-informed neural networks"
- NASJAX documentation: See repository `README.md`
