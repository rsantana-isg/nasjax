# PINN Examples - Troubleshooting Guide

This guide helps you resolve common issues when running PINN examples.

---

## GPU/CUDA Issues

### Error: "DNN library initialization failed" with CuDNN version mismatch

**Symptoms:**
```
E0103 00:04:05.525670 cuda_dnn.cc:456] Loaded runtime CuDNN library: 9.10.2 but source was compiled with: 9.12.0
FAILED_PRECONDITION: DNN library initialization failed
```

**Cause:** Your JAX installation was compiled with CuDNN 9.12.0, but your system has CuDNN 9.10.2.

**Solutions:**

#### Option 1: Force CPU Execution (Quickest Fix)

Run examples with CPU-only execution:

```bash
JAX_PLATFORMS=cpu python examples/pinn/evolve_linear_burgers.py
JAX_PLATFORMS=cpu python examples/pinn/evolve_convection_diffusion.py
```

Or set it in your shell permanently:
```bash
export JAX_PLATFORMS=cpu
python examples/pinn/evolve_linear_burgers.py
```

Or add to the top of each script:
```python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
```

#### Option 2: Update CuDNN (System Fix)

Update your CuDNN library to version 9.12.0 or higher:

1. Download CuDNN 9.12+ from NVIDIA:
   https://developer.nvidia.com/cudnn

2. Install following NVIDIA's instructions for your system

3. Verify installation:
   ```bash
   ldconfig -p | grep cudnn
   ```

#### Option 3: Reinstall JAX with Matching CUDA Version

Reinstall JAX to match your CuDNN version:

```bash
# For CUDA 11.x
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CPU only
pip install --upgrade "jax[cpu]"
```

#### Option 4: Use requirements-cuda11.txt

If you have CUDA 11.x and older CuDNN:

```bash
pip install -r requirements-cuda11.txt
```

---

## Memory Issues

### Error: "Out of memory"

**Solutions:**

1. **Reduce population size:**
   ```python
   config = EvolvingConfig(
       pop_size=4,  # Reduce from 8 or 20
       ...
   )
   ```

2. **Reduce collocation points:**
   ```python
   problem = LinearBurgersProblem(
       n_collocation=1000,  # Reduce from 2000
       ...
   )
   ```

3. **Reduce training iterations:**
   ```python
   evaluator = PINNEvaluator(
       problem=problem,
       n_train_iters=100,  # Reduce from 200
       ...
   )
   ```

---

## Import Errors

### Error: "ModuleNotFoundError: No module named 'jax'"

**Solution:** Install dependencies:
```bash
# For GPU
pip install -r requirements.txt

# For CPU only
pip install -r requirements-cpu.txt
```

### Error: "ModuleNotFoundError: No module named 'matplotlib'"

**Solution:** Install visualization dependencies:
```bash
pip install matplotlib
```

For all visualization features:
```bash
pip install matplotlib scikit-learn tqdm
```

---

## Performance Issues

### Examples running very slowly

**Solutions:**

1. **Use GPU if available** (after fixing CUDA issues)
   - GPU is 10-100x faster than CPU for these workloads

2. **Reduce problem size for testing:**
   ```python
   # Quick test configuration
   config = EvolvingConfig(
       pop_size=4,           # Small population
       n_generations=3,      # Few generations
   )

   evaluator = PINNEvaluator(
       n_train_iters=50,     # Few iterations
   )

   problem = LinearBurgersProblem(
       n_collocation=500,    # Fewer points
   )
   ```

3. **Enable JIT compilation** (should be automatic)
   - First run will be slow (compilation)
   - Subsequent runs will be fast

---

## Numerical Issues

### Physics loss is NaN or very high

**Solutions:**

1. **Reduce learning rate:**
   ```python
   evaluator = PINNEvaluator(
       learning_rate=1e-4,  # Lower from 1e-3
   )
   ```

2. **Increase training iterations:**
   ```python
   evaluator = PINNEvaluator(
       n_train_iters=500,  # More training
   )
   ```

3. **Adjust loss weights:**
   ```python
   problem = LinearBurgersProblem(
       loss_weight_pde=1.0,
       loss_weight_ic=10.0,   # Emphasize IC more
       loss_weight_bc=1.0,
   )
   ```

4. **Try different activation functions:**
   - PINNs often work better with `tanh` than `relu`
   - This is handled automatically by evolution

---

## Verification

### Test if JAX is working correctly

```python
import jax
import jax.numpy as jnp

# Check JAX version
print(f"JAX version: {jax.__version__}")

# Check devices
print(f"Devices: {jax.devices()}")

# Simple computation
x = jnp.array([1.0, 2.0, 3.0])
print(f"Sum: {jnp.sum(x)}")

# Check if GPU is available
if jax.devices()[0].platform == 'gpu':
    print("GPU is available!")
else:
    print("Using CPU")
```

### Test if NASJAX imports work

```python
from nasjax.pinn import LinearBurgersProblem, PINNEvaluator, PINNNetwork
from nasjax.evolution import Evolving, EvolvingConfig
from nasjax.descriptors import MLPDescriptor

print("All imports successful!")
```

---

## Getting Help

If you're still experiencing issues:

1. **Check the error message carefully** - it often contains the solution
2. **Search for similar issues** on the JAX GitHub repository
3. **Verify your environment:**
   ```bash
   python --version
   pip list | grep jax
   nvidia-smi  # For GPU info
   ```
4. **Try the minimal example** in README.md first
5. **Open an issue** on GitHub with:
   - Full error message
   - Your environment info (Python version, JAX version, CUDA version)
   - Minimal code to reproduce the issue

---

## Quick Reference

### Recommended Configurations

**For quick testing (CPU or GPU):**
```python
config = EvolvingConfig(pop_size=4, n_generations=3)
evaluator = PINNEvaluator(n_train_iters=50)
problem = LinearBurgersProblem(n_collocation=500)
```

**For production (GPU recommended):**
```python
config = EvolvingConfig(pop_size=20, n_generations=30)
evaluator = PINNEvaluator(n_train_iters=500)
problem = LinearBurgersProblem(n_collocation=2000)
```

**For research (GPU required):**
```python
config = EvolvingConfig(pop_size=50, n_generations=100)
evaluator = PINNEvaluator(n_train_iters=1000)
problem = LinearBurgersProblem(n_collocation=5000)
```

---

## Platform-Specific Notes

### Linux
- CuDNN libraries are in `/usr/local/cuda/lib64` or `/usr/lib/x86_64-linux-gnu/`
- Use `ldconfig -p | grep cudnn` to check installation
- May need to set `LD_LIBRARY_PATH`

### macOS
- No NVIDIA GPU support (use CPU)
- Metal GPU support is experimental in JAX
- Always use `JAX_PLATFORMS=cpu`

### Windows
- CUDA setup can be more complex
- Recommend WSL2 for GPU support
- Or use CPU mode

---

**Last Updated:** 2026-01-02
