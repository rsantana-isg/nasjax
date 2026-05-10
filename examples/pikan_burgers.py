"""
PIKAN example: Burgers' equation.

Demonstrates how to configure, instantiate, and train a Physics-Informed
Kolmogorov-Arnold Network (PIKAN) for Burgers' equation using the
nasjax.pikans sub-package.

Layer types available from jaxkan:
  chebyshev   — Chebyshev polynomial basis (D: degree)
  legendre    — Legendre polynomial basis (D: degree)
  fourier     — Fourier (sin/cos) basis (D: order)
  rbf         — Radial basis functions (D: n_basis, kernel: dict)
  sine        — Sine-based basis (D: n_basis)
  spline      — B-spline basis (k: order, G: grid size)

Per-layer configuration allows fully heterogeneous KAN architectures.

Input convention (jaxkan):
  Collocation points are (N, 2) with column 0 = t, column 1 = x.

Run from the repo root:
    python examples/pikan_burgers.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import json

from nasjax.pikans import (
    PIKANConfig,
    PIKANLayerSpec,
    GlobalLambdaConfig,
    make_layer_specs,
    PIKANNetwork,
    BurgersProblem,
    train_pikan,
    PIKANEvaluator,
)
from nasjax.bpinn.config import CollocationConfig


# ============================================================================
# 1.  Architecture: list-of-lists representation
# ============================================================================
# Each sublist: [n_out, layer_type, layer_params_dict]
#
# Layer-type-specific params:
#   chebyshev / legendre: D (degree), flavor ('default'/'modified'/'exact')
#   fourier:              D (order), smooth_init (bool)
#   rbf:                  D (n_basis), kernel (dict, e.g. {'type': 'gaussian'})
#   sine:                 D (n_basis)
#   spline:               k (order), G (grid size), grid_range (tuple)

layer_specs = [
    [8, "chebyshev", {"D": 5}],    # layer 1: 8 outputs, Chebyshev degree 5
    [8, "chebyshev", {"D": 5}],    # layer 2: 8 outputs
    [1, "chebyshev", {"D": 5}],    # output:  1 output (scalar u)
]

# Alternative: uniform stack using convenience constructor
# layer_specs = make_layer_specs(
#     hidden_widths=[8, 8],
#     output_dim=1,
#     layer_type='chebyshev',
#     layer_params={'D': 5},
# )

# Alternative: heterogeneous mixed architecture
# layer_specs = [
#     [8, "chebyshev", {"D": 5}],
#     [8, "fourier",   {"D": 4, "smooth_init": True}],
#     [1, "chebyshev", {"D": 5}],
# ]

# Alternative: Legendre-based PIKAN
# layer_specs = [
#     [8, "legendre", {"D": 5, "flavor": "exact"}],
#     [8, "legendre", {"D": 5, "flavor": "exact"}],
#     [1, "legendre", {"D": 5, "flavor": "exact"}],
# ]

# ============================================================================
# 2.  Full PIKANConfig
# ============================================================================
config = PIKANConfig(
    # --- Input dimension ---------------------------------------------------
    input_dim=2,              # (t, x) in jaxkan convention

    # --- Architecture ------------------------------------------------------
    layer_specs=layer_specs,

    # --- Training ----------------------------------------------------------
    n_epochs=1000,
    learning_rate=0.001,
    optimiser_name="adam",    # 'adam', 'adamw', 'sgd'

    # --- Collocation points ------------------------------------------------
    collocation=CollocationConfig(
        n_pde=4096,           # power of 2 for Sobol sampling
        n_ic=64,
        n_bc_l=64,
        n_bc_r=64,
    ),

    # --- Loss term weights -------------------------------------------------
    global_lambdas=GlobalLambdaConfig(
        lambda_pde=1.0,
        lambda_ic=100.0,
        lambda_bc_l=1.0,
        lambda_bc_r=1.0,
    ),

    # --- Seed ---------------------------------------------------------------
    seed=42,
)

print("Configuration summary")
print("=====================")
flat = config.to_flat_dict()
for k, v in flat.items():
    print(f"  {k}: {v}")
print()

# ============================================================================
# 3.  Validate configuration
# ============================================================================
errors = config.validation_errors()
if errors:
    print("Configuration errors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Configuration is valid.")
    print(f"  Full layer dims:  {config.full_layer_dims}")
    print()

# ============================================================================
# 4.  Instantiate PDE problem
# ============================================================================
problem = BurgersProblem()
print(f"PDE problem: {problem.name}")
print(f"  BC keys:  {problem.bc_keys}")
print()

# ============================================================================
# 5.  Train
# ============================================================================
print("Starting training …")
result = train_pikan(
    config,
    problem,
    verbose=True,
    print_every=100,
)

print()
print("Training complete.")
print(f"  Total time:       {result.total_time_s:.1f} s")
print(f"  Mean epoch time:  {result.mean_epoch_s*1000:.1f} ms")
print(f"  Final loss:       {result.final_loss:.4e}")

# ============================================================================
# 6.  Evaluation on the reference grid
# ============================================================================
key = jax.random.PRNGKey(config.seed + 1)
data_key, _ = jax.random.split(key)
batch_data = problem.prepare_data(data_key, config)

from nasjax.pikans import relative_l2_error
rel_l2 = relative_l2_error(result.model, batch_data)
print(f"  Relative L2 error: {rel_l2:.4e}")

# ============================================================================
# 7.  Visualization
# ============================================================================
import os

from nasjax.visualization import (
    predict_on_grid,
    plot_burgers_slices,
    plot_burgers_heatmaps,
    save_burgers_npz,
)

print()
print("Generating visualizations …")
x_vec, t_vec, u_exact = problem.load_reference_grid()

# PIKAN input convention: (N, 2) with column 0 = t, column 1 = x
def pikan_predict(x, t):
    inp = jnp.concatenate([t, x], axis=1)   # (N, 2): [:, 0]=t, [:, 1]=x
    return np.array(result.model(inp)).ravel()

u_pred_grid = predict_on_grid(pikan_predict, x_vec, t_vec)

os.makedirs("results", exist_ok=True)
save_burgers_npz(x_vec, t_vec, u_exact, u_pred_grid,
                 model_name="PIKAN", path="results/pikan_burgers.npz")
plot_burgers_slices(x_vec, t_vec, u_exact, u_pred_grid,
                    model_name="PIKAN",
                    save_path="results/pikan_burgers_slices.pdf")
plot_burgers_heatmaps(x_vec, t_vec, u_exact, u_pred_grid,
                      model_name="PIKAN",
                      save_path="results/pikan_burgers_heatmap.pdf")

# ============================================================================
# 9.  Config serialisation round-trip
# ============================================================================
config_json = json.dumps(config.to_dict(), indent=2, default=str)
config_restored = PIKANConfig.from_dict(json.loads(config_json))
assert config_restored.full_layer_dims == config.full_layer_dims
print()
print("Config round-trip serialisation: OK")

# ============================================================================
# 10. Using PIKANEvaluator for NAS / evolutionary search
# ============================================================================
print()
print("PIKANEvaluator demo …")
evaluator = PIKANEvaluator(
    problem=problem,
    base_config=config,
    metric="rel_l2",
    verbose=False,
)

# Evaluate a different (wider) architecture
key_eval = jax.random.PRNGKey(99)
fitness = evaluator.evaluate_layer_specs(
    layer_specs=[
        [16, "chebyshev", {"D": 5}],
        [16, "chebyshev", {"D": 5}],
        [1,  "chebyshev", {"D": 5}],
    ],
    key=key_eval,
    n_epochs=500,
)
print(f"Evaluator fitness (16×2 arch, 500 epochs): {fitness:.4e}")

# ============================================================================
# 9.  Heterogeneous DIY architecture (Tutorial 5 style)
# ============================================================================
print()
print("Heterogeneous KAN architecture demo …")
mixed_config = PIKANConfig(
    layer_specs=[
        [8, "fourier",    {"D": 4, "smooth_init": True}],
        [8, "sine",       {"D": 6}],
        [1, "chebyshev",  {"D": 5}],
    ],
    n_epochs=500,
    seed=7,
)
mixed_result = train_pikan(mixed_config, problem, verbose=True, print_every=100)
rel_l2_mixed = relative_l2_error(mixed_result.model, batch_data)
print(f"Mixed-arch relative L2 error: {rel_l2_mixed:.4e}")
