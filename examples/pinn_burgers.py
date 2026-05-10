"""
PINN example: Burgers' equation.

Demonstrates how to configure, instantiate, and train a standard Physics-
Informed Neural Network (PINN) for Burgers' equation using the
nasjax.pinns sub-package.

Supports four network types:
  FCN         — standard tanh MLP
  AdaptiveFCN — MLP with per-neuron trainable activation scales
  SIREN       — MLP with sinusoidal activations and SIREN init
  FourierFCN  — MLP preceded by random Fourier feature embedding

Run from the repo root:
    python examples/pinn_burgers.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import json

from nasjax.pinns import (
    PINNConfig,
    PINNLayerSpec,
    FourierInputConfig,
    GlobalLambdaConfig,
    make_layer_specs,
    BurgersProblem,
    train_pinn,
    PINNEvaluator,
)
from nasjax.bpinn.config import CollocationConfig


# ============================================================================
# 1.  Architecture: list-of-lists representation
# ============================================================================
# Each sublist: [n_neurons, activation]
# Activation options: 'tanh', 'relu', 'gelu', 'silu', 'sin', 'sigmoid',
#                     'softplus', None (linear, for output layer)

layer_specs = [
    [50, "tanh"],   # hidden 1
    [50, "tanh"],   # hidden 2
    [50, "tanh"],   # hidden 3
    [1,  None ],    # output (linear)
]

# Alternative: convenience constructor for uniform stacks
# layer_specs = make_layer_specs(hidden_widths=[50, 50, 50], output_dim=1)

# ============================================================================
# 2.  Full PINNConfig
# ============================================================================
config = PINNConfig(
    # --- Input dimension ---------------------------------------------------
    input_dim=2,                     # (x, t) for Burgers

    # --- Architecture (network_type selects the MLP variant) ---------------
    layer_specs=layer_specs,
    network_type="FCN",              # 'FCN', 'AdaptiveFCN', 'SIREN', 'FourierFCN'

    # --- Random Fourier features (only active when network_type='FourierFCN')
    fourier_input=FourierInputConfig(
        enabled=False,               # set True to enable
        n_features=64,
        mu=0.0,
        sd=1.0,
    ),

    # --- Training ----------------------------------------------------------
    n_epochs=5000,
    learning_rate=0.001,
    optimiser_name="adam",           # 'adam', 'adamw', 'sgd'

    # --- Collocation points ------------------------------------------------
    collocation=CollocationConfig(
        n_pde=10_000,
        n_ic=400,
        n_bc_l=400,
        n_bc_r=400,
    ),

    # --- Loss term weights -------------------------------------------------
    global_lambdas=GlobalLambdaConfig(
        lambda_pde=1.0,
        lambda_ic=100.0,   # IC weighted more heavily — standard practice
        lambda_bc_l=1.0,
        lambda_bc_r=1.0,
    ),

    # --- Seed ---------------------------------------------------------------
    seed=1234,
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
    print(f"  Effective input dim: {config.effective_input_dim}")
    print(f"  Full layer dims:     {config.full_layer_dims}")
    print()

# ============================================================================
# 4.  Instantiate the PDE problem
# ============================================================================
problem = BurgersProblem()
print(f"PDE problem: {problem.name}")
print(f"  BC keys: {problem.bc_keys}")
print()

# ============================================================================
# 5.  Train
# ============================================================================
print("Starting training …")
result = train_pinn(
    config,
    problem,
    verbose=True,
    print_every=500,
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

u_pred_flat = result.model.out_u_only(batch_data["x_data"], batch_data["t_data"]).ravel()
u_true = batch_data["u_data"].ravel()
rel_l2 = float(jnp.linalg.norm(u_pred_flat - u_true) / jnp.linalg.norm(u_true))
print(f"  Relative L2 error: {rel_l2:.4e}")

# ============================================================================
# 7.  Visualization
# ============================================================================
from nasjax.visualization import (
    predict_on_grid,
    plot_burgers_slices,
    plot_burgers_heatmaps,
    save_burgers_npz,
)

print()
print("Generating visualizations …")
x_vec, t_vec, u_exact = problem.load_reference_grid()

# Prediction closure: takes (N,1) JAX arrays, returns (N,) numpy
pinn_predict = lambda x, t: np.array(result.model.out_u_only(x, t)).ravel()
u_pred_grid = predict_on_grid(pinn_predict, x_vec, t_vec)

os.makedirs("results", exist_ok=True)
save_burgers_npz(x_vec, t_vec, u_exact, u_pred_grid,
                 model_name="PINN", path="results/pinn_burgers.npz")
plot_burgers_slices(x_vec, t_vec, u_exact, u_pred_grid,
                    model_name="PINN",
                    save_path="results/pinn_burgers_slices.pdf")
plot_burgers_heatmaps(x_vec, t_vec, u_exact, u_pred_grid,
                      model_name="PINN",
                      save_path="results/pinn_burgers_heatmap.pdf")

# ============================================================================
# 9.  Config serialisation round-trip
# ============================================================================
config_json = json.dumps(config.to_dict(), indent=2, default=str)
config_restored = PINNConfig.from_dict(json.loads(config_json))
assert config_restored.full_layer_dims == config.full_layer_dims
print()
print("Config round-trip serialisation: OK")

# ============================================================================
# 10. Using PINNEvaluator for NAS / evolutionary search
# ============================================================================
print()
print("PINNEvaluator demo …")
evaluator = PINNEvaluator(
    problem=problem,
    base_config=config,
    metric="rel_l2",
    verbose=False,
)

# Evaluate a different architecture
key_eval = jax.random.PRNGKey(42)
fitness = evaluator.evaluate_layer_specs(
    layer_specs=[
        [64, "tanh"],
        [64, "tanh"],
        [1,  None],
    ],
    key=key_eval,
    n_epochs=1000,
)
print(f"Evaluator fitness (64×2 arch, 1000 epochs): {fitness:.4e}")

# ============================================================================
# 9.  SIREN variant (alternative)
# ============================================================================
print()
print("SIREN variant demo …")
siren_config = PINNConfig(
    layer_specs=make_layer_specs([50, 50, 50]),
    network_type="SIREN",
    n_epochs=2000,
    learning_rate=0.0001,  # SIREN typically needs a smaller lr
    collocation=CollocationConfig(n_pde=5000, n_ic=200, n_bc_l=200, n_bc_r=200),
    seed=42,
)
siren_result = train_pinn(siren_config, problem, verbose=True, print_every=500)
u_pred_siren = siren_result.model.out_u_only(batch_data["x_data"], batch_data["t_data"]).ravel()
rel_l2_siren = float(jnp.linalg.norm(u_pred_siren - u_true) / jnp.linalg.norm(u_true))
print(f"SIREN relative L2 error: {rel_l2_siren:.4e}")
