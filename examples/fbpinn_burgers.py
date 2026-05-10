"""
FBPINN example: Burgers' equation.

Demonstrates how to configure, instantiate, and train a Finite Basis Physics-
Informed Neural Network (FBPINN) for Burgers' equation using the
nasjax.fbpinns sub-package.

Key FBPINN parameters
---------------------
n_subdomains_x, n_subdomains_t — grid of overlapping subdomains
overlap                         — fractional overlap between adjacent subdomains
window_fn                       — 'cosine' or 'sigmoid' partition-of-unity
layer_specs                     — shared network architecture per subdomain

The domain [-1,1] × [0,1] is partitioned into a grid of n_subdomains_x ×
n_subdomains_t overlapping rectangles.  Each rectangle has its own small FCN.
The global solution is the partition-of-unity weighted sum of all local
predictions.

Run from the repo root:
    python examples/fbpinn_burgers.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import json

from nasjax.fbpinns import (
    FBPINNConfig,
    FBPINNLayerSpec,
    DecompositionConfig,
    GlobalLambdaConfig,
    make_layer_specs,
    FBPINNModel,
    BurgersProblem,
    train_fbpinn,
    FBPINNEvaluator,
    relative_l2_error,
)
from nasjax.bpinn.config import CollocationConfig


# ============================================================================
# 1.  Architecture: list-of-lists representation (same as PINN)
# ============================================================================
# Each sublist: [n_neurons, activation]
# All subdomains share the same network architecture.

layer_specs = [
    [32, "tanh"],  # hidden 1
    [32, "tanh"],  # hidden 2
    [1,  None],    # output (linear)
]

# ============================================================================
# 2.  Domain decomposition
# ============================================================================
decomposition = DecompositionConfig(
    n_subdomains_x=4,     # subdomains along x ∈ [-1, 1]
    n_subdomains_t=4,     # subdomains along t ∈ [0, 1]
    overlap=0.5,          # 50% overlap between adjacent subdomains
    x_min=-1.0,
    x_max=1.0,
    t_min=0.0,
    t_max=1.0,
    window_fn="cosine",   # 'cosine' or 'sigmoid'
)

# ============================================================================
# 3.  Full FBPINNConfig
# ============================================================================
config = FBPINNConfig(
    # --- Input dimension ---------------------------------------------------
    input_dim=2,               # (x, t) for Burgers

    # --- Subdomain network architecture ------------------------------------
    layer_specs=layer_specs,
    network_type="FCN",        # 'FCN', 'AdaptiveFCN', 'SIREN'

    # --- Domain decomposition ----------------------------------------------
    decomposition=decomposition,

    # --- Training ----------------------------------------------------------
    n_epochs=2000,
    learning_rate=0.001,
    optimiser_name="adam",

    # --- Collocation points ------------------------------------------------
    collocation=CollocationConfig(
        n_pde=5000,
        n_ic=200,
        n_bc_l=200,
        n_bc_r=200,
    ),

    # --- Loss term weights -------------------------------------------------
    global_lambdas=GlobalLambdaConfig(
        lambda_pde=1.0,
        lambda_ic=100.0,
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
print(f"  Total subdomains: {config.n_subdomains}")
print(f"  Full layer dims:  {config.full_layer_dims}")
print()

# ============================================================================
# 4.  Validate configuration
# ============================================================================
errors = config.validation_errors()
if errors:
    print("Configuration errors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Configuration is valid.")
    print()

# ============================================================================
# 5.  Instantiate PDE problem
# ============================================================================
problem = BurgersProblem()
print(f"PDE problem: {problem.name}")
print(f"  BC keys:  {problem.bc_keys}")
print()

# ============================================================================
# 6.  Train
# ============================================================================
print("Starting training …")
result = train_fbpinn(
    config,
    problem,
    verbose=True,
    print_every=200,
)

print()
print("Training complete.")
print(f"  Total subdomains: {result.model.n_subdomains}")
print(f"  Total time:       {result.total_time_s:.1f} s")
print(f"  Mean epoch time:  {result.mean_epoch_s*1000:.1f} ms")
print(f"  Final loss:       {result.final_loss:.4e}")

# ============================================================================
# 7.  Evaluation on the reference grid
# ============================================================================
key = jax.random.PRNGKey(config.seed + 1)
data_key, _ = jax.random.split(key)
batch_data = problem.prepare_data(data_key, config)

rel_l2 = relative_l2_error(result.model, result.params, batch_data)
print(f"  Relative L2 error: {rel_l2:.4e}")

# ============================================================================
# 8.  Visualization
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

# FBPINN prediction: model.out_u_only(params, x, t)
fbpinn_predict = lambda x, t: np.array(
    result.model.out_u_only(result.params, x, t)
).ravel()
u_pred_grid = predict_on_grid(fbpinn_predict, x_vec, t_vec)

os.makedirs("results", exist_ok=True)
save_burgers_npz(x_vec, t_vec, u_exact, u_pred_grid,
                 model_name="FBPINN", path="results/fbpinn_burgers.npz")
plot_burgers_slices(x_vec, t_vec, u_exact, u_pred_grid,
                    model_name="FBPINN",
                    save_path="results/fbpinn_burgers_slices.pdf")
plot_burgers_heatmaps(x_vec, t_vec, u_exact, u_pred_grid,
                      model_name="FBPINN",
                      save_path="results/fbpinn_burgers_heatmap.pdf")

# ============================================================================
# 10. Config serialisation round-trip
# ============================================================================
config_json = json.dumps(config.to_dict(), indent=2, default=str)
config_restored = FBPINNConfig.from_dict(json.loads(config_json))
assert config_restored.full_layer_dims == config.full_layer_dims
assert config_restored.n_subdomains == config.n_subdomains
print()
print("Config round-trip serialisation: OK")

# ============================================================================
# 11. Using FBPINNEvaluator for NAS / evolutionary search
# ============================================================================
print()
print("FBPINNEvaluator demo …")
evaluator = FBPINNEvaluator(
    problem=problem,
    base_config=config,
    metric="rel_l2",
    verbose=False,
)

# Evaluate a finer decomposition (more subdomains)
key_eval = jax.random.PRNGKey(77)
config_fine = FBPINNConfig(
    layer_specs=make_layer_specs([16, 16]),
    decomposition=DecompositionConfig(n_subdomains_x=6, n_subdomains_t=6, overlap=0.5),
    n_epochs=1000,
    seed=77,
)
fitness_fine = evaluator.evaluate(config_fine, key=key_eval)
print(f"Evaluator fitness (6×6 subdomains, 16×2 net, 1000 epochs): {fitness_fine:.4e}")

# ============================================================================
# 10. Effect of overlap parameter
# ============================================================================
print()
print("Effect of overlap parameter …")
for overlap in [0.25, 0.5, 0.75]:
    cfg_ov = FBPINNConfig(
        layer_specs=layer_specs,
        decomposition=DecompositionConfig(
            n_subdomains_x=4, n_subdomains_t=4, overlap=overlap,
        ),
        n_epochs=500,
        seed=0,
    )
    f_ov = evaluator.evaluate(cfg_ov, key=jax.random.PRNGKey(0))
    print(f"  overlap={overlap:.2f} → rel_L2 = {f_ov:.4e}")
