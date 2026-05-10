"""
BPINN example: Burgers' equation.

Demonstrates how to configure, instantiate, and train a Bayesian PINN for
Burgers' equation using the nasjax.bpinn sub-package.

All aspects of the BPINN that appear in
    docs/BPINN_docs/parameters_which_affect_BPINNs.tex
are shown explicitly so they can be used as starting points for experiments.

Run from the repo root:
    python examples/bpinn_burgers.py
"""

import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np

from nasjax.bpinn import (
    BPINNConfig,
    BPINNLayerSpec,
    CollocationConfig,
    FourierEmbeddingConfig,
    GlobalLambdaConfig,
    RBAConfig,
    SigmaConfig,
    make_layer_specs,
    BurgersProblem,
    train_bpinn,
)


# ============================================================================
# 1.  Architecture: list-of-lists representation
# ============================================================================
# Each sublist: [n_neurons, activation, prior_type, prior_params_dict]
#
# Prior types:
#   'Gaussian' → {'mean': float, 'sigma': float}
#   'Laplace'  → {'mean': float, 'beta': float}
#   'SnS'      → {'pi': float, 'sigma_spike': float, 'sigma_slab': float}

layer_specs = [
    [50, "tanh", "Gaussian", {"mean": 0.0, "sigma": 1.0}],   # hidden 1
    [50, "tanh", "Gaussian", {"mean": 0.0, "sigma": 1.0}],   # hidden 2
    [25, "relu", "Gaussian", {"mean": 0.0, "sigma": 1.0}],   # hidden 3    
    [1,  None,  "Gaussian", {"mean": 0.0, "sigma": 1.0}],   # output (linear)
]

# Alternative: convenience constructor for uniform stacks
# layer_specs = make_layer_specs(
#     hidden_widths=[50, 50],
#     output_dim=1,
#     activation='tanh',
#     prior_type='Gaussian',
#     prior_params={'mean': 0.0, 'sigma': 1.0},
# )

# Alternative with Spike-and-Slab prior:
# import math
# layer_specs = make_layer_specs(
#     hidden_widths=[50, 50],
#     output_dim=1,
#     activation='tanh',
#     prior_type='SnS',
#     prior_params={'pi': 0.5, 'sigma_spike': math.exp(-6), 'sigma_slab': math.exp(0)},
# )

# ============================================================================
# 2.  Full BPINNConfig
# ============================================================================
config = BPINNConfig(
    # --- Input dimension ---------------------------------------------------
    input_dim=2,               # (x, t) for Burgers

    # --- Architecture (Section: Width and depth) ---------------------------
    layer_specs=layer_specs,
    #enhanced_mlp=True,         # Modified MLP with U/V residual connections
    enhanced_mlp=False,         # Modified MLP with U/V residual connections    

    # --- Hard-coded BCs (Section: Hard-coded BCs) --------------------------
    # Fourier embedding enforces periodic BCs automatically.
    # Set enabled=True only for periodic problems (e.g. Allen-Cahn).
    fourier_embedding=FourierEmbeddingConfig(enabled=False, L=2.0, M=10),

    # --- Training (Section: Number of epochs, Optimisation) ----------------
    n_epochs=500,
    learning_rate=0.001,
    optimiser_name="adam",

    # --- MC samples (Section: Number of MC samples per epoch) --------------
    k_samples_per_epoch=1,     # K=1 is fast; K=3 reduces posterior variance

    # --- Collocation points (Section: Number of collocation points) --------
    collocation=CollocationConfig(
        n_pde=10_000,
        n_ic=400,
        n_bc_l=400,
        n_bc_r=400,
    ),

    # --- Likelihood sigmas (Section: Likelihood sigmas) --------------------
    sigma=SigmaConfig(
        sigma_pde=0.001,
        sigma_ic=0.005,
        sigma_bc_l=0.005,
        sigma_bc_r=0.005,
    ),

    # --- Local lambdas (Section: Local lambdas) ----------------------------
    # Options: 'None', 'Residual_RBA', 'Likelihood_RBA'
    local_lambda_tech="None",

    # --- Global lambdas (Section: Global lambdas) --------------------------
    global_lambdas=GlobalLambdaConfig(
        lambda_lik_pde=1.0,
        lambda_lik_ic=100.0,   # IC weighted more heavily
        lambda_lik_bc_l=1.0,
        lambda_lik_bc_r=1.0,
        lambda_prior=1.0,
        lambda_var_posterior=1.0,
    ),

    # --- RBA configuration -------------------------------------------------
    rba=RBAConfig(gamma=0.999, lr=0.001, eps=1e-12),

    # --- Resampling (Section: Resampling strategies) -----------------------
    resample_each_epoch=False, # True conflicts with RBA

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
# BurgersProblem loads the reference solution from burgers_shock.mat.
# The data_path is resolved relative to JAX_BPINN-main/ by default.
problem = BurgersProblem()
print(f"PDE problem: {problem.name}")
print(f"  BC keys: {problem.bc_keys}")
print()

# ============================================================================
# 5.  Train
# ============================================================================
print("Starting training …")
result = train_bpinn(
    config,
    problem,
    verbose=True,
    print_every=50,
)

print()
print("Training complete.")
print(f"  Total time:       {result.total_time_s:.1f} s")
print(f"  Mean epoch time:  {result.mean_epoch_s*1000:.1f} ms")
print(f"  Final -ELBO:      {result.final_neg_elbo:.4e}")

# ============================================================================
# 6.  Posterior predictive samples on the reference grid
# ============================================================================
key = jax.random.PRNGKey(config.seed + 1)
data_key, sample_key = jax.random.split(key)
batch_data = problem.prepare_data(data_key, config)

n_samples = 50
sample_keys = jax.random.split(sample_key, n_samples)

def single_pred(sk):
    params, _, _ = result.model.sample_params(sk, sample=True)
    return result.model.out_u_only(
        batch_data["x_data"], batch_data["t_data"], params
    ).ravel()

preds = jax.vmap(single_pred)(sample_keys)          # (n_samples, N)
u_mean = jnp.mean(preds, axis=0)
u_std  = jnp.std(preds,  axis=0)
u_true = batch_data["u_data"].ravel()

rel_l2 = float(jnp.linalg.norm(u_mean - u_true) / jnp.linalg.norm(u_true))
print(f"  Relative L2 error: {rel_l2:.4e}")
print(f"  Mean posterior std: {float(jnp.mean(u_std)):.4e}")

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

# Prediction closure: MC posterior mean over n_samples weight samples
_n_vis_samples = 50
_vis_keys = jax.random.split(sample_key, _n_vis_samples)

def bpinn_predict(x, t):
    def one_sample(sk):
        params, _, _ = result.model.sample_params(sk, sample=True)
        return result.model.out_u_only(x, t, params).ravel()
    preds = jax.vmap(one_sample)(_vis_keys)   # (n_samples, N)
    return np.array(jnp.mean(preds, axis=0))

u_pred_grid = predict_on_grid(bpinn_predict, x_vec, t_vec)

os.makedirs("results", exist_ok=True)
save_burgers_npz(x_vec, t_vec, u_exact, u_pred_grid,
                 model_name="BPINN", path="results/bpinn_burgers.npz")
plot_burgers_slices(x_vec, t_vec, u_exact, u_pred_grid,
                    model_name="BPINN",
                    save_path="results/bpinn_burgers_slices.pdf")
plot_burgers_heatmaps(x_vec, t_vec, u_exact, u_pred_grid,
                      model_name="BPINN",
                      save_path="results/bpinn_burgers_heatmap.pdf")

# ============================================================================
# 9.  Serialise / deserialise config (for reproducibility / search)
# ============================================================================
import json

config_json = json.dumps(config.to_dict(), indent=2, default=str)
config_restored = BPINNConfig.from_dict(json.loads(config_json))
assert config_restored.full_layer_dims == config.full_layer_dims
print()
print("Config round-trip serialisation: OK")

# ============================================================================
# 10. Using BPINNEvaluator for evolutionary search
# ============================================================================
from nasjax.bpinn import BPINNEvaluator

evaluator = BPINNEvaluator(
    problem=problem,
    base_config=config,
    metric="neg_elbo",
    verbose=False,
)

# Example: evaluate a slightly different architecture
key_eval = jax.random.PRNGKey(42)
fitness = evaluator.evaluate_layer_specs(
    layer_specs=[
        [64, "tanh", "Gaussian", {"mean": 0.0, "sigma": 1.0}],
        [64, "tanh", "Gaussian", {"mean": 0.0, "sigma": 1.0}],
        [1,  None,  "Gaussian", {"mean": 0.0, "sigma": 1.0}],
    ],
    key=key_eval,
    n_epochs=100,
)
print(f"Evaluator fitness (64×2 arch, 100 epochs): {fitness:.4e}")
