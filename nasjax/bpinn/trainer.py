"""
BPINN training loop.

Public API
----------
:func:`make_step`
    One JIT-compiled gradient step (used by both the training loop and the
    evolutionary evaluator).

:func:`train_bpinn`
    Full training run.  Accepts a :class:`~nasjax.bpinn.config.BPINNConfig`
    and a :class:`~nasjax.bpinn.pde.base.PDEProblem`, returns a
    :class:`TrainingResult`.

JAX / Equinox design
---------------------
- ``make_step`` uses ``eqx.filter_value_and_grad`` so that *only* JAX-array
  leaves in the model are differentiated.  Static fields (prior objects,
  activation functions, etc.) are safely ignored.
- ``local_lambdas`` and ``global_lambdas`` are captured as Python-level
  closures / arguments to ``make_step``.  They must be plain dicts of
  JAX arrays / floats — not Equinox modules.
- ``local_lambda_tech`` is a Python string; it controls a Python-level
  if/else inside :func:`~nasjax.bpinn.losses.single_sample_elbo_terms`,
  which is resolved at JIT-compilation time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from .config import BPINNConfig
from .model import BPINNModel
from .losses import compute_loss, init_global_lambdas, init_local_lambdas, init_sigma_dict
from .pde.base import PDEProblem


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Result returned by :func:`train_bpinn`.

    Attributes:
        model:          The trained :class:`~nasjax.bpinn.model.BPINNModel`.
        logs:           Training history (lists of per-epoch values).
        total_time_s:   Wall-clock training time in seconds.
        mean_epoch_s:   Mean per-epoch time.
        std_epoch_s:    Std dev of per-epoch times.
        config:         The :class:`~nasjax.bpinn.config.BPINNConfig` used.
        final_neg_elbo: Final negative ELBO value.
    """
    model: BPINNModel
    logs: Dict[str, List[float]]
    total_time_s: float
    mean_epoch_s: float
    std_epoch_s: float
    config: BPINNConfig
    final_neg_elbo: float = 0.0


# ---------------------------------------------------------------------------
# Single optimiser step (JIT-compiled)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def make_step(
    model: BPINNModel,
    optimiser: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch_data: Dict[str, jnp.ndarray],
    local_lambdas: Dict[str, jnp.ndarray],
    global_lambdas: Dict[str, float],
    local_lambda_tech: str,
    sigma: Dict[str, float],
    rba_gamma: float,
    rba_lr: float,
    rba_eps: float,
    fwd_key: jax.random.PRNGKey,
    k_samples: int,
    problem: PDEProblem,
) -> Tuple[BPINNModel, optax.OptState, Dict, Dict]:
    """One JIT-compiled training step.

    Computes the negative ELBO, back-propagates through model parameters
    (μ, ρ for all layers), and updates the optimiser state.

    Args:
        model:               Current BPINN model.
        optimiser:           Optax optimiser.
        opt_state:           Current optimiser state.
        batch_data:          Collocation data dict.
        local_lambdas:       Per-point lambda arrays.
        global_lambdas:      Global ELBO weights.
        local_lambda_tech:   ``'None'`` | ``'Residual_RBA'`` | ``'Likelihood_RBA'``.
        sigma:               Likelihood σ values.
        rba_gamma, rba_lr, rba_eps: RBA hyper-parameters.
        fwd_key:             JAX PRNG key for this step.
        k_samples:           Number of MC weight samples.
        problem:             PDE problem instance (for residuals).

    Returns:
        ``(new_model, new_opt_state, log_loss_dict, new_local_lambdas)``
    """
    (loss, (log_loss, new_local_lambdas)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(
        model, problem, batch_data,
        local_lambdas, global_lambdas,
        local_lambda_tech, sigma,
        rba_gamma, rba_lr, rba_eps,
        fwd_key, k_samples,
    )

    updates, new_opt_state = optimiser.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, log_loss, new_local_lambdas


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def _build_optimiser(name: str, lr: float) -> optax.GradientTransformation:
    """Return an Optax optimiser by name."""
    if name == "adam":
        return optax.adam(lr)
    if name == "sgd":
        return optax.sgd(lr)
    if name == "rmsprop":
        return optax.rmsprop(lr)
    raise ValueError(f"Unknown optimiser '{name}'. Choose from 'adam', 'sgd', 'rmsprop'.")


# ---------------------------------------------------------------------------
# Log initialisation / update helpers
# ---------------------------------------------------------------------------

def _init_logs(bc_keys: List[str]) -> Dict[str, List[float]]:
    base = {
        "negELBO":                       [],
        "log_lik_pde":                   [],
        "log_lik_ic":                    [],
        "total_log_prior":               [],
        "total_log_var_posterior":       [],
        "unaltered_log_lik_pde":         [],
        "unaltered_log_lik_ic":          [],
        "lambda_l_pde_mean":             [],
        "lambda_l_ic_mean":              [],
    }
    for k in bc_keys:
        base[f"log_lik_{k}"] = []
        base[f"unaltered_log_lik_{k}"] = []
        base[f"lambda_l_{k}_mean"] = []
    return base


def _update_logs(
    logs: Dict[str, List[float]],
    loss_log: Dict[str, jnp.ndarray],
    local_lambdas: Dict[str, jnp.ndarray],
    bc_keys: List[str],
) -> None:
    """Append current-epoch values to the logs dict (in-place)."""
    logs["negELBO"].append(float(loss_log["negative_ELBO"]))
    logs["log_lik_pde"].append(float(loss_log["log_lik_pde"]))
    logs["log_lik_ic"].append(float(loss_log["log_lik_ic"]))
    logs["total_log_prior"].append(float(loss_log["total_log_prior"]))
    logs["total_log_var_posterior"].append(float(loss_log["total_log_var_posterior"]))
    logs["unaltered_log_lik_pde"].append(float(loss_log["unaltered_log_lik_pde"]))
    logs["unaltered_log_lik_ic"].append(float(loss_log["unaltered_log_lik_ic"]))
    logs["lambda_l_pde_mean"].append(float(jnp.mean(local_lambdas["lambda_pde"])))
    logs["lambda_l_ic_mean"].append(float(jnp.mean(local_lambdas["lambda_ic"])))
    for k in bc_keys:
        logs[f"log_lik_{k}"].append(float(loss_log[f"log_lik_{k}"]))
        logs[f"unaltered_log_lik_{k}"].append(float(loss_log[f"unaltered_log_lik_{k}"]))
        logs[f"lambda_l_{k}_mean"].append(float(jnp.mean(local_lambdas[f"lambda_{k}"])))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_bpinn(
    config: BPINNConfig,
    problem: PDEProblem,
    key: Optional[jax.random.PRNGKey] = None,
    verbose: bool = True,
    print_every: int = 25,
) -> TrainingResult:
    """Train a Bayesian PINN specified by *config* on *problem*.

    This is the **single entry-point** function:  all BPINN hyperparameters
    are contained in *config*; all equation-specific logic is in *problem*.

    Steps
    -----
    1. [Python] Validate config.
    2. [Python] Initialise PRNG key.
    3. [Python] Build model (:class:`~nasjax.bpinn.model.BPINNModel`).
    4. [Python] Build optimiser.
    5. [Python] Prepare collocation data (calls ``problem.prepare_data``).
    6. [Python] Initialise local and global lambdas.
    7. [JAX loop] Run training epochs via :func:`make_step`.
    8. [Python] Collect and return :class:`TrainingResult`.

    Args:
        config:       Full BPINN specification.
        problem:      The PDE problem (e.g. :class:`~nasjax.bpinn.pde.BurgersProblem`).
        key:          Optional JAX PRNG key.  If ``None``, ``config.seed`` is used.
        verbose:      Print progress every ``print_every`` epochs.
        print_every:  Epoch interval for console output.

    Returns:
        :class:`TrainingResult` with trained model, training logs, and timing.

    Raises:
        ValueError: If ``config.validate()`` fails.
    """
    # --- 1. Validate ---------------------------------------------------------
    errors = config.validation_errors()
    if errors:
        raise ValueError("Invalid BPINNConfig:\n  " + "\n  ".join(errors))

    # --- 2. PRNG keys --------------------------------------------------------
    if key is None:
        key = jax.random.PRNGKey(config.seed)
    key, init_key, data_key = jax.random.split(key, 3)

    # --- 3. Build model ------------------------------------------------------
    model = BPINNModel(config, init_key)

    # --- 4. Build optimiser --------------------------------------------------
    optimiser = _build_optimiser(config.optimiser_name, config.learning_rate)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

    # --- 5. Prepare data -----------------------------------------------------
    batch_data = problem.prepare_data(data_key, config)

    # --- 6. Initialise lambdas -----------------------------------------------
    bc_keys = problem.bc_keys
    local_lambdas  = init_local_lambdas(batch_data, bc_keys)
    global_lambdas = init_global_lambdas(config.global_lambdas, bc_keys)
    sigma          = init_sigma_dict(config.sigma)

    rba_gamma = config.rba.gamma
    rba_lr    = config.rba.lr
    rba_eps   = config.rba.eps
    k_samples = config.k_samples_per_epoch
    local_lambda_tech = config.local_lambda_tech

    # --- 7. Training loop ----------------------------------------------------
    logs = _init_logs(bc_keys)
    epoch_times: List[float] = []

    t_start = time.time()

    for epoch in range(config.n_epochs):
        t_epoch_start = time.time()

        key, fwd_key = jax.random.split(key)

        model, opt_state, loss_log, local_lambdas = make_step(
            model, optimiser, opt_state,
            batch_data, local_lambdas, global_lambdas,
            local_lambda_tech, sigma,
            rba_gamma, rba_lr, rba_eps,
            fwd_key, k_samples, problem,
        )

        t_epoch_end = time.time()
        if epoch > 0:
            epoch_times.append(t_epoch_end - t_epoch_start)

        _update_logs(logs, loss_log, local_lambdas, bc_keys)

        if verbose and epoch % print_every == 0:
            neg_elbo = float(loss_log["negative_ELBO"])
            print(f"Epoch {epoch:5d} | -ELBO: {neg_elbo:.4e}")

    t_end = time.time()
    total_time = t_end - t_start

    if epoch_times:
        mean_t = sum(epoch_times) / len(epoch_times)
        std_t  = (sum((t - mean_t) ** 2 for t in epoch_times) / len(epoch_times)) ** 0.5
    else:
        mean_t = std_t = 0.0

    # --- 8. Return result ----------------------------------------------------
    return TrainingResult(
        model=model,
        logs=logs,
        total_time_s=total_time,
        mean_epoch_s=mean_t,
        std_epoch_s=std_t,
        config=config,
        final_neg_elbo=float(logs["negELBO"][-1]) if logs["negELBO"] else float("inf"),
    )
