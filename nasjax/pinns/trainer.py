"""
PINN training loop.

Public API
----------
:func:`make_step`   One JIT-compiled gradient step.
:func:`train_pinn`  Full training run returning a :class:`TrainingResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from .config import PINNConfig
from .model import PINNModel
from .losses import compute_loss
from nasjax.bpinn.pde.base import PDEProblem


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Result returned by :func:`train_pinn`.

    Attributes:
        model:          The trained :class:`~nasjax.pinns.model.PINNModel`.
        logs:           Per-epoch loss history.
        total_time_s:   Wall-clock training time in seconds.
        mean_epoch_s:   Mean per-epoch time.
        std_epoch_s:    Std dev of per-epoch times.
        config:         The :class:`~nasjax.pinns.config.PINNConfig` used.
        final_loss:     Final total loss value.
    """
    model: PINNModel
    logs: Dict[str, List[float]]
    total_time_s: float
    mean_epoch_s: float
    std_epoch_s: float
    config: PINNConfig
    final_loss: float = 0.0


# ---------------------------------------------------------------------------
# Single optimiser step
# ---------------------------------------------------------------------------

@eqx.filter_jit
def make_step(
    model: PINNModel,
    optimiser: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch_data: Dict[str, jnp.ndarray],
    global_lambdas: Dict[str, float],
    nu: float,
) -> tuple:
    """One gradient step.

    Returns:
        ``(model, opt_state, loss)``
    """
    loss, grads = eqx.filter_value_and_grad(compute_loss)(
        model, batch_data, global_lambdas, nu
    )
    updates, new_opt_state = optimiser.update(
        eqx.filter(grads, eqx.is_array), opt_state
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_pinn(
    config: PINNConfig,
    problem: PDEProblem,
    key: Optional[jax.random.PRNGKey] = None,
    verbose: bool = True,
    print_every: int = 100,
) -> TrainingResult:
    """Train a PINN defined by *config* on *problem*.

    Args:
        config:       :class:`~nasjax.pinns.config.PINNConfig`.
        problem:      :class:`~nasjax.bpinn.pde.base.PDEProblem` instance.
        key:          JAX PRNG key.  If ``None``, ``config.seed`` is used.
        verbose:      Print progress every ``print_every`` epochs.
        print_every:  Epoch interval for progress printing.

    Returns:
        :class:`TrainingResult`.
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)
    model_key, data_key = jax.random.split(key)

    # Build model
    model = PINNModel(config, model_key)

    # Build optimiser
    if config.optimiser_name == "adam":
        optimiser = optax.adam(config.learning_rate)
    elif config.optimiser_name == "adamw":
        optimiser = optax.adamw(config.learning_rate)
    else:
        optimiser = optax.sgd(config.learning_rate)

    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

    # Global lambdas dict
    gl = config.global_lambdas
    global_lambdas = {
        "lambda_pde":   gl.lambda_pde,
        "lambda_ic":    gl.lambda_ic,
        "lambda_bc_l":  gl.lambda_bc_l,
        "lambda_bc_r":  gl.lambda_bc_r,
    }

    # Kinematic viscosity from problem (BurgersProblem exposes .nu)
    nu = getattr(problem, "nu", 0.01 / float(np.pi))

    # Prepare data
    batch_data = problem.prepare_data(data_key, config)

    # Training loop
    loss_history: List[float] = []
    epoch_times: List[float] = []
    t0 = time.perf_counter()

    for epoch in range(config.n_epochs):
        te = time.perf_counter()
        model, opt_state, loss = make_step(
            model, optimiser, opt_state, batch_data, global_lambdas, nu
        )
        epoch_times.append(time.perf_counter() - te)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if verbose and (epoch == 0 or (epoch + 1) % print_every == 0):
            print(f"Epoch {epoch+1:5d}/{config.n_epochs}  loss = {loss_val:.4e}")

    total_time = time.perf_counter() - t0
    epoch_arr = np.array(epoch_times)

    return TrainingResult(
        model=model,
        logs={"loss": loss_history},
        total_time_s=total_time,
        mean_epoch_s=float(epoch_arr.mean()),
        std_epoch_s=float(epoch_arr.std()),
        config=config,
        final_loss=loss_history[-1],
    )
