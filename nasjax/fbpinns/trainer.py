"""
FBPINN training loop (pure JAX / optax).

Unlike the equinox-based PINN trainer, parameters here are a plain pytree
dict.  ``jax.value_and_grad`` differentiates through all subdomain networks
simultaneously via the stacked-matmul batched forward pass.

Public API
----------
:func:`make_step`    One JIT-compiled gradient step.
:func:`train_fbpinn` Full training run returning :class:`TrainingResult`.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import numpy as np

from .config import FBPINNConfig
from .model import FBPINNModel
from .networks import init_params
from .losses import compute_loss
from nasjax.bpinn.pde.base import PDEProblem


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Result returned by :func:`train_fbpinn`.

    Attributes:
        model:          :class:`~nasjax.fbpinns.model.FBPINNModel` (static geometry).
        params:         Trained subdomain network parameter pytree.
        logs:           Per-epoch loss history.
        total_time_s:   Wall-clock training time in seconds.
        mean_epoch_s:   Mean per-epoch time.
        std_epoch_s:    Std dev of per-epoch times.
        config:         The :class:`~nasjax.fbpinns.config.FBPINNConfig` used.
        final_loss:     Final total loss value.
    """
    model: FBPINNModel
    params: Dict[str, Any]
    logs: Dict[str, List[float]]
    total_time_s: float
    mean_epoch_s: float
    std_epoch_s: float
    config: FBPINNConfig
    final_loss: float = 0.0


# ---------------------------------------------------------------------------
# Single optimiser step
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2, 3))
def make_step(
    params: Dict[str, Any],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_fn,
) -> Tuple[Dict, optax.OptState, jnp.ndarray]:
    """One gradient step.

    Args:
        params:     Current parameter pytree.
        opt_state:  Optimiser state.
        optimizer:  Optax gradient transformation (static).
        loss_fn:    Partial loss function ``fn(params) → scalar`` (static).

    Returns:
        ``(new_params, new_opt_state, loss)``.
    """
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_fbpinn(
    config: FBPINNConfig,
    problem: PDEProblem,
    key: Optional[jax.random.PRNGKey] = None,
    verbose: bool = True,
    print_every: int = 100,
) -> TrainingResult:
    """Train a FBPINN defined by *config* on *problem*.

    Args:
        config:       :class:`~nasjax.fbpinns.config.FBPINNConfig`.
        problem:      :class:`~nasjax.bpinn.pde.base.PDEProblem` instance.
        key:          JAX PRNG key.  If ``None``, ``config.seed`` is used.
        verbose:      Print progress every ``print_every`` epochs.
        print_every:  Epoch print interval.

    Returns:
        :class:`TrainingResult`.
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)
    init_key, data_key = jax.random.split(key)

    # Build model (geometry) and initial parameters
    fbpinn = FBPINNModel(config)
    params = init_params(config, init_key)

    # Build optimiser
    if config.optimiser_name == "adam":
        optimizer = optax.adam(config.learning_rate)
    elif config.optimiser_name == "adamw":
        optimizer = optax.adamw(config.learning_rate)
    else:
        optimizer = optax.sgd(config.learning_rate)

    opt_state = optimizer.init(params)

    # Global lambdas dict
    gl = config.global_lambdas
    global_lambdas = {
        "lambda_pde":   gl.lambda_pde,
        "lambda_ic":    gl.lambda_ic,
        "lambda_bc_l":  gl.lambda_bc_l,
        "lambda_bc_r":  gl.lambda_bc_r,
    }

    # Prepare data
    batch_data = problem.prepare_data(data_key, config)

    # Partial loss (params is the only dynamic argument)
    loss_fn = functools.partial(
        compute_loss,
        model=fbpinn,
        problem=problem,
        batch_data=batch_data,
        global_lambdas=global_lambdas,
    )

    # Training loop
    loss_history: List[float] = []
    epoch_times: List[float] = []
    t0 = time.perf_counter()

    for epoch in range(config.n_epochs):
        te = time.perf_counter()
        params, opt_state, loss = make_step(params, opt_state, optimizer, loss_fn)
        epoch_times.append(time.perf_counter() - te)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if verbose and (epoch == 0 or (epoch + 1) % print_every == 0):
            print(f"Epoch {epoch+1:5d}/{config.n_epochs}  loss = {loss_val:.4e}")

    total_time = time.perf_counter() - t0
    epoch_arr = np.array(epoch_times)

    return TrainingResult(
        model=fbpinn,
        params=params,
        logs={"loss": loss_history},
        total_time_s=total_time,
        mean_epoch_s=float(epoch_arr.mean()),
        std_epoch_s=float(epoch_arr.std()),
        config=config,
        final_loss=loss_history[-1],
    )
