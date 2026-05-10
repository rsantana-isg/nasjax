"""
PIKAN training loop using flax.nnx.

The flax.nnx training paradigm differs from equinox in two ways:
1. Gradient computation uses ``nnx.value_and_grad`` instead of
   ``eqx.filter_value_and_grad``.
2. The optimizer wraps the model via ``nnx.Optimizer`` and updates in-place.

Public API
----------
:func:`train_pikan`  Full training run returning :class:`TrainingResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

from .config import PIKANConfig
from .networks import PIKANNetwork
from .losses import compute_loss
from nasjax.bpinn.pde.base import PDEProblem


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Result returned by :func:`train_pikan`.

    Attributes:
        model:          Trained :class:`~nasjax.pikans.networks.PIKANNetwork`.
        logs:           Per-epoch loss history.
        total_time_s:   Wall-clock training time in seconds.
        mean_epoch_s:   Mean per-epoch time.
        std_epoch_s:    Std dev of per-epoch times.
        config:         The :class:`~nasjax.pikans.config.PIKANConfig` used.
        final_loss:     Final total loss value.
    """
    model: PIKANNetwork
    logs: Dict[str, List[float]]
    total_time_s: float
    mean_epoch_s: float
    std_epoch_s: float
    config: PIKANConfig
    final_loss: float = 0.0


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_pikan(
    config: PIKANConfig,
    problem: PDEProblem,
    key: Optional[jax.random.PRNGKey] = None,
    verbose: bool = True,
    print_every: int = 50,
) -> TrainingResult:
    """Train a PIKAN defined by *config* on *problem*.

    Note: the ``problem`` must be a :class:`~nasjax.bpinn.pde.burgers.BurgersProblem`
    or compatible (providing ``prepare_data``).  The Burgers PDE residual is
    computed via :func:`~nasjax.pikans.losses.burgers_pde_residual` using
    jaxkan-style scalar autodiff.

    Args:
        config:       :class:`~nasjax.pikans.config.PIKANConfig`.
        problem:      :class:`~nasjax.bpinn.pde.base.PDEProblem` instance.
        key:          JAX PRNG key.  If ``None``, ``config.seed`` is used.
        verbose:      Print progress every ``print_every`` epochs.
        print_every:  Epoch interval for progress printing.

    Returns:
        :class:`TrainingResult`.
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)
    data_key = jax.random.split(key)[0]

    # Build model and optimizer
    model = PIKANNetwork(config)
    if config.optimiser_name == "adam":
        tx = optax.adam(config.learning_rate)
    elif config.optimiser_name == "adamw":
        tx = optax.adamw(config.learning_rate)
    else:
        tx = optax.sgd(config.learning_rate)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Global lambda dict
    gl = config.global_lambdas
    global_lambdas = {
        "lambda_pde":   gl.lambda_pde,
        "lambda_ic":    gl.lambda_ic,
        "lambda_bc_l":  gl.lambda_bc_l,
        "lambda_bc_r":  gl.lambda_bc_r,
    }

    # Prepare data
    batch_data = problem.prepare_data(data_key, config)
    nu = getattr(problem, "nu", 0.01 / float(np.pi))

    # JIT-compiled training step
    _diff_state = nnx.DiffState(0, nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, batch_data):
        loss, grads = nnx.value_and_grad(compute_loss, argnums=_diff_state)(
            model, batch_data, global_lambdas, nu
        )
        optimizer.update(model, grads)
        return loss

    # Training loop
    loss_history: List[float] = []
    epoch_times: List[float] = []
    t0 = time.perf_counter()

    for epoch in range(config.n_epochs):
        te = time.perf_counter()
        loss = train_step(model, optimizer, batch_data)
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
