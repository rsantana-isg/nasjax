"""
PINNEvaluator — bridge between the PINN trainer and the NAS evolutionary loop.

Fitness function: relative L2 error or final MSE loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from .config import PINNConfig
from .model import PINNModel
from .trainer import TrainingResult, train_pinn
from nasjax.bpinn.pde.base import PDEProblem


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def relative_l2_error(
    model: PINNModel,
    batch_data: Dict[str, jnp.ndarray],
) -> float:
    """Relative L2 error ||u_pred − u_true|| / ||u_true||."""
    u_pred = model.out_u_only(batch_data["x_data"], batch_data["t_data"]).ravel()
    u_true = batch_data["u_data"].ravel()
    return float(
        jnp.linalg.norm(u_pred - u_true) / (jnp.linalg.norm(u_true) + 1e-12)
    )


# ---------------------------------------------------------------------------
# PINNEvaluator
# ---------------------------------------------------------------------------

@dataclass
class PINNEvaluator:
    """Fitness evaluator for NAS / hyperparameter search over PINN configs.

    Args:
        problem:       PDE problem shared across evaluations.
        base_config:   Default :class:`~nasjax.pinns.config.PINNConfig` used
                       as a starting point when overriding individual fields.
        metric:        ``'rel_l2'`` (relative L2 error, lower is better) or
                       ``'loss'`` (final MSE loss).
        verbose:       Pass to :func:`~nasjax.pinns.trainer.train_pinn`.
        print_every:   Epoch print interval.
    """
    problem: PDEProblem
    base_config: PINNConfig = field(default_factory=PINNConfig)
    metric: str = "rel_l2"
    verbose: bool = False
    print_every: int = 100

    def evaluate(
        self,
        config: PINNConfig,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """Train a PINN with *config* and return the fitness value."""
        result = train_pinn(
            config, self.problem,
            key=key, verbose=self.verbose, print_every=self.print_every,
        )
        if self.metric == "rel_l2":
            if key is None:
                key = jax.random.PRNGKey(config.seed)
            data_key = jax.random.split(key)[0]
            batch_data = self.problem.prepare_data(data_key, config)
            return relative_l2_error(result.model, batch_data)
        return result.final_loss

    def evaluate_config_dict(
        self,
        config_dict: Dict[str, Any],
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """Evaluate from a nested dict (output of :meth:`PINNConfig.to_dict`)."""
        config = PINNConfig.from_dict(config_dict)
        return self.evaluate(config, key)

    def evaluate_layer_specs(
        self,
        layer_specs: list,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs: Any,
    ) -> float:
        """Convenience: evaluate differing only in architecture.

        Copies :attr:`base_config` but replaces ``layer_specs`` and any
        additional keyword arguments.
        """
        base_dict = self.base_config.to_dict()
        base_dict["layer_specs"] = [
            s.to_dict() if hasattr(s, "to_dict") else
            {"n_neurons": s[0], "activation": s[1]}
            for s in layer_specs
        ]
        base_dict.update(kwargs)
        return self.evaluate_config_dict(base_dict, key)
