"""
FBPINNEvaluator — NAS bridge for FBPINN hyperparameter search.

Covers both architecture (subdomain network depth/width/activation) and
domain decomposition (number of subdomains, overlap) as searchable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from .config import FBPINNConfig
from .model import FBPINNModel
from .trainer import TrainingResult, train_fbpinn
from nasjax.bpinn.pde.base import PDEProblem


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def relative_l2_error(
    model: FBPINNModel,
    params: Dict[str, Any],
    batch_data: Dict[str, jnp.ndarray],
) -> float:
    """Relative L2 error ||u_pred − u_true|| / ||u_true||."""
    u_pred = model.out_u_only(params, batch_data["x_data"], batch_data["t_data"]).ravel()
    u_true = batch_data["u_data"].ravel()
    return float(
        jnp.linalg.norm(u_pred - u_true) / (jnp.linalg.norm(u_true) + 1e-12)
    )


# ---------------------------------------------------------------------------
# FBPINNEvaluator
# ---------------------------------------------------------------------------

@dataclass
class FBPINNEvaluator:
    """Fitness evaluator for evolutionary FBPINN architecture/config search.

    Args:
        problem:       PDE problem shared across evaluations.
        base_config:   Default :class:`~nasjax.fbpinns.config.FBPINNConfig`.
        metric:        ``'rel_l2'`` or ``'loss'``.
        verbose:       Print training progress.
        print_every:   Epoch print interval.
    """
    problem: PDEProblem
    base_config: FBPINNConfig = field(default_factory=FBPINNConfig)
    metric: str = "rel_l2"
    verbose: bool = False
    print_every: int = 100

    def evaluate(
        self,
        config: FBPINNConfig,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """Train a FBPINN with *config* and return the fitness value."""
        result = train_fbpinn(
            config, self.problem,
            key=key, verbose=self.verbose, print_every=self.print_every,
        )
        if self.metric == "rel_l2":
            if key is None:
                key = jax.random.PRNGKey(config.seed)
            data_key = jax.random.split(key)[0]
            batch_data = self.problem.prepare_data(data_key, config)
            return relative_l2_error(result.model, result.params, batch_data)
        return result.final_loss

    def evaluate_config_dict(
        self,
        config_dict: Dict[str, Any],
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """Evaluate from a dict produced by :meth:`FBPINNConfig.to_dict`."""
        config = FBPINNConfig.from_dict(config_dict)
        return self.evaluate(config, key)

    def evaluate_layer_specs(
        self,
        layer_specs: list,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate differing only in subdomain network architecture."""
        base_dict = self.base_config.to_dict()
        base_dict["layer_specs"] = [
            s.to_dict() if hasattr(s, "to_dict") else
            {"n_neurons": s[0], "activation": s[1]}
            for s in layer_specs
        ]
        base_dict.update(kwargs)
        return self.evaluate_config_dict(base_dict, key)
