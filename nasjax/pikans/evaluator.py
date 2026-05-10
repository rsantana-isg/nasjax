"""
PIKANEvaluator — NAS bridge for PIKAN architecture search.

Provides the same ``evaluate / evaluate_config_dict / evaluate_layer_specs``
interface as ``BPINNEvaluator`` and ``PINNEvaluator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .config import PIKANConfig
from .networks import PIKANNetwork
from .trainer import TrainingResult, train_pikan
from nasjax.bpinn.pde.base import PDEProblem


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def relative_l2_error(
    model: PIKANNetwork,
    batch_data: Dict[str, jnp.ndarray],
) -> float:
    """Relative L2 error on full reference grid.

    Args:
        model:       Trained :class:`~nasjax.pikans.networks.PIKANNetwork`.
        batch_data:  Data dict containing ``'x_data'``, ``'t_data'``, ``'u_data'``.

    Returns:
        ``||u_pred − u_true|| / ||u_true||``.
    """
    import jax.numpy as jnp
    inp = jnp.concatenate([batch_data["t_data"], batch_data["x_data"]], axis=1)
    u_pred = model(inp).ravel()
    u_true = batch_data["u_data"].ravel()
    return float(
        jnp.linalg.norm(u_pred - u_true) / (jnp.linalg.norm(u_true) + 1e-12)
    )


# ---------------------------------------------------------------------------
# PIKANEvaluator
# ---------------------------------------------------------------------------

@dataclass
class PIKANEvaluator:
    """Fitness evaluator for evolutionary PIKAN architecture search.

    Args:
        problem:       PDE problem shared across evaluations.
        base_config:   Default :class:`~nasjax.pikans.config.PIKANConfig`.
        metric:        ``'rel_l2'`` or ``'loss'``.
        verbose:       Print training progress.
        print_every:   Epoch print interval.
    """
    problem: PDEProblem
    base_config: PIKANConfig = field(default_factory=PIKANConfig)
    metric: str = "rel_l2"
    verbose: bool = False
    print_every: int = 100

    def evaluate(
        self,
        config: PIKANConfig,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """Train a PIKAN with *config* and return the fitness value."""
        result = train_pikan(
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
        """Evaluate from a dict produced by :meth:`PIKANConfig.to_dict`."""
        config = PIKANConfig.from_dict(config_dict)
        return self.evaluate(config, key)

    def evaluate_layer_specs(
        self,
        layer_specs: list,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs: Any,
    ) -> float:
        """Evaluate differing only in layer specs.

        Each spec is ``[n_out, layer_type, layer_params_dict]``.
        """
        base_dict = self.base_config.to_dict()
        base_dict["layer_specs"] = [
            s.to_dict() if hasattr(s, "to_dict") else
            {"n_out": s[0], "layer_type": s[1], "layer_params": s[2] if len(s) > 2 else {}}
            for s in layer_specs
        ]
        base_dict.update(kwargs)
        return self.evaluate_config_dict(base_dict, key)
