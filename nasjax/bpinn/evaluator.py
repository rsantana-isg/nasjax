"""
BPINNEvaluator — bridge between the BPINN trainer and the NAS evolutionary loop.

This module provides :class:`BPINNEvaluator`, which wraps :func:`train_bpinn`
so that any evolutionary search algorithm can treat BPINN training as a
*black-box fitness function*:

    fitness = evaluator.evaluate(config, key)

The evaluator also exposes a *relative L2 validation error* as the fitness
metric, computed from a held-out subset of the reference data.

Design notes
------------
- Decouples *what to optimise* (config) from *how to train* (problem + hyperparams).
- The ``problem`` instance is shared across evaluations; only the config changes.
- A ``base_config`` provides default training settings; the caller can supply a
  partial config that overrides only the architecture / hyperparameters of interest.
- The evaluator is intentionally **not** an ``eqx.Module`` — it lives entirely
  outside JAX tracing.  Only the inner :func:`train_bpinn` call runs under JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from .config import BPINNConfig
from .model import BPINNModel
from .pde.base import PDEProblem
from .trainer import TrainingResult, train_bpinn


# ---------------------------------------------------------------------------
# Validation metric helpers
# ---------------------------------------------------------------------------

def relative_l2_error(
    model: BPINNModel,
    batch_data: Dict[str, jnp.ndarray],
    sample_key: jax.random.PRNGKey,
    n_samples: int = 50,
) -> float:
    """Compute the mean relative L2 error on the full reference data.

    Uses the *posterior predictive mean* (average over ``n_samples`` weight
    samples) as the point estimate.

    Args:
        model:       Trained :class:`~nasjax.bpinn.model.BPINNModel`.
        batch_data:  Data dict containing ``'x_data'``, ``'t_data'``, ``'u_data'``.
        sample_key:  JAX PRNG key.
        n_samples:   Number of MC samples for the predictive mean.

    Returns:
        Relative L2 error  ||ū − u_true|| / ||u_true||.
    """
    x_data = batch_data["x_data"]
    t_data = batch_data["t_data"]
    u_true = batch_data["u_data"].ravel()

    keys = jax.random.split(sample_key, n_samples)

    def single_pred(key: jax.random.PRNGKey) -> jnp.ndarray:
        params_list, _, _ = model.sample_params(key, sample=True)
        return model.out_u_only(x_data, t_data, params_list).ravel()

    preds = jax.vmap(single_pred)(keys)   # (n_samples, N)
    u_mean = jnp.mean(preds, axis=0)

    return float(jnp.linalg.norm(u_mean - u_true) / (jnp.linalg.norm(u_true) + 1e-12))


# ---------------------------------------------------------------------------
# BPINNEvaluator
# ---------------------------------------------------------------------------

@dataclass
class BPINNEvaluator:
    """Fitness evaluator for evolutionary BPINN architecture search.

    Trains a BPINN specified by a :class:`~nasjax.bpinn.config.BPINNConfig`
    and returns a scalar fitness value.

    Args:
        problem:         :class:`~nasjax.bpinn.pde.base.PDEProblem` instance
                         shared across all evaluations.
        base_config:     Default :class:`~nasjax.bpinn.config.BPINNConfig`.
                         Used as the starting point when ``evaluate`` is called
                         with a partial override dict.
        metric:          Fitness metric to use (``'neg_elbo'`` or ``'rel_l2'``).
                         ``'rel_l2'`` is more meaningful but requires reference
                         data in batch_data.
        n_val_samples:   Number of MC weight samples for validation metric.
        verbose:         Pass to :func:`~nasjax.bpinn.trainer.train_bpinn`.
        print_every:     Epoch print interval.
    """

    problem: PDEProblem
    base_config: BPINNConfig = field(default_factory=BPINNConfig)
    metric: str = "neg_elbo"          # 'neg_elbo' or 'rel_l2'
    n_val_samples: int = 50
    verbose: bool = False
    print_every: int = 100

    def evaluate(
        self,
        config: BPINNConfig,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """Train a BPINN with *config* and return the fitness value.

        Smaller is better for both ``'neg_elbo'`` and ``'rel_l2'``.

        Args:
            config: Full :class:`~nasjax.bpinn.config.BPINNConfig`.
            key:    JAX PRNG key.  If ``None``, ``config.seed`` is used.

        Returns:
            Scalar fitness (lower is better).
        """
        result = train_bpinn(
            config,
            self.problem,
            key=key,
            verbose=self.verbose,
            print_every=self.print_every,
        )

        if self.metric == "rel_l2":
            if key is None:
                key = jax.random.PRNGKey(config.seed + 9999)
            _, val_key = jax.random.split(key)
            # We need batch_data to evaluate; re-prepare with same seed
            data_key = jax.random.PRNGKey(config.seed)
            batch_data = self.problem.prepare_data(data_key, config)
            return relative_l2_error(
                result.model, batch_data, val_key, self.n_val_samples
            )

        # Default: negative ELBO (final epoch value)
        return result.final_neg_elbo

    def evaluate_config_dict(
        self,
        config_dict: Dict[str, Any],
        key: Optional[jax.random.PRNGKey] = None,
    ) -> float:
        """Evaluate from a flat or nested dict (useful for search algorithms).

        Merges *config_dict* on top of :attr:`base_config` via
        :meth:`~nasjax.bpinn.config.BPINNConfig.from_dict`.

        Args:
            config_dict: Nested dict produced by
                         :meth:`~nasjax.bpinn.config.BPINNConfig.to_dict`.
            key:         JAX PRNG key.

        Returns:
            Scalar fitness.
        """
        config = BPINNConfig.from_dict(config_dict)
        return self.evaluate(config, key)

    def evaluate_layer_specs(
        self,
        layer_specs: list,
        key: Optional[jax.random.PRNGKey] = None,
        **kwargs: Any,
    ) -> float:
        """Convenience: evaluate a config that differs only in architecture.

        Copies :attr:`base_config` but replaces ``layer_specs`` (and any
        additional keyword arguments).

        Args:
            layer_specs: List of ``[n_neurons, activation, prior_type, params]``
                         lists, or :class:`~nasjax.bpinn.config.BPINNLayerSpec`
                         objects.
            key:         JAX PRNG key.
            **kwargs:    Additional fields to override in the config dict.

        Returns:
            Scalar fitness.
        """
        base_dict = self.base_config.to_dict()
        base_dict["layer_specs"] = [
            s.to_dict() if hasattr(s, "to_dict") else
            {"n_neurons": s[0], "activation": s[1], "prior_type": s[2], "prior_params": s[3]}
            for s in layer_specs
        ]
        base_dict.update(kwargs)
        return self.evaluate_config_dict(base_dict, key)
