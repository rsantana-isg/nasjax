"""
ELBO loss computation for BPINN.

This module implements the negative ELBO:

    -ELBO = λ_vp · log q(θ|φ) − λ_pr · log p(θ)
            − (λ_pde · log p_pde + λ_ic · log p_ic + Σ_k λ_k · log p_k)

where ``log p(D|θ)`` for each term uses a Gaussian likelihood:

    log p_m(D_m|θ) = Σ_n [−½ log(2πσ²) − residual²/(2σ²)]

Two local-lambda (per-collocation-point) weighting strategies are available:

- **Residual RBA** (Residual-Based Attention): Scale each residual by a
  per-point weight λ_n that amplifies high-residual regions.
- **Likelihood RBA**: Scale the log-likelihood itself by a per-point weight
  derived from the deviation of the current likelihood from an optimal value.

All functions are pure JAX (JIT- and vmap-compatible).

Public API
----------
- :func:`gaussian_log_likelihood`          — baseline Gaussian log-likelihood
- :func:`rba_residual_apply`               — Residual-Based Attention update
- :func:`likelihood_rba_apply`             — Likelihood-RBA update
- :func:`single_sample_elbo_terms`         — ELBO terms for one MC sample
- :func:`compute_loss`                     — JIT-compiled multi-sample loss
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Gaussian likelihood utilities
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(
    residual: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Standard Gaussian log-likelihood (sum over all points).

        log p = Σ_n [ −½ log(2πσ²) − residual_n²/(2σ²) ]

    Args:
        residual: Residual array (any shape).
        sigma:    Likelihood standard deviation.

    Returns:
        Scalar log-likelihood.
    """
    return jnp.sum(
        -0.5 * jnp.log(2.0 * jnp.pi * sigma ** 2)
        - jnp.square(residual) / (2.0 * sigma ** 2)
    )


def gaussian_log_likelihood_rba_residual(
    scaled_residual: jnp.ndarray,
    sigma: float,
    lambdas: jnp.ndarray,
) -> jnp.ndarray:
    """Gaussian log-likelihood after Residual-RBA scaling.

    Assumes the residual has been scaled to ``λ_n · r_n`` and the effective
    variance is ``(σ/λ_n)²``:

        log p = Σ_n [ −½ log(2π(σ/λ_n)²) − (λ_n·r_n)²/(2σ²) ]

    Args:
        scaled_residual: ``λ_n · r_n``, shape ``(N, 1)``.
        sigma:           Likelihood σ.
        lambdas:         Per-point weights λ_n, shape ``(N, 1)``.

    Returns:
        Scalar log-likelihood.
    """
    return jnp.sum(
        -0.5 * jnp.log(2.0 * jnp.pi * (sigma / lambdas) ** 2)
        - jnp.square(scaled_residual) / (2.0 * sigma ** 2)
    )


# ---------------------------------------------------------------------------
# Residual-Based Attention (RBA)
# ---------------------------------------------------------------------------

def rba_residual_apply(
    residual: jnp.ndarray,
    old_lambda: jnp.ndarray,
    gamma: float = 0.999,
    lr: float = 0.001,
    eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update per-point weights via Residual-Based Attention.

    λ_n ← γ · λ_n + lr · |r_n| / (max |r| + ε)

    Weights are normalised to (0, 1) by dividing by the max residual.
    ``jax.lax.stop_gradient`` ensures λ updates don't pass gradients back to
    the model parameters.

    Args:
        residual:    PDE/IC/BC residual array.
        old_lambda:  Previous epoch's per-point weights.
        gamma:       EMA decay.
        lr:          Lambda learning rate.
        eps:         Numerical stability constant.

    Returns:
        ``(scaled_residual, new_lambda)``
    """
    new_lambda = (
        gamma * old_lambda
        + lr * jnp.abs(residual / (jnp.max(jnp.abs(residual)) + eps))
    )
    new_lambda = jax.lax.stop_gradient(new_lambda)
    scaled_residual = new_lambda * residual
    return scaled_residual, new_lambda


def likelihood_rba_apply(
    residual: jnp.ndarray,
    sigma: float,
    old_lambda: jnp.ndarray,
    gamma: float = 0.999,
    lr: float = 0.001,
    eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update per-point weights via Likelihood-Based Attention.

    λ_n ← γ · λ_n + lr · |ℓ_n| / (max |ℓ| + ε)
    where ℓ_n = (1 / 2σ²) · r_n² (the pointwise likelihood cost).

    The total log-likelihood then becomes Σ_n λ_n · log p_n(D|θ).

    Args:
        residual:    Residual array.
        sigma:       Likelihood σ.
        old_lambda:  Previous epoch's per-point weights.
        gamma:       EMA decay.
        lr:          Lambda learning rate.
        eps:         Numerical stability constant.

    Returns:
        ``(weighted_log_likelihood_scalar, new_lambda)``
    """
    lik_cost = (1.0 / (2.0 * sigma ** 2)) * jnp.square(residual)
    new_lambda = (
        gamma * old_lambda
        + lr * jnp.abs(lik_cost / (jnp.max(jnp.abs(lik_cost)) + eps))
    )
    new_lambda = jax.lax.stop_gradient(new_lambda)

    # Pointwise log-likelihood (without λ)
    log_lik_per_point = (
        -0.5 * jnp.log(2.0 * jnp.pi * sigma ** 2)
        - jnp.square(residual) / (2.0 * sigma ** 2)
    )
    total_weighted_log_lik = jnp.sum(new_lambda * log_lik_per_point)
    return total_weighted_log_lik, new_lambda


# ---------------------------------------------------------------------------
# Single Monte-Carlo sample ELBO terms
# ---------------------------------------------------------------------------

def single_sample_elbo_terms(
    model: eqx.Module,        # BPINNModel
    problem: object,           # PDEProblem
    batch_data: Dict[str, jnp.ndarray],
    local_lambdas: Dict[str, jnp.ndarray],
    global_lambdas: Dict[str, float],
    local_lambda_tech: str,    # 'None' | 'Residual_RBA' | 'Likelihood_RBA'
    sigma: Dict[str, float],   # {'sigma_pde': ..., 'sigma_ic': ..., ...}
    rba_gamma: float,
    rba_lr: float,
    rba_eps: float,
    fwd_key: jax.random.PRNGKey,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Compute ELBO terms for a *single* MC sample of network weights.

    This function is mapped over ``K`` keys by :func:`compute_loss`.

    Returns:
        ``(loss_terms_dict, new_local_lambdas_dict)``

        ``loss_terms_dict`` keys:
            ``'log_lik_pde'``, ``'log_lik_ic'``,
            + one ``'log_lik_{k}'`` per ``problem.bc_keys``,
            ``'log_prior'``, ``'log_var_posterior'``,
            ``'unaltered_log_lik_pde'``, ``'unaltered_log_lik_ic'``,
            + one ``'unaltered_log_lik_{k}'`` per ``problem.bc_keys``.
    """
    # --- Sample weights ------------------------------------------------
    params_list, log_prior, log_q = model.sample_params(fwd_key, sample=True)

    # --- PDE residual --------------------------------------------------
    x_pde, t_pde = batch_data["x_pde"], batch_data["t_pde"]
    res_pde = problem.pde_residual(model, x_pde, t_pde, params_list)

    # --- IC residual ---------------------------------------------------
    x_ic, t_ic, u_ic = batch_data["x_ic"], batch_data["t_ic"], batch_data["u_ic"]
    res_ic = problem.ic_residual(model, x_ic, t_ic, u_ic, params_list)

    # --- BC residuals --------------------------------------------------
    res_bcs = problem.bc_residuals(model, batch_data, params_list)  # dict

    # --- Apply local lambda technique ----------------------------------
    s_pde = float(sigma["sigma_pde"])
    s_ic  = float(sigma["sigma_ic"])

    if local_lambda_tech == "Residual_RBA":
        res_pde_used, new_ll_pde = rba_residual_apply(
            res_pde, local_lambdas["lambda_pde"], rba_gamma, rba_lr, rba_eps
        )
        res_ic_used, new_ll_ic = rba_residual_apply(
            res_ic, local_lambdas["lambda_ic"], rba_gamma, rba_lr, rba_eps
        )
        log_lik_pde = gaussian_log_likelihood_rba_residual(res_pde_used, s_pde, new_ll_pde)
        log_lik_ic  = gaussian_log_likelihood_rba_residual(res_ic_used, s_ic, new_ll_ic)

        new_ll_bcs: Dict[str, jnp.ndarray] = {}
        log_lik_bcs: Dict[str, jnp.ndarray] = {}
        for k in problem.bc_keys:
            s_bc = float(sigma[f"sigma_{k}"])
            res_used, new_ll = rba_residual_apply(
                res_bcs[k], local_lambdas[f"lambda_{k}"], rba_gamma, rba_lr, rba_eps
            )
            new_ll_bcs[f"lambda_{k}"] = new_ll
            log_lik_bcs[f"log_lik_{k}"] = gaussian_log_likelihood_rba_residual(
                res_used, s_bc, new_ll
            )

    elif local_lambda_tech == "Likelihood_RBA":
        log_lik_pde, new_ll_pde = likelihood_rba_apply(
            res_pde, s_pde, local_lambdas["lambda_pde"], rba_gamma, rba_lr, rba_eps
        )
        log_lik_ic, new_ll_ic = likelihood_rba_apply(
            res_ic, s_ic, local_lambdas["lambda_ic"], rba_gamma, rba_lr, rba_eps
        )

        new_ll_bcs = {}
        log_lik_bcs = {}
        for k in problem.bc_keys:
            s_bc = float(sigma[f"sigma_{k}"])
            ll_k, new_ll = likelihood_rba_apply(
                res_bcs[k], s_bc, local_lambdas[f"lambda_{k}"], rba_gamma, rba_lr, rba_eps
            )
            new_ll_bcs[f"lambda_{k}"] = new_ll
            log_lik_bcs[f"log_lik_{k}"] = ll_k

    else:  # 'None' — no local weighting
        new_ll_pde = local_lambdas["lambda_pde"]
        new_ll_ic  = local_lambdas["lambda_ic"]
        log_lik_pde = gaussian_log_likelihood(res_pde, s_pde)
        log_lik_ic  = gaussian_log_likelihood(res_ic, s_ic)

        new_ll_bcs = {}
        log_lik_bcs = {}
        for k in problem.bc_keys:
            s_bc = float(sigma[f"sigma_{k}"])
            new_ll_bcs[f"lambda_{k}"] = local_lambdas[f"lambda_{k}"]
            log_lik_bcs[f"log_lik_{k}"] = gaussian_log_likelihood(res_bcs[k], s_bc)

    # Unaltered likelihoods (for logging, always without RBA scaling)
    unalt_log_lik_pde = gaussian_log_likelihood(res_pde, s_pde)
    unalt_log_lik_ic  = gaussian_log_likelihood(res_ic,  s_ic)
    unalt_log_lik_bcs = {
        f"unaltered_log_lik_{k}": gaussian_log_likelihood(res_bcs[k], float(sigma[f"sigma_{k}"]))
        for k in problem.bc_keys
    }

    # --- Assemble outputs ----------------------------------------------
    loss_terms = {
        "log_lik_pde":          log_lik_pde,
        "log_lik_ic":           log_lik_ic,
        "log_prior":            log_prior,
        "log_var_posterior":    log_q,
        "unaltered_log_lik_pde": unalt_log_lik_pde,
        "unaltered_log_lik_ic":  unalt_log_lik_ic,
        **log_lik_bcs,
        **unalt_log_lik_bcs,
    }

    new_local_lambdas = {
        "lambda_pde": new_ll_pde,
        "lambda_ic":  new_ll_ic,
        **new_ll_bcs,
    }

    return loss_terms, new_local_lambdas


# ---------------------------------------------------------------------------
# Multi-sample ELBO loss (JIT-compiled training objective)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def compute_loss(
    model: eqx.Module,
    problem: object,
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
) -> Tuple[jnp.ndarray, Tuple[Dict, Dict]]:
    """Compute the negative ELBO over K Monte-Carlo weight samples.

    This is the top-level training objective differentiated by
    :func:`~nasjax.bpinn.trainer.make_step`.

    The expectation in the ELBO is approximated as an average over K samples:

        ELBO ≈ (1/K) Σ_k [ log p(D|θ_k) + log p(θ_k) − log q(θ_k|φ) ]

    Args:
        model:               :class:`~nasjax.bpinn.model.BPINNModel`.
        problem:             :class:`~nasjax.bpinn.pde.base.PDEProblem`.
        batch_data:          Collocation data dict.
        local_lambdas:       Current per-point lambda arrays.
        global_lambdas:      Global ELBO term weights.
        local_lambda_tech:   ``'None'`` | ``'Residual_RBA'`` | ``'Likelihood_RBA'``.
        sigma:               Likelihood σ values dict.
        rba_gamma, rba_lr, rba_eps: RBA hyper-parameters.
        fwd_key:             JAX PRNG key (split into K sub-keys internally).
        k_samples:           Number of MC samples K.

    Returns:
        ``(negative_elbo, (log_loss_dict, new_local_lambdas))``
    """
    keys = jax.random.split(fwd_key, k_samples)

    def single_sample(key):
        return single_sample_elbo_terms(
            model, problem, batch_data,
            local_lambdas, global_lambdas,
            local_lambda_tech, sigma,
            rba_gamma, rba_lr, rba_eps, key,
        )

    # vmap over K samples
    all_terms, all_new_ll = jax.vmap(single_sample)(keys)

    # Average over K samples
    avg = {k: jnp.mean(v) for k, v in all_terms.items()}
    new_local_lambdas = {k: jnp.mean(v, axis=0) for k, v in all_new_ll.items()}

    # --- Weighted total log-likelihood ---------------------------------
    g = global_lambdas
    total_log_lik = (
        g["lambda_lik_pde"] * avg["log_lik_pde"]
        + g["lambda_lik_ic"]  * avg["log_lik_ic"]
    )
    for k in problem.bc_keys:
        total_log_lik += g.get(f"lambda_lik_{k}", 1.0) * avg[f"log_lik_{k}"]

    # --- KL divergence (variational posterior − prior) -----------------
    kl = (
        g["lambda_var_posterior"] * avg["log_var_posterior"]
        - g["lambda_prior"]       * avg["log_prior"]
    )

    neg_elbo = kl - total_log_lik

    # Detailed log dict for diagnostics
    log_loss = {
        "negative_ELBO":          neg_elbo,
        "log_lik_pde":            avg["log_lik_pde"],
        "log_lik_ic":             avg["log_lik_ic"],
        "total_log_prior":        avg["log_prior"],
        "total_log_var_posterior": avg["log_var_posterior"],
        "unaltered_log_lik_pde":  avg["unaltered_log_lik_pde"],
        "unaltered_log_lik_ic":   avg["unaltered_log_lik_ic"],
        **{f"log_lik_{k}":             avg[f"log_lik_{k}"]          for k in problem.bc_keys},
        **{f"unaltered_log_lik_{k}":   avg[f"unaltered_log_lik_{k}"] for k in problem.bc_keys},
    }

    return neg_elbo, (log_loss, new_local_lambdas)


# ---------------------------------------------------------------------------
# Lambda initialisation helpers
# ---------------------------------------------------------------------------

def init_local_lambdas(
    batch_data: Dict[str, jnp.ndarray],
    bc_keys: List[str],
) -> Dict[str, jnp.ndarray]:
    """Initialise per-point local lambdas to 1 for all terms.

    Shapes match the corresponding collocation arrays in ``batch_data``.
    """
    return {
        "lambda_pde": jnp.ones_like(batch_data["x_pde"]),
        "lambda_ic":  jnp.ones_like(batch_data["x_ic"]),
        **{f"lambda_{k}": jnp.ones_like(batch_data[f"x_{k}"]) for k in bc_keys},
    }


def init_global_lambdas(
    global_lambda_config: object,
    bc_keys: List[str],
) -> Dict[str, float]:
    """Convert :class:`~nasjax.bpinn.config.GlobalLambdaConfig` to a plain dict.

    Args:
        global_lambda_config: :class:`~nasjax.bpinn.config.GlobalLambdaConfig`.
        bc_keys:              List of boundary condition names.

    Returns:
        Dict with keys ``'lambda_lik_pde'``, ``'lambda_lik_ic'``,
        ``'lambda_lik_{k}'`` for each k in bc_keys,
        ``'lambda_prior'``, ``'lambda_var_posterior'``.
    """
    cfg = global_lambda_config
    d: Dict[str, float] = {
        "lambda_lik_pde":        cfg.lambda_lik_pde,
        "lambda_lik_ic":         cfg.lambda_lik_ic,
        "lambda_prior":          cfg.lambda_prior,
        "lambda_var_posterior":  cfg.lambda_var_posterior,
    }
    for k in bc_keys:
        # Use bc_l / bc_r specific values if present, otherwise default to 1.0
        attr = f"lambda_lik_{k}"
        d[f"lambda_lik_{k}"] = getattr(cfg, attr, 1.0)
    return d


def init_sigma_dict(sigma_config: object) -> Dict[str, float]:
    """Convert :class:`~nasjax.bpinn.config.SigmaConfig` to a plain dict.

    Returns:
        Dict with keys ``'sigma_pde'``, ``'sigma_ic'``,
        ``'sigma_bc_l'``, ``'sigma_bc_r'`` (and any extras on the config).
    """
    from dataclasses import asdict
    return asdict(sigma_config)
