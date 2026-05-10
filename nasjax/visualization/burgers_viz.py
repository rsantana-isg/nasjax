"""
Visualization helpers for Burgers' equation PINN results.

All plotting functions are *model-agnostic*: they accept plain numpy arrays
produced by a caller-supplied ``predict_fn`` closure, so the same code works
for BPINN, PINN, PIKAN, and FBPINN.

Typical workflow in an example script
--------------------------------------
::

    from nasjax.visualization import (
        predict_on_grid, plot_burgers_slices, plot_burgers_heatmaps,
        save_burgers_npz,
    )

    x_vec, t_vec, u_exact = problem.load_reference_grid()

    # Model-specific prediction closure (x, t are (N,1) JAX arrays)
    predict_fn = lambda x, t: np.array(trained_model.out_u_only(x, t)).ravel()

    u_pred = predict_on_grid(predict_fn, x_vec, t_vec)          # (Nt, Nx)
    save_burgers_npz(x_vec, t_vec, u_exact, u_pred,
                     model_name="PINN", path="results/pinn_burgers.npz")
    plot_burgers_slices(x_vec, t_vec, u_exact, u_pred,
                        model_name="PINN",
                        save_path="results/pinn_burgers_slices.eps")
    plot_burgers_heatmaps(x_vec, t_vec, u_exact, u_pred,
                          model_name="PINN",
                          save_path="results/pinn_burgers_heatmap.eps")

Figure design
-------------
* ``plot_burgers_slices``   — 2-row × N-column panel.
  Row 0: u(x) at each time slice (exact = black dashed, model = solid colour).
  Row 1: |u_exact − u_pred|(x) on a log scale.
* ``plot_burgers_heatmaps`` — 1-row × 3-column panel.
  Exact | Predicted | Absolute error (colour maps: RdBu_r / RdBu_r / hot_r).

Both figures are paper-quality (LaTeX-ready, no titles, large fonts) and saved
as EPS by default (change the ``save_path`` extension to ``.pdf`` if preferred).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

# Colour sequence for the five time slices.
_SLICE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# ---------------------------------------------------------------------------
# matplotlib defaults (LaTeX-ready, readable in papers)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        12,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


# ---------------------------------------------------------------------------
# Grid prediction
# ---------------------------------------------------------------------------

def predict_on_grid(
    predict_fn: Callable,
    x_vec: np.ndarray,
    t_vec: np.ndarray,
) -> np.ndarray:
    """Evaluate *predict_fn* on the full (x, t) meshgrid.

    Args:
        predict_fn: ``f(x, t) -> array`` where ``x`` and ``t`` are
                    ``(N, 1)`` JAX arrays and the output is ``(N,)`` or
                    ``(N, 1)`` (numpy or JAX).  The function must be
                    model-agnostic — callers wrap model-specific APIs in a
                    closure before passing here.
        x_vec:      ``(Nx,)`` spatial grid (numpy).
        t_vec:      ``(Nt,)`` time grid (numpy).

    Returns:
        ``u_pred`` of shape ``(Nt, Nx)`` — row *i* is the prediction at
        ``t_vec[i]``.
    """
    import jax.numpy as jnp

    X, T = np.meshgrid(x_vec, t_vec)          # (Nt, Nx) each
    x_flat = jnp.array(X.flatten()[:, None])  # (Nt*Nx, 1)
    t_flat = jnp.array(T.flatten()[:, None])  # (Nt*Nx, 1)
    u_flat = predict_fn(x_flat, t_flat)
    return np.array(u_flat).ravel().reshape(len(t_vec), len(x_vec))


# ---------------------------------------------------------------------------
# NPZ persistence
# ---------------------------------------------------------------------------

def save_burgers_npz(
    x_vec:      np.ndarray,
    t_vec:      np.ndarray,
    u_exact:    np.ndarray,
    u_pred:     np.ndarray,
    model_name: str,
    path:       str,
) -> None:
    """Save Burgers grid results to a ``.npz`` file for later re-visualization.

    Args:
        x_vec:      ``(Nx,)`` spatial grid.
        t_vec:      ``(Nt,)`` time grid.
        u_exact:    ``(Nt, Nx)`` exact solution.
        u_pred:     ``(Nt, Nx)`` model prediction.
        model_name: Display name used in figure labels (e.g. ``"PINN"``).
        path:       Output path (e.g. ``"results/pinn_burgers.npz"``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        x_vec=x_vec,
        t_vec=t_vec,
        u_exact=u_exact,
        u_pred=u_pred,
        model_name=np.array(model_name),
    )
    print(f"  Results saved → {path}")


def load_burgers_npz(path: str) -> dict:
    """Load Burgers results saved by :func:`save_burgers_npz`.

    Returns:
        Dict with keys ``x_vec``, ``t_vec``, ``u_exact``, ``u_pred``,
        ``model_name``.
    """
    data = np.load(path, allow_pickle=True)
    return {
        "x_vec":      data["x_vec"],
        "t_vec":      data["t_vec"],
        "u_exact":    data["u_exact"],
        "u_pred":     data["u_pred"],
        "model_name": str(data["model_name"]),
    }


# ---------------------------------------------------------------------------
# Figure: time-slice comparison (solution + error)
# ---------------------------------------------------------------------------

def plot_burgers_slices(
    x_vec:      np.ndarray,
    t_vec:      np.ndarray,
    u_exact:    np.ndarray,
    u_pred:     np.ndarray,
    model_name: str,
    save_path:  str,
    n_times:    int = 5,
) -> None:
    """2-row figure comparing the exact and predicted solutions at *n_times*
    evenly-spaced time snapshots.

    * **Row 0** — solution: exact (black dashed) vs predicted (solid colour).
    * **Row 1** — absolute point-wise error on a semi-log scale.

    Args:
        x_vec:      ``(Nx,)`` spatial grid.
        t_vec:      ``(Nt,)`` time grid.
        u_exact:    ``(Nt, Nx)`` exact solution.
        u_pred:     ``(Nt, Nx)`` model prediction.
        model_name: String used in the legend.
        save_path:  Output path (e.g. ``"results/pinn_burgers_slices.eps"``).
        n_times:    Number of time slices to show (default 5).
    """
    Nt = len(t_vec)
    t_indices = np.round(np.linspace(0, Nt - 1, n_times)).astype(int)

    fig, axes = plt.subplots(
        2, n_times,
        figsize=(3.4 * n_times, 5.2),
        sharex=True,
    )

    for col, ti in enumerate(t_indices):
        color = _SLICE_COLORS[col % len(_SLICE_COLORS)]
        t_val = t_vec[ti]
        u_ex  = u_exact[ti]
        u_pr  = u_pred[ti]
        err   = np.abs(u_ex - u_pr)

        # ---- row 0: solution comparison ----
        ax0 = axes[0, col]
        ax0.plot(x_vec, u_ex, "k--", lw=1.6, label="Exact")
        ax0.plot(x_vec, u_pr, "-",   lw=1.6, color=color, label=model_name)
        ax0.set_title(f"$t = {t_val:.2f}$", fontsize=12)
        ax0.set_xlim(x_vec[0], x_vec[-1])
        if col == 0:
            ax0.set_ylabel("$u(x,\\,t)$")
            ax0.legend(loc="best")

        # ---- row 1: absolute error (log scale) ----
        ax1 = axes[1, col]
        ax1.semilogy(x_vec, np.maximum(err, 1e-10), "-", lw=1.6, color=color)
        ax1.set_xlabel("$x$")
        ax1.set_xlim(x_vec[0], x_vec[-1])
        if col == 0:
            ax1.set_ylabel("$|u_{\\rm exact} - u_{\\rm pred}|$")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Slice figure   → {save_path}")


# ---------------------------------------------------------------------------
# Figure: 2-D heatmaps (exact | predicted | error)
# ---------------------------------------------------------------------------

def plot_burgers_heatmaps(
    x_vec:      np.ndarray,
    t_vec:      np.ndarray,
    u_exact:    np.ndarray,
    u_pred:     np.ndarray,
    model_name: str,
    save_path:  str,
) -> None:
    """1-row × 3-column heatmap: exact solution | model prediction | absolute error.

    The first two panels share the same colour scale; the error panel uses an
    independent scale and the ``hot_r`` colour map.

    Args:
        x_vec:      ``(Nx,)`` spatial grid.
        t_vec:      ``(Nt,)`` time grid.
        u_exact:    ``(Nt, Nx)`` exact solution.
        u_pred:     ``(Nt, Nx)`` model prediction.
        model_name: String label for the prediction panel.
        save_path:  Output path (e.g. ``"results/pinn_burgers_heatmap.eps"``).
    """
    err  = np.abs(u_exact - u_pred)
    vmin = min(float(u_exact.min()), float(u_pred.min()))
    vmax = max(float(u_exact.max()), float(u_pred.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    panels = [
        (u_exact,    "Exact",       "RdBu_r",  {"vmin": vmin, "vmax": vmax}),
        (u_pred,     model_name,    "RdBu_r",  {"vmin": vmin, "vmax": vmax}),
        (err,        "Absolute error", "hot_r", {}),
    ]

    for ax, (arr, label, cmap, kwargs) in zip(axes, panels):
        im = ax.pcolormesh(x_vec, t_vec, arr, shading="auto", cmap=cmap, **kwargs)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_title(label)
        cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=20)
        cb.ax.tick_params(labelsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Heatmap figure → {save_path}")
