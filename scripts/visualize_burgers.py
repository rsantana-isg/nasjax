"""
Standalone re-visualization of saved Burgers' equation results.

Loads a ``.npz`` file written by any of the four example scripts
(``examples/*_burgers.py``) and regenerates both figures without retraining.

Usage
-----
    python scripts/visualize_burgers.py <results.npz> [output_prefix]

Arguments
---------
results.npz     Path to the saved numpy archive (required).
output_prefix   Optional prefix for output figure paths.
                Defaults to the stem of the input path
                (e.g. ``results/pinn_burgers`` → ``pinn_burgers_slices.eps``
                and ``pinn_burgers_heatmap.eps`` in the same directory).

Examples
--------
    python scripts/visualize_burgers.py results/pinn_burgers.npz
    python scripts/visualize_burgers.py results/fbpinn_burgers.npz results/figs/fbpinn
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nasjax.visualization import (
    load_burgers_npz,
    plot_burgers_slices,
    plot_burgers_heatmaps,
)


def _output_prefix(npz_path: str, user_prefix: str | None) -> str:
    if user_prefix:
        return user_prefix
    stem = os.path.splitext(npz_path)[0]   # e.g. "results/pinn_burgers"
    return stem


def main(npz_path: str, output_prefix: str | None = None) -> None:
    print(f"Loading  {npz_path} …")
    data = load_burgers_npz(npz_path)

    x_vec      = data["x_vec"]
    t_vec      = data["t_vec"]
    u_exact    = data["u_exact"]
    u_pred     = data["u_pred"]
    model_name = data["model_name"]

    rel_l2 = float(
        np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-12)
    )
    print(f"Model:   {model_name}")
    print(f"Grid:    Nx={len(x_vec)}, Nt={len(t_vec)}")
    print(f"Rel L2:  {rel_l2:.4e}")

    prefix = _output_prefix(npz_path, output_prefix)
    slices_path  = f"{prefix}_slices.pdf"
    heatmap_path = f"{prefix}_heatmap.pdf"

    plot_burgers_slices(
        x_vec, t_vec, u_exact, u_pred,
        model_name=model_name,
        save_path=slices_path,
    )
    plot_burgers_heatmaps(
        x_vec, t_vec, u_exact, u_pred,
        model_name=model_name,
        save_path=heatmap_path,
    )

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    npz_path = sys.argv[1]
    prefix   = sys.argv[2] if len(sys.argv) > 2 else None
    main(npz_path, prefix)
