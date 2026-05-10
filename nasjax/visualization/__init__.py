"""
Visualization utilities for nasjax physics-informed neural network models.

Public API
----------
:mod:`nasjax.visualization.burgers_viz`
    Plotting helpers for the Burgers' equation benchmark:

    * :func:`predict_on_grid`       — evaluate any model on a 2-D mesh.
    * :func:`plot_burgers_slices`   — solution + error at N time slices (.eps).
    * :func:`plot_burgers_heatmaps` — 3-panel heatmap: exact | pred | error (.eps).
    * :func:`save_burgers_npz`      — persist grid results to a .npz file.
    * :func:`load_burgers_npz`      — load a previously saved .npz file.
"""

from .burgers_viz import (
    predict_on_grid,
    plot_burgers_slices,
    plot_burgers_heatmaps,
    save_burgers_npz,
    load_burgers_npz,
)

__all__ = [
    "predict_on_grid",
    "plot_burgers_slices",
    "plot_burgers_heatmaps",
    "save_burgers_npz",
    "load_burgers_npz",
]
