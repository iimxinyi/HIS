"""Sigmoid fitting helpers for the HIQM curves."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sigmoid_func(x, L, k, x0, C):
    return L / (1 + np.exp(k * (x - x0))) + C


def fit_sigmoid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    initial_guess = [0.1, 0.1, 0.1, 0.1]
    params, _ = curve_fit(sigmoid_func, x, y, p0=initial_guess, maxfev=1_000_000)
    return tuple(float(p) for p in params)


def plot_fits(
    x: np.ndarray,
    series: Dict[float, np.ndarray],
    out_path: str | Path,
    title: str = "",
    xlabel: str = "common inference step",
    ylabel: str = "score",
) -> Dict[float, Tuple[float, float, float, float]]:
    """Fit one sigmoid per series, save a combined plot, return fitted params."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params_by_bin: Dict[float, Tuple[float, float, float, float]] = {}
    plt.figure(figsize=(8, 5))
    x_dense = np.linspace(x.min(), x.max(), 200)
    for bin_value, y in sorted(series.items()):
        try:
            params = fit_sigmoid(x, y)
        except RuntimeError:
            continue
        params_by_bin[bin_value] = params
        plt.scatter(x, y, s=14, label=f"sim={bin_value:.1f}")
        plt.plot(x_dense, sigmoid_func(x_dense, *params), linewidth=1.0)
    plt.legend(fontsize=8, loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return params_by_bin
