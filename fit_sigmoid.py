"""Standalone sigmoid fitter for the (sim_bin x common_step) point tables.

Reads an Excel produced by fit_similarity.py, fits one sigmoid per sim_bin in
the chosen metric sheet, and writes:
  * plot.png        : all sim_bin curves on one figure (points + fit)
  * params.csv      : (sim_bin, L, k, x0, C) per bin
  * curves.xlsx     : resampled curves at x_step=0.03 (one column per sim_bin)

Default sigmoid: y = L / (1 + exp(k * (x - x0))) + C

Usage:
    # Fit CLIP curves from SD3 wSIM, group 1
    python fit_sigmoid.py --input sd3-medium/results/fitting/group1/wSIM_points.xlsx \\
        --metric clip --out sd3-medium/results/fitting/group1/wSIM_clip_sigmoid

    # Only fit specific sim bins
    python fit_sigmoid.py --input ... --metric clip --sim-bins 0.2 0.3 0.4 0.5 0.6 0.7

    # Change resample step
    python fit_sigmoid.py --input ... --metric clip --x-step 0.05
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def sigmoid_func(x, L, k, x0, C):
    return L / (1 + np.exp(k * (x - x0))) + C


def fit_one(x: np.ndarray, y: np.ndarray, initial_guess=(0.1, 0.1, 0.1, 0.1)):
    params, _ = curve_fit(sigmoid_func, x, y, p0=initial_guess, maxfev=1_000_000)
    return params  # L, k, x0, C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Excel produced by fit_similarity.py (e.g. wSIM_points.xlsx)")
    p.add_argument("--metric", required=True, choices=("clip", "image_reward", "brisque", "musiq"),
                   help="which sheet of the input Excel to fit")
    p.add_argument("--sim-bins", type=float, nargs="+", default=None,
                   help="subset of sim bins to fit (default: every row in the sheet)")
    p.add_argument("--x-step", type=float, default=0.03,
                   help="resample step for the fitted curve output (default 0.03)")
    p.add_argument("--x-max", type=float, default=None,
                   help="upper x bound for the resampled curve (default: max common_step in data)")
    p.add_argument("--out", default=None,
                   help="output directory (default: same dir as --input, stem + '_<metric>_sigmoid')")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    out_dir = Path(args.out or in_path.parent / f"{in_path.stem}_{args.metric}_sigmoid")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_path, sheet_name=args.metric, index_col=0)
    # df: index = sim_bin (float), columns = common_step (int), values = mean metric

    available_bins = [float(b) for b in df.index]
    if args.sim_bins:
        bins = [b for b in args.sim_bins if b in available_bins]
        missing = [b for b in args.sim_bins if b not in available_bins]
        if missing:
            print(f"warning: sim_bins not in data, skipping: {missing}")
    else:
        bins = available_bins

    if not bins:
        raise SystemExit("no sim bins to fit")

    x_all = np.array([int(c) for c in df.columns], dtype=float)
    x_min, x_max = float(x_all.min()), float(args.x_max or x_all.max())
    x_fine = np.arange(x_min, x_max + args.x_step / 2, args.x_step)

    params_rows = []
    fine_curves = pd.DataFrame({"x": x_fine})

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(bins)))

    for color, sim_bin in zip(colors, bins):
        y = df.loc[sim_bin].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 4:
            print(f"skip sim_bin={sim_bin}: only {mask.sum()} data points")
            continue
        x_local = x_all[mask]
        y_local = y[mask]
        try:
            L, k, x0, C = fit_one(x_local, y_local)
        except Exception as e:
            print(f"skip sim_bin={sim_bin}: fit failed ({e})")
            continue

        # Plot
        ax.scatter(x_local, y_local, color=color, s=30)
        y_fit_local = sigmoid_func(x_local, L, k, x0, C)
        ax.plot(x_local, y_fit_local, color=color, alpha=0.6,
                label=f"sim={sim_bin:.1f}")

        # Resampled curve
        y_fine = sigmoid_func(x_fine, L, k, x0, C)
        fine_curves[f"sim={sim_bin:.1f}"] = y_fine

        params_rows.append({"sim_bin": sim_bin, "L": L, "k": k, "x0": x0, "C": C})
        print(f"sim_bin={sim_bin:.1f}: L={L:.6f} k={k:.6f} x0={x0:.6f} C={C:.6f}")

    ax.set_xlabel("common_step")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{args.metric} vs common_step (from {in_path.name})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plot_path = out_dir / "plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"saved {plot_path}")

    params_df = pd.DataFrame(params_rows)
    params_path = out_dir / "params.csv"
    params_df.to_csv(params_path, index=False)
    print(f"saved {params_path}")

    curves_path = out_dir / "curves.xlsx"
    fine_curves.to_excel(curves_path, index=False)
    print(f"saved {curves_path}")


if __name__ == "__main__":
    main()
