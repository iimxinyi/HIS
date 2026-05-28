"""Fit sigmoid curves per similarity column for one sheet of
``Similarity_Fitting_Results.xlsx``.

Input sheet layout (rows x cols):
    Row 0:  "Common Inference Step" | sim_1 | sim_2 | ... | sim_K
    Row N:  step_N                  | y_N_1 | y_N_2 | ... | y_N_K  (N >= 1)

For each similarity column we fit y = L / (1 + exp(k * (x - x0))) + C and write
the curve resampled at x_step (default 0.03) into a new Excel placed next to
the input file, plus a PNG plot of points + fits.

Usage (copy-paste):

    # === sd3-medium ===
    python fit_sigmoid.py --input final-results/sd3-medium/SImilarity_Fitting_Results.xlsx --sheet "Alignment Dot wSIM"
    python fit_sigmoid.py --input final-results/sd3-medium/SImilarity_Fitting_Results.xlsx --sheet "Alignment Dot woSIM"
    python fit_sigmoid.py --input final-results/sd3-medium/SImilarity_Fitting_Results.xlsx --sheet "Fidelity Dot wSIM"
    python fit_sigmoid.py --input final-results/sd3-medium/SImilarity_Fitting_Results.xlsx --sheet "Fidelity Dot woSIM"

    # === flux.1-dev ===
    python fit_sigmoid.py --input final-results/flux.1-dev/SImilarity_Fitting_Results.xlsx --sheet "Alignment Dot wSIM"
    python fit_sigmoid.py --input final-results/flux.1-dev/SImilarity_Fitting_Results.xlsx --sheet "Alignment Dot woSIM"
    python fit_sigmoid.py --input final-results/flux.1-dev/SImilarity_Fitting_Results.xlsx --sheet "Fidelity Dot wSIM"
    python fit_sigmoid.py --input final-results/flux.1-dev/SImilarity_Fitting_Results.xlsx --sheet "Fidelity Dot woSIM"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


SHEET_CHOICES = (
    "Alignment Dot wSIM",
    "Alignment Dot woSIM",
    "Fidelity Dot wSIM",
    "Fidelity Dot woSIM",
)


def sigmoid_func(x, L, k, x0, C):
    return L / (1 + np.exp(k * (x - x0))) + C


def initial_guess_from_data(x: np.ndarray, y: np.ndarray):
    """Estimate (L, k, x0, C) from the data so curve_fit doesn't get stuck in
    the flat-line local optimum.

    For the paper's HIQM curve, y is monotonic in x (decreasing for alignment,
    increasing for fidelity). We set:
        L  = amplitude       (y_max - y_min, signed by direction)
        C  = baseline        (the asymptote opposite the inflection)
        x0 = inflection x    (where y is closest to its midpoint)
        k  = slope sign / scale, derived from end-to-end slope
    """
    y_max, y_min = float(np.max(y)), float(np.min(y))
    amplitude = y_max - y_min
    if amplitude < 1e-9:
        return (0.0, 0.1, float(np.median(x)), y_min)

    # End-to-end slope tells us whether y goes up or down with x.
    dy_dx = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0
    if dy_dx >= 0:
        # increasing sigmoid: y -> y_min at x->-inf, y -> y_min+L at x->+inf, need k<0
        L_init = amplitude
        C_init = y_min
    else:
        # decreasing sigmoid: y -> y_min+L at x->-inf, y -> y_min at x->+inf, need k>0
        L_init = amplitude
        C_init = y_min

    y_mid = (y_max + y_min) / 2.0
    x0_init = float(x[np.argmin(np.abs(y - y_mid))])
    # Slope at midpoint of L/(1+exp(k(x-x0))) + C is -L*k/4; solve for k.
    k_init = -4.0 * dy_dx / L_init
    return (L_init, k_init, x0_init, C_init)


def fit_one(x: np.ndarray, y: np.ndarray):
    p0 = initial_guess_from_data(x, y)
    params, _ = curve_fit(sigmoid_func, x, y, p0=p0, maxfev=1_000_000)
    return params  # L, k, x0, C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to SImilarity_Fitting_Results.xlsx")
    p.add_argument("--sheet", required=True, choices=SHEET_CHOICES,
                   help="Which sheet to fit")
    p.add_argument("--x-step", type=float, default=0.03,
                   help="Resample step for the fitted curve output (default 0.03)")
    p.add_argument("--x-max", type=float, default=None,
                   help="Upper x bound for the resampled curve (default: max step in data)")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    out_dir = in_path.parent
    safe_sheet = args.sheet.replace(" ", "_")
    plot_path = out_dir / f"{safe_sheet}_sigmoid_plot.png"
    curves_path = out_dir / f"{safe_sheet}_sigmoid_curves.xlsx"

    raw = pd.read_excel(in_path, sheet_name=args.sheet, header=None)
    sims = raw.iloc[0, 1:].astype(float).tolist()
    x_all = raw.iloc[1:, 0].astype(float).to_numpy()
    y_matrix = raw.iloc[1:, 1:].astype(float).to_numpy()

    x_min, x_max = float(x_all.min()), float(args.x_max or x_all.max())
    x_fine = np.arange(x_min, x_max + args.x_step / 2, args.x_step)

    fine_curves = pd.DataFrame({"common_step": x_fine})

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sims)))

    for i, (color, sim) in enumerate(zip(colors, sims)):
        y = y_matrix[:, i]
        mask = ~np.isnan(y)
        if mask.sum() < 4:
            print(f"skip sim={sim:.1f}: only {mask.sum()} data points")
            continue
        x_local = x_all[mask]
        y_local = y[mask]
        try:
            L, k, x0, C = fit_one(x_local, y_local)
        except Exception as e:
            print(f"skip sim={sim:.1f}: fit failed ({e})")
            continue

        ax.scatter(x_local, y_local, color=color, s=30)
        y_fit_local = sigmoid_func(x_local, L, k, x0, C)
        ax.plot(x_local, y_fit_local, color=color, alpha=0.6, label=f"sim={sim:.1f}")

        fine_curves[f"sim={sim:.1f}"] = sigmoid_func(x_fine, L, k, x0, C)
        print(f"sim={sim:.1f}: L={L:.6f} k={k:.6f} x0={x0:.6f} C={C:.6f}")

    ax.set_xlabel("Common Inference Step")
    ax.set_ylabel(args.sheet)
    ax.set_title(f"{args.sheet} (from {in_path.name})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"saved {plot_path}")

    fine_curves.to_excel(curves_path, index=False)
    print(f"saved {curves_path}")


if __name__ == "__main__":
    main()
