"""Fit sigmoid curves: one per (variant, metric, sentence-similarity bin).

Reads the per-variant Excel produced by evaluate.py for the similarity
experiment, joins each (public, personal) pair to its sentence-similarity
bin (rounded to 0.1), averages the metric across pairs in each bin per
common_step, and fits a sigmoid. Plots + fitted params are written under
results/fitting/group{n}/{variant}/.

Usage:
    python fit_similarity.py --group 1                 # fits both variants
    python fit_similarity.py --group 1 --variants wSIM # one variant only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.fitting import plot_fits
from common.sentence_similarity import compute_similarity_matrix, round_to_bin
from prompts import load_group


VARIANT_CHOICES = ("wSIM", "woSIM")


def fit_one_variant(group: int, variant: str, eval_xlsx: Path, out_dir: Path):
    if not eval_xlsx.exists():
        print(f"skip variant {variant}: {eval_xlsx} does not exist (run evaluate.py first)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    public_prompts, personal_prompts = load_group(group)
    all_prompts = list(public_prompts) + list(personal_prompts)
    sim_matrix = compute_similarity_matrix(all_prompts)

    raw = pd.read_excel(eval_xlsx, sheet_name="raw")

    n_pub = len(public_prompts)

    def bin_for_row(row) -> float:
        i = int(row["public_prompt"])
        j = int(row["personal_prompt"]) + n_pub
        return round_to_bin(float(sim_matrix[i, j]))

    raw["sim_bin"] = raw.apply(bin_for_row, axis=1)

    fit_summary_rows = []
    for metric in ("clip", "image_reward", "brisque", "musiq"):
        grouped = raw.groupby(["sim_bin", "common_step"])[metric].mean().reset_index()
        series: dict[float, np.ndarray] = {}
        x = None
        for sim_bin, df in grouped.groupby("sim_bin"):
            df_sorted = df.sort_values("common_step")
            if len(df_sorted) < 4:
                continue
            series[float(sim_bin)] = df_sorted[metric].to_numpy()
            x = df_sorted["common_step"].to_numpy()
        if not series or x is None:
            print(f"  skip {metric}: not enough data")
            continue
        out_png = out_dir / f"{metric}.png"
        params = plot_fits(
            x=x,
            series=series,
            out_path=out_png,
            title=f"{metric} (group {group}, {variant})",
            ylabel=metric,
        )
        for sim_bin, (L, k, x0, C) in params.items():
            fit_summary_rows.append({
                "metric": metric,
                "sim_bin": sim_bin,
                "L": L,
                "k": k,
                "x0": x0,
                "C": C,
            })
        print(f"  saved {out_png}")

    params_df = pd.DataFrame(fit_summary_rows)
    params_path = out_dir / "params.csv"
    params_df.to_csv(params_path, index=False)
    print(f"  wrote {params_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, choices=[1, 2], default=1)
    p.add_argument("--variants", nargs="+", choices=VARIANT_CHOICES, default=list(VARIANT_CHOICES))
    args = p.parse_args()

    here = Path(__file__).parent
    for variant in args.variants:
        print(f"=== variant {variant} ===")
        eval_xlsx = here / "results" / "eval" / f"similarity_group{args.group}_{variant}.xlsx"
        out_dir = here / "results" / "fitting" / f"group{args.group}" / variant
        fit_one_variant(args.group, variant, eval_xlsx, out_dir)


if __name__ == "__main__":
    main()
