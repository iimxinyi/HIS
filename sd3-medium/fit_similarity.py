"""Aggregate similarity-experiment metrics into (sim_bin x common_step) tables.

Reads the per-variant Excel produced by evaluate.py, joins each (anchor, target)
pair to its sentence-similarity bin (rounded to 0.1), averages each metric over
all (anchor, target, seed) tuples that fall into the same bin per common_step,
and writes a single Excel per variant.

Output: results/fitting/group{n}/{variant}_points.xlsx
  Sheets: clip, image_reward, brisque, musiq
  Each sheet: index = sim_bin, columns = common_step, cells = mean metric value

No sigmoid fitting here — use fit_sigmoid.py downstream if you want the curves.

Usage:
    python fit_similarity.py --group 1                # both variants
    python fit_similarity.py --group 1 --variants wSIM
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.sentence_similarity import compute_similarity_matrix, round_to_bin
from prompts import load_group


VARIANT_CHOICES = ("wSIM", "woSIM")


def aggregate_one_variant(group: int, variant: str, eval_xlsx: Path, out_xlsx: Path):
    if not eval_xlsx.exists():
        print(f"skip variant {variant}: {eval_xlsx} does not exist (run evaluate.py first)")
        return

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    public_prompts, personal_prompts = load_group(group)
    n_pub = len(public_prompts)
    # We embed everything in one matrix: [public_0..N-1, personal_0..M-1].
    sim_matrix = compute_similarity_matrix(list(public_prompts) + list(personal_prompts))

    raw = pd.read_excel(eval_xlsx, sheet_name="raw")

    def sim_bin(row) -> float:
        kind = row["anchor_kind"]
        a = int(row["anchor_idx"])
        t = int(row["target_idx"])
        if kind == "public":
            i, j = a, t + n_pub
        else:  # personal
            i, j = a + n_pub, t + n_pub
        return round_to_bin(float(sim_matrix[i, j]))

    raw["sim_bin"] = raw.apply(sim_bin, axis=1)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        for metric in ("clip", "image_reward", "brisque", "musiq"):
            pivot = raw.pivot_table(
                values=metric,
                index="sim_bin",
                columns="common_step",
                aggfunc="mean",
            )
            # Sort sim_bin ascending; sort columns by common_step ascending
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)
            pivot.to_excel(w, sheet_name=metric)
    print(f"wrote {out_xlsx}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, choices=[1, 2], default=1)
    p.add_argument("--variants", nargs="+", choices=VARIANT_CHOICES, default=list(VARIANT_CHOICES))
    args = p.parse_args()

    here = Path(__file__).parent
    for variant in args.variants:
        print(f"=== variant {variant} ===")
        eval_xlsx = here / "results" / "eval" / f"similarity_group{args.group}_{variant}.xlsx"
        out_xlsx = here / "results" / "fitting" / f"group{args.group}" / f"{variant}_points.xlsx"
        aggregate_one_variant(args.group, variant, eval_xlsx, out_xlsx)


if __name__ == "__main__":
    main()
