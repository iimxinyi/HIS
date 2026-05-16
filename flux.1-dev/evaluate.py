"""Run all 4 metrics (CLIP, ImageReward, BRISQUE, MUSIQ) on a results dir.

Usage:
    python evaluate.py --exp similarity --group 1
    python evaluate.py --exp sim
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common import metrics_brisque, metrics_clip, metrics_image_reward, metrics_musiq
from prompts import load_group


SIMILARITY_RE = re.compile(r"^Public(\d+)_Personal(\d+)_CommonStep(\d+)\.png$")
SIM_RE = re.compile(r"^scale(\d+)_prompt(\d+)_seed(\d+)\.png$")


def evaluate_similarity(group: int, root: Path, out: Path):
    _, personal_prompts = load_group(group)
    if not personal_prompts:
        raise SystemExit(f"group {group} personal prompts are empty")

    rows = []
    files = sorted(root.glob("Public*_Personal*_CommonStep*.png"))
    for k, img_path in enumerate(files, start=1):
        m = SIMILARITY_RE.match(img_path.name)
        if not m:
            continue
        i_pub, i_per, common_step = (int(g) for g in m.groups())
        prompt = personal_prompts[i_per]
        rows.append({
            "public_prompt": i_pub,
            "personal_prompt": i_per,
            "common_step": common_step,
            "clip": metrics_clip.score(str(img_path), prompt),
            "image_reward": metrics_image_reward.score(str(img_path), prompt),
            "brisque": metrics_brisque.score(str(img_path)),
            "musiq": metrics_musiq.score(str(img_path)),
            "filename": img_path.name,
        })
        if k % 25 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] {img_path.name}")

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="raw", index=False)
        for metric in ("clip", "image_reward", "brisque", "musiq"):
            pivot = df.pivot_table(
                values=metric,
                index=["public_prompt", "personal_prompt"],
                columns="common_step",
                aggfunc="mean",
            )
            pivot.to_excel(w, sheet_name=metric)
    print(f"wrote {out}")


def evaluate_sim(root: Path, out: Path):
    _, personal_prompts = load_group(1)
    if not personal_prompts:
        raise SystemExit("group 1 personal prompts are empty")

    rows = []
    files = sorted(root.glob("Guidance_Scale=*/scale*_prompt*_seed*.png"))
    for k, img_path in enumerate(files, start=1):
        m = SIM_RE.match(img_path.name)
        if not m:
            continue
        scale, prompt_idx, seed = (int(g) for g in m.groups())
        prompt = personal_prompts[prompt_idx]
        rows.append({
            "scale": scale,
            "prompt": prompt_idx,
            "seed": seed,
            "clip": metrics_clip.score(str(img_path), prompt),
            "image_reward": metrics_image_reward.score(str(img_path), prompt),
            "brisque": metrics_brisque.score(str(img_path)),
            "musiq": metrics_musiq.score(str(img_path)),
            "filename": img_path.name,
        })
        if k % 25 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] {img_path.name}")

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="raw", index=False)
        for metric in ("clip", "image_reward", "brisque", "musiq"):
            pivot = df.pivot_table(values=metric, index="prompt", columns="scale", aggfunc="mean")
            pivot.to_excel(w, sheet_name=metric)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["similarity", "sim"], required=True)
    p.add_argument("--group", type=int, choices=[1, 2], default=1, help="similarity only")
    p.add_argument("--root", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    here = Path(__file__).parent
    if args.exp == "similarity":
        root = Path(args.root or here / "results" / "similarity" / f"group{args.group}")
        out = Path(args.out or here / "results" / "eval" / f"similarity_group{args.group}.xlsx")
        evaluate_similarity(args.group, root, out)
    else:
        root = Path(args.root or here / "results" / "sim")
        out = Path(args.out or here / "results" / "eval" / "sim.xlsx")
        evaluate_sim(root, out)


if __name__ == "__main__":
    main()
