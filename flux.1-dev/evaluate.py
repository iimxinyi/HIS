"""Run all 4 metrics (CLIP, ImageReward, BRISQUE, MUSIQ) on a results dir.

Usage:
    c
    python evaluate.py --exp similarity --group 1 --variants wSIM
    python evaluate.py --exp sim
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common import metrics_brisque, metrics_clip, metrics_image_reward, metrics_musiq
from prompts import load_group


PUBLIC_ANCHORED_RE = re.compile(r"^Public(\d+)_Personal(\d+)_CommonStep(\d+)_Seed(\d+)\.png$")
PERSONAL_ANCHORED_RE = re.compile(r"^PAnchor(\d+)_POther(\d+)_CommonStep(\d+)_Seed(\d+)\.png$")
SIM_RE = re.compile(r"^scale(\d+(?:\.\d+)?)_step(\d+)_prompt(\d+)_seed(\d+)\.png$")
VARIANT_CHOICES = ("wSIM", "woSIM")

BATCH_SIZE = 32
CHECKPOINT_EVERY = 10  # save checkpoint every N batches


def _load_checkpoint(ckpt_path):
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            rows = pickle.load(f)
        done = {r["filename"] for r in rows}
        print(f"  resumed from checkpoint: {len(done)} already scored")
        return rows, done
    return [], set()


def _save_checkpoint(ckpt_path, rows):
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(rows, f)
    tmp.rename(ckpt_path)


def _score_batch(pil_images, prompts):
    clip_scores = metrics_clip.score_batch(pil_images, prompts)
    ir_scores = metrics_image_reward.score_batch(pil_images, prompts)
    brisque_scores = metrics_brisque.score_batch(pil_images)
    musiq_scores = metrics_musiq.score_batch(pil_images)
    return clip_scores, ir_scores, brisque_scores, musiq_scores


def evaluate_similarity(group: int, root: Path, out: Path):
    _, personal_prompts = load_group(group)
    if not personal_prompts:
        raise SystemExit(f"group {group} personal prompts are empty")

    files = sorted(list(root.glob("Public*_Personal*_CommonStep*_Seed*.png"))
                   + list(root.glob("PAnchor*_POther*_CommonStep*_Seed*.png")))
    items = []
    for img_path in files:
        m_pub = PUBLIC_ANCHORED_RE.match(img_path.name)
        m_per = PERSONAL_ANCHORED_RE.match(img_path.name)
        if m_pub:
            i_anchor, i_target, common_step, seed = (int(g) for g in m_pub.groups())
            anchor_kind = "public"
        elif m_per:
            i_anchor, i_target, common_step, seed = (int(g) for g in m_per.groups())
            anchor_kind = "personal"
        else:
            continue
        items.append({
            "img_path": img_path,
            "anchor_kind": anchor_kind,
            "anchor_idx": i_anchor,
            "target_idx": i_target,
            "common_step": common_step,
            "seed": seed,
            "prompt": personal_prompts[i_target],
        })

    ckpt_path = out.with_suffix(".ckpt.pkl")
    rows, done = _load_checkpoint(ckpt_path)
    remaining = [item for item in items if item["img_path"].name not in done]
    total = len(items)

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]

        pil_images = []
        valid = []
        for i, item in enumerate(batch):
            try:
                pil_images.append(Image.open(item["img_path"]).convert("RGB"))
                valid.append(i)
            except Exception as e:
                print(f"  SKIP {item['img_path'].name}: {e}")

        if not pil_images:
            continue

        prompts = [batch[i]["prompt"] for i in valid]
        clip_s, ir_s, bri_s, mus_s = _score_batch(pil_images, prompts)

        for j, idx in enumerate(valid):
            item = batch[idx]
            rows.append({
                "anchor_kind": item["anchor_kind"],
                "anchor_idx": item["anchor_idx"],
                "target_idx": item["target_idx"],
                "common_step": item["common_step"],
                "seed": item["seed"],
                "clip": clip_s[j],
                "image_reward": ir_s[j],
                "brisque": bri_s[j],
                "musiq": mus_s[j],
                "filename": item["img_path"].name,
            })

        progress = len(done) + batch_start + len(valid)
        print(f"  [{progress}/{total}] {batch[valid[-1]]['img_path'].name}")

        batch_num = batch_start // BATCH_SIZE + 1
        if batch_num % CHECKPOINT_EVERY == 0:
            _save_checkpoint(ckpt_path, rows)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="raw", index=False)
        for metric in ("clip", "image_reward", "brisque", "musiq"):
            pivot = df.pivot_table(
                values=metric,
                index=["anchor_kind", "anchor_idx", "target_idx"],
                columns="common_step",
                aggfunc="mean",
            )
            pivot.to_excel(w, sheet_name=metric)
    print(f"wrote {out}")
    if ckpt_path.exists():
        ckpt_path.unlink()


def evaluate_sim(root: Path, out: Path):
    _, personal_prompts = load_group(1)
    if not personal_prompts:
        raise SystemExit("group 1 personal prompts are empty")

    files = sorted(root.glob("Guidance_Scale=*/scale*_step*_prompt*_seed*.png"))
    items = []
    for img_path in files:
        m = SIM_RE.match(img_path.name)
        if not m:
            continue
        scale_str, common_step_str, prompt_idx_str, seed_str = m.groups()
        items.append({
            "img_path": img_path,
            "scale": float(scale_str),
            "common_step": int(common_step_str),
            "prompt_idx": int(prompt_idx_str),
            "seed": int(seed_str),
            "prompt": personal_prompts[int(prompt_idx_str)],
        })

    ckpt_path = out.with_suffix(".ckpt.pkl")
    rows, done = _load_checkpoint(ckpt_path)
    remaining = [item for item in items if item["img_path"].name not in done]
    total = len(items)

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]

        pil_images = []
        valid = []
        for i, item in enumerate(batch):
            try:
                pil_images.append(Image.open(item["img_path"]).convert("RGB"))
                valid.append(i)
            except Exception as e:
                print(f"  SKIP {item['img_path'].name}: {e}")

        if not pil_images:
            continue

        prompts = [batch[i]["prompt"] for i in valid]
        clip_s, ir_s, bri_s, mus_s = _score_batch(pil_images, prompts)

        for j, idx in enumerate(valid):
            item = batch[idx]
            rows.append({
                "scale": item["scale"],
                "common_step": item["common_step"],
                "prompt": item["prompt_idx"],
                "seed": item["seed"],
                "clip": clip_s[j],
                "image_reward": ir_s[j],
                "brisque": bri_s[j],
                "musiq": mus_s[j],
                "filename": item["img_path"].name,
            })

        progress = len(done) + batch_start + len(valid)
        print(f"  [{progress}/{total}] {batch[valid[-1]]['img_path'].name}")

        batch_num = batch_start // BATCH_SIZE + 1
        if batch_num % CHECKPOINT_EVERY == 0:
            _save_checkpoint(ckpt_path, rows)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="raw", index=False)
        for metric in ("clip", "image_reward", "brisque", "musiq"):
            pivot = df.pivot_table(
                values=metric,
                index="common_step",
                columns="scale",
                aggfunc="mean",
            )
            pivot.to_excel(w, sheet_name=metric)
    print(f"wrote {out}")
    if ckpt_path.exists():
        ckpt_path.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["similarity", "sim"], required=True)
    p.add_argument("--group", type=int, choices=[1, 2], default=1, help="similarity only")
    p.add_argument("--variants", nargs="+", choices=VARIANT_CHOICES, default=list(VARIANT_CHOICES),
                   help="similarity only; defaults to both wSIM and woSIM")
    p.add_argument("--root", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    here = Path(__file__).parent
    if args.exp == "similarity":
        for variant in args.variants:
            default_root = here / "results" / "similarity" / f"group{args.group}" / variant
            default_out = here / "results" / "eval" / f"similarity_group{args.group}_{variant}.xlsx"
            root = Path(args.root or default_root) if len(args.variants) == 1 and args.root else default_root
            out = Path(args.out or default_out) if len(args.variants) == 1 and args.out else default_out
            if not root.exists():
                print(f"skip variant {variant}: {root} does not exist")
                continue
            print(f"=== variant {variant}: reading {root} ===")
            evaluate_similarity(args.group, root, out)
    else:
        root = Path(args.root or here / "results" / "sim")
        out = Path(args.out or here / "results" / "eval" / "sim.xlsx")
        evaluate_sim(root, out)


if __name__ == "__main__":
    main()
