"""SIM experiment on SD3-medium: sweep the public-prompt guidance scale.

Generates `scale{i}_prompt{j}_seed{s}.png` for i in [0..7], j in [0..N-1],
seed in [1..3]. Uses Group 1 only (per the experimental design). Restartable
- existing PNGs are skipped.

Usage:
    python generate_sim.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.io import already_done, ensure_parent, reserve_latents_path
from common.seed import seed_everywhere
from prompts import load_group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("SD3_MODEL_PATH", "stabilityai/stable-diffusion-3-medium-diffusers"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--total-step", type=int, default=28)
    p.add_argument("--common-step", type=int, default=3)
    p.add_argument("--personal-scale", type=float, default=8.0)
    p.add_argument("--public-scales", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    public_prompts, personal_prompts = load_group(1)
    if not public_prompts or not personal_prompts:
        print("group 1 is empty; cannot run SIM")
        return

    out_root = Path(args.out_dir or (Path(__file__).parent / "results" / "sim"))
    out_root.mkdir(parents=True, exist_ok=True)

    reserve_latents_path()

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")

    # Convention from the original code: first half of personal prompts use
    # public_prompts[0], second half use public_prompts[1].
    half = len(personal_prompts) // 2

    total = len(args.public_scales) * len(personal_prompts) * len(args.seeds)
    done = 0
    skipped = 0
    for scale in args.public_scales:
        scale_dir = out_root / f"Guidance_Scale={scale}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        for j, personal_prompt in enumerate(personal_prompts):
            public_prompt = public_prompts[0] if j < half else public_prompts[min(1, len(public_prompts) - 1)]
            for seed in args.seeds:
                done += 1
                out_path = scale_dir / f"scale{scale}_prompt{j}_seed{seed}.png"
                if already_done(out_path):
                    skipped += 1
                    continue

                generator = seed_everywhere(seed)
                pipe(
                    prompt=public_prompt,
                    num_inference_steps=args.total_step,
                    guidance_scale=float(scale),
                    common_step=args.common_step,
                    prompt_unchanged=True,
                    generator=generator,
                )
                generator = seed_everywhere(seed)
                image = pipe(
                    prompt=personal_prompt,
                    num_inference_steps=args.total_step,
                    guidance_scale=args.personal_scale,
                    common_step=args.common_step,
                    prompt_unchanged=False,
                    generator=generator,
                ).images[0]
                image.save(ensure_parent(out_path))
                print(f"[{done}/{total}] saved {out_path.relative_to(out_root)}")

    print(f"done. generated {done - skipped} new images, skipped {skipped} existing")


if __name__ == "__main__":
    main()
