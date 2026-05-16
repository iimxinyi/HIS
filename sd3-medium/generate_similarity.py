"""Generate similarity-experiment images on SD3-medium.

For each (public_prompt_idx, personal_prompt_idx, common_step) the script
runs the public phase to produce the shared latents, then the personal phase
to produce the final image. Output PNGs that already exist are skipped, so
the script is restartable.

Usage:
    python generate_similarity.py --group 1
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
    p.add_argument("--group", type=int, choices=[1, 2], default=1)
    p.add_argument("--model", default=os.environ.get("SD3_MODEL_PATH", "stabilityai/stable-diffusion-3-medium-diffusers"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="seeds to generate per sample; metrics are averaged across them")
    p.add_argument("--total-step", type=int, default=28)
    p.add_argument("--public-scale", type=float, default=2.0)
    p.add_argument("--personal-scale", type=float, default=8.0)
    args = p.parse_args()

    public_prompts, personal_prompts = load_group(args.group)
    if not public_prompts or not personal_prompts:
        print(f"group {args.group} has empty prompts; fill them in before running")
        return

    out_dir = Path(args.out_dir or (Path(__file__).parent / "results" / "similarity" / f"group{args.group}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    reserve_latents_path()

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")

    total = len(public_prompts) * len(personal_prompts) * args.total_step * len(args.seeds)
    done = 0
    skipped = 0
    for i_pub, public_prompt in enumerate(public_prompts):
        for i_per, personal_prompt in enumerate(personal_prompts):
            for common_step in range(args.total_step):
                for seed in args.seeds:
                    done += 1
                    out_path = out_dir / f"Public{i_pub}_Personal{i_per}_CommonStep{common_step}_Seed{seed}.png"
                    if already_done(out_path):
                        skipped += 1
                        continue

                    generator = seed_everywhere(seed)
                    pipe(
                        prompt=public_prompt,
                        num_inference_steps=args.total_step,
                        guidance_scale=args.public_scale,
                        common_step=common_step,
                        prompt_unchanged=True,
                        generator=generator,
                    )
                    generator = seed_everywhere(seed)
                    image = pipe(
                        prompt=personal_prompt,
                        num_inference_steps=args.total_step,
                        guidance_scale=args.personal_scale,
                        common_step=common_step,
                        prompt_unchanged=False,
                        generator=generator,
                    ).images[0]
                    image.save(ensure_parent(out_path))
                    print(f"[{done}/{total}] saved {out_path.name}")

    print(f"done. generated {done - skipped} new images, skipped {skipped} existing")


if __name__ == "__main__":
    main()
