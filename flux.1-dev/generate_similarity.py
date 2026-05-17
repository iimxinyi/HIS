"""Generate similarity-experiment images on FLUX.1-dev.

Runs two variants:
  * wSIM : public_scale != personal_scale
  * woSIM: public_scale == personal_scale (baseline)

Each variant has its own subdirectory and is evaluated / fitted separately.

The public denoising trajectory depends only on (public_prompt, seed, variant),
so the patched pipeline saves latents at every step during a single public-phase
call; the public phase runs once per (variant, public, seed) and the inner
(personal, common_step) loop reuses the saved checkpoints.

Resumable: skips per-block when every output PNG already exists, and skips
individual personal-phase images that exist.

Usage:
    python generate_similarity.py --group 1
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import FluxPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.io import already_done, ensure_parent, reserve_latents_path
from common.seed import seed_everywhere
from prompts import load_group


VARIANT_CHOICES = ("wSIM", "woSIM")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, choices=[1, 2], default=1)
    p.add_argument("--model", default=os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="seeds to generate per sample; metrics are averaged across them")
    p.add_argument("--total-step", type=int, default=28)
    p.add_argument("--public-scale", type=float, default=2.0,
                   help="public-phase guidance scale for the wSIM variant")
    p.add_argument("--personal-scale", type=float, default=3.5,
                   help="personal-phase guidance scale (also used as public_scale for woSIM)")
    p.add_argument("--variants", nargs="+", choices=VARIANT_CHOICES, default=list(VARIANT_CHOICES))
    args = p.parse_args()

    public_prompts, personal_prompts = load_group(args.group)
    if not public_prompts or not personal_prompts:
        print(f"group {args.group} has empty prompts; fill them in before running")
        return

    base_out = Path(args.out_dir or (Path(__file__).parent / "results" / "similarity" / f"group{args.group}"))

    latents_base = reserve_latents_path()

    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    variant_scales = {
        "wSIM": args.public_scale,
        "woSIM": args.personal_scale,
    }

    total_per_variant = len(public_prompts) * len(personal_prompts) * args.total_step * len(args.seeds)
    total = total_per_variant * len(args.variants)
    done = 0
    skipped = 0
    public_runs = 0

    for variant in args.variants:
        public_scale = variant_scales[variant]
        out_dir = base_out / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== variant {variant}: public_scale={public_scale}, personal_scale={args.personal_scale} ===")

        for i_pub, public_prompt in enumerate(public_prompts):
            for seed in args.seeds:
                block_outputs = [
                    out_dir / f"Public{i_pub}_Personal{i_per}_CommonStep{k}_Seed{seed}.png"
                    for i_per in range(len(personal_prompts))
                    for k in range(args.total_step)
                ]
                if all(already_done(p) for p in block_outputs):
                    done += len(block_outputs)
                    skipped += len(block_outputs)
                    continue

                generator = seed_everywhere(seed)
                pipe(
                    prompt=public_prompt,
                    num_inference_steps=args.total_step,
                    guidance_scale=public_scale,
                    common_step=0,
                    prompt_unchanged=True,
                    generator=generator,
                )
                public_runs += 1
                print(f"[public {public_runs}] variant={variant} public_prompt={i_pub} seed={seed}")

                for i_per, personal_prompt in enumerate(personal_prompts):
                    for common_step in range(args.total_step):
                        done += 1
                        out_path = out_dir / f"Public{i_pub}_Personal{i_per}_CommonStep{common_step}_Seed{seed}.png"
                        if already_done(out_path):
                            skipped += 1
                            continue
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
                        print(f"[{done}/{total}] saved {variant}/{out_path.name}")

                for k in range(args.total_step):
                    p_step = Path(f"{latents_base}.step{k}")
                    if p_step.exists():
                        p_step.unlink()

    print(f"done. generated {done - skipped} new images, skipped {skipped} existing, "
          f"public phase ran {public_runs} times")


if __name__ == "__main__":
    main()
