"""Single-image sanity check for the patched FLUX.1-dev pipeline."""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import FluxPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.io import reserve_latents_path
from common.seed import seed_everywhere


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev"))
    p.add_argument("--out", default="demo_flux.png")
    p.add_argument("--seed", type=int, default=203)
    p.add_argument("--total-step", type=int, default=28)
    p.add_argument("--common-step", type=int, default=8)
    p.add_argument("--public-scale", type=float, default=2.0)
    p.add_argument("--personal-scale", type=float, default=3.5)
    args = p.parse_args()

    reserve_latents_path()

    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    public_prompt = "A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur."
    personal_prompt = (
        "A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, "
        "with a serene garden visible through the window."
    )

    generator = seed_everywhere(args.seed)
    pipe(
        prompt=public_prompt,
        num_inference_steps=args.total_step,
        guidance_scale=args.public_scale,
        common_step=args.common_step,
        prompt_unchanged=True,
        generator=generator,
    )

    generator = seed_everywhere(args.seed)
    image = pipe(
        prompt=personal_prompt,
        num_inference_steps=args.total_step,
        guidance_scale=args.personal_scale,
        common_step=args.common_step,
        prompt_unchanged=False,
        generator=generator,
    ).images[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
