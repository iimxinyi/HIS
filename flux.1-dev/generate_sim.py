"""SIM experiment on FLUX.1-dev: sweep the public-prompt guidance scale across
common-inference-step values.

NOTE: FLUX.1-dev is a guidance-distilled model. ``guidance_scale`` is embedded
into the transformer, not implemented as a two-pass classifier-free guidance
like SD3. The SIM scan therefore explores a different mechanism than on SD3;
trends should be interpreted accordingly.

Files land at:

    results/sim/Guidance_Scale={scale}/scale{scale}_step{k}_prompt{j}_seed{s}.png

Restartable - existing PNGs are skipped.

Usage:
    python generate_sim.py
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--total-step", type=int, default=28)
    p.add_argument("--common-steps", type=int, nargs="+",
                   default=[3, 6, 9, 12, 15, 18, 21, 24, 27])
    p.add_argument("--personal-scale", type=float, default=3.5)
    p.add_argument("--public-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    public_prompts, personal_prompts = load_group(1)
    if not public_prompts or not personal_prompts:
        print("group 1 is empty; cannot run SIM")
        return

    out_root = Path(args.out_dir or (Path(__file__).parent / "results" / "sim"))
    out_root.mkdir(parents=True, exist_ok=True)

    latents_base = reserve_latents_path()

    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    half = len(personal_prompts) // 2
    pub_to_personals = {
        0: list(range(half)),
        1: list(range(half, len(personal_prompts))),
    }

    total = (len(args.public_scales) * len(personal_prompts)
             * len(args.seeds) * len(args.common_steps))
    done = 0
    skipped = 0
    public_runs = 0

    for i_pub in (0, 1):
        public_prompt = public_prompts[min(i_pub, len(public_prompts) - 1)]
        j_indices = pub_to_personals[i_pub]
        if not j_indices:
            continue
        for scale in args.public_scales:
            scale_dir = out_root / f"Guidance_Scale={scale}"
            scale_dir.mkdir(parents=True, exist_ok=True)
            for seed in args.seeds:
                block_outputs = [
                    scale_dir / f"scale{scale}_step{k}_prompt{j}_seed{seed}.png"
                    for j in j_indices
                    for k in args.common_steps
                ]
                if all(already_done(p) for p in block_outputs):
                    done += len(block_outputs)
                    skipped += len(block_outputs)
                    continue

                generator = seed_everywhere(seed)
                pipe(
                    prompt=public_prompt,
                    num_inference_steps=args.total_step,
                    guidance_scale=float(scale),
                    common_step=0,
                    prompt_unchanged=True,
                    generator=generator,
                )
                public_runs += 1
                print(f"[public {public_runs}] i_pub={i_pub} scale={scale} seed={seed}")

                for j in j_indices:
                    personal_prompt = personal_prompts[j]
                    for common_step in args.common_steps:
                        done += 1
                        out_path = scale_dir / f"scale{scale}_step{common_step}_prompt{j}_seed{seed}.png"
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
                        print(f"[{done}/{total}] saved {out_path.relative_to(out_root)}")

                for k in range(args.total_step):
                    p_step = Path(f"{latents_base}.step{k}")
                    if p_step.exists():
                        p_step.unlink()

    print(f"done. generated {done - skipped} new images, skipped {skipped} existing, "
          f"public phase ran {public_runs} times")


if __name__ == "__main__":
    main()
