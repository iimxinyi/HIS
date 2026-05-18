"""Generate similarity-experiment images on SD3-medium.

Two variants (per existing code):
  * wSIM : public_scale != personal_scale
  * woSIM: public_scale == personal_scale (baseline)

Two anchoring modes (run both by default; resume-safe):
  * public-anchored   : public_prompts[i] anchors common phase, then every personal[j]
                         swaps in as personal phase. Filenames:
                         Public{i}_Personal{j}_CommonStep{k}_Seed{s}.png
  * personal-anchored : personal_prompts[i] anchors common phase, then every OTHER
                         personal_prompts[j] (j != i) swaps in. Filenames:
                         PAnchor{i}_POther{j}_CommonStep{k}_Seed{s}.png

The patched pipeline saves latents at every step during a single public-phase
call, so each (anchor, seed) pair runs the common phase only once and the
inner (target, common_step) loop reuses the saved checkpoints.

Resumable: skips per-block when every output PNG already exists.

Usage:
    python generate_similarity.py --group 1                              # both variants, both modes
    python generate_similarity.py --group 1 --modes personal-anchored    # only the new mode
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


VARIANT_CHOICES = ("wSIM", "woSIM")
MODE_CHOICES = ("public-anchored", "personal-anchored")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, choices=[1, 2], default=2)
    p.add_argument("--model", default=os.environ.get("SD3_MODEL_PATH", "stabilityai/stable-diffusion-3-medium-diffusers"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="seeds to generate per sample; metrics are averaged across them")
    p.add_argument("--total-step", type=int, default=28)
    p.add_argument("--public-scale", type=float, default=2.0,
                   help="anchor-phase guidance scale for wSIM (common phase)")
    p.add_argument("--personal-scale", type=float, default=7.0,
                   help="personal-phase guidance scale (also used as anchor scale for woSIM)")
    p.add_argument("--variants", nargs="+", choices=VARIANT_CHOICES, default=list(VARIANT_CHOICES))
    p.add_argument("--modes", nargs="+", choices=MODE_CHOICES, default=list(MODE_CHOICES),
                   help="which anchoring modes to run; defaults to both")
    p.add_argument("--cross-anchors", type=int, nargs="+", default=None,
                   help="subset of personal indices to use as anchors in personal-anchored mode "
                        "(default: all personal prompts)")
    args = p.parse_args()

    public_prompts, personal_prompts = load_group(args.group)
    if not public_prompts or not personal_prompts:
        print(f"group {args.group} has empty prompts; fill them in before running")
        return

    base_out = Path(args.out_dir or (Path(__file__).parent / "results" / "similarity" / f"group{args.group}"))

    latents_base = reserve_latents_path()

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")

    variant_scales = {
        "wSIM": args.public_scale,
        "woSIM": args.personal_scale,
    }

    cross_anchors = args.cross_anchors if args.cross_anchors is not None else list(range(len(personal_prompts)))

    def cleanup_steps():
        for k in range(args.total_step):
            p_step = Path(f"{latents_base}.step{k}")
            if p_step.exists():
                p_step.unlink()

    def run_block(anchor_prompt, anchor_scale, targets, name_for_log, filename_pattern, seed):
        """One common-phase call followed by all target personal-phase calls."""
        nonlocal done, skipped, public_runs

        block_outputs = [
            out_dir / filename_pattern.format(target=i_target, common_step=k, seed=seed)
            for i_target, _ in targets
            for k in range(args.total_step)
        ]
        if all(already_done(pp) for pp in block_outputs):
            done += len(block_outputs)
            skipped += len(block_outputs)
            return

        generator = seed_everywhere(seed)
        pipe(
            prompt=anchor_prompt,
            num_inference_steps=args.total_step,
            guidance_scale=anchor_scale,
            common_step=0,  # ignored when prompt_unchanged=True
            prompt_unchanged=True,
            generator=generator,
        )
        public_runs += 1
        print(f"[anchor {public_runs}] {name_for_log}")

        for i_target, target_prompt in targets:
            for common_step in range(args.total_step):
                done += 1
                out_path = out_dir / filename_pattern.format(target=i_target, common_step=common_step, seed=seed)
                if already_done(out_path):
                    skipped += 1
                    continue
                generator = seed_everywhere(seed)
                image = pipe(
                    prompt=target_prompt,
                    num_inference_steps=args.total_step,
                    guidance_scale=args.personal_scale,
                    common_step=common_step,
                    prompt_unchanged=False,
                    generator=generator,
                ).images[0]
                image.save(ensure_parent(out_path))
                print(f"[{done}/{total}] saved {variant}/{out_path.name}")

        cleanup_steps()

    # Pre-count total
    total = 0
    for variant in args.variants:
        for mode in args.modes:
            if mode == "public-anchored":
                total += len(public_prompts) * len(personal_prompts) * args.total_step * len(args.seeds)
            else:
                total += sum(len(personal_prompts) - 1 for i in cross_anchors if i < len(personal_prompts)) * args.total_step * len(args.seeds)

    done = 0
    skipped = 0
    public_runs = 0

    for variant in args.variants:
        anchor_scale = variant_scales[variant]
        out_dir = base_out / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== variant {variant}: anchor_scale={anchor_scale}, personal_scale={args.personal_scale} ===")

        if "public-anchored" in args.modes:
            for i_pub, public_prompt in enumerate(public_prompts):
                for seed in args.seeds:
                    targets = list(enumerate(personal_prompts))
                    pattern = f"Public{i_pub}_Personal{{target}}_CommonStep{{common_step}}_Seed{{seed}}.png"
                    run_block(public_prompt, anchor_scale, targets, f"public={i_pub} seed={seed}", pattern, seed)

        if "personal-anchored" in args.modes:
            for i_anchor in cross_anchors:
                if i_anchor >= len(personal_prompts):
                    continue
                anchor_prompt = personal_prompts[i_anchor]
                for seed in args.seeds:
                    targets = [(j, personal_prompts[j]) for j in range(len(personal_prompts)) if j != i_anchor]
                    pattern = f"PAnchor{i_anchor}_POther{{target}}_CommonStep{{common_step}}_Seed{{seed}}.png"
                    run_block(anchor_prompt, anchor_scale, targets, f"panchor={i_anchor} seed={seed}", pattern, seed)

    print(f"done. generated {done - skipped} new images, skipped {skipped} existing, "
          f"common phase ran {public_runs} times")


if __name__ == "__main__":
    main()
