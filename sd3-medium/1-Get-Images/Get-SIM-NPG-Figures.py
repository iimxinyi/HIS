import os
import torch
import random
import numpy as np
from diffusers import StableDiffusion3Pipeline

# Function to set seed for all random operations
def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return generator

# Local path to Stable Diffusion 3
model_directory = "/root/autodl-tmp/Stable-Diffusion-3-Medium"

# Stable Diffusion 3 pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(model_directory, torch_dtype=torch.float16).to("cuda")

# Basic parameters
total_step = 28  # 28
common_step = 3  # [3 6 9 12 15 18 21]
personal_scale = 8.0  # 8.0
results_dir2 = "Results_wNPG"
results_dir3 = "Results_woNPG"

# Public prompts
public_prompts = [
    'A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur.',
    'A beautifully detailed dog with expressive eyes and a unique coat stands in a scenic natural setting.',
    ]

# prompts
positive_personal_prompts = [
    'A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.',
    'A majestic orange cat with a crown, sitting on a throne in a medieval castle, surrounded by intricate tapestries and candlelight.',
    'A gray cat with green eyes, sitting on a wooden porch, with soft sunlight highlighting its fur and a blurred garden in the background.',
    'A white cat with round, expressive eyes, sitting on a leather armchair in a cozy library filled with books and warm lighting.',
    'A black cat sitting on a stack of old books in a dusty attic, its multicolored fur contrasting with the vintage surroundings, with beams of sunlight filtering through a small window.',
    'A fluffy gray-and-white cat with golden eyes sits curled up on a cozy knitted blanket by a crackling fireplace.',
    'A brown cat with soft fur and green eyes sits calmly on a rustic wooden table in a sunlit kitchen.',
    'A black-and-white cat with curious eyes sits on a wooden porch, surrounded by autumn leaves and soft sunlight.',
    'A sleek brown cat with golden eyes sits on a polished grand piano, its fur reflecting the soft light of the room.',
    'A sleek black cat with yellow eyes sits on a cobblestone street at dusk, its fur glowing under the light of a streetlamp.',
    'A majestic dog with striking blue eyes and a muscular build stands alert on a rocky cliff edge, its thick, wavy fur glowing in the golden hour sunlight.',
    'A graceful dog with silky, well-groomed fur and deep, soulful eyes sits calmly in a sunlit meadow, its alert ears perked up and a subtle smile hinting at its friendly nature.',
    'A lively dog with a lean, athletic physique dashes through a field of tall grass, its wagging tail and bright, inquisitive eyes capturing pure joy as sunlight streams through the blades.',
    'A contented dog with soft, fluffy fur and gentle, half-closed eyes lies comfortably on a cozy couch.',
    'A curious dog with a finely detailed coat marked by subtle brindle patterns carefully sniffs the ground in an autumn forest.',
    'A dog with a sleek, black coat and bright, alert eyes runs through a shallow stream, water splashing around its paws, with sunlight reflecting off the ripples.',
    'A dog with a short, brindle coat and a strong jawline sits by a campfire, its eyes reflecting the flickering flames and ears twitching at the sound of crackling wood.',
    'A dog with a thick, double-layered coat stands in a snowy field, its breath visible in the cold air and snowflakes clinging to its fur, looking intently at something ahead.',
    'A dog with a curly, white coat and a pink nose plays in a field of wildflowers, its tongue out and tail wagging energetically, surrounded by vibrant colors.',
    'A lively dog with a glossy, golden coat and a slightly tilted head looks up with curious eyes, its ears perked and nose twitching, standing in a sunlit garden filled with vibrant flowers.',
    ]
negative_personal_prompts = [
    'A sleek black cat with yellow eyes.',
    'A humble gray cat with a hood.',
    'A white cat with red eyes',
    'A black cat with narrow, dull eyes.',
    'A white cat with solid-colored fur.',
    'A sleek black-and-brown cat with icy blue eyes.',
    'A gray cat with coarse fur and red eyes.',
    'A brown-and-gray cat with dull eyes.',
    'A fluffy gray cat with pale blue eyes.',
    'A fluffy white cat with blue eyes.',
    'A timid dog with dull brown eyes, a slender frame, and short, coarse fur.',
    'A clumsy dog with rough, unkempt fur and shallow, vacant eyes.',
    'A sluggish dog with a stocky, heavy build, a still tail, and dull, indifferent eyes.',
    'A restless dog with coarse, wiry fur and sharp, wide-open eyes.',
    'A indifferent dog with a plain, uniform coat and bold, patchy markings.',
    'A dog with a shaggy, white coat and dull, drowsy eyes.',
    'A dog with a long, solid-colored coat and a delicate muzzle.',
    'A dog with a thin, single-layered coat.',
    'A dog with a straight, black coat and a dark nose.',
    'A sluggish dog with a dull, dark coat and a rigid head.',
    ]

# Two files to save results
os.makedirs(results_dir2, exist_ok=True)
os.makedirs(results_dir3, exist_ok=True)

# SIM with classifier-free guidance
for i in range(0,8):
    public_scale = i
    for j in range(0,20):
        for seed in range (1,4):
            generator = seed_everywhere(seed)
            if 0 <= j <= 9:
                image1 = pipe(prompt=public_prompts[0], num_inference_steps=total_step, guidance_scale=public_scale, common_step=common_step, prompt_unchanged=True, generator=generator, skip=True).images[0]
            else:
                image1 = pipe(prompt=public_prompts[1], num_inference_steps=total_step, guidance_scale=public_scale, common_step=common_step, prompt_unchanged=True, generator=generator, skip=True).images[0]
            image2 = pipe(prompt=positive_personal_prompts[j], negative_prompt=negative_personal_prompts[j], num_inference_steps=total_step, guidance_scale=personal_scale, common_step=common_step, prompt_unchanged=False, generator=generator).images[0]  # with NPG
            image3 = pipe(prompt=positive_personal_prompts[j], num_inference_steps=total_step, guidance_scale=personal_scale, common_step=common_step, prompt_unchanged=False, generator=generator).images[0]  # without NPG
            file_path2 = os.path.join(results_dir2, f"scale{i}_prompt{j}_seed{seed}_wNPG.png")
            file_path3 = os.path.join(results_dir3, f"scale{i}_prompt{j}_seed{seed}_woNPG.png")
            image2.save(file_path2)
            image3.save(file_path3)