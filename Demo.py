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

# Local path to Stable Diffusion 3
model_directory = "/root/autodl-tmp/Stable-Diffusion-3-Medium"

# Stable Diffusion 3 pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(model_directory, torch_dtype=torch.float16).to("cuda")

# Basic parameters
seed = 2009
seed_everywhere(seed)
total_step = 28  # 28
common_step = 6  # 8
public_scale = 2.0  # 2.0
personal_scale = 8.0  # 8.0
positive_prompts = [
    'A cat in a detailed environment, showcasing its unique personality and actions, with a focus on lighting, colors, and atmosphere to create a visually engaging scene.', # public prompt
    'A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.', # personal prompt 1
    'A majestic orange tabby cat with a crown, sitting on a throne in a medieval castle, surrounded by intricate tapestries and candlelight.', # personal prompt 2
    ]
negative_prompts = [
    'A sleek black cat with green eyes sitting awkwardly on the floor.',  # negative prompt 1
    'A humble gray tabby cat sits on the floor.'  # negative prompt 2
    ]

# Demo
positive_public_prompt1 = positive_prompts[0]
image1 = pipe(prompt=positive_public_prompt1, num_inference_steps=total_step, guidance_scale=public_scale, common_step=common_step, prompt_unchanged=True).images[0]
image1.save("1-1.png")

positive_personal_prompt2 = positive_prompts[2]
negative_personal_prompt2 = negative_prompts[1]
image2 = pipe(prompt=positive_personal_prompt2, num_inference_steps=total_step, guidance_scale=personal_scale, common_step=common_step, prompt_unchanged=False).images[0]
image2.save("2-1.png")
image3 = pipe(prompt=positive_personal_prompt2, negative_prompt=negative_personal_prompt2, num_inference_steps=total_step, guidance_scale=personal_scale, common_step=common_step, prompt_unchanged=False).images[0]
image3.save("2-2.png")
image4 = pipe(prompt=negative_personal_prompt2, num_inference_steps=total_step, guidance_scale=personal_scale, common_step=common_step, prompt_unchanged=False).images[0]
image4.save("2-3.png")

image5 = pipe(prompt=positive_personal_prompt2, num_inference_steps=total_step - common_step, guidance_scale=7.5, common_step=0).images[0]
image5.save("3-1.png")

