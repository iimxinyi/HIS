import os
import glob
import pandas as pd
import numpy as np
import clip
import torch
from PIL import Image

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device=device)

personal_prompts = [
    'A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur.',
    'A beautifully detailed dog with expressive eyes and a unique coat stands in a scenic natural setting.',
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

def forward_modality(model, preprocess, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        data = preprocess(data).unsqueeze(0)
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        data = clip.tokenize(data)
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features

@torch.no_grad()
def calculate_clip_score(model, preprocess, first_data, second_data, first_flag='txt', second_flag='img'):
    first_features = forward_modality(model, preprocess, first_data, first_flag)
    second_features = forward_modality(model, preprocess, second_data, second_flag)

    first_features = first_features / first_features.norm(dim=1, keepdim=True).to(torch.float32)
    second_features = second_features / second_features.norm(dim=1, keepdim=True).to(torch.float32)

    return (second_features * first_features).sum().item()

def process_images(image_folder, output_file):
    # Create full index matrix
    public_range = range(22)  # 0-21
    personal_range = range(22)
    steps = range(28)  # 0-27

    # Create DataFrame with all combinations
    index = pd.MultiIndex.from_product(
        [public_range, personal_range],
        names=['Public Prompt', 'Personal Prompt']
    )
    df = pd.DataFrame(
        np.nan,
        index=index,
        columns=steps
    ).reset_index()

    i = 0
    # Process image files
    for img_path in glob.glob(os.path.join(image_folder, 'Public*_Personal*_CommonStep*.png')):
        try:
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            x = int(parts[0][6:])      # Number after 'Public'
            y = int(parts[1][8:])      # Number after 'Personal'
            z = int(parts[2][10:-4])   # Number after 'CommonStep'
        except:
            continue

        # Calculate CLIP score
        image = Image.open(img_path)
        text = personal_prompts[y]
        score = calculate_clip_score(model, preprocess, text, image)
        print(f"{i}/{13552}")
        i += 1

        # Locate corresponding row and column
        mask = (df['Public Prompt'] == x) & (df['Personal Prompt'] == y)
        df.loc[mask, z] = score

    # Sort by convention
    df = df.sort_values(by=['Public Prompt', 'Personal Prompt'])

    # Save results
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    process_images(
        image_folder="./",
        output_file="results-clip.xlsx"
    )
