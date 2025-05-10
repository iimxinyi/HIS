import os
import clip
import torch
import numpy as np
from PIL import Image

# Initialize the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device=device)

def forward_modality(model, preprocess, data, flag):
    """
    Extract features based on data type (image or text).
    :param model: CLIP model
    :param preprocess: Image preprocessing function
    :param data: Input data (image or text)
    :param flag: Data type, 'img' or 'txt'
    :return: Extracted features
    """
    device = next(model.parameters()).device
    if flag == 'img':
        data = preprocess(data).unsqueeze(0)
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        data = clip.tokenize(data)
        features = model.encode_text(data.to(device))
    else:
        raise TypeError("Invalid flag. Use 'img' or 'txt'.")
    return features

@torch.no_grad()
def calculate_clip_score(model, preprocess, first_data, second_data, first_flag='txt', second_flag='img'):
    """
    Compute the CLIP similarity score.
    :param model: CLIP model
    :param preprocess: Image preprocessing function
    :param first_data: First input (text or image)
    :param second_data: Second input (text or image)
    :param first_flag: First input type, 'txt' or 'img'
    :param second_flag: Second input type, 'txt' or 'img'
    :return: CLIP similarity score
    """
    first_features = forward_modality(model, preprocess, first_data, first_flag)
    second_features = forward_modality(model, preprocess, second_data, second_flag)

    # Normalize features
    first_features = first_features / first_features.norm(dim=1, keepdim=True).to(torch.float32)
    second_features = second_features / second_features.norm(dim=1, keepdim=True).to(torch.float32)

    # Compute similarity score
    score = (second_features * first_features).sum()
    return score

def compute_image_text_alignment(image_path, text):
    """
    Calculate the CLIP score between an image and text.
    :param image_path: Path to the image file
    :param text: Text prompt
    :return: CLIP score
    """
    image = Image.open(image_path)
    score = calculate_clip_score(
        model,
        preprocess,
        first_data=text,
        second_data=image,
        first_flag='txt',
        second_flag='img'
    ).cpu().numpy()
    return score

def extract_prompt_index(filename):
    """
    Extract the prompt index from the filename.
    :param filename: Filename string
    :return: Prompt index as integer
    """
    # Assume filename format is scaleX_promptY_seedZ.png
    parts = filename.split('_')
    prompt_part = parts[1]  # Get the 'promptY' part
    prompt_index = int(prompt_part.replace('prompt', ''))  # Extract Y
    return prompt_index

def compute_average_clip_score(image_folder, positive_personal_prompts):
    """
    Compute the average CLIP score for all images in a folder.
    :param image_folder: Path to the image folder
    :param positive_personal_prompts: List of text prompts
    :return: Average CLIP score, or None if no valid images
    """
    image_files = [f for f in os.listdir(image_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    scores = []
    for image_file in image_files:
        try:
            prompt_index = extract_prompt_index(image_file)
            text = positive_personal_prompts[prompt_index]
            image_path = os.path.join(image_folder, image_file)
            score = compute_image_text_alignment(image_path, text)
            scores.append(score)
            print(f"Image: {image_file}, Prompt Index: {prompt_index}, CLIP Score: {score}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    if scores:
        average_score = np.mean(scores)
        print(f"Average CLIP Score for {image_folder}: {average_score}")
        return average_score
    else:
        print(f"No valid images found in {image_folder}.")
        return None

def compute_clip_for_all_folders(base_folder, positive_personal_prompts):
    """
    Compute the average CLIP score for each Guidance_Scale folder.
    :param base_folder: Root directory containing Guidance_Scale folders
    :param positive_personal_prompts: List of text prompts
    :return: Dictionary mapping folder names to average CLIP scores
    """
    results = {}

    # Iterate through folders Guidance_Scale=1 to Guidance_Scale=7
    for i in range(1, 8):
        folder_name = f"Guidance_Scale={i}"
        folder_path = os.path.join(base_folder, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_name} does not exist, skipping")
            results[folder_name] = None
            continue
        
        print(f"Processing folder: {folder_name}")
        try:
            average_clip_score = compute_average_clip_score(folder_path, positive_personal_prompts)
            results[folder_name] = average_clip_score
        except Exception as e:
            print(f"Error computing folder {folder_name}: {e}")
            results[folder_name] = None

    return results

# Example usage
base_folder = "./"  # Replace with the root directory path
positive_personal_prompts = [
    'A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.',
    'A majestic orange cat with a crown, sitting on a throne in a medieval castle, surrounded by intricate tapestries and candlelight.',
    # ... (additional prompts) ...
]

# Compute average CLIP scores for all folders
results = compute_clip_for_all_folders(base_folder, positive_personal_prompts)

# Print the results
print("\nAverage CLIP Scores for all folders:")
for folder_name, score in results.items():
    if score is not None:
        print(f"{folder_name}: {score}")
    else:
        print(f"{folder_name}: Unable to compute")
