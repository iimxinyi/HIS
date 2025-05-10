import os
import cv2
import numpy as np
from piq import brisque
import torch
from torchvision import transforms

def compute_brisque(image_path):
    """
    Compute the BRISQUE score of an image
    :param image_path: Path to the image file
    :return: BRISQUE score
    """
    # Read the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to load image: {image_path}, please check if the path is correct")

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert to PyTorch tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Compute BRISQUE score
    brisque_score = brisque(image_tensor)
    
    return brisque_score.item()

def compute_average_brisque(folder_path):
    """
    Compute the average BRISQUE score of all images in a folder (ignoring zero scores)
    :param folder_path: Path to the folder containing images
    :return: Average BRISQUE score (excluding zero scores), and count of valid images
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)
                  if os.path.splitext(fname)[1].lower() in image_extensions]

    if not image_paths:
        raise ValueError("No supported image files found in the folder")

    brisque_scores = []
    for image_path in image_paths:
        score = compute_brisque(image_path)

        # Keep only valid scores greater than 0
        if score > 0:
            brisque_scores.append(score)
            print(f"Image: {os.path.basename(image_path)}, BRISQUE score: {score}")
        else:
            print(f"Image: {os.path.basename(image_path)}, BRISQUE score is 0 and will be ignored")

    if not brisque_scores:
        raise ValueError("All images have BRISQUE score 0, unable to compute valid average")

    average_score = np.mean(brisque_scores)
    return average_score

def compute_brisque_for_all_folders(base_folder):
    """
    Compute the average BRISQUE score for all 'Guidance_Scale' folders under base_folder
    :param base_folder: Root directory containing Guidance_Scale folders
    :return: Dictionary with folder names and their average BRISQUE scores
    """
    results = {}  # Store results for each folder

    # Iterate through folders Guidance_Scale=1 to Guidance_Scale=7
    for i in range(1, 8):
        folder_name = f"Guidance_Scale={i}"
        folder_path = os.path.join(base_folder, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_name} does not exist, skipping")
            results[folder_name] = None  # Store None if folder not found
            continue
        
        print(f"Processing folder: {folder_name}")
        try:
            average_brisque = compute_average_brisque(folder_path)
            results[folder_name] = average_brisque  # Store result
            print(f"Average BRISQUE score for folder {folder_name}: {average_brisque}")
        except Exception as e:
            print(f"Error computing folder {folder_name}: {e}")
            results[folder_name] = None  # Store None if error occurs

    return results

# Example usage
base_folder = "./"  # Replace with your root directory path
results = compute_brisque_for_all_folders(base_folder)

# Print the average BRISQUE scores for all folders
print("\nAverage BRISQUE scores for all folders:")
for folder_name, score in results.items():
    if score is not None:
        print(f"{folder_name}: {score}")
    else:
        print(f"{folder_name}: Unable to compute")
