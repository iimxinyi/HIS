import cv2
import numpy as np
from piq import brisque
import torch
from torchvision import transforms


def compute_brisque(image_path):
    """
    Compute the BRISQUE score of an image
    :param image_path: Path to the image
    :return: BRISQUE score
    """
    # Read the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to load image, please check if the path is correct")

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert to PyTorch tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Calculate the BRISQUE score
    brisque_score = brisque(image_tensor)
    
    return brisque_score.item()


image_path = "scale6_prompt1_seed3.png"  # Replace with your image path
score = compute_brisque(image_path)
print(f"BRISQUE Quality Score: {score}")
