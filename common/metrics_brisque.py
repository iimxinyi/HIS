"""BRISQUE no-reference image quality score (via piq).

Lower = better quality. Output is in [0, 100] for natural images.
"""

import cv2
import numpy as np
import torch
from piq import brisque
from torchvision import transforms


def score(image_path: str) -> float:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unable to load image: {image_path}")
    img = img.astype(np.float32) / 255.0
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return float(brisque(tensor).item())
