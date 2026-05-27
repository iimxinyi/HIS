"""BRISQUE no-reference image quality score (via piq).

Lower = better quality. Output is in [0, 100] for natural images.
"""

# Side effect: sets TORCH_HOME so piq's torch.hub download lands under
# pretrained/torch/ instead of /root/.cache/torch/.
from common import pretrained_util  # noqa: F401

import cv2
import numpy as np
import torch
from piq import brisque
from torchvision import transforms


_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def score(image_path: str) -> float:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unable to load image: {image_path}")
    img = img.astype(np.float32) / 255.0
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(_get_device())
    return float(brisque(tensor).item())


def score_pil(image) -> float:
    img = np.array(image.convert("L")).astype(np.float32) / 255.0
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(_get_device())
    return float(brisque(tensor).item())


def score_batch(images: list) -> list:
    return [score_pil(img) for img in images]
