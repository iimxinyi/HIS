import random

import numpy as np
import torch


def seed_everywhere(seed: int) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) and return a CUDA generator."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)
