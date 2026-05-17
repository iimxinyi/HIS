"""Shared paths and download caches under ./pretrained/."""

import os
from pathlib import Path

PRETRAINED_DIR = Path(__file__).resolve().parent.parent / "pretrained"
HF_CACHE_DIR = PRETRAINED_DIR / "huggingface"
TORCH_CACHE_DIR = PRETRAINED_DIR / "torch"

# Redirect torch.hub downloads (piq BRISQUE SVM weights, etc.) to a persistent
# location so they survive container restarts. Set at import time so any code
# that triggers torch.hub picks it up.
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))


def configure_hf_cache() -> Path:
    """Keep huggingface_hub downloads under ./pretrained/ (not ~/.cache/huggingface/)."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    return PRETRAINED_DIR
