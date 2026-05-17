"""Shared paths and Hugging Face cache under ./pretrained/."""

import os
from pathlib import Path

PRETRAINED_DIR = Path(__file__).resolve().parent.parent / "pretrained"
HF_CACHE_DIR = PRETRAINED_DIR / "huggingface"


def configure_hf_cache() -> Path:
    """Keep huggingface_hub downloads under ./pretrained/ (not ~/.cache/huggingface/)."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    return PRETRAINED_DIR
