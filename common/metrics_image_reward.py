"""ImageReward score (BLIP-based, trained on human preferences).

Higher = better image-text alignment / preference. Output is a real number
roughly in [-3, 3]; positive values indicate the image is preferred for the
given prompt.
"""

from pathlib import Path
from typing import Union

from PIL import Image

from common.pretrained_util import PRETRAINED_DIR, configure_hf_cache

_IMAGEREWARD_DIR = PRETRAINED_DIR / "ImageReward-v1.0"

_model = None


def _load():
    global _model
    if _model is None:
        import ImageReward as RM

        configure_hf_cache()
        _IMAGEREWARD_DIR.mkdir(parents=True, exist_ok=True)
        _model = RM.load("ImageReward-v1.0", download_root=str(_IMAGEREWARD_DIR))
    return _model


def score(image: Union[str, Image.Image], prompt: str) -> float:
    model = _load()
    # ImageReward.score accepts either an image path (str) or a PIL.Image.
    # PIL is preferred when we already have it open.
    return float(model.score(prompt, image))


def score_batch(images: list, prompts: list) -> list:
    model = _load()
    return [float(model.score(p, img)) for img, p in zip(images, prompts)]
