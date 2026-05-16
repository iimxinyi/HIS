"""ImageReward score (BLIP-based, trained on human preferences).

Higher = better image-text alignment / preference. Output is a real number
roughly in [-3, 3]; positive values indicate the image is preferred for the
given prompt.
"""

from typing import Union

from PIL import Image

_model = None


def _load():
    global _model
    if _model is None:
        import ImageReward as RM
        _model = RM.load("ImageReward-v1.0")
    return _model


def score(image: Union[str, Image.Image], prompt: str) -> float:
    model = _load()
    # ImageReward.score accepts either an image path (str) or a PIL.Image.
    # PIL is preferred when we already have it open.
    return float(model.score(prompt, image))
