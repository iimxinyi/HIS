"""MUSIQ no-reference image quality score (via pyiqa, SPAQ checkpoint).

Higher = better quality. Output is normalised to roughly [0, 1] by dividing
the raw model output by 100.
"""

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

_DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "pretrained" / "musiq_spaq_ckpt-358bb6af.pth"

_model = None
_device = None


def _load(device: str | None = None, ckpt_path: str | None = None):
    global _model, _device
    if _model is None:
        from pyiqa.archs.musiq_arch import MUSIQ
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = ckpt_path or str(_DEFAULT_CKPT)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"MUSIQ checkpoint not found at {path}. "
                f"Copy musiq_spaq_ckpt-358bb6af.pth into ./pretrained/ first."
            )
        _model = MUSIQ(pretrained_model_path=path).to(_device).eval()
    return _model, _device


def _prep(image: Union[str, Image.Image]) -> torch.Tensor:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    t = transforms.ToTensor()(image).unsqueeze(0)
    _, _, h, w = t.size()
    if max(h, w) > 512:
        scale = 512.0 / max(h, w)
        t = transforms.Resize((int(scale * h), int(scale * w)), antialias=False)(t)
    return t


@torch.no_grad()
def score(image: Union[str, Image.Image], device: str | None = None) -> float:
    model, dev = _load(device)
    tensor = _prep(image).to(dev)
    raw = model(tensor)
    return float(raw) / 100.0
