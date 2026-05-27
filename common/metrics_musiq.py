"""MUSIQ no-reference image quality score (via pyiqa, SPAQ checkpoint).

Higher = better quality. Output is normalised to roughly [0, 1] by dividing
the raw model output by 100.
"""

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

from common.pretrained_util import PRETRAINED_DIR, configure_hf_cache

_CKPT_NAME = "musiq_spaq_ckpt-358bb6af.pth"
_HF_REPO = "chaofengc/IQA-PyTorch-Weights"
_DEFAULT_CKPT = PRETRAINED_DIR / _CKPT_NAME

_model = None
_device = None


def _ensure_ckpt() -> Path:
    configure_hf_cache()
    if _DEFAULT_CKPT.exists():
        return _DEFAULT_CKPT
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=_HF_REPO,
        filename=_CKPT_NAME,
        local_dir=str(PRETRAINED_DIR),
    )
    if not _DEFAULT_CKPT.exists():
        raise FileNotFoundError(
            f"MUSIQ checkpoint download failed; expected {_DEFAULT_CKPT}"
        )
    return _DEFAULT_CKPT


def _load(device: str | None = None, ckpt_path: str | None = None):
    global _model, _device
    if _model is None:
        from pyiqa.archs.musiq_arch import MUSIQ

        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = ckpt_path or str(_ensure_ckpt())
        print(f"Loading MUSIQ from {path} on {_device} ...")
        _model = MUSIQ(pretrained_model_path=path).to(_device).eval()
        print("MUSIQ loaded")
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


@torch.no_grad()
def score_batch(images: list, device: str | None = None) -> list:
    model, dev = _load(device)
    batch = torch.cat([_prep(img) for img in images], dim=0).to(dev)
    raw = model(batch)
    if raw.dim() == 0:
        return [float(raw) / 100.0]
    return [float(r) / 100.0 for r in raw]
