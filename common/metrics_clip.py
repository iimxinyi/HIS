"""CLIP image-text alignment score (ViT-L/14@336px).

Higher = better alignment. Output is the cosine similarity in [-1, 1].
"""

from typing import Union

import clip
import torch
from PIL import Image

_model = None
_preprocess = None


def _load(device: str | None = None):
    global _model, _preprocess
    if _model is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _model, _preprocess = clip.load("ViT-L/14@336px", device=device)
        _model.eval()
    return _model, _preprocess


@torch.no_grad()
def score(image: Union[str, Image.Image], prompt: str, device: str | None = None) -> float:
    model, preprocess = _load(device)
    dev = next(model.parameters()).device

    if isinstance(image, (str,)):
        image = Image.open(image).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(dev)
    txt_tensor = clip.tokenize(prompt, truncate=True).to(dev)

    img_feat = model.encode_image(img_tensor)
    txt_feat = model.encode_text(txt_tensor)

    img_feat = img_feat / img_feat.norm(dim=1, keepdim=True).to(torch.float32)
    txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True).to(torch.float32)

    return float((img_feat * txt_feat).sum().item())
