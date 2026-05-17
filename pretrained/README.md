# Pretrained model weights

Evaluation metrics download checkpoints into this folder on first use.

## MUSIQ (objective image quality)

On first use, `common/metrics_musiq.py` downloads from
[chaofengc/IQA-PyTorch-Weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights)
to `./pretrained/musiq_spaq_ckpt-358bb6af.pth`.

Loaded via `pyiqa.archs.musiq_arch.MUSIQ(pretrained_model_path=...)`.

## ImageReward (subjective image-text alignment)

On first use, `common/metrics_image_reward.py` downloads `ImageReward-v1.0` from
[THUDM/ImageReward](https://huggingface.co/THUDM/ImageReward) into:

- `./pretrained/ImageReward-v1.0/ImageReward.pt`
- `./pretrained/ImageReward-v1.0/med_config.json`

## Hugging Face cache

Hub downloads for the above use `./pretrained/huggingface/` (not `~/.cache/huggingface/`).
