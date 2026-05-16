# Pretrained model weights

Drop the following checkpoints into this folder before running evaluation.

## MUSIQ (objective image quality)

File: `musiq_spaq_ckpt-358bb6af.pth` (≈ 109 MB)

Copy from `~/Desktop/bench/pretrained/musiq_spaq_ckpt-358bb6af.pth` to
`./pretrained/musiq_spaq_ckpt-358bb6af.pth`.

The MUSIQ score is loaded via `pyiqa.archs.musiq_arch.MUSIQ(pretrained_model_path=...)`
inside `common/metrics_musiq.py`.

## ImageReward (subjective image-text alignment)

No manual setup required - the `image-reward` package downloads weights
(`ImageReward-v1.0`) into `~/.cache/huggingface/` on first use.
