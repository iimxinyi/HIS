FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Keep venv outside /workspace so `docker run -v $PWD:/workspace` does not hide it.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_HTTP_TIMEOUT=120 \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    libxcb1 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock* ./

# Lock file optional on first clone; sync resolves from pyproject.toml
RUN uv sync --no-install-project

COPY . .

RUN uv sync --no-install-project

# openai-clip 1.0.1 uses `from pkg_resources import packaging`, which broke
# in setuptools 80+ (pkg_resources removed). The vendored `packaging` is API-
# identical to the standalone PyPI `packaging` package, so swap the import.
RUN sed -i 's/from pkg_resources import packaging/import packaging/' \
    /opt/venv/lib/python3.10/site-packages/clip/clip.py

# ImageReward bundles an old BLIP copy that imports apply_chunking_to_forward
# from transformers.modeling_utils. transformers 4.30+ moved those helpers
# to transformers.pytorch_utils, so we redirect the import.
RUN sed -i '1i from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer' \
        /opt/venv/lib/python3.10/site-packages/ImageReward/models/BLIP/med.py \
 && sed -i '/^\s*apply_chunking_to_forward,\?\s*$/d; /^\s*find_pruneable_heads_and_indices,\?\s*$/d; /^\s*prune_linear_layer,\?\s*$/d' \
        /opt/venv/lib/python3.10/site-packages/ImageReward/models/BLIP/med.py

# ImageReward.__init__ pulls in the ReFL training module, which depends on
# old `datasets` + new `pyarrow` (broken). We only need .score() for inference.
RUN sed -i '/^from \.ReFL import \*/d' /opt/venv/lib/python3.10/site-packages/ImageReward/__init__.py

# ImageReward's BLIP needs the bert-base-uncased tokenizer. Inside the container
# we have no HF access, so the user pre-downloads tokenizer files to
# pretrained/bert-base-uncased/ on the host and we redirect the hardcoded id.
RUN sed -i "s|'bert-base-uncased'|'/workspace/pretrained/bert-base-uncased'|g; \
            s|\"bert-base-uncased\"|\"/workspace/pretrained/bert-base-uncased\"|g" \
    /opt/venv/lib/python3.10/site-packages/ImageReward/models/BLIP/blip.py

# Replace bundled SD3 + FLUX pipelines with this repository's modified versions
RUN /opt/venv/bin/python - <<'PY'
import pathlib
import shutil

import diffusers

root = pathlib.Path(diffusers.__file__).resolve().parent
patches = [
    ("/workspace/sd3-medium/pipeline_stable_diffusion_3.py", "pipeline_stable_diffusion_3.py"),
    ("/workspace/flux.1-dev/pipeline_flux.py", "pipeline_flux.py"),
]
for src_path, dst_name in patches:
    src = pathlib.Path(src_path)
    matches = sorted(root.glob(f"**/{dst_name}"))
    if not matches:
        raise SystemExit(f"diffusers: {dst_name} not found")
    dst = matches[0]
    shutil.copy(src, dst)
    print("patched", dst)
PY

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
CMD ["bash"]
