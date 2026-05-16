FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./

# Lock file optional on first clone; sync resolves from pyproject.toml
RUN uv sync --no-install-project

COPY . .

RUN uv sync --no-install-project

# Replace bundled SD3 + FLUX pipelines with this repository's modified versions
RUN /app/.venv/bin/python - <<'PY'
import pathlib
import shutil

import diffusers

root = pathlib.Path(diffusers.__file__).resolve().parent
patches = [
    ("/app/sd3-medium/pipeline_stable_diffusion_3.py", "pipeline_stable_diffusion_3.py"),
    ("/app/flux.1-dev/pipeline_flux.py", "pipeline_flux.py"),
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

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app
CMD ["bash"]
