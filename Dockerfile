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

# Replace bundled SD3 pipeline with this repository’s modified version
RUN /app/.venv/bin/python - <<'PY'
import pathlib
import shutil

import diffusers

root = pathlib.Path(diffusers.__file__).resolve().parent
src = pathlib.Path("/app/sd3-medium/pipeline_stable_diffusion_3.py")
matches = sorted(root.glob("**/pipeline_stable_diffusion_3.py"))
if not matches:
    raise SystemExit("diffusers: pipeline_stable_diffusion_3.py not found")
dst = matches[0]
shutil.copy(src, dst)
print("patched", dst)
PY

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app/sd3-medium
CMD ["bash"]
