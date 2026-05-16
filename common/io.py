import os
import tempfile
from pathlib import Path


def already_done(path: str | os.PathLike) -> bool:
    """Return True if a previous run already produced this output file."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def ensure_parent(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def reserve_latents_path() -> str:
    """Create a per-process temp file path for the shared-latents handoff.

    Setting ``HIS_LATENTS_PATH`` lets two pipelines (e.g. SD3 and FLUX) run in
    parallel without colliding on a fixed filename. Returns the path so the
    caller can also clean it up at the end of the run.
    """
    if "HIS_LATENTS_PATH" in os.environ:
        return os.environ["HIS_LATENTS_PATH"]
    fd, path = tempfile.mkstemp(prefix="his_latents_", suffix=".pth")
    os.close(fd)
    os.environ["HIS_LATENTS_PATH"] = path
    return path
