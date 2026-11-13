"""Small artifact path helpers."""

from pathlib import Path


def get_latest_model(model_name: str, artifacts_dir: str | Path | None = None) -> Path:
    """Return the newest matching model artifact, or raise if none exists."""
    directory = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else Path(__file__).resolve().parents[2] / "artifacts"
    )
    model_files = list(directory.glob(f"*{model_name}*.bin"))
    if not model_files:
        raise FileNotFoundError(f"No model found for {model_name!r} in {directory}")
    return max(model_files, key=lambda file: file.stat().st_mtime)
