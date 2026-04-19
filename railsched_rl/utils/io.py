"""File-system helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
