"""Shared storage utilities for the memory system.

Reuses the same project-slug scheme as :mod:`miniharness.sessions.storage`
so every project gets its own isolated memory directory.
"""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path


def _project_slug(cwd: str | Path) -> str:
    """Build a stable directory name for a project directory."""
    resolved = str(Path(cwd).resolve())
    digest = sha1(resolved.encode("utf-8")).hexdigest()[:12]
    name = Path(resolved).name
    return f"{name}-{digest}"


def _memory_root() -> Path:
    """Return (and create) the MiniHarness memory root."""
    path = Path.home() / ".miniharness" / "memory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_memory_dir(cwd: str | Path) -> Path:
    """Return (and create) the per-project memory directory."""
    mem_dir = _memory_root() / _project_slug(cwd)
    mem_dir.mkdir(parents=True, exist_ok=True)
    return mem_dir
