"""Atomic file-write helpers for persistent state."""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes, *, mode: int | None = None) -> None:
    """Write bytes atomically using a same-directory temp file and rename."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    target_mode = _resolve_target_mode(dst, mode)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(data)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        _apply_mode(tmp_path, target_mode)
        os.replace(tmp_path, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def atomic_write_text(
    path: str | os.PathLike[str],
    data: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
) -> None:
    """Text variant of :func:`atomic_write_bytes`."""
    atomic_write_bytes(path, data.encode(encoding), mode=mode)


def _resolve_target_mode(path: Path, explicit_mode: int | None) -> int:
    if explicit_mode is not None:
        return explicit_mode
    try:
        st = path.stat()
    except FileNotFoundError:
        current_umask = os.umask(0)
        os.umask(current_umask)
        return 0o666 & ~current_umask
    return stat.S_IMODE(st.st_mode)


def _apply_mode(path: Path, target_mode: int) -> None:
    try:
        os.chmod(path, target_mode)
    except OSError:
        pass
