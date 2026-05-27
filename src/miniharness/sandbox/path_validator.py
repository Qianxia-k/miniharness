"""Filesystem path boundary enforcement.

Mirrors OpenHarness's sandbox/path_validator.py.  Resolves symlinks before
checking that a path falls within the allowed workspace.
"""

from __future__ import annotations

from pathlib import Path


def validate_sandbox_path(
    path: Path,
    cwd: Path,
    extra_allowed: list[str] | None = None,
) -> tuple[bool, str]:
    """Check whether *path* is inside the sandbox boundary.

    Returns ``(True, "")`` if allowed, or ``(False, reason)`` otherwise.

    Unlike a plain ``path.relative_to(cwd)``, this resolves both the candidate
    and the workspace root first, so symlink escapes and ``..`` traversal are
    caught.
    """
    resolved = path.resolve()
    resolved_cwd = cwd.resolve()

    # Primary check: path must be within the project directory.
    try:
        resolved.relative_to(resolved_cwd)
        return True, ""
    except ValueError:
        pass

    # Secondary: extra allowed directories (e.g. /tmp, external data mounts).
    for allowed in extra_allowed or []:
        allowed_path = Path(allowed).expanduser().resolve()
        try:
            resolved.relative_to(allowed_path)
            return True, ""
        except ValueError:
            continue

    return False, f"path {resolved} is outside sandbox boundary ({resolved_cwd})"
