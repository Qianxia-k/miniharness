"""Shared helpers for text-file tools."""

from __future__ import annotations

import difflib
from pathlib import Path

from miniharness.sandbox import is_sandbox_active, validate_sandbox_path


def resolve_path(cwd: Path, candidate: str) -> Path:
    """Resolve a user-provided path against the tool workspace."""
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def validate_write_boundary(path: Path, cwd: Path) -> str | None:
    """Return an error message if a write target is outside the allowed root."""
    if is_sandbox_active():
        allowed, reason = validate_sandbox_path(path, cwd)
        if not allowed:
            return f"Sandbox: {reason}"
        return None

    try:
        path.relative_to(cwd)
    except ValueError:
        return f"Refusing to write outside workspace: {path}"
    return None


def read_existing_text(path: Path) -> tuple[str, str | None]:
    """Read an existing UTF-8 text file and reject directories/binary files."""
    if not path.exists():
        return "", None
    if path.is_dir():
        return "", f"Cannot write directory: {path}"
    raw = path.read_bytes()
    if b"\x00" in raw:
        return "", f"Binary file cannot be edited as text: {path}"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return "", f"File is not valid UTF-8 text: {path}: {exc}"


def compute_diff(filename: str, original: str, updated: str) -> tuple[str, int, int]:
    """Return unified diff text and +/- line counts."""
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=filename,
            tofile=filename,
            lineterm="",
        )
    )
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return "".join(diff_lines), added, removed


def format_diff_permission_prompt(
    *,
    tool_name: str,
    action: str,
    path: Path,
    diff_text: str,
    added: int,
    removed: int,
    max_chars: int = 4000,
) -> str:
    """Format a bounded diff preview for permission prompts."""
    preview = diff_text
    omitted = len(preview) - max_chars
    if omitted > 0:
        preview = preview[:max_chars] + f"\n... ({omitted} diff chars omitted)"
    return (
        f"Allow {tool_name} to {action} {path}? (+{added} -{removed})\n\n"
        f"{preview}"
    )
