"""Shared read-only git helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_git(cwd: Path, *args: str, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    """Run git with prompts disabled and return the completed process."""
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
    }
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def git_text(cwd: Path, *args: str) -> str:
    """Return stripped stdout for a successful git command, else empty."""
    result = run_git(cwd, *args)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def repo_root(cwd: Path) -> Path | None:
    """Return the repository top-level path, or None outside a git repo."""
    result = run_git(cwd, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    return Path(text).resolve()


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    """Format stdout/stderr from a git command."""
    output = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stderr:
        output = f"{output}\n[stderr]\n{stderr}" if output else stderr
    return output or f"git exited with code {result.returncode}"
