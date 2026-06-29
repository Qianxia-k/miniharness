"""Shared utilities for spawning MiniHarness teammate processes."""

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any
from pathlib import Path


TEAMMATE_COMMAND_ENV_VAR = "MINIHARNESS_TEAMMATE_COMMAND"
AGENT_HOOKS_ENV_VAR = "MINIHARNESS_AGENT_HOOKS_JSON"

_TEAMMATE_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "DASHSCOPE_API_KEY",
    "MINIHARNESS_API_KEY",
    "MINIHARNESS_BASE_URL",
    "MINIHARNESS_MODEL",
    "MINIHARNESS_PROFILE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "CURL_CA_BUNDLE",
]


def get_teammate_command() -> str:
    """Return the executable used to spawn teammate processes."""
    override = os.environ.get(TEAMMATE_COMMAND_ENV_VAR)
    if override:
        return override
    if sys.executable:
        return sys.executable
    entry_point = shutil.which("miniharness") or shutil.which("mh")
    if entry_point:
        return entry_point
    return "python"


def build_teammate_argv(
    *,
    cwd: str | Path,
    model: str | None = None,
    system_prompt: str | None = None,
    system_prompt_mode: str | None = None,
) -> list[str]:
    """Build the default direct-exec argv for a MiniHarness task worker."""
    command = get_teammate_command()
    resolved_cwd = str(Path(cwd).expanduser().resolve())
    if _looks_like_python(command):
        argv = [command, "-m", "miniharness", "--cwd", resolved_cwd, "--task-worker"]
    else:
        argv = [command, "--cwd", resolved_cwd, "--task-worker"]
    if model and model != "inherit":
        argv.extend(["--model", model])
    if system_prompt:
        flag = "--system-prompt" if system_prompt_mode == "replace" else "--append-system-prompt"
        argv.extend([flag, system_prompt])
    return argv


def build_inherited_env_vars() -> dict[str, str]:
    """Build environment variables forwarded to spawned teammates."""
    env = {"MINIHARNESS_AGENT_TEAMS": "1"}
    for key in _TEAMMATE_ENV_VARS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def encode_agent_hooks_env(hooks: dict[str, Any] | None) -> dict[str, str]:
    """Encode session-scoped agent hooks for a spawned worker process."""
    if not hooks:
        return {}
    return {
        AGENT_HOOKS_ENV_VAR: json.dumps(
            hooks,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    }


def _looks_like_python(command: str) -> bool:
    lowered = command.lower()
    return (
        lowered.endswith("python")
        or lowered.endswith("python3")
        or lowered.endswith("python.exe")
        or lowered.endswith("python3.exe")
        or "/python" in lowered
        or "\\python" in lowered
    )
