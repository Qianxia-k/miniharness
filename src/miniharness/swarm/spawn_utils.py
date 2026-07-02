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
AGENT_TOOL_POLICY_ENV_VAR = "MINIHARNESS_AGENT_TOOL_POLICY_JSON"
AGENT_PERMISSION_MODE_ENV_VAR = "MINIHARNESS_AGENT_PERMISSION_MODE"
AGENT_MAX_TURNS_ENV_VAR = "MINIHARNESS_AGENT_MAX_TURNS"
AGENT_ID_ENV_VAR = "MINIHARNESS_AGENT_ID"
AGENT_NAME_ENV_VAR = "MINIHARNESS_AGENT_NAME"
AGENT_TEAM_ENV_VAR = "MINIHARNESS_AGENT_TEAM"

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


def encode_agent_tool_policy_env(
    *,
    tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
) -> dict[str, str]:
    """Encode session-scoped tool visibility policy for a spawned worker."""
    policy: dict[str, list[str]] = {}
    if tools is not None:
        policy["tools"] = [str(item).strip() for item in tools if str(item).strip()]
    if disallowed_tools:
        policy["disallowed_tools"] = [
            str(item).strip() for item in disallowed_tools if str(item).strip()
        ]
    if not policy:
        return {}
    return {
        AGENT_TOOL_POLICY_ENV_VAR: json.dumps(
            policy,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    }


def encode_agent_permission_mode_env(permission_mode: str | None) -> dict[str, str]:
    """Encode session-scoped permission mode for a spawned worker."""
    mode = _normalize_permission_mode(permission_mode)
    if mode is None:
        return {}
    return {AGENT_PERMISSION_MODE_ENV_VAR: mode}


def encode_agent_max_turns_env(max_turns: int | None) -> dict[str, str]:
    """Encode session-scoped max-turn override for a spawned worker."""
    if max_turns is None:
        return {}
    try:
        value = int(max_turns)
    except (TypeError, ValueError):
        return {}
    if value < 1:
        return {}
    return {AGENT_MAX_TURNS_ENV_VAR: str(value)}


def encode_agent_identity_env(
    *,
    agent_id: str,
    agent_name: str,
    team: str,
) -> dict[str, str]:
    """Encode delegated-agent identity for cross-process coordination."""
    return {
        AGENT_ID_ENV_VAR: agent_id,
        AGENT_NAME_ENV_VAR: agent_name,
        AGENT_TEAM_ENV_VAR: team,
    }


def _normalize_permission_mode(permission_mode: str | None) -> str | None:
    if not permission_mode:
        return None
    normalized = permission_mode.strip()
    aliases = {
        "default": "default",
        "accept-edits": "accept-edits",
        "accept_edits": "accept-edits",
        "acceptEdits": "accept-edits",
        "bypass": "bypass",
        "bypassPermissions": "bypass",
        "dangerously-skip-permissions": "bypass",
        "full_auto": "bypass",
        "plan": "plan",
        "dontAsk": "bypass",
    }
    return aliases.get(normalized)


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
