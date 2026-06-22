"""Configuration loading.

Mirrors the layered config resolution in OpenHarness:

    defaults  →  env vars  →  provider auto-detect  →  CLI overrides
    (lowest)                                          (highest)

After loading, the rest of the codebase reads from a Settings object instead
of reaching for os.environ or CLI args directly.
"""

from __future__ import annotations

import os
from dataclasses import replace

from miniharness.config.settings import (
    AgentSettings as AgentSettings,
    HookSettings as HookSettings,
    ProviderSettings as ProviderSettings,
    SandboxSettings as SandboxSettings,
    Settings,
)


def load_settings() -> Settings:
    """Build a Settings object from the full config chain.

    Resolution order (later layers override earlier):

        1. Defaults (dataclass field defaults)
        2. ~/.miniharness/mcp.json (user-level MCP config)
        3. .miniharness/mcp.json (project-level MCP config)
        4. MINIHARNESS_MCP_SERVERS env var (JSON string)
        5. Environment variables (model, provider, etc.)
        6. Provider auto-detection
    """
    settings = Settings()

    # ---- Layer 1: defaults ------------------------------------------------
    # Already set by dataclass field defaults.

    # ---- Layer 2: MCP config files ----------------------------------------
    settings = _load_mcp_config_files(settings)

    # ---- Layer 3: MCP env var ---------------------------------------------
    settings = _apply_mcp_env_var(settings)

    # ---- Layer 4: environment variables -----------------------------------
    settings = _apply_env_vars(settings)

    # ---- Layer 5: provider auto-detection ---------------------------------
    settings = _auto_detect_provider(settings)

    return settings


def apply_cli_overrides(
    settings: Settings,
    *,
    profile: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_turns: int | None = None,
    context_budget_ratio: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    sandbox: bool | None = None,
    sandbox_image: str | None = None,
) -> Settings:
    """Apply CLI arguments on top of an already-resolved Settings.

    CLI args are the highest-priority layer.  Pass only the flags the user
    explicitly provided (None = "user didn't pass this flag").
    """
    if profile is not None:
        settings.provider = replace(settings.provider, name=profile)
    if model is not None:
        settings.provider = replace(settings.provider, model=model)
    if base_url is not None:
        settings.provider = replace(settings.provider, base_url=base_url)
    if max_turns is not None:
        settings = replace(settings, max_turns=max_turns)
    if context_budget_ratio is not None:
        settings = replace(settings, context_budget_ratio=_clamp_context_budget_ratio(context_budget_ratio))
    # LLM sampling params — write through the shared AgentSettings reference.
    if temperature is not None:
        settings.agent.temperature = temperature
    if top_p is not None:
        settings.agent.top_p = top_p
    if max_tokens is not None:
        settings.agent.max_tokens = max_tokens
    if sandbox is not None:
        settings.sandbox = replace(settings.sandbox, enabled=sandbox)
    if sandbox_image is not None:
        settings.sandbox = replace(settings.sandbox, image=sandbox_image)
    return settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_env_vars(settings: Settings) -> Settings:
    """Layer env vars onto the settings object."""

    # Provider overrides
    if os.environ.get("MINIHARNESS_MODEL"):
        settings.provider = replace(settings.provider, model=os.environ["MINIHARNESS_MODEL"])
    if os.environ.get("MINIHARNESS_BASE_URL"):
        settings.provider = replace(settings.provider, base_url=os.environ["MINIHARNESS_BASE_URL"])
    if os.environ.get("MINIHARNESS_PROFILE"):
        settings.provider = replace(settings.provider, name=os.environ["MINIHARNESS_PROFILE"])

    # Agent
    max_turns_str = os.environ.get("MINIHARNESS_MAX_TURNS", "")
    if max_turns_str.isdigit():
        settings = replace(settings, max_turns=int(max_turns_str))

    ratio_str = os.environ.get("MINIHARNESS_CONTEXT_BUDGET_RATIO", "")
    if ratio_str:
        try:
            settings = replace(
                settings,
                context_budget_ratio=_clamp_context_budget_ratio(float(ratio_str)),
            )
        except ValueError:
            pass

    # Agent — LLM sampling params
    temp_str = os.environ.get("MINIHARNESS_TEMPERATURE", "")
    if temp_str:
        try:
            settings.agent.temperature = float(temp_str)
        except ValueError:
            pass
    top_p_str = os.environ.get("MINIHARNESS_TOP_P", "")
    if top_p_str:
        try:
            settings.agent.top_p = float(top_p_str)
        except ValueError:
            pass
    tokens_str = os.environ.get("MINIHARNESS_MAX_TOKENS", "")
    if tokens_str.isdigit():
        settings.agent.max_tokens = int(tokens_str)

    # Sandbox
    if os.environ.get("MINIHARNESS_SANDBOX_ENABLED", "").lower() in ("1", "true", "yes"):
        settings.sandbox = replace(settings.sandbox, enabled=True)
    if os.environ.get("MINIHARNESS_SANDBOX_IMAGE"):
        settings.sandbox = replace(settings.sandbox, image=os.environ["MINIHARNESS_SANDBOX_IMAGE"])

    # Plugins
    allow_project_plugins = os.environ.get("MINIHARNESS_ALLOW_PROJECT_PLUGINS", "").lower()
    if allow_project_plugins in ("1", "true", "yes", "on"):
        settings = replace(settings, allow_project_plugins=True)
    elif allow_project_plugins in ("0", "false", "no", "off"):
        settings = replace(settings, allow_project_plugins=False)

    return settings


def _clamp_context_budget_ratio(value: float) -> float:
    """Validate the soft context-budget ratio.

    Keep this intentionally broad so developers can force compact testing with
    tiny ratios, while still rejecting values that make no semantic sense.
    """
    if value <= 0 or value > 1:
        raise ValueError("context budget ratio must be > 0 and <= 1")
    return value


def _auto_detect_provider(settings: Settings) -> Settings:
    """Infer the provider from available API keys when not explicitly set.

    Priority (first match wins):
        1. MINIHARNESS_PROFILE env var (already applied in _apply_env_vars)
        2. DASHSCOPE_API_KEY set → profile "qwen"
        3. OPENAI_API_KEY set    → profile "openai"
        4. MINIHARNESS_API_KEY + base_url keyword matching
        5. Fallback to "openai"
    """
    # If an explicit profile was already set (via env var), don't override.
    if os.environ.get("MINIHARNESS_PROFILE"):
        return settings

    detected: str | None = None

    if os.environ.get("DASHSCOPE_API_KEY"):
        detected = "qwen"
    elif os.environ.get("OPENAI_API_KEY"):
        detected = "openai"
    elif os.environ.get("MINIHARNESS_API_KEY"):
        base = os.environ.get("MINIHARNESS_BASE_URL", "")
        if "dashscope" in base:
            detected = "qwen"
        else:
            detected = "openai"

    if detected is not None:
        settings.provider = replace(settings.provider, name=detected)

    return settings


# ---------------------------------------------------------------------------
# MCP config loading
# ---------------------------------------------------------------------------


def _parse_json_with_comments(text: str) -> dict:
    """Parse JSON text that may contain ``//`` or ``#`` comment lines."""
    import json as _json
    import re
    text = re.sub(r'^\s*//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*#.*$', '', text, flags=re.MULTILINE)
    return _json.loads(text)


def _load_mcp_config_files(settings: Settings) -> Settings:
    """Load MCP server configs from ``~/.miniharness/mcp.json`` and
    project ``.miniharness/mcp.json``.

    Project-level config overrides user-level for servers with the same name.
    """
    import json
    from pathlib import Path

    merged: dict = {}

    # 1. User-level: ~/.miniharness/mcp.json
    user_path = Path.home() / ".miniharness" / "mcp.json"
    if user_path.is_file():
        try:
            user_config = _parse_json_with_comments(user_path.read_text(encoding="utf-8"))
            if isinstance(user_config, dict):
                servers = user_config.get("mcpServers", user_config)
                if isinstance(servers, dict):
                    merged.update(servers)
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Project-level: .miniharness/mcp.json (cwd-relative)
    cwd = Path.cwd()
    proj_path = cwd / ".miniharness" / "mcp.json"
    if proj_path.is_file():
        try:
            proj_config = _parse_json_with_comments(proj_path.read_text(encoding="utf-8"))
            if isinstance(proj_config, dict):
                servers = proj_config.get("mcpServers", proj_config)
                if isinstance(servers, dict):
                    merged.update(servers)  # project overrides user
        except (json.JSONDecodeError, OSError):
            pass

    # Merge into settings (don't overwrite programmatically-set configs).
    if merged:
        existing = dict(settings.mcp_servers)
        merged.update(existing)  # programmatic configs win
        settings.mcp_servers = merged

    return settings


def _apply_mcp_env_var(settings: Settings) -> Settings:
    """Load MCP servers from MINIHARNESS_MCP_SERVERS env var (JSON string).

    Example::

        export MINIHARNESS_MCP_SERVERS='{"filesystem":{"type":"stdio","command":"npx","args":["-y","@mcp/server-filesystem","/tmp"]}}'
    """
    import json

    raw = os.environ.get("MINIHARNESS_MCP_SERVERS", "")
    if not raw.strip():
        return settings

    try:
        servers = json.loads(raw)
    except json.JSONDecodeError:
        return settings

    if not isinstance(servers, dict):
        return settings

    existing = dict(settings.mcp_servers)
    existing.update(servers)  # env var overrides files
    settings.mcp_servers = existing
    return settings
