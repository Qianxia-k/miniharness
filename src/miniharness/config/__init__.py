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
    ProviderSettings as ProviderSettings,
    SandboxSettings as SandboxSettings,
    Settings,
)


def load_settings() -> Settings:
    """Build a Settings object from the full config chain.

    Returns a resolved Settings that the CLI, loop, and tools can consume.
    """
    settings = Settings()

    # ---- Layer 1: defaults ------------------------------------------------
    # Already set by dataclass field defaults.

    # ---- Layer 2: environment variables -----------------------------------
    settings = _apply_env_vars(settings)

    # ---- Layer 3: provider auto-detection ---------------------------------
    settings = _auto_detect_provider(settings)

    return settings


def apply_cli_overrides(
    settings: Settings,
    *,
    profile: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_turns: int | None = None,
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

    # Sandbox
    if os.environ.get("MINIHARNESS_SANDBOX_ENABLED", "").lower() in ("1", "true", "yes"):
        settings.sandbox = replace(settings.sandbox, enabled=True)
    if os.environ.get("MINIHARNESS_SANDBOX_IMAGE"):
        settings.sandbox = replace(settings.sandbox, image=os.environ["MINIHARNESS_SANDBOX_IMAGE"])

    return settings


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
