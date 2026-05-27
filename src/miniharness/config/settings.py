"""Settings data model.

Mirrors OpenHarness's config/settings.py.  Each concern (provider, sandbox,
agent) owns its section so defaults, env vars, and overrides compose cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderSettings:
    """Resolved provider configuration."""

    name: str = "qwen"          # profile key: "qwen", "openai", ...
    model: str = ""               # empty = use profile's default_model
    base_url: str | None = None   # None = use profile's default base_url


@dataclass
class SandboxSettings:
    """Sandbox / container-isolation configuration."""

    enabled: bool = False
    image: str = "miniharness-sandbox:latest"
    fail_if_unavailable: bool = False


@dataclass
class Settings:
    """Top-level settings bag passed through the whole agent lifecycle.

    Every layer reads from this instead of reaching for env vars or CLI args
    directly — that way the loading chain is the single source of truth.
    """

    provider: ProviderSettings = field(default_factory=ProviderSettings)
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    max_turns: int = 8
