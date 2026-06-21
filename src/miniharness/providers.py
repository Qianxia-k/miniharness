"""Provider profiles for MiniHarness.

This is the small, beginner-friendly version of OpenHarness's provider layer.
OpenHarness keeps provider workflows in `config/settings.py` and provider
metadata in `api/registry.py`. MiniHarness starts with the same idea in one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any


@dataclass(frozen=True)
class ProviderProfile:
    """A model provider workflow."""

    name: str
    label: str
    provider: str
    api_format: str
    api_key_env: str
    default_model: str
    base_url: str | None = None
    context_window: int = 131072
    extra_body: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """Resolve this profile's API key from environment variables."""
        api_key = os.environ.get("MINIHARNESS_API_KEY") or os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                "No API key configured. "
                f"Set MINIHARNESS_API_KEY or {self.api_key_env} for profile '{self.name}'."
            )
        return api_key


PROFILES: dict[str, ProviderProfile] = {
    "qwen": ProviderProfile(
        name="qwen",
        label="Qwen (DashScope)",
        provider="dashscope",
        api_format="openai",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3.7-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        context_window=131072,  # 128K
        extra_body={"enable_thinking": False},
    ),
    "openai": ProviderProfile(
        name="openai",
        label="OpenAI",
        provider="openai",
        api_format="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
        context_window=131072,  # 128K
    ),
    "anthropic": ProviderProfile(
        name="anthropic",
        label="Anthropic (Claude)",
        provider="anthropic",
        api_format="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-6",
        context_window=200000,
    ),
    "openai-compatible": ProviderProfile(
        name="openai-compatible",
        label="OpenAI-Compatible API",
        provider="openai",
        api_format="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
        context_window=131072,  # 128K
    ),
}


def get_profile(name: str) -> ProviderProfile:
    """Return a provider profile by name."""
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise RuntimeError(f"Unknown provider profile '{name}'. Known profiles: {known}") from None

