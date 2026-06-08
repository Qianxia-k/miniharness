"""Loaded plugin — what a plugin contributes after loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniharness.plugins.schemas import PluginManifest


@dataclass
class LoadedPlugin:
    """A plugin that has been loaded from disk and validated.

    Attributes
    ----------
    manifest:
        The parsed and validated ``plugin.json``.
    path:
        Absolute filesystem path to the plugin directory.
    enabled:
        Whether this plugin is trusted/configured and allowed to contribute
        capabilities. Runtime exposure is tracked separately as ``active`` in
        AgentLoop's plugin index.
    skills:
        Skill definitions loaded from the skills subdirectory.
    hooks:
        Hook definitions keyed by event name.
    mcp_servers:
        MCP server configs from the plugin's mcp.json.
    """

    manifest: PluginManifest
    path: Path
    enabled: bool = True
    skills: list[Any] = field(default_factory=list)     # list[SkillDefinition]
    hooks: dict[str, list[dict]] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def description(self) -> str:
        return self.manifest.description
