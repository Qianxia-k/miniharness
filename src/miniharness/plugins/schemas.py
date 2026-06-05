"""Plugin manifest — the plugin.json schema.

Each plugin directory contains a ``plugin.json`` file that declares what
the plugin contributes.  This Pydantic model validates that file.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PluginManifest(BaseModel):
    """Schema for ``plugin.json`` in a plugin directory.

    Only ``name`` is required.  All other fields have sensible defaults.

    Example::

        {
            "name": "my-plugin",
            "version": "1.0.0",
            "description": "Adds code review and deployment skills",
            "enabled_by_default": true
        }
    """

    name: str = Field(description="Unique plugin identifier (e.g. 'my-plugin')")
    version: str = Field(default="0.1.0", description="SemVer version string")
    description: str = Field(default="", description="Human-readable description")
    enabled_by_default: bool = Field(
        default=True,
        description="Whether the plugin is enabled when first discovered",
    )

    # ── Subdirectory names (relative to plugin root) ──────────────
    skills_dir: str = Field(
        default="skills",
        description="Subdirectory for SKILL.md-based skills",
    )
    hooks_file: str = Field(
        default="hooks.json",
        description="JSON file with hook definitions",
    )
    mcp_file: str = Field(
        default="mcp.json",
        description="JSON file with MCP server definitions",
    )
