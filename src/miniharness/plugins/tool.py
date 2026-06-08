"""Plugin activation tool — controls runtime capability visibility.

Installed/enabled plugins are trusted configuration. Activating a plugin is a
per-conversation context decision: it exposes that plugin's namespaced skills
and MCP tools, and execution-side gates reject inactive plugin capabilities.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class PluginToolInput(BaseModel):
    name: str = Field(description="Plugin name to activate (from Available Plugins)")


class PluginTool(BaseTool):
    """Activate a plugin so its capabilities appear in the runtime context.

    Plugins not yet activated show only their description in the system
    prompt.  Activating one makes its skills and MCP tools visible in the
    next turn.
    """

    name = "plugin"
    description = (
        "Activate a trusted plugin by name to expose its namespaced skills "
        "and MCP tools in the current runtime context."
    )
    input_model = PluginToolInput

    def __init__(
        self,
        *,
        cwd: Path,
        permissions=None,
        plugin_index: list[dict] | None = None,  # [{name, description, active, skills}]
    ) -> None:
        super().__init__(cwd=cwd, permissions=permissions)
        self._index = plugin_index or []

    async def execute(self, arguments: PluginToolInput) -> ToolResult:
        plugin_name = arguments.name.strip()

        for entry in self._index:
            if entry["name"] != plugin_name:
                continue

            if entry.get("active"):
                return ToolResult(f"Plugin '{plugin_name}' is already active.")

            entry["active"] = True
            skills = entry.get("skills", [])
            plugin_obj = entry.get("_plugin")
            mcp_servers = getattr(plugin_obj, "mcp_servers", {}) if plugin_obj else {}
            if not skills and not mcp_servers:
                return ToolResult(f"Plugin '{plugin_name}' has no skills or MCP servers.")

            lines = [
                f"Plugin '{plugin_name}' activated.",
                "Capabilities become visible on the next model turn.",
                "",
            ]
            if skills:
                lines.append("Skills:")
                for s in skills:
                    lines.append(f"- **{s.invocation_name}**: {s.description}")
            if mcp_servers:
                if skills:
                    lines.append("")
                lines.append("MCP servers:")
                for server_name in mcp_servers:
                    lines.append(f"- **{plugin_name}:{server_name}**")
            return ToolResult("\n".join(lines))

        available = ", ".join(e["name"] for e in self._index)
        return ToolResult(
            f"Plugin '{plugin_name}' not found. Available: {available or '(none)'}",
            is_error=True,
        )
