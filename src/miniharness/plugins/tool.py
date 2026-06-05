"""Plugin activation tool — controls plugin capability visibility.

Unlike skills (always injected), plugin skills are COLLAPSED behind a
single plugin description by default.  ``plugin(name="x")`` activates a
plugin so its skills and MCP tools become visible in the next turn's
system prompt and tool schema.

Skills are always registered in the skill_registry (the model COULD call
them directly), but the system prompt only lists activated plugins'
skills, saving tokens when there are many plugins.
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
        "Activate a plugin by name to see its available skills.  "
        "Use this when a user's request matches a plugin description."
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
                    lines.append(f"- **{s.name}**: {s.description}")
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
