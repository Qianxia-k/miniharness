"""MCP resource tools — expose MCP server resources and tools to the model.

Resources are data that MCP servers expose (files, DB rows, API responses).
Tools are functions the model can call.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.plugins.gating import is_mcp_server_visible
from miniharness.tools.base import BaseTool, ToolResult


class ListMcpResourcesTool(BaseTool):
    """List all MCP servers, their tools, and resources.

    This is the primary discovery tool — the model calls it to understand
    what external MCP capabilities are available.
    """

    name = "list_mcp_resources"
    description = (
        "List all connected MCP servers and their available tools and resources. "
        "Use this to discover what external capabilities are available via MCP."
    )

    class InputModel(BaseModel):
        pass

    input_model = InputModel

    def __init__(
        self,
        *,
        cwd: Path,
        manager=None,
        permissions=None,
        plugin_index: list[dict] | None = None,
    ) -> None:
        super().__init__(cwd=cwd, permissions=permissions)
        self._manager = manager
        self._plugin_index = plugin_index

    async def execute(self, arguments: InputModel) -> ToolResult:
        if self._manager is None:
            return ToolResult("MCP manager not available.", is_error=True)

        statuses = [
            s for s in self._manager.list_statuses()
            if is_mcp_server_visible(s.name, self._plugin_index)
        ]
        if not statuses:
            return ToolResult("No active MCP servers configured.")

        lines: list[str] = ["**Connected MCP Servers**", ""]
        for s in statuses:
            icon = {"connected": "✅", "failed": "❌", "pending": "⏳", "disabled": "⚫"}.get(s.state, "?")
            lines.append(f"{icon} **{s.name}** [{s.state}] ({s.transport})")

            # List tools.
            if s.tools:
                lines.append(f"   Tools ({len(s.tools)}):")
                for t in s.tools:
                    desc = t.description[:100] if t.description else "(no description)"
                    lines.append(f"     - `{t.name}`: {desc}")
            else:
                lines.append("   Tools: (none)")

            # List resources (if any).
            if hasattr(s, "resources") and s.resources:
                lines.append(f"   Resources ({len(s.resources)}):")
                for r in s.resources[:10]:
                    lines.append(f"     - {r.uri} — {r.description[:80]}")

            if s.state == "failed" and s.detail:
                lines.append(f"   Error: {s.detail[:120]}")

            lines.append("")

        if not lines:
            return ToolResult("No MCP servers connected.")
        return ToolResult("\n".join(lines))


class ReadMcpResourceInput(BaseModel):
    server: str = Field(description="MCP server name")
    uri: str = Field(description="Resource URI to read")


class ReadMcpResourceTool(BaseTool):
    """Read a resource from an MCP server by server name and URI."""

    name = "read_mcp_resource"
    description = "Read a resource (data) from a connected MCP server by URI."
    input_model = ReadMcpResourceInput

    def __init__(
        self,
        *,
        cwd: Path,
        manager=None,
        permissions=None,
        plugin_index: list[dict] | None = None,
    ) -> None:
        super().__init__(cwd=cwd, permissions=permissions)
        self._manager = manager
        self._plugin_index = plugin_index

    async def execute(self, arguments: ReadMcpResourceInput) -> ToolResult:
        if self._manager is None:
            return ToolResult("MCP manager not available.", is_error=True)
        if not is_mcp_server_visible(arguments.server, self._plugin_index):
            return ToolResult(
                f"MCP server '{arguments.server}' is not active in the current runtime context.",
                is_error=True,
            )

        session = getattr(self._manager, "_sessions", {}).get(arguments.server)
        if session is None:
            return ToolResult(
                f"MCP server '{arguments.server}' is not connected.",
                is_error=True,
            )

        try:
            result = await session.read_resource(arguments.uri)
        except Exception as exc:
            return ToolResult(f"Failed to read MCP resource: {exc}", is_error=True)

        parts: list[str] = []
        for item in getattr(result, "contents", []) or []:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return ToolResult("\n".join(parts) if parts else "(empty resource)")
