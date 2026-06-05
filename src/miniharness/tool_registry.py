"""Tool registry.

The registry is the bridge between model-facing schemas and Python tool code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniharness.permissions import PermissionChecker
from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult
from miniharness.tools.bash import BashTool
from miniharness.tools.grep import GrepTool
from miniharness.tools.ls import LsTool
from miniharness.tools.memory_tool import (
    MemoryAddTool,
    MemoryLogTool,
    MemorySearchTool,
)
from miniharness.tools.read_file import ReadFileTool
from miniharness.tools.task import TaskTool
from miniharness.tools.web_fetch import WebFetchTool
from miniharness.tools.write_file import WriteFileTool
from miniharness.tools.edit_file import EditFileTool


class ToolRegistry:
    """Keep track of available tools."""

    def __init__(self, *, permissions: PermissionChecker | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._permissions = permissions

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(output=f"Unknown tool: {name}", is_error=True)
        try:
            parsed = tool.input_model(**arguments)
        except Exception as exc:
            return ToolResult(output=f"Invalid arguments for {name}: {exc}", is_error=True)

        if self._permissions is not None:
            permission_result = self._check_permission_requests(name, tool, parsed)
            if permission_result is not None:
                return permission_result
        return await tool.execute(parsed)

    def _check_permission_requests(
        self,
        name: str,
        tool: BaseTool,
        parsed: Any,
    ) -> ToolResult | None:
        try:
            requests = tool.permission_requests(parsed)
        except Exception as exc:
            return ToolResult(
                output=f"Permission analysis failed for {name}: {exc}",
                is_error=True,
            )

        for request in requests:
            if not isinstance(request, ToolPermissionRequest):
                return ToolResult(
                    output=f"Invalid permission request from {name}.",
                    is_error=True,
                )
            file_path = _resolve_permission_path(
                request.file_path,
                self._permissions.cwd,
            )
            decision = self._permissions.evaluate(
                tool_name=name,
                file_path=file_path,
                command=request.command,
                is_read_only=request.is_read_only,
            )
            if not decision.allowed:
                if decision.requires_confirmation:
                    decision = self._permissions.resolve_interactive(
                        decision,
                        _permission_prompt(name, request),
                    )
                if not decision.allowed:
                    return ToolResult(decision.reason, is_error=True)
        return None


def _permission_prompt(tool_name: str, request: ToolPermissionRequest) -> str:
    if request.reason:
        return request.reason
    if request.command:
        preview = request.command[:120] + "..." if len(request.command) > 120 else request.command
        return f"Allow {tool_name} to run command: {preview}?"
    if request.file_path:
        action = "read" if request.is_read_only else "access/change"
        return f"Allow {tool_name} to {action} {request.file_path}?"
    return f"Allow {tool_name} to execute?"


def _resolve_permission_path(file_path: str | None, cwd: Path) -> str | None:
    if not file_path:
        return None
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return str(path.resolve())


def create_default_registry(
    *,
    cwd: Path,
    permissions: PermissionChecker,
    mcp_manager=None,  # McpClientManager | None
) -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools + MCP adapters.

    Parameters
    ----------
    cwd:
        Working directory.
    permissions:
        Permission checker instance.
    mcp_manager:
        Optional McpClientManager.  If provided, resource tools and MCP
        tool adapters are registered automatically.
    """
    registry = ToolRegistry(permissions=permissions)
    # Read-only tools
    registry.register(ReadFileTool(cwd=cwd, permissions=permissions))
    registry.register(LsTool(cwd=cwd, permissions=permissions))
    registry.register(GrepTool(cwd=cwd, permissions=permissions))
    # Write tools
    registry.register(WriteFileTool(cwd=cwd, permissions=permissions))
    registry.register(EditFileTool(cwd=cwd, permissions=permissions))
    # Shell
    registry.register(BashTool(cwd=cwd, permissions=permissions))
    # External & meta
    registry.register(WebFetchTool(cwd=cwd, permissions=permissions))
    registry.register(TaskTool(cwd=cwd, permissions=permissions))
    # Memory (agent-managed, always allowed)
    registry.register(MemorySearchTool(cwd=cwd, permissions=permissions))
    registry.register(MemoryAddTool(cwd=cwd, permissions=permissions))
    registry.register(MemoryLogTool(cwd=cwd, permissions=permissions))

    # MCP — resource tools + auth tool + per-server adapters
    if mcp_manager is not None:
        from miniharness.mcp.auth_tool import McpAuthTool
        from miniharness.mcp.resource_tools import (
            ListMcpResourcesTool,
            ReadMcpResourceTool,
        )
        from miniharness.mcp.tool_adapter import McpToolAdapter

        registry.register(ListMcpResourcesTool(
            cwd=cwd, manager=mcp_manager, permissions=permissions
        ))
        registry.register(ReadMcpResourceTool(
            cwd=cwd, manager=mcp_manager, permissions=permissions
        ))
        registry.register(McpAuthTool(
            cwd=cwd, manager=mcp_manager, permissions=permissions
        ))
        for tool_info in mcp_manager.list_tools():
            registry.register(McpToolAdapter(
                manager=mcp_manager, tool_info=tool_info,
                cwd=cwd, permissions=permissions,
            ))

    return registry
