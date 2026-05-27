"""Tool registry.

The registry is the bridge between model-facing schemas and Python tool code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniharness.permissions import PermissionChecker
from miniharness.tools.base import BaseTool, ToolResult
from miniharness.tools.bash import BashTool
from miniharness.tools.grep import GrepTool
from miniharness.tools.read_file import ReadFileTool
from miniharness.tools.write_file import WriteFileTool
from miniharness.tools.edit_file import EditFileTool


class ToolRegistry:
    """Keep track of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

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
        return await tool.execute(parsed)


def create_default_registry(*, cwd: Path, permissions: PermissionChecker) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool(cwd=cwd, permissions=permissions))
    registry.register(GrepTool(cwd=cwd, permissions=permissions))
    registry.register(BashTool(cwd=cwd, permissions=permissions))
    registry.register(WriteFileTool(cwd=cwd, permissions=permissions))
    registry.register(EditFileTool(cwd=cwd, permissions=permissions))
    return registry

