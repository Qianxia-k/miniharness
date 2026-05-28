"""Read a text file from the workspace."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class ReadFileInput(BaseModel):
    """Arguments for read_file."""

    path: str = Field(description="File path to read")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a UTF-8 text file from the current workspace."
    input_model = ReadFileInput

    async def execute(self, arguments: ReadFileInput) -> ToolResult:
        raw_path = arguments.path.strip()
        if not raw_path:
            return ToolResult("path is required", is_error=True)

        path = Path(raw_path)
        if not path.is_absolute():
            path = self.cwd / path
        path = path.resolve()

        # Sandbox-aware boundary check.
        from miniharness.sandbox import is_sandbox_active, validate_sandbox_path

        if is_sandbox_active():
            allowed, reason = validate_sandbox_path(path, self.cwd)
            if not allowed:
                return ToolResult(f"Sandbox: {reason}", is_error=True)

        decision = self.permissions.can_read(path)
        if not decision.allowed:
            return ToolResult(decision.reason, is_error=True)
        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)
        if path.is_dir():
            return ToolResult(f"Cannot read directory: {path}", is_error=True)
        return ToolResult(path.read_text(encoding="utf-8", errors="replace"))
