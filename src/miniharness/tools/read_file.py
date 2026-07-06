"""Read a text file from the workspace."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class ReadFileInput(BaseModel):
    """Arguments for read_file."""

    path: str = Field(description="File path to read")
    offset: int = Field(default=0, ge=0, description="Zero-based starting line")
    limit: int = Field(default=200, ge=1, le=2000, description="Number of lines to return")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a UTF-8 text file from the current workspace with line numbers."
    input_model = ReadFileInput

    def is_read_only(self, arguments: ReadFileInput) -> bool:
        del arguments
        return True

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
        raw = path.read_bytes()
        if b"\x00" in raw:
            return ToolResult(f"Binary file cannot be read as text: {path}", is_error=True)

        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[arguments.offset : arguments.offset + arguments.limit]
        numbered = [
            f"{arguments.offset + index + 1:>6}\t{line}"
            for index, line in enumerate(selected)
        ]
        if not numbered:
            return ToolResult(f"(no content in selected range for {path})")
        return ToolResult("\n".join(numbered))
