"""Write content to a file in the workspace."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


class WriteFileInput(BaseModel):
    """Arguments for write_file."""

    path: str = Field(description="File path to write, relative to workspace")
    content: str = Field(description="Text content to write to the file")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write or overwrite a text file in the workspace. Creates parent directories if needed."
    input_model = WriteFileInput

    async def execute(self, arguments: WriteFileInput) -> ToolResult:
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
        else:
            try:
                path.relative_to(self.cwd)
            except ValueError:
                return ToolResult(
                    f"Refusing to write outside workspace: {path}", is_error=True
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments.content, encoding="utf-8")
        return ToolResult(f"Wrote {path.stat().st_size} bytes to {path}")

    def permission_requests(self, arguments: WriteFileInput) -> list[ToolPermissionRequest]:
        raw_path = arguments.path.strip()
        if not raw_path:
            return []
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.cwd / path
        return [ToolPermissionRequest(
            is_read_only=False,
            file_path=str(path.resolve()),
            reason=f"Allow write_file to access/change {path.resolve()}?",
        )]
