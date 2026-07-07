"""Write content to a file in the workspace."""

from __future__ import annotations

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult
from miniharness.tools.file_ops import (
    compute_diff,
    format_diff_permission_prompt,
    read_existing_text,
    resolve_path,
    validate_write_boundary,
)
from miniharness.utils.fs import atomic_write_text


class WriteFileInput(BaseModel):
    """Arguments for write_file."""

    path: str = Field(description="File path to write, relative to workspace")
    content: str = Field(description="Text content to write to the file")
    create_directories: bool = Field(default=True, description="Create parent directories if needed")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a text file in the workspace."
    input_model = WriteFileInput

    async def execute(self, arguments: WriteFileInput) -> ToolResult:
        raw_path = arguments.path.strip()
        if not raw_path:
            return ToolResult("path is required", is_error=True)

        path = resolve_path(self.cwd, raw_path)
        boundary_error = validate_write_boundary(path, self.cwd)
        if boundary_error is not None:
            return ToolResult(boundary_error, is_error=True)
        if path.exists() and path.is_dir():
            return ToolResult(f"Cannot write directory: {path}", is_error=True)
        if path.exists():
            _, read_error = read_existing_text(path)
            if read_error is not None:
                return ToolResult(read_error, is_error=True)
        if not arguments.create_directories and not path.parent.exists():
            return ToolResult(f"Parent directory does not exist: {path.parent}", is_error=True)

        atomic_write_text(path, arguments.content, encoding="utf-8")
        return ToolResult(f"Wrote {path.stat().st_size} bytes to {path}")

    def permission_requests(self, arguments: WriteFileInput) -> list[ToolPermissionRequest]:
        raw_path = arguments.path.strip()
        if not raw_path:
            return []
        path = resolve_path(self.cwd, raw_path)
        reason = f"Allow write_file to access/change {path}?"
        try:
            original, read_error = read_existing_text(path)
            if read_error is None:
                diff_text, added, removed = compute_diff(str(path), original, arguments.content)
                reason = format_diff_permission_prompt(
                    tool_name="write_file",
                    action="write",
                    path=path,
                    diff_text=diff_text,
                    added=added,
                    removed=removed,
                )
        except OSError:
            pass
        return [ToolPermissionRequest(
            is_read_only=False,
            file_path=str(path),
            reason=reason,
        )]
