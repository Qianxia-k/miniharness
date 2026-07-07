"""Edit a file by replacing an exact string match."""

from __future__ import annotations

from pathlib import Path

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


class EditFileInput(BaseModel):
    """Arguments for edit_file."""

    path: str = Field(description="Path of the file to edit")
    old_str: str = Field(description="Existing text to replace")
    new_str: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace all occurrences instead of just the first")


class EditFileTool(BaseTool):
    name = "edit_file"
    description = "Edit an existing file by replacing a string."
    input_model = EditFileInput

    async def execute(self, arguments: EditFileInput) -> ToolResult:
        raw_path = arguments.path.strip()
        if not raw_path:
            return ToolResult("path is required", is_error=True)
        if not arguments.old_str:
            return ToolResult("old_str is required", is_error=True)

        path = self._resolve_path(raw_path)
        boundary_error = validate_write_boundary(path, self.cwd)
        if boundary_error is not None:
            return ToolResult(boundary_error.replace("write", "edit"), is_error=True)

        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)

        original, read_error = read_existing_text(path)
        if read_error is not None:
            return ToolResult(read_error, is_error=True)
        if arguments.old_str not in original:
            return ToolResult("old_str was not found in the file", is_error=True)

        if arguments.replace_all:
            updated = original.replace(arguments.old_str, arguments.new_str)
        else:
            updated = original.replace(arguments.old_str, arguments.new_str, 1)

        atomic_write_text(path, updated, encoding="utf-8")
        return ToolResult(f"Updated {path}")

    def _resolve_path(self, raw_path: str) -> Path:
        return resolve_path(self.cwd, raw_path)

    def permission_requests(self, arguments: EditFileInput) -> list[ToolPermissionRequest]:
        raw_path = arguments.path.strip()
        if not raw_path:
            return []
        path = self._resolve_path(raw_path)
        reason = f"Allow edit_file to access/change {path}?"
        try:
            if path.exists() and path.is_file() and arguments.old_str:
                original, read_error = read_existing_text(path)
                if read_error is None and arguments.old_str in original:
                    updated = (
                        original.replace(arguments.old_str, arguments.new_str)
                        if arguments.replace_all
                        else original.replace(arguments.old_str, arguments.new_str, 1)
                    )
                    diff_text, added, removed = compute_diff(str(path), original, updated)
                    reason = format_diff_permission_prompt(
                        tool_name="edit_file",
                        action="update",
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
