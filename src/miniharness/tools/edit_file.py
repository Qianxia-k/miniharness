"""Edit a file by replacing an exact string match.

OpenHarness uses this old_str/new_str pattern (not unified diffs) because:
- It's robust: no line numbers to drift, no context to mismatch.
- It's idempotent: re-running the same edit won't break anything.
- It's simpler for the model to generate correctly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


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
                    f"Refusing to edit outside workspace: {path}", is_error=True
                )

        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)

        original = path.read_text(encoding="utf-8")
        if arguments.old_str not in original:
            return ToolResult("old_str was not found in the file", is_error=True)

        if arguments.replace_all:
            updated = original.replace(arguments.old_str, arguments.new_str)
        else:
            updated = original.replace(arguments.old_str, arguments.new_str, 1)

        path.write_text(updated, encoding="utf-8")
        return ToolResult(f"Updated {path}")

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.cwd / path
        return path.resolve()

    def permission_requests(self, arguments: EditFileInput) -> list[ToolPermissionRequest]:
        raw_path = arguments.path.strip()
        if not raw_path:
            return []
        path = self._resolve_path(raw_path)
        return [ToolPermissionRequest(
            is_read_only=False,
            file_path=str(path),
            reason=f"Allow edit_file to access/change {path}?",
        )]
