"""Tool for maintaining a project TODO checklist file."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


class TodoWriteInput(BaseModel):
    """Arguments for TODO writes."""

    item: str = Field(description="TODO item text")
    checked: bool = Field(default=False)
    path: str = Field(default="TODO.md")


class TodoWriteTool(BaseTool):
    """Add or update an item in a markdown checklist file."""

    name = "todo_write"
    description = "Add a new TODO item or mark an existing one as done in a markdown checklist file."
    input_model = TodoWriteInput

    def permission_requests(self, arguments: TodoWriteInput) -> list[ToolPermissionRequest]:
        path = _resolve_todo_path(self.cwd, arguments.path)
        return [
            ToolPermissionRequest(
                is_read_only=False,
                file_path=str(path),
                reason=f"Allow todo_write to update {path}?",
            )
        ]

    async def execute(self, arguments: TodoWriteInput) -> ToolResult:
        item = arguments.item.strip()
        if not item:
            return ToolResult("item is required", is_error=True)

        path = _resolve_todo_path(self.cwd, arguments.path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = path.read_text(encoding="utf-8") if path.exists() else "# TODO\n"

            unchecked_line = f"- [ ] {item}"
            checked_line = f"- [x] {item}"
            target_line = checked_line if arguments.checked else unchecked_line

            if unchecked_line in existing and arguments.checked:
                updated = existing.replace(unchecked_line, checked_line, 1)
            elif target_line in existing:
                return ToolResult(f"No change needed in {path}")
            else:
                updated = existing.rstrip() + f"\n{target_line}\n"

            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"Failed to update {path}: {exc}", is_error=True)

        return ToolResult(f"Updated {path}")


def _resolve_todo_path(cwd: Path, raw_path: str) -> Path:
    path = Path(raw_path.strip() or "TODO.md").expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()
