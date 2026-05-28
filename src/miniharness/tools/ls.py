"""List files and directories in the workspace."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class LsInput(BaseModel):
    """Arguments for ls."""

    path: str = Field(default=".", description="Directory path to list")


class LsTool(BaseTool):
    name = "ls"
    description = (
        "List files and directories in a directory. "
        "Shows name, size, and type for each entry. "
        "Use this to explore the project structure before reading or editing files."
    )
    input_model = LsInput

    _MAX_ENTRIES = 200

    async def execute(self, arguments: LsInput) -> ToolResult:
        raw = arguments.path.strip() or "."
        path = Path(raw)
        if not path.is_absolute():
            path = self.cwd / path
        path = path.resolve()

        # Sandbox check.
        from miniharness.sandbox import is_sandbox_active, validate_sandbox_path

        if is_sandbox_active():
            allowed, reason = validate_sandbox_path(path, self.cwd)
            if not allowed:
                return ToolResult(f"Sandbox: {reason}", is_error=True)

        decision = self.permissions.can_read(path)
        if not decision.allowed:
            return ToolResult(decision.reason, is_error=True)
        if not path.exists():
            return ToolResult(f"Directory not found: {path}", is_error=True)
        if not path.is_dir():
            return ToolResult(f"Not a directory: {path}", is_error=True)

        lines: list[str] = []
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        for entry in entries[: self._MAX_ENTRIES]:
            try:
                stat = entry.lstat()
                size = stat.st_size if entry.is_file() else 0
            except OSError:
                size = 0

            type_tag = "/" if entry.is_dir() else "@" if entry.is_symlink() else ""
            name = entry.name + type_tag

            if entry.is_dir():
                lines.append(f"[dir]  {name}")
            elif entry.is_symlink():
                lines.append(f"[link] {name} -> {entry.readlink()}")
            else:
                lines.append(f"[file] {name}  ({_format_size(size)})")

        if len(entries) > self._MAX_ENTRIES:
            lines.append(f"... ({len(entries) - self._MAX_ENTRIES} more entries)")

        rel = path.relative_to(self.cwd) if path.is_relative_to(self.cwd) else path
        header = f"Contents of {rel}/ ({len(entries)} entries):"
        return ToolResult("\n".join([header, ""] + lines) if lines else f"{header}\n(empty)")


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TB"
