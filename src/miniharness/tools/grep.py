"""Search text files in the workspace."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class GrepInput(BaseModel):
    """Arguments for grep."""

    query: str = Field(description="Literal text to search for")
    root: str = Field(default=".", description="Directory to search")


class GrepTool(BaseTool):
    name = "grep"
    description = "Search for a literal string under the current workspace."
    input_model = GrepInput

    async def execute(self, arguments: GrepInput) -> ToolResult:
        query = arguments.query.strip()
        if not query:
            return ToolResult("query is required", is_error=True)

        root = Path(arguments.root.strip() or ".")
        if not root.is_absolute():
            root = self.cwd / root
        root = root.resolve()

        decision = self.permissions.can_read(root)
        if not decision.allowed:
            return ToolResult(decision.reason, is_error=True)
        if not root.exists():
            return ToolResult(f"Search root not found: {root}", is_error=True)

        matches: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file() or len(matches) >= 50:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if query in line:
                    rel = path.relative_to(self.cwd) if path.is_relative_to(self.cwd) else path
                    matches.append(f"{rel}:{line_no}: {line}")
                    break
        return ToolResult("\n".join(matches) if matches else "(no matches)")
