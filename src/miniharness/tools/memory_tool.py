"""Memory tools — the agent's system calls for long-term memory.

Three tools:

- ``memory_search``  — read-only, always allowed.  Searches both stores.
- ``memory_add``    — append a fact to semantic memory.  Auto-allowed.
- ``memory_log``    — record a completed task to episodic memory.  Auto-allowed.

Memory writes are auto-allowed because memory is agent-managed metadata,
not user files.  The user can inspect and edit memory via ``/memory``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.memory.episodic import EpisodicStore
from miniharness.memory.semantic import SemanticStore
from miniharness.permissions import PermissionChecker
from miniharness.tools.base import BaseTool, ToolResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_search_results(
    semantic: list[dict],
    episodic: list[dict],
) -> str:
    """Render search results from both stores into a single text block."""
    parts: list[str] = []

    if semantic:
        parts.append("── Semantic Memory (facts) ──")
        for entry in semantic:
            tags = ", ".join(entry.get("tags", []))
            tag_str = f"  [{tags}]" if tags else ""
            parts.append(f"• {entry['fact']}{tag_str}")

    if episodic:
        parts.append("\n── Episodic Memory (task traces) ──")
        for entry in episodic:
            parts.append(
                f"• [{entry.get('id', '?')}] {entry.get('task', '')}\n"
                f"  {entry.get('summary', '')}\n"
                f"  outcome: {entry.get('outcome', '?')}"
            )

    if not parts:
        return "(no matching memory entries found)"
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------


class MemorySearchInput(BaseModel):
    query: str = Field(description="Keywords to search for in memory")


class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = (
        "Search the agent's long-term memory for relevant facts and past task traces. "
        "Use this before starting a task to recall what you already know about the "
        "project or how you solved similar tasks before."
    )
    input_model = MemorySearchInput

    async def execute(self, arguments: MemorySearchInput) -> ToolResult:
        cwd = str(self.cwd)
        semantic = SemanticStore(cwd).search(arguments.query.strip())
        episodic = EpisodicStore(cwd).search(arguments.query.strip())
        return ToolResult(_format_search_results(semantic, episodic))


# ---------------------------------------------------------------------------
# memory_add
# ---------------------------------------------------------------------------


class MemoryAddInput(BaseModel):
    fact: str = Field(description="A single fact to remember about this project")
    tags: str = Field(
        default="",
        description="Comma-separated tags for categorisation, e.g. 'build,python'",
    )


class MemoryAddTool(BaseTool):
    name = "memory_add"
    description = (
        "Store a fact in persistent semantic memory. "
        "Use this when you learn something important about the project that you "
        "want to remember across sessions — tech stack, architecture patterns, "
        "file conventions, user preferences that aren't in core memory, etc."
    )
    input_model = MemoryAddInput

    async def execute(self, arguments: MemoryAddInput) -> ToolResult:
        if not arguments.fact.strip():
            return ToolResult("fact is required", is_error=True)
        tags = [t.strip() for t in arguments.tags.split(",") if t.strip()]
        store = SemanticStore(str(self.cwd))
        entry_id = store.add(arguments.fact.strip(), tags=tags)
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return ToolResult(
            f"semantic → {store._path}\n"
            f"         id={entry_id}\n"
            f"         fact: {arguments.fact.strip()}{tag_str}"
        )


# ---------------------------------------------------------------------------
# memory_log
# ---------------------------------------------------------------------------


class MemoryLogInput(BaseModel):
    task: str = Field(description="Short title of the completed task")
    summary: str = Field(description="What was done and how")
    files_touched: str = Field(default="", description="Comma-separated list of files modified")
    outcome: str = Field(default="", description="Result — success, failure, or notes")


class MemoryLogTool(BaseTool):
    name = "memory_log"
    description = (
        "Record a completed task in episodic memory. "
        "Call this after finishing a non-trivial task so you can recall what you "
        "did and how you did it in future sessions.  Include the task title, a "
        "brief summary, files you touched, and the outcome."
    )
    input_model = MemoryLogInput

    async def execute(self, arguments: MemoryLogInput) -> ToolResult:
        if not arguments.task.strip():
            return ToolResult("task is required", is_error=True)
        files = [f.strip() for f in arguments.files_touched.split(",") if f.strip()]
        store = EpisodicStore(str(self.cwd))
        entry_id = store.log(
            task=arguments.task.strip(),
            summary=arguments.summary.strip(),
            files_touched=files,
            outcome=arguments.outcome.strip(),
        )
        file_str = ", ".join(files) if files else "(none)"
        outcome_str = arguments.outcome.strip() or "(not specified)"
        return ToolResult(
            f"episodic → {store._path}\n"
            f"         id={entry_id}\n"
            f"         task: {arguments.task.strip()}\n"
            f"         summary: {arguments.summary.strip()}\n"
            f"         files: {file_str}\n"
            f"         outcome: {outcome_str}"
        )
