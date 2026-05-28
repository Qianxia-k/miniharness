"""Task management tool — the agent tracks progress on complex work.

Mirrors the ``TodoWrite`` tool in Claude Code / OpenHarness: the agent passes
the **complete** task list on every invocation (replace-all semantics).  This
is simpler for the model than individual CRUD operations.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class TaskInput(BaseModel):
    """Arguments for task."""

    tasks: str = Field(
        description=(
            "JSON array of ALL tasks the agent is working on. "
            'Each task: {"content": "...", "status": "pending|in_progress|completed"}. '
            "Include completed tasks so the user can see progress. "
            "Pass the complete list every time — tasks not in the list are removed."
        )
    )


class TaskTool(BaseTool):
    name = "task"
    description = (
        "Manage a task list (todo list) to track progress on complex multi-step work. "
        "Pass the COMPLETE task list every time — this replaces all previous tasks. "
        "Each task has a 'content' (description) and 'status' "
        "(pending / in_progress / completed). "
        "Keep completed tasks in the list to show progress. "
        "Use this for any work that requires 3+ distinct steps."
    )
    input_model = TaskInput

    async def execute(self, arguments: TaskInput) -> ToolResult:
        raw = arguments.tasks.strip()
        if not raw:
            return ToolResult("tasks is required — pass a JSON array of task objects", is_error=True)

        try:
            tasks: list[dict] = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolResult(f"Invalid JSON: {exc}", is_error=True)

        if not isinstance(tasks, list):
            return ToolResult("tasks must be a JSON array", is_error=True)

        valid_statuses = {"pending", "in_progress", "completed"}
        cleaned: list[dict] = []
        for i, t in enumerate(tasks):
            if not isinstance(t, dict):
                return ToolResult(f"Task {i} is not an object", is_error=True)
            content = str(t.get("content", "")).strip()
            status = str(t.get("status", "pending")).strip()
            if not content:
                return ToolResult(f"Task {i} has empty content", is_error=True)
            if status not in valid_statuses:
                return ToolResult(
                    f"Task {i} has invalid status '{status}' — use: {', '.join(sorted(valid_statuses))}",
                    is_error=True,
                )
            cleaned.append({"content": content, "status": status})

        # Render the task list back so the model sees the canonical state.
        lines = ["Tasks:"]
        status_icon = {"pending": "  ⬜", "in_progress": "  🔄", "completed": "  ✅"}
        for i, t in enumerate(cleaned, 1):
            icon = status_icon.get(t["status"], "  ❓")
            lines.append(f"{i}.{icon} [{t['status']}] {t['content']}")

        pending = sum(1 for t in cleaned if t["status"] == "pending")
        in_progress = sum(1 for t in cleaned if t["status"] == "in_progress")
        completed = sum(1 for t in cleaned if t["status"] == "completed")
        lines.append(
            f"\nSummary: {len(cleaned)} tasks "
            f"({pending} pending, {in_progress} in progress, {completed} completed)"
        )

        return ToolResult("\n".join(lines))
