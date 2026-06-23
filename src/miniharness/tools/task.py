"""Task management tool.

The model-facing tool uses replace-all semantics: every call submits the full
task list.  The canonical state is owned by ``TaskListManager`` and stored in
session ``tool_metadata`` so it survives compaction and session switching.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from miniharness.tasks import TaskListManager, format_task_list
from miniharness.tools.base import BaseTool, ToolResult


class TaskItemInput(BaseModel):
    """One model-submitted task item."""

    id: str | None = Field(
        default=None,
        description="Stable task id from the previous task list. Omit for new tasks.",
    )
    content: str = Field(description="Concrete task description")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Task status. Use at most one in_progress item."
    )


class TaskInput(BaseModel):
    """Arguments for task."""

    tasks: list[TaskItemInput] = Field(
        description=(
            "Complete task list. Each task has content and status "
            "(pending, in_progress, completed). "
            "Include completed tasks so the user can see progress. "
            "Pass the complete list every time; omitted tasks are removed. "
            "Use at most one in_progress task."
        )
    )


class TaskTool(BaseTool):
    name = "task"
    description = (
        "Update the current task list for complex multi-step work. "
        "Pass the COMPLETE task list every time; this replaces previous tasks. "
        "Use statuses pending, in_progress, completed. "
        "Use at most one in_progress item. "
        "Keep completed tasks in the list to show progress. "
        "Use this for any work that requires 3+ distinct steps."
    )
    input_model = TaskInput

    def __init__(self, *args, manager: TaskListManager | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.manager = manager

    async def execute(self, arguments: TaskInput) -> ToolResult:
        if self.manager is None:
            return ToolResult("Task manager is not available for this session.", is_error=True)

        try:
            tasks = self.manager.replace_all([item.model_dump() for item in arguments.tasks])
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)

        return ToolResult(format_task_list(tasks))
