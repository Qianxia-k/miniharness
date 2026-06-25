"""Tools for local background tasks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from miniharness.tasks import get_background_task_manager
from miniharness.tools.bash import _reject_probable_markup
from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


class TaskCreateInput(BaseModel):
    """Arguments for creating a background task."""

    type: str = Field(default="local_bash", description="Task type: local_bash or local_agent.")
    description: str = Field(description="Short human-readable task description")
    command: str | None = Field(default=None, description="Shell command for local_bash tasks")
    prompt: str | None = Field(default=None, description="Prompt for local_agent tasks")
    model: str | None = Field(default=None, description="Optional model override for local_agent tasks")


class TaskCreateTool(BaseTool):
    """Create a background task."""

    name = "task_create"
    description = "Create a local background shell task for long-running work."
    input_model = TaskCreateInput

    def permission_requests(self, arguments: TaskCreateInput) -> list[ToolPermissionRequest]:
        if arguments.type == "local_bash" and arguments.command:
            return [ToolPermissionRequest(
                is_read_only=False,
                command=arguments.command,
                reason=f"Allow task_create to run background command: {arguments.command[:120]}?",
            )]
        if arguments.type == "local_agent":
            preview = arguments.command or f"local_agent: {arguments.description}"
            return [ToolPermissionRequest(
                is_read_only=False,
                command=preview,
                reason=f"Allow task_create to spawn local agent task: {arguments.description[:120]}?",
            )]
        return [ToolPermissionRequest(
            is_read_only=False,
            reason=f"Allow task_create to create background task type {arguments.type}?",
        )]

    async def execute(self, arguments: TaskCreateInput) -> ToolResult:
        try:
            if arguments.type == "local_bash":
                if not arguments.command:
                    return ToolResult("command is required for local_bash tasks", is_error=True)
                markup_error = _reject_probable_markup(arguments.command)
                if markup_error:
                    return ToolResult(markup_error, is_error=True)
                task = await get_background_task_manager().create_shell_task(
                    command=arguments.command,
                    description=arguments.description,
                    cwd=self.cwd,
                )
            elif arguments.type == "local_agent":
                if not arguments.prompt:
                    return ToolResult("prompt is required for local_agent tasks", is_error=True)
                task = await get_background_task_manager().create_agent_task(
                    prompt=arguments.prompt,
                    description=arguments.description,
                    cwd=self.cwd,
                    model=arguments.model,
                    command=arguments.command,
                )
            else:
                return ToolResult(f"unsupported task type: {arguments.type}", is_error=True)
        except Exception as exc:
            return ToolResult(str(exc), is_error=True)
        return ToolResult(f"Created background task {task.id} ({task.type})")


class TaskListInput(BaseModel):
    """Arguments for listing background tasks."""

    status: str | None = Field(default=None, description="Optional status filter")


class TaskListTool(BaseTool):
    """List background tasks."""

    name = "task_list"
    description = "List local background tasks."
    input_model = TaskListInput

    def is_read_only(self, arguments: TaskListInput) -> bool: return True

    async def execute(self, arguments: TaskListInput) -> ToolResult:
        tasks = get_background_task_manager().list_tasks(status=arguments.status)
        if not tasks:
            return ToolResult("(no background tasks)")
        return ToolResult("\n".join(task.to_summary() for task in tasks))


class TaskGetInput(BaseModel):
    """Arguments for reading one background task record."""

    task_id: str = Field(description="Background task id")


class TaskGetTool(BaseTool):
    """Get one background task."""

    name = "task_get"
    description = "Get details for a background task."
    input_model = TaskGetInput

    def is_read_only(self, arguments: TaskGetInput) -> bool: return True

    async def execute(self, arguments: TaskGetInput) -> ToolResult:
        task = get_background_task_manager().get_task(arguments.task_id)
        if task is None:
            return ToolResult(f"No background task found with ID: {arguments.task_id}", is_error=True)
        lines = [
            f"id: {task.id}",
            f"type: {task.type}",
            f"status: {task.status}",
            f"description: {task.description}",
            f"cwd: {task.cwd}",
            f"command: {task.command}",
            f"prompt: {(task.prompt or '')[:500]}",
            f"argv: {task.argv or []}",
            f"output_file: {task.output_file}",
            f"return_code: {task.return_code}",
        ]
        if task.metadata:
            lines.append(f"metadata: {task.metadata}")
        return ToolResult("\n".join(lines))


class TaskOutputInput(BaseModel):
    """Arguments for reading background task output."""

    task_id: str = Field(description="Background task id")
    max_bytes: int = Field(default=12000, ge=1, le=100000)


class TaskOutputTool(BaseTool):
    """Read background task output."""

    name = "task_output"
    description = "Read the captured output log for a background task."
    input_model = TaskOutputInput

    def is_read_only(self, arguments: TaskOutputInput) -> bool: return True

    async def execute(self, arguments: TaskOutputInput) -> ToolResult:
        try:
            output = get_background_task_manager().read_output(
                arguments.task_id,
                max_bytes=arguments.max_bytes,
            )
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)
        return ToolResult(output or "(no output)")


class TaskStopInput(BaseModel):
    """Arguments for stopping a background task."""

    task_id: str = Field(description="Background task id")


class TaskStopTool(BaseTool):
    """Stop a background task."""

    name = "task_stop"
    description = "Stop a running background task."
    input_model = TaskStopInput

    async def execute(self, arguments: TaskStopInput) -> ToolResult:
        try:
            task = await get_background_task_manager().stop_task(arguments.task_id)
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)
        return ToolResult(f"Stopped background task {task.id} ({task.status})")


class TaskUpdateInput(BaseModel):
    """Arguments for updating background task metadata."""

    task_id: str = Field(description="Background task id")
    description: str | None = Field(default=None, description="Updated task description")
    progress: int | None = Field(default=None, ge=0, le=100, description="Progress percentage")
    status_note: str | None = Field(default=None, description="Short status note")


class TaskUpdateTool(BaseTool):
    """Update background task metadata."""

    name = "task_update"
    description = "Update a background task description, progress, or status note."
    input_model = TaskUpdateInput

    async def execute(self, arguments: TaskUpdateInput) -> ToolResult:
        try:
            task = get_background_task_manager().update_task(
                arguments.task_id,
                description=arguments.description,
                progress=arguments.progress,
                status_note=arguments.status_note,
            )
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)
        parts = [f"Updated background task {task.id}: {task.to_summary()}"]
        progress_val = task.metadata.get("progress")
        note_val = task.metadata.get("status_note")
        if progress_val:
            parts.append(f"progress={progress_val}%")
        if note_val:
            parts.append(f"note={note_val}")
        return ToolResult(" ".join(parts))
