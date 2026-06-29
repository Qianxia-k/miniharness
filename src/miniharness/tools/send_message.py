"""Tool for sending follow-up messages to running agent tasks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from miniharness.swarm.registry import get_backend_registry
from miniharness.swarm.types import TeammateMessage
from miniharness.tasks import get_background_task_manager
from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


class SendMessageInput(BaseModel):
    """Arguments for sending a message to a running agent task."""

    task_id: str = Field(description="Target background task id, agent-* id, or name@team agent id")
    message: str = Field(description="Message to write to the target task stdin")


class SendMessageTool(BaseTool):
    """Send a follow-up message to a running local agent task."""

    name = "send_message"
    description = (
        "Send a follow-up message to a running local agent task. The target can "
        "be a bg-* task id, agent-* compatibility id, or name@team id returned by the agent tool."
    )
    input_model = SendMessageInput

    def permission_requests(self, arguments: SendMessageInput) -> list[ToolPermissionRequest]:
        target = arguments.task_id.strip()
        preview = arguments.message.strip().replace("\n", " ")[:120]
        return [ToolPermissionRequest(
            is_read_only=False,
            command=f"send_message {target}: {preview}",
            reason=f"Allow send_message to write to {target}?",
        )]

    async def execute(self, arguments: SendMessageInput) -> ToolResult:
        target = arguments.task_id.strip()
        message = arguments.message.strip()
        if not target:
            return ToolResult("task_id is required", is_error=True)
        if not message:
            return ToolResult("message is required", is_error=True)

        if "@" in target:
            executor = get_backend_registry().get_executor()
            try:
                await executor.send_message(
                    target,
                    TeammateMessage(text=message, from_agent="coordinator"),
                )
            except ValueError as exc:
                return ToolResult(str(exc), is_error=True)
            task_id = executor.get_task_id(target) or target
            return ToolResult(f"Sent message to {target} (task_id={task_id})")

        manager = get_background_task_manager()
        task_id = manager.resolve_task_id(target)
        try:
            await manager.write_to_task(task_id, message)
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)
        return ToolResult(f"Sent message to {target} (task_id={task_id})")
