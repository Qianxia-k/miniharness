"""Tool for spawning delegated local agent tasks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from miniharness.swarm.registry import get_backend_registry
from miniharness.swarm.types import TeammateSpawnConfig
from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


class AgentInput(BaseModel):
    """Arguments for spawning a delegated local agent."""

    description: str = Field(description="Short description of the delegated work")
    prompt: str = Field(description="Full prompt for the delegated local agent")
    subagent_type: str | None = Field(
        default=None,
        description="Optional agent role label, such as reviewer, researcher, or general-purpose",
    )
    team: str | None = Field(default=None, description="Optional team namespace for the delegated agent")
    model: str | None = Field(default=None, description="Optional model override for the delegated agent")
    command: str | None = Field(default=None, description="Optional custom command for tests or deployments")


class AgentListInput(BaseModel):
    """Arguments for listing delegated local agents."""

    team: str | None = Field(default=None, description="Optional team filter")


class AgentTool(BaseTool):
    """Spawn a local background agent task.

    This is the semantic model-facing entry point.  It deliberately reuses the
    background task manager so spawned agents are pollable through task_get and
    task_output, matching OpenHarness's subprocess-agent shape.
    """

    name = "agent"
    description = (
        "Delegate independent or long-running work to a local background MiniHarness "
        "agent. Returns an agent_id plus task_id; inspect output with task_output "
        "and send follow-up messages with send_message."
    )
    input_model = AgentInput

    def permission_requests(self, arguments: AgentInput) -> list[ToolPermissionRequest]:
        description = arguments.description.strip()
        preview = arguments.command or f"local_agent: {description}"
        return [ToolPermissionRequest(
            is_read_only=False,
            command=preview,
            reason=f"Allow agent to spawn local agent task: {description[:120]}?",
        )]

    async def execute(self, arguments: AgentInput) -> ToolResult:
        description = arguments.description.strip()
        prompt = arguments.prompt.strip()
        if not description:
            return ToolResult("description is required", is_error=True)
        if not prompt:
            return ToolResult("prompt is required", is_error=True)

        executor = get_backend_registry().get_executor("subprocess")
        result = await executor.spawn(
            TeammateSpawnConfig(
                name=arguments.subagent_type or "agent",
                team=arguments.team or "default",
                prompt=prompt,
                description=description,
                cwd=self.cwd,
                model=arguments.model,
                command=arguments.command,
            )
        )
        if not result.success:
            return ToolResult(result.error or "Failed to spawn agent", is_error=True)
        return ToolResult(
            f"Spawned agent {result.agent_id} "
            f"(task_id={result.task_id}, backend={result.backend_type})"
        )


class AgentListTool(BaseTool):
    """List delegated local agents and their backing task state."""

    name = "agent_list"
    description = "List delegated local agents, their teams, task IDs, and current task status."
    input_model = AgentListInput

    def is_read_only(self, arguments: AgentListInput) -> bool: return True

    async def execute(self, arguments: AgentListInput) -> ToolResult:
        from miniharness.tasks import (
            get_agent_registry,
            get_background_task_manager,
            get_team_registry,
            team_store_path,
        )

        manager = get_background_task_manager()
        agents = get_agent_registry()
        teams = get_team_registry()
        agents.restore_from_tasks(manager.list_tasks())
        teams.load(team_store_path(manager.tasks_dir))
        teams.restore_from_agents(agents.list_agents())

        team = (arguments.team or "").strip()
        records = agents.list_agents()
        if team:
            records = [record for record in records if record.team == team]
        if not records:
            return ToolResult("(no delegated agents)")

        lines: list[str] = []
        for record in records:
            task = manager.get_task(record.task_id)
            status = task.status if task is not None else "missing"
            note = ""
            if task is not None:
                raw_note = task.metadata.get("status_note", "")
                note = f" ({raw_note})" if raw_note else ""
            lines.append(
                f"{record.agent_id} task_id={record.task_id} "
                f"status={status}{note} backend={record.backend_type} "
                f"description={record.description}"
            )
        return ToolResult("\n".join(lines))
