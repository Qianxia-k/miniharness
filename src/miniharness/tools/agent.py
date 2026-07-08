"""Tool for spawning delegated local agent tasks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from miniharness.coordinator.agent_definitions import get_agent_definition
from miniharness.hooks import HookEvent
from miniharness.swarm.registry import get_backend_registry
from miniharness.swarm.types import TeammateSpawnConfig
from miniharness.tasks import get_background_task_manager
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
    isolation: str | None = Field(
        default=None,
        description="Optional isolation mode override. Supported: worktree",
    )


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

    def __init__(
        self,
        *,
        cwd,
        permissions,
        plugin_index: list[dict] | None = None,
        hook_executor=None,
    ) -> None:
        super().__init__(cwd=cwd, permissions=permissions)
        self.plugin_index = plugin_index
        self.hook_executor = hook_executor

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

        agent_def = get_agent_definition(
            arguments.subagent_type,
            cwd=self.cwd,
            plugins=self._plugins_from_index(),
        )
        agent_name = (
            agent_def.spawn_name()
            if agent_def is not None
            else (arguments.subagent_type or "agent")
        )
        model = arguments.model
        if model is None and agent_def is not None:
            model = agent_def.model
        if agent_def is not None and agent_def.initial_prompt:
            prompt = f"{agent_def.initial_prompt.strip()}\n\n{prompt}"
        metadata = {}
        if agent_def is not None:
            metadata = {
                "agent_definition": agent_def.name,
                "agent_definition_source": agent_def.source,
            }
            if agent_def.permission_mode:
                metadata["agent_permission_mode"] = agent_def.permission_mode
            if agent_def.path:
                metadata["agent_definition_path"] = agent_def.path
            if agent_def.disallowed_tools:
                metadata["agent_disallowed_tools"] = ",".join(agent_def.disallowed_tools)
            if agent_def.isolation:
                metadata["agent_isolation"] = agent_def.isolation

        isolation = _normalize_isolation(arguments.isolation)
        if isolation is None and agent_def is not None:
            isolation = agent_def.isolation

        executor = get_backend_registry().get_executor()
        result = await executor.spawn(
            TeammateSpawnConfig(
                name=agent_name,
                team=arguments.team or "default",
                prompt=prompt,
                description=description,
                cwd=self.cwd,
                model=model,
                command=arguments.command,
                system_prompt=agent_def.system_prompt if agent_def is not None else None,
                system_prompt_mode=(
                    agent_def.system_prompt_mode if agent_def is not None else None
                ),
                max_turns=agent_def.max_turns if agent_def is not None else None,
                tools=agent_def.tools if agent_def is not None else None,
                disallowed_tools=agent_def.disallowed_tools if agent_def is not None else None,
                permission_mode=agent_def.permission_mode if agent_def is not None else None,
                hooks=agent_def.hooks if agent_def is not None else None,
                isolation=isolation,
                metadata=metadata,
            )
        )
        if not result.success:
            return ToolResult(result.error or "Failed to spawn agent", is_error=True)
        await self._register_subagent_stop_hook(
            result=result,
            description=description,
            subagent_type=arguments.subagent_type or "agent",
            team=arguments.team or "default",
        )
        return ToolResult(
            f"Spawned agent {result.agent_id} "
            f"(task_id={result.task_id}, backend={result.backend_type})",
            metadata={
                "agent_id": result.agent_id,
                "task_id": result.task_id,
                "backend_type": result.backend_type,
                "description": description,
            },
        )

    async def _register_subagent_stop_hook(
        self,
        *,
        result,
        description: str,
        subagent_type: str,
        team: str,
    ) -> None:
        if self.hook_executor is None:
            return
        manager = get_background_task_manager()
        unregister = None

        async def emit_subagent_stop(task_record) -> None:
            nonlocal unregister
            if task_record.id != result.task_id:
                return
            if unregister is not None:
                unregister()
                unregister = None
            await self.hook_executor.execute(
                HookEvent.SUBAGENT_STOP,
                {
                    "agent_id": result.agent_id,
                    "task_id": result.task_id,
                    "backend_type": result.backend_type,
                    "status": task_record.status,
                    "return_code": task_record.return_code,
                    "description": description,
                    "subagent_type": subagent_type,
                    "team": team,
                },
            )

        unregister = manager.register_completion_listener(emit_subagent_stop)
        task_record = manager.get_task(result.task_id)
        if task_record is not None and task_record.status in {"completed", "failed", "killed"}:
            await emit_subagent_stop(task_record)

    def _plugins_from_index(self) -> list | None:
        if self.plugin_index is None:
            return None
        plugins = []
        for entry in self.plugin_index:
            plugin = entry.get("_plugin") if isinstance(entry, dict) else None
            if plugin is not None:
                plugins.append(plugin)
        return plugins


class AgentListTool(BaseTool):
    """List delegated local agents and their backing task state."""

    name = "agent_list"
    description = "List delegated local agents, their teams, task IDs, and current task status."
    input_model = AgentListInput

    async def execute(self, arguments: AgentListInput) -> ToolResult:
        executor = get_backend_registry().get_executor()
        statuses = executor.list_agents(team=arguments.team)
        if not statuses:
            return ToolResult("(no delegated agents)")

        lines: list[str] = []
        for status in statuses:
            note = f" ({status.status_note})" if status.status_note else ""
            lines.append(
                f"{status.agent_id} task_id={status.task_id} "
                f"status={status.status}{note} backend={status.backend_type} "
                f"{_worktree_note(status.worktree_path)}"
                f"description={status.description}"
            )
        return ToolResult("\n".join(lines))


def _normalize_isolation(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    return normalized if normalized in {"worktree"} else None


def _worktree_note(worktree_path: str) -> str:
    return f"worktree={worktree_path} " if worktree_path else ""
