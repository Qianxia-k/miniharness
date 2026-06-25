"""Tools for lightweight delegated-agent teams."""

from __future__ import annotations

from pydantic import BaseModel, Field

from miniharness.tasks import (
    get_agent_registry,
    get_background_task_manager,
    get_team_registry,
    team_store_path,
)
from miniharness.tools.base import BaseTool, ToolResult


class TeamCreateInput(BaseModel):
    """Arguments for creating a team."""

    name: str = Field(description="Team name")
    description: str = Field(default="", description="Team description")


class TeamDeleteInput(BaseModel):
    """Arguments for deleting a team."""

    name: str = Field(description="Team name")


class TeamListInput(BaseModel):
    """Arguments for listing teams."""

    include_agents: bool = Field(default=True, description="Whether to include team members")


class TeamCreateTool(BaseTool):
    """Create a lightweight in-memory team."""

    name = "team_create"
    description = "Create a lightweight team namespace for delegated local agents."
    input_model = TeamCreateInput

    async def execute(self, arguments: TeamCreateInput) -> ToolResult:
        manager = get_background_task_manager()
        teams = get_team_registry()
        path = team_store_path(manager.tasks_dir)
        teams.load(path)
        try:
            team = teams.create_team(arguments.name, arguments.description)
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)
        teams.save(path)
        return ToolResult(f"Created team {team.name}")


class TeamDeleteTool(BaseTool):
    """Delete an empty lightweight team."""

    name = "team_delete"
    description = "Delete an empty lightweight delegated-agent team."
    input_model = TeamDeleteInput

    async def execute(self, arguments: TeamDeleteInput) -> ToolResult:
        manager = get_background_task_manager()
        teams = get_team_registry()
        agents = get_agent_registry()
        path = team_store_path(manager.tasks_dir)
        teams.load(path)
        agents.restore_from_tasks(manager.list_tasks())
        teams.restore_from_agents(agents.list_agents())

        team = teams.get_team(arguments.name)
        if team is None:
            return ToolResult(f"Team '{arguments.name.strip()}' does not exist", is_error=True)
        if team.agents:
            return ToolResult(
                f"Team '{team.name}' is not empty; stop or move agents before deleting it.",
                is_error=True,
            )
        try:
            teams.delete_team(arguments.name)
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)
        teams.save(path)
        return ToolResult(f"Deleted team {team.name}")


class TeamListTool(BaseTool):
    """List lightweight delegated-agent teams."""

    name = "team_list"
    description = "List delegated-agent teams and their current agent members."
    input_model = TeamListInput

    async def execute(self, arguments: TeamListInput) -> ToolResult:
        manager = get_background_task_manager()
        agents = get_agent_registry()
        teams = get_team_registry()
        teams.load(team_store_path(manager.tasks_dir))
        agents.restore_from_tasks(manager.list_tasks())
        teams.restore_from_agents(agents.list_agents())
        teams.save(team_store_path(manager.tasks_dir))

        records = teams.list_teams()
        if not records:
            return ToolResult("(no teams)")

        lines: list[str] = []
        for team in records:
            description = f" description={team.description}" if team.description else ""
            line = f"{team.name} agents={len(team.agents)}{description}"
            lines.append(line)
            if arguments.include_agents:
                for agent_id in team.agents:
                    agent = agents.get(agent_id)
                    task_id = agent.task_id if agent is not None else "-"
                    status = "missing"
                    if agent is not None:
                        task = manager.get_task(agent.task_id)
                        status = task.status if task is not None else "missing"
                    lines.append(f"  - {agent_id} task_id={task_id} status={status}")
        return ToolResult("\n".join(lines))
