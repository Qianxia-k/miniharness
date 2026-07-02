"""Subprocess backend for delegated MiniHarness agents."""

from __future__ import annotations

from miniharness.tasks import (
    get_agent_registry,
    get_background_task_manager,
    get_team_registry,
    team_store_path,
)
from miniharness.swarm.types import (
    SpawnResult,
    TeammateMessage,
    TeammateSpawnConfig,
    TeammateStatus,
)
from miniharness.swarm.spawn_utils import (
    build_inherited_env_vars,
    build_teammate_argv,
    encode_agent_hooks_env,
    encode_agent_identity_env,
    encode_agent_max_turns_env,
    encode_agent_permission_mode_env,
    encode_agent_tool_policy_env,
)


class SubprocessBackend:
    """Run each delegated agent as a local background task."""

    backend_type = "subprocess"

    def is_available(self) -> bool:
        """Subprocess teammates are the portable fallback backend."""
        return True

    async def spawn(self, config: TeammateSpawnConfig) -> SpawnResult:
        agents = get_agent_registry()
        teams = get_team_registry()
        manager = get_background_task_manager()
        teams.load(team_store_path(manager.tasks_dir))
        agent_id, agent_name, team = agents.allocate_agent_id(
            name=config.name,
            team=config.team,
        )
        teams.ensure_team(team)
        task_description = f"{agent_name}: {config.description.strip()}"
        argv = None if config.command is not None else build_teammate_argv(
            cwd=config.cwd,
            model=config.model,
            system_prompt=config.system_prompt,
            system_prompt_mode=config.system_prompt_mode,
        )
        extra_env = build_inherited_env_vars()
        extra_env.update(encode_agent_identity_env(
            agent_id=agent_id,
            agent_name=agent_name,
            team=team,
        ))
        extra_env.update(encode_agent_hooks_env(config.hooks))
        extra_env.update(encode_agent_max_turns_env(config.max_turns))
        extra_env.update(encode_agent_permission_mode_env(config.permission_mode))
        extra_env.update(encode_agent_tool_policy_env(
            tools=config.tools,
            disallowed_tools=config.disallowed_tools,
        ))
        try:
            task = await manager.create_agent_task(
                prompt=config.prompt,
                description=task_description,
                cwd=config.cwd,
                model=config.model,
                command=config.command,
                argv=argv,
                keep_stdin_open=True,
                extra_env=extra_env,
            )
        except Exception as exc:
            return SpawnResult(
                task_id="",
                agent_id=agent_id,
                backend_type=self.backend_type,
                success=False,
                error=str(exc),
            )

        agents.register(
            agent_id=agent_id,
            name=agent_name,
            team=team,
            task_id=task.id,
            backend_type=self.backend_type,
            description=config.description.strip(),
        )
        manager.update_task_metadata(task.id, {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "team": team,
            "backend_type": self.backend_type,
            "agent_description": config.description.strip(),
            **(config.metadata or {}),
        })
        teams.add_agent(team, agent_id)
        teams.save(team_store_path(manager.tasks_dir))
        return SpawnResult(
            task_id=task.id,
            agent_id=agent_id,
            backend_type=self.backend_type,
        )

    async def send_message(self, agent_id: str, message: TeammateMessage) -> None:
        task_id = self.get_task_id(agent_id)
        if task_id is None:
            raise ValueError(f"No active subprocess for agent {agent_id!r}")
        await get_background_task_manager().write_to_task(task_id, message.text)

    async def shutdown(self, agent_id: str, *, force: bool = False) -> bool:
        del force
        task_id = self.get_task_id(agent_id)
        if task_id is None:
            return False
        try:
            await get_background_task_manager().stop_task(task_id)
        except ValueError:
            return False
        return True

    def get_task_id(self, agent_id: str) -> str | None:
        manager = get_background_task_manager()
        agents = get_agent_registry()
        task_id = agents.resolve_task_id(agent_id)
        if task_id is not None:
            return task_id
        agents.restore_from_tasks(manager.list_tasks())
        return agents.resolve_task_id(agent_id)

    def list_agents(self, *, team: str | None = None) -> list[TeammateStatus]:
        manager = get_background_task_manager()
        agents = get_agent_registry()
        teams = get_team_registry()
        agents.restore_from_tasks(manager.list_tasks())
        teams.load(team_store_path(manager.tasks_dir))
        teams.restore_from_agents(agents.list_agents())

        team_filter = (team or "").strip()
        records = agents.list_agents()
        if team_filter:
            records = [record for record in records if record.team == team_filter]
        statuses: list[TeammateStatus] = []
        for record in records:
            task = manager.get_task(record.task_id)
            statuses.append(
                TeammateStatus(
                    agent_id=record.agent_id,
                    task_id=record.task_id,
                    status=task.status if task is not None else "missing",
                    backend_type=record.backend_type,
                    description=record.description,
                    status_note=task.metadata.get("status_note", "") if task is not None else "",
                )
            )
        return statuses
