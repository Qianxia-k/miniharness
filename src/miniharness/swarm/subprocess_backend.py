"""Subprocess backend for delegated MiniHarness agents."""

from __future__ import annotations

import os

from miniharness.tasks import (
    get_agent_registry,
    get_background_task_manager,
    get_team_registry,
    team_store_path,
)
from miniharness.swarm.types import SpawnResult, TeammateMessage, TeammateSpawnConfig


def _inherited_env(config: TeammateSpawnConfig) -> dict[str, str]:
    """Build env vars to forward to spawned sub-agents.

    Mirrors OpenHarness's ``build_inherited_env_vars()`` to ensure spawned
    agents have access to API keys, proxy settings, and SSL config.
    """
    env: dict[str, str] = {}
    for key in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "DASHSCOPE_API_KEY",
        "MINIHARNESS_API_KEY", "MINIHARNESS_BASE_URL",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy",
        "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
        "PATH", "HOME", "USER",
    ):
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


class SubprocessBackend:
    """Run each delegated agent as a local background task."""

    backend_type = "subprocess"

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
        try:
            task = await manager.create_agent_task(
                prompt=config.prompt,
                description=task_description,
                cwd=config.cwd,
                model=config.model,
                command=config.command,
                keep_stdin_open=True,
                extra_env=_inherited_env(config),
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
