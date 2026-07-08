"""Runtime registry for delegated local agents.

The background task manager owns process lifecycle.  This registry owns the
coordination-facing identity layer: stable ``name@team`` agent IDs that route
back to background task IDs.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


class AgentTaskRecord(Protocol):
    """Minimal task-record shape needed to restore agent routing."""

    id: str
    description: str
    created_at: float
    metadata: dict[str, str]


@dataclass(frozen=True)
class AgentRecord:
    """Mapping from a coordinator-facing agent id to a background task."""

    agent_id: str
    name: str
    team: str
    task_id: str
    backend_type: str
    description: str
    created_at: float


@dataclass
class TeamRecord:
    """Lightweight team for delegated local agents."""

    name: str
    description: str = ""
    agents: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class TeamRegistry:
    """Store teams and agent memberships."""

    def __init__(self) -> None:
        self._teams: dict[str, TeamRecord] = {}

    def create_team(self, name: str, description: str = "") -> TeamRecord:
        normalized = _normalize_segment(name, fallback="default")
        if normalized in self._teams:
            raise ValueError(f"Team '{normalized}' already exists")
        team = TeamRecord(name=normalized, description=description.strip())
        self._teams[normalized] = team
        return team

    def ensure_team(self, name: str, description: str = "") -> TeamRecord:
        normalized = _normalize_segment(name, fallback="default")
        team = self._teams.get(normalized)
        if team is not None:
            return team
        return self.create_team(normalized, description)

    def delete_team(self, name: str) -> None:
        normalized = _normalize_segment(name, fallback="default")
        if normalized not in self._teams:
            raise ValueError(f"Team '{normalized}' does not exist")
        del self._teams[normalized]

    def add_agent(self, team_name: str, agent_id: str) -> None:
        team = self.ensure_team(team_name)
        if agent_id not in team.agents:
            team.agents.append(agent_id)

    def remove_agent(self, team_name: str, agent_id: str) -> bool:
        team = self.get_team(team_name)
        if team is None or agent_id not in team.agents:
            return False
        team.agents = [existing for existing in team.agents if existing != agent_id]
        return True

    def remove_agent_everywhere(self, agent_id: str) -> int:
        removed = 0
        for team in self._teams.values():
            if agent_id in team.agents:
                team.agents = [existing for existing in team.agents if existing != agent_id]
                removed += 1
        return removed

    def get_team(self, name: str) -> TeamRecord | None:
        return self._teams.get(_normalize_segment(name, fallback="default"))

    def list_teams(self) -> list[TeamRecord]:
        return sorted(self._teams.values(), key=lambda team: team.name)

    def load(self, path: str | Path) -> int:
        """Load persisted teams from *path* and merge them into the registry."""
        team_path = Path(path)
        if not team_path.exists():
            return 0
        try:
            payload = json.loads(team_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0
        raw_teams = payload.get("teams") if isinstance(payload, dict) else None
        if not isinstance(raw_teams, list):
            return 0

        loaded = 0
        for raw in raw_teams:
            if not isinstance(raw, dict):
                continue
            name = _normalize_segment(str(raw.get("name") or ""), fallback="")
            if not name:
                continue
            description = str(raw.get("description") or "")
            agents_raw = raw.get("agents")
            agents = [
                str(agent).strip()
                for agent in agents_raw
                if str(agent).strip()
            ] if isinstance(agents_raw, list) else []
            created_at = _coerce_float(raw.get("created_at"), default=time.time())
            existing = self._teams.get(name)
            if existing is None:
                self._teams[name] = TeamRecord(
                    name=name,
                    description=description,
                    agents=list(dict.fromkeys(agents)),
                    created_at=created_at,
                )
                loaded += 1
            else:
                if description and not existing.description:
                    existing.description = description
                for agent in agents:
                    if agent not in existing.agents:
                        existing.agents.append(agent)
        return loaded

    def save(self, path: str | Path) -> None:
        """Persist all teams to *path* atomically."""
        team_path = Path(path)
        team_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "teams": [
                {
                    "name": team.name,
                    "description": team.description,
                    "agents": list(team.agents),
                    "created_at": team.created_at,
                }
                for team in self.list_teams()
            ],
        }
        data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        tmp_path = team_path.with_name(f".{team_path.name}.{int(time.time() * 1000000)}.tmp")
        tmp_path.write_text(data, encoding="utf-8")
        tmp_path.replace(team_path)

    def restore_from_agents(self, agents: Iterable[AgentRecord]) -> int:
        restored = 0
        for agent in agents:
            before = self.get_team(agent.team)
            team = self.ensure_team(agent.team)
            if before is None:
                restored += 1
            if agent.agent_id not in team.agents:
                team.agents.append(agent.agent_id)
        return restored


class AgentRegistry:
    """In-process registry for local delegated agents."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._task_to_agent: dict[str, str] = {}

    def allocate_agent_id(self, *, name: str | None, team: str | None) -> tuple[str, str, str]:
        """Return a unique ``agent_id, normalized_name, normalized_team`` tuple."""
        normalized_name = _normalize_segment(name, fallback="agent")
        normalized_team = _normalize_segment(team, fallback="default")
        base_id = f"{normalized_name}@{normalized_team}"
        if base_id not in self._agents:
            return base_id, normalized_name, normalized_team

        index = 2
        while True:
            candidate = f"{normalized_name}-{index}@{normalized_team}"
            if candidate not in self._agents:
                return candidate, f"{normalized_name}-{index}", normalized_team
            index += 1

    def register(
        self,
        *,
        agent_id: str,
        name: str,
        team: str,
        task_id: str,
        backend_type: str,
        description: str,
    ) -> AgentRecord:
        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            team=team,
            task_id=task_id,
            backend_type=backend_type,
            description=description,
            created_at=time.time(),
        )
        self._agents[agent_id] = record
        self._task_to_agent[task_id] = agent_id
        return record

    def get(self, agent_id: str) -> AgentRecord | None:
        return self._agents.get(agent_id.strip())

    def get_by_task_id(self, task_id: str) -> AgentRecord | None:
        agent_id = self._task_to_agent.get(task_id.strip())
        if not agent_id:
            return None
        return self._agents.get(agent_id)

    def resolve_task_id(self, target: str) -> str | None:
        record = self.get(target)
        if record is not None:
            return record.task_id
        return None

    def remove(self, target: str) -> AgentRecord | None:
        normalized = target.strip()
        record = self._agents.pop(normalized, None)
        if record is None:
            agent_id = self._task_to_agent.get(normalized)
            if agent_id:
                record = self._agents.pop(agent_id, None)
        if record is None:
            return None
        self._task_to_agent.pop(record.task_id, None)
        return record

    def restore_from_tasks(self, tasks: Iterable[AgentTaskRecord]) -> int:
        """Restore agent routing entries from persisted task metadata."""
        restored = 0
        for task in tasks:
            if str(task.metadata.get("agent_removed") or "").lower() == "true":
                continue
            agent_id = str(task.metadata.get("agent_id") or "").strip()
            if not agent_id or agent_id in self._agents:
                continue
            name = str(task.metadata.get("agent_name") or agent_id.split("@", 1)[0] or "agent").strip()
            team = str(task.metadata.get("team") or (agent_id.split("@", 1)[1] if "@" in agent_id else "default")).strip()
            backend_type = str(task.metadata.get("backend_type") or "local_agent").strip()
            description = str(task.metadata.get("agent_description") or task.description or agent_id).strip()
            self._agents[agent_id] = AgentRecord(
                agent_id=agent_id,
                name=name or "agent",
                team=team or "default",
                task_id=task.id,
                backend_type=backend_type or "local_agent",
                description=description,
                created_at=float(task.created_at or time.time()),
            )
            self._task_to_agent[task.id] = agent_id
            restored += 1
        return restored

    def list_agents(self) -> list[AgentRecord]:
        return sorted(self._agents.values(), key=lambda record: record.created_at)


_GLOBAL_AGENT_REGISTRY: AgentRegistry | None = None
_GLOBAL_TEAM_REGISTRY: TeamRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _GLOBAL_AGENT_REGISTRY
    if _GLOBAL_AGENT_REGISTRY is None:
        _GLOBAL_AGENT_REGISTRY = AgentRegistry()
    return _GLOBAL_AGENT_REGISTRY


def get_team_registry() -> TeamRegistry:
    global _GLOBAL_TEAM_REGISTRY
    if _GLOBAL_TEAM_REGISTRY is None:
        _GLOBAL_TEAM_REGISTRY = TeamRegistry()
    return _GLOBAL_TEAM_REGISTRY


def reset_agent_registry_for_tests(registry: AgentRegistry | None = None) -> AgentRegistry:
    global _GLOBAL_AGENT_REGISTRY
    _GLOBAL_AGENT_REGISTRY = registry or AgentRegistry()
    return _GLOBAL_AGENT_REGISTRY


def reset_team_registry_for_tests(registry: TeamRegistry | None = None) -> TeamRegistry:
    global _GLOBAL_TEAM_REGISTRY
    _GLOBAL_TEAM_REGISTRY = registry or TeamRegistry()
    return _GLOBAL_TEAM_REGISTRY


def team_store_path(tasks_dir: str | Path) -> Path:
    return Path(tasks_dir) / "teams.json"


def _normalize_segment(value: str | None, *, fallback: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raw = fallback
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")
    return normalized or fallback


def _coerce_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
