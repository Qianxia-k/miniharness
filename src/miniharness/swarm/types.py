"""Types shared by MiniHarness agent backends."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


BackendType = Literal["subprocess", "in_process", "tmux", "iterm2", "remote"]


@dataclass(frozen=True)
class BackendStatus:
    """Availability information for one teammate backend."""

    backend_type: str
    available: bool
    active: bool = False
    reason: str = ""


@dataclass(frozen=True)
class TeammateSpawnConfig:
    """Configuration for spawning a delegated local agent."""

    name: str
    team: str
    prompt: str
    description: str
    cwd: str | Path
    model: str | None = None
    command: str | None = None
    system_prompt: str | None = None
    system_prompt_mode: str | None = None
    hooks: dict | None = None
    permissions: list[str] | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class SpawnResult:
    """Result returned by a teammate backend after spawn."""

    task_id: str
    agent_id: str
    backend_type: str
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class TeammateMessage:
    """Message sent from the coordinator to a delegated agent."""

    text: str
    from_agent: str = "coordinator"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time.time())


@dataclass(frozen=True)
class TeammateStatus:
    """Runtime status for one delegated agent."""

    agent_id: str
    task_id: str
    status: str
    backend_type: str
    description: str
    status_note: str = ""


class TeammateExecutor(Protocol):
    """Backend interface for delegated agents."""

    backend_type: str

    def is_available(self) -> bool:
        """Return whether this backend can run in the current process."""
        ...

    async def spawn(self, config: TeammateSpawnConfig) -> SpawnResult:
        """Spawn a delegated agent."""
        ...

    async def send_message(self, agent_id: str, message: TeammateMessage) -> None:
        """Send a follow-up message to a delegated agent."""
        ...

    async def shutdown(self, agent_id: str, *, force: bool = False) -> bool:
        """Terminate a delegated agent."""
        ...

    def get_task_id(self, agent_id: str) -> str | None:
        """Return the backing background task id for an agent id."""
        ...

    def list_agents(self, *, team: str | None = None) -> list[TeammateStatus]:
        """Return delegated agents known to this backend."""
        ...
