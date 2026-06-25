"""Types shared by MiniHarness agent backends."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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


class TeammateExecutor(Protocol):
    """Backend interface for delegated agents."""

    backend_type: str

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
