"""Sandbox session management.

OpenHarness uses a module-level singleton to track the active sandbox.
MiniHarness mirrors this pattern so tools can query sandbox state at call time
without passing a session object through every layer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from miniharness.sandbox.docker import DockerSandbox

logger = logging.getLogger(__name__)

# Module-level singleton — exactly one sandbox per process.
_active_session: DockerSandbox | None = None


def is_sandbox_active() -> bool:
    """Return True when a sandbox container is running."""
    return _active_session is not None and _active_session.is_running


def get_sandbox() -> DockerSandbox | None:
    """Return the active sandbox, or None."""
    return _active_session if is_sandbox_active() else None


async def start_sandbox(*, cwd: Path, image: str = "openharness-sandbox:latest") -> DockerSandbox:
    """Start a Docker sandbox container and register it as the active session.

    Returns the running session.  If a session is already active, it is
    returned as-is (idempotent).
    """
    global _active_session

    if is_sandbox_active():
        return _active_session  # type: ignore[return-value]

    session = DockerSandbox(cwd=cwd, image=image)
    await session.start()
    _active_session = session
    logger.info("Sandbox started (container=%s)", session.container_name)
    return session


async def stop_sandbox() -> None:
    """Stop and remove the active sandbox container."""
    global _active_session

    if _active_session is not None:
        await _active_session.stop()
        _active_session = None
        logger.info("Sandbox stopped")
