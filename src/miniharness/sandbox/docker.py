"""Docker sandbox backend.

Runs bash commands inside an isolated Docker container.

Mirrors OpenHarness's DockerSandboxSession:
    - Container is named ``miniharness-sandbox-{id}``.
    - Network is disabled by default (--network none).
    - The workspace is bind-mounted at the same absolute path.
    - Commands are executed via ``docker exec``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path


class SandboxError(RuntimeError):
    """Raised when a sandbox operation fails."""


class DockerSandbox:
    """Manage a single Docker sandbox container."""

    def __init__(self, *, cwd: Path, image: str) -> None:
        self._cwd = cwd.resolve()
        self._image = image
        self._session_id = uuid.uuid4().hex[:12]
        self.container_name = f"miniharness-sandbox-{self._session_id}"
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the Docker container in detached mode."""
        if not shutil.which("docker"):
            raise SandboxError("Docker is not installed or not on PATH")

        argv = [
            "docker", "run",
            "--detach",#后台运行容器
            "--rm",#容器停止立即消除
            "--name", self.container_name,
            "--network", "none",
            "--volume", f"{self._cwd}:{self._cwd}",#把宿主机目录挂载到容器中的相同路径
            "--workdir", str(self._cwd),#设置容器启动后的工作目录
            self._image,
            "tail", "-f", "/dev/null",   # keep the container alive
        ]

        result = await _run(argv, timeout=30)
        if result.returncode != 0:
            raise SandboxError(
                f"Docker run failed (exit {result.returncode}):\n{result.stderr}"
            )
        self._running = True

    async def stop(self) -> None:
        """Stop and remove the container."""
        if not self._running:
            return

        argv = ["docker", "stop", "--time", "5", self.container_name]
        await _run(argv, timeout=15)
        self._running = False

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    async def exec_command(self, command: str) -> str:
        """Run *command* inside the sandbox container and return stdout.

        Raises SandboxError if the container is not running or the command fails.
        """
        if not self._running:
            raise SandboxError("Sandbox container is not running")

        argv = [
            "docker", "exec",
            "--workdir", str(self._cwd),
            self.container_name,
            "bash", "-lc", command,
        ]

        result = await _run(argv, timeout=60)
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code {result.returncode})"
        return output.strip()


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

async def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a subprocess asynchronously (delegates to a thread so the event
    loop stays responsive)."""
    import asyncio

    return await asyncio.to_thread(
        subprocess.run,
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
