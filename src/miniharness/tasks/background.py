"""Background task manager for long-running local work.

This mirrors the OpenHarness shape without pretending to support every backend
up front.  The first production slice supports local shell tasks with:

- stable task records;
- output captured to disk;
- status transitions;
- cooperative stop;
- read-only inspection tools.

Local agent/subagent tasks can be added on top of this manager later.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BackgroundTaskStatus = Literal["running", "completed", "failed", "killed"]
BackgroundTaskType = Literal["local_bash"]


@dataclass
class BackgroundTaskRecord:
    """Runtime record for one background task."""

    id: str
    type: BackgroundTaskType
    status: BackgroundTaskStatus
    description: str
    cwd: str
    output_file: Path
    command: str
    created_at: float
    started_at: float
    ended_at: float | None = None
    return_code: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_summary(self) -> str:
        note = self.metadata.get("status_note", "")
        suffix = f" ({note})" if note else ""
        return f"{self.id} {self.type} {self.status} {self.description}{suffix}"


class BackgroundTaskManager:
    """Manage local background shell tasks."""

    def __init__(self, *, tasks_dir: Path | None = None) -> None:
        self.tasks_dir = tasks_dir or _default_tasks_dir()
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, BackgroundTaskRecord] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._watchers: dict[str, asyncio.Task[None]] = {}
        self._output_locks: dict[str, asyncio.Lock] = {}

    async def create_shell_task(
        self,
        *,
        command: str,
        description: str,
        cwd: str | Path,
    ) -> BackgroundTaskRecord:
        command = command.strip()
        description = description.strip()
        if not command:
            raise ValueError("command is required")
        if not description:
            raise ValueError("description is required")

        task_id = f"bg-{uuid.uuid4().hex[:8]}"
        output_file = self.tasks_dir / f"{task_id}.log"
        output_file.write_text("", encoding="utf-8")
        now = time.time()
        record = BackgroundTaskRecord(
            id=task_id,
            type="local_bash",
            status="running",
            description=description,
            cwd=str(Path(cwd).expanduser().resolve()),
            output_file=output_file,
            command=command,
            created_at=now,
            started_at=now,
        )
        self._tasks[task_id] = record
        self._output_locks[task_id] = asyncio.Lock()

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=record.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._processes[task_id] = process
        self._watchers[task_id] = asyncio.create_task(
            self._watch_process(task_id, process)
        )
        return record

    def list_tasks(self, *, status: str | None = None) -> list[BackgroundTaskRecord]:
        records = list(self._tasks.values())
        if status:
            records = [record for record in records if record.status == status]
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def get_task(self, task_id: str) -> BackgroundTaskRecord | None:
        return self._tasks.get(task_id)

    def read_output(self, task_id: str, *, max_bytes: int = 12000) -> str:
        record = self._require_task(task_id)
        data = record.output_file.read_text(encoding="utf-8", errors="replace")
        if len(data) > max_bytes:
            return data[-max_bytes:]
        return data

    def update_task(
        self,
        task_id: str,
        *,
        description: str | None = None,
        progress: int | None = None,
        status_note: str | None = None,
    ) -> BackgroundTaskRecord:
        record = self._require_task(task_id)
        if description is not None and description.strip():
            record.description = description.strip()
        if progress is not None:
            if progress < 0 or progress > 100:
                raise ValueError("progress must be between 0 and 100")
            record.metadata["progress"] = str(progress)
        if status_note is not None:
            note = status_note.strip()
            if note:
                record.metadata["status_note"] = note
            else:
                record.metadata.pop("status_note", None)
        return record

    async def stop_task(self, task_id: str) -> BackgroundTaskRecord:
        record = self._require_task(task_id)
        process = self._processes.get(task_id)
        if process is None or process.returncode is not None:
            return record

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

        record.status = "killed"
        record.ended_at = time.time()
        watcher = self._watchers.get(task_id)
        if watcher is not None:
            await asyncio.gather(watcher, return_exceptions=True)
        return record

    async def close(self) -> None:
        """Terminate running tasks owned by this process."""
        running = [
            task_id
            for task_id, record in self._tasks.items()
            if record.status == "running"
        ]
        for task_id in running:
            await self.stop_task(task_id)

    async def _watch_process(
        self,
        task_id: str,
        process: asyncio.subprocess.Process,
    ) -> None:
        await self._copy_output(task_id, process)
        return_code = await process.wait()
        record = self._tasks[task_id]
        record.return_code = return_code
        if record.status != "killed":
            record.status = "completed" if return_code == 0 else "failed"
        record.ended_at = time.time()
        self._processes.pop(task_id, None)

    async def _copy_output(
        self,
        task_id: str,
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.stdout is None:
            return
        while True:
            chunk = await process.stdout.read(4096)
            if not chunk:
                return
            async with self._output_locks[task_id]:
                with self._tasks[task_id].output_file.open("ab") as handle:
                    handle.write(chunk)

    def _require_task(self, task_id: str) -> BackgroundTaskRecord:
        record = self._tasks.get(task_id)
        if record is None:
            raise ValueError(f"No background task found with ID: {task_id}")
        return record


_GLOBAL_MANAGER: BackgroundTaskManager | None = None


def get_background_task_manager() -> BackgroundTaskManager:
    """Return the process-global background task manager."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = BackgroundTaskManager()
    return _GLOBAL_MANAGER


def reset_background_task_manager_for_tests(
    manager: BackgroundTaskManager | None = None,
) -> BackgroundTaskManager:
    """Replace the global manager.  Intended for tests only."""
    global _GLOBAL_MANAGER
    _GLOBAL_MANAGER = manager or BackgroundTaskManager()
    return _GLOBAL_MANAGER


def _default_tasks_dir() -> Path:
    path = Path.home() / ".miniharness" / "background_tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path
