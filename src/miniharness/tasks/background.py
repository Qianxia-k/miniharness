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
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from miniharness.swarm.spawn_utils import build_teammate_argv
from miniharness.tasks.worker_protocol import encode_worker_message


BackgroundTaskStatus = Literal["running", "completed", "failed", "killed"]
BackgroundTaskType = Literal["local_bash", "local_agent"]


@dataclass
class BackgroundTaskRecord:
    """Runtime record for one background task."""

    id: str
    type: BackgroundTaskType
    status: BackgroundTaskStatus
    description: str
    cwd: str
    output_file: Path
    command: str | None
    created_at: float
    started_at: float
    prompt: str | None = None
    argv: list[str] | None = None
    env: dict[str, str] | None = None
    ended_at: float | None = None
    return_code: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_summary(self) -> str:
        note = self.metadata.get("status_note", "")
        suffix = f" ({note})" if note else ""
        return f"{self.id} {self.type} {self.status} {self.description}{suffix}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "description": self.description,
            "cwd": self.cwd,
            "output_file": str(self.output_file),
            "command": self.command,
            "prompt": self.prompt,
            "argv": list(self.argv) if self.argv is not None else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "return_code": self.return_code,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> BackgroundTaskRecord | None:
        try:
            task_id = str(payload["id"]).strip()
            task_type = str(payload.get("type") or "local_bash")
            status = str(payload.get("status") or "failed")
            description = str(payload.get("description") or "").strip()
            cwd = str(payload.get("cwd") or "").strip()
            output_file = Path(str(payload.get("output_file") or ""))
            command = str(payload.get("command") or "")
            argv_raw = payload.get("argv")
            created_at = float(payload.get("created_at") or 0.0)
            started_at = float(payload.get("started_at") or created_at)
        except (KeyError, TypeError, ValueError):
            return None

        if not task_id or task_type not in {"local_bash", "local_agent"} or status not in {"running", "completed", "failed", "killed"}:
            return None
        if not description or not cwd or not output_file:
            return None
        argv = [str(item) for item in argv_raw] if isinstance(argv_raw, list) else None

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            id=task_id,
            type=task_type,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            description=description,
            cwd=cwd,
            output_file=output_file,
            command=command or None,
            prompt=str(payload.get("prompt") or "") or None,
            argv=argv,
            created_at=created_at,
            started_at=started_at,
            ended_at=_optional_float(payload.get("ended_at")),
            return_code=_optional_int(payload.get("return_code")),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )


class BackgroundTaskManager:
    """Manage local background shell tasks."""

    def __init__(self, *, tasks_dir: Path | None = None) -> None:
        self.tasks_dir = tasks_dir or _default_tasks_dir()
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.tasks_dir / "tasks.json"
        self._tasks: dict[str, BackgroundTaskRecord] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._watchers: dict[str, asyncio.Task[None]] = {}
        self._output_locks: dict[str, asyncio.Lock] = {}
        self._input_locks: dict[str, asyncio.Lock] = {}
        self._runtime_envs: dict[str, dict[str, str]] = {}
        self._completion_listeners: dict[str, Callable[[BackgroundTaskRecord], Any]] = {}
        self._load_index()

    async def create_shell_task(
        self,
        *,
        command: str,
        description: str,
        cwd: str | Path,
    ) -> BackgroundTaskRecord:
        return await self._create_process_task(
            task_type="local_bash",
            command=command,
            argv=None,
            prompt=None,
            description=description,
            cwd=cwd,
            input_data=None,
        )

    async def create_agent_task(
        self,
        *,
        prompt: str,
        description: str,
        cwd: str | Path,
        model: str | None = None,
        command: str | None = None,
        argv: list[str] | None = None,
        keep_stdin_open: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> BackgroundTaskRecord:
        """Start a local MiniHarness agent task.

        By default this launches the current Python interpreter with
        ``-m miniharness`` and passes the prompt as a one-shot CLI prompt.  A
        command/argv override is supported for tests and custom deployments.
        Command overrides receive the prompt on stdin.  By default stdin is
        closed after the initial prompt for one-shot compatibility; interactive
        agent tools can keep it open for follow-up messages.
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")
        if command is not None and argv is not None:
            raise ValueError("create_agent_task accepts only one of command or argv")
        if command is None and argv is None:
            argv = _default_agent_argv(cwd=cwd, model=model)
            input_data = prompt + "\n"
        else:
            input_data = prompt + "\n"
        return await self._create_process_task(
            task_type="local_agent",
            command=command,
            argv=argv,
            prompt=prompt,
            description=description,
            cwd=cwd,
            input_data=input_data,
            close_stdin_after_input=not keep_stdin_open,
            extra_env=extra_env,
        )

    async def _create_process_task(
        self,
        *,
        task_type: BackgroundTaskType,
        command: str | None,
        argv: list[str] | None,
        prompt: str | None,
        description: str,
        cwd: str | Path,
        input_data: str | None,
        close_stdin_after_input: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> BackgroundTaskRecord:
        if command is not None:
            command = command.strip()
        description = description.strip()
        if command is None and argv is None:
            raise ValueError("command or argv is required")
        if command is not None and argv is not None:
            raise ValueError("only one of command or argv is allowed")
        if command is not None and not command:
            raise ValueError("command is required")
        if argv is not None and not argv:
            raise ValueError("argv is required")
        if not description:
            raise ValueError("description is required")

        task_id = f"bg-{uuid.uuid4().hex[:8]}"
        output_file = self.tasks_dir / f"{task_id}.log"
        output_file.write_text("", encoding="utf-8")
        now = time.time()
        record = BackgroundTaskRecord(
            id=task_id,
            type=task_type,
            status="running",
            description=description,
            cwd=str(Path(cwd).expanduser().resolve()),
            output_file=output_file,
            command=command,
            prompt=prompt,
            argv=list(argv) if argv is not None else None,
            created_at=now,
            started_at=now,
        )
        self._tasks[task_id] = record
        self._output_locks[task_id] = asyncio.Lock()
        if extra_env:
            self._runtime_envs[task_id] = dict(extra_env)

        process = await self._start_process(record, env=extra_env)
        self._processes[task_id] = process
        self._watchers[task_id] = asyncio.create_task(
            self._watch_process(task_id, process)
        )
        if input_data is not None and process.stdin is not None:
            process.stdin.write(encode_worker_message(input_data))
            await process.stdin.drain()
            if close_stdin_after_input:
                process.stdin.close()
        self._persist_index()
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

    async def write_to_task(self, task_id: str, message: str) -> None:
        """Write a message line to a running task's stdin."""
        resolved_task_id = self.resolve_task_id(task_id)
        record = self._require_task(resolved_task_id)
        process = await self._ensure_writable_process(record)
        if process.stdin is None or process.stdin.is_closing():
            raise ValueError(f"Task {resolved_task_id} does not accept stdin messages")

        lock = self._input_locks.setdefault(resolved_task_id, asyncio.Lock())
        async with lock:
            process.stdin.write(encode_worker_message(message))
            await process.stdin.drain()

    def register_completion_listener(self, listener) -> Callable[[], None]:
        """Register a callback invoked when any task completes.

        The listener receives ``(record: BackgroundTaskRecord)``.
        """
        listener_id = uuid.uuid4().hex
        self._completion_listeners[listener_id] = listener

        def unregister() -> None:
            self._completion_listeners.pop(listener_id, None)

        return unregister

    async def _fire_completion_listeners(self, record: BackgroundTaskRecord) -> None:
        for listener in list(self._completion_listeners.values()):
            try:
                maybe_awaitable = listener(record)
                if maybe_awaitable is not None:
                    await maybe_awaitable
            except Exception:
                pass

    def resolve_task_id(self, task_or_agent_id: str) -> str:
        """Resolve a task id or MiniHarness agent id to a background task id."""
        value = task_or_agent_id.strip()
        if value.startswith("agent-"):
            return "bg-" + value.removeprefix("agent-")
        return value

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
        self._persist_index()
        return record

    def update_task_metadata(
        self,
        task_id: str,
        values: dict[str, str],
    ) -> BackgroundTaskRecord:
        """Merge string metadata into a task record and persist it."""
        record = self._require_task(task_id)
        for key, value in values.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            record.metadata[normalized_key] = str(value)
        self._persist_index()
        return record

    def delete_task_metadata(
        self,
        task_id: str,
        keys: list[str] | tuple[str, ...] | set[str],
    ) -> BackgroundTaskRecord:
        """Remove selected metadata keys from a task record and persist it."""
        record = self._require_task(task_id)
        for key in keys:
            normalized_key = str(key).strip()
            if normalized_key:
                record.metadata.pop(normalized_key, None)
        self._persist_index()
        return record

    async def stop_task(self, task_id: str) -> BackgroundTaskRecord:
        record = self._require_task(task_id)
        process = self._processes.get(task_id)
        if process is None or process.returncode is not None:
            return record

        self._terminate_process_group(process)
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self._kill_process_group(process)
            await process.wait()

        record.status = "killed"
        record.ended_at = time.time()
        self._persist_index()
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
        self._persist_index()
        await self._fire_completion_listeners(record)

    async def _ensure_writable_process(
        self,
        record: BackgroundTaskRecord,
    ) -> asyncio.subprocess.Process:
        process = self._processes.get(record.id)
        if (
            record.status == "running"
            and process is not None
            and process.returncode is None
            and process.stdin is not None
            and not process.stdin.is_closing()
        ):
            return process
        if record.type != "local_agent":
            if record.status != "running":
                raise ValueError(f"Task {record.id} is not running (status={record.status})")
            raise ValueError(f"Task {record.id} is not attached to this MiniHarness process")
        return await self._restart_agent_task(record)

    async def _restart_agent_task(
        self,
        record: BackgroundTaskRecord,
    ) -> asyncio.subprocess.Process:
        if record.command is None and record.argv is None:
            raise ValueError(f"Task {record.id} does not have a command or argv to restart")

        watcher = self._watchers.get(record.id)
        if watcher is not None and not watcher.done():
            await asyncio.gather(watcher, return_exceptions=True)

        restart_count = int(record.metadata.get("restart_count", "0") or "0") + 1
        record.metadata["restart_count"] = str(restart_count)
        record.metadata["status_note"] = "Task restarted; prior interactive context was not preserved."
        record.status = "running"
        record.started_at = time.time()
        record.ended_at = None
        record.return_code = None
        with record.output_file.open("ab") as handle:
            handle.write(
                b"\n[MiniHarness] Agent task restarted; prior interactive context was not preserved.\n"
            )
        process = await self._start_process(record)
        self._processes[record.id] = process
        self._watchers[record.id] = asyncio.create_task(
            self._watch_process(record.id, process)
        )
        self._persist_index()
        return process

    async def _start_process(self, record: BackgroundTaskRecord, *, env: dict[str, str] | None = None) -> asyncio.subprocess.Process:
        process_env = dict(os.environ)
        process_env.update(env or self._runtime_envs.get(record.id) or {})
        process_env["MINIHARNESS_BACKGROUND_TASK_ID"] = record.id
        process_env["MINIHARNESS_BACKGROUND_TASK_TYPE"] = record.type
        if record.argv is not None:
            return await asyncio.create_subprocess_exec(
                *record.argv,
                cwd=record.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=process_env,
                start_new_session=True,
            )
        if record.command is None:
            raise ValueError(f"Task {record.id} has no command or argv")
        return await asyncio.create_subprocess_shell(
            record.command,
            cwd=record.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=process_env,
            start_new_session=True,
        )

    def _terminate_process_group(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()

    def _kill_process_group(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            process.kill()

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

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        raw_tasks = payload.get("tasks") if isinstance(payload, dict) else None
        if not isinstance(raw_tasks, list):
            return

        changed = False
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue
            record = BackgroundTaskRecord.from_json(raw)
            if record is None:
                continue
            if record.status == "running":
                record.status = "failed"
                record.ended_at = record.ended_at or time.time()
                record.return_code = record.return_code if record.return_code is not None else -1
                record.metadata["status_note"] = (
                    "Task process is not attached to this MiniHarness process."
                )
                changed = True
            self._tasks[record.id] = record
            self._output_locks[record.id] = asyncio.Lock()
        if changed:
            self._persist_index()

    def _persist_index(self) -> None:
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "tasks": [
                record.to_json()
                for record in sorted(
                    self._tasks.values(),
                    key=lambda item: item.created_at,
                )
            ],
        }
        data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        tmp_path = self.index_path.with_name(f".{self.index_path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(data, encoding="utf-8")
        tmp_path.replace(self.index_path)


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


def _default_agent_argv(*, cwd: str | Path, model: str | None = None) -> list[str]:
    return build_teammate_argv(cwd=cwd, model=model)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
