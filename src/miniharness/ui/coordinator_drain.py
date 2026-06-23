"""Drain completed background tasks back into the runtime.

OpenHarness uses coordinator drain for async agents: once a background worker
finishes, the frontend/backend notices and feeds a structured notification back
to the coordinator.  MiniHarness does not have subagents yet, but the same
primitive is useful now for background shell tasks:

- find task IDs remembered in ``tool_metadata``;
- check the process-global background task manager;
- format one deterministic notification;
- mark each terminal task as notified so users are not spammed.

This module intentionally does not call the LLM.  It is the first safe layer:
surface completed background work to the user/runtime.  Follow-up coordinator
turn submission can be layered on top once local-agent tasks exist.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from miniharness.tasks import BackgroundTaskManager, get_background_task_manager


_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "killed"})


def background_task_entries(tool_metadata: dict | None) -> list[dict]:
    """Return mutable background task entries from session metadata."""
    if not isinstance(tool_metadata, dict):
        return []
    value = tool_metadata.get("background_task_state")
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def pending_background_task_entries(tool_metadata: dict | None) -> list[dict]:
    """Return remembered background tasks that have not sent a notification."""
    pending: list[dict] = []
    for entry in background_task_entries(tool_metadata):
        task_id = str(entry.get("id") or "").strip()
        if not task_id:
            continue
        if bool(entry.get("notification_sent")):
            continue
        pending.append(entry)
    return pending


def collect_completed_background_tasks(
    tool_metadata: dict | None,
    *,
    manager: BackgroundTaskManager | None = None,
) -> list[dict]:
    """Poll remembered background tasks and return newly terminal entries."""
    manager = manager or get_background_task_manager()
    completed: list[dict] = []
    for entry in pending_background_task_entries(tool_metadata):
        task_id = str(entry.get("id") or "").strip()
        task = manager.get_task(task_id)
        if task is None:
            entry["status"] = "missing"
            entry["notification_sent"] = True
            continue

        entry["status"] = task.status
        entry["return_code"] = task.return_code
        if task.status in _TERMINAL_TASK_STATUSES:
            entry["description"] = task.description
            entry["type"] = task.type
            entry["updated_at"] = task.ended_at or task.started_at
            completed.append(entry)
    return completed


def format_completed_background_task_notifications(
    completed: list[dict],
    *,
    manager: BackgroundTaskManager | None = None,
    max_output_bytes: int = 8000,
) -> str:
    """Format completed task notifications and mark them as sent."""
    manager = manager or get_background_task_manager()
    blocks: list[str] = []
    for entry in completed:
        task_id = str(entry.get("id") or "").strip()
        if not task_id:
            continue
        task = manager.get_task(task_id)
        if task is None:
            continue

        output = manager.read_output(task_id, max_bytes=max_output_bytes).strip()
        status = task.status
        if status == "completed":
            headline = f"Background task completed: {task.description}"
        elif status == "killed":
            headline = f"Background task stopped: {task.description}"
        else:
            code = f" with exit code {task.return_code}" if task.return_code is not None else ""
            headline = f"Background task failed{code}: {task.description}"

        lines = [
            "<background-task-notification>",
            f"task_id: {task.id}",
            f"status: {status}",
            f"summary: {headline}",
        ]
        if output:
            lines.extend(["output:", output])
        lines.append("</background-task-notification>")
        blocks.append("\n".join(lines))

        entry["notification_sent"] = True
        entry["notified_status"] = status
        entry["last_output_preview"] = output[:240]

    return "\n\n".join(blocks)


async def drain_completed_background_tasks(
    tool_metadata: dict | None,
    *,
    print_system: Callable[[str], Awaitable[None]],
    manager: BackgroundTaskManager | None = None,
) -> str:
    """Notify the current frontend about completed background tasks.

    Returns the notification text for tests and future coordinator submission.
    """
    manager = manager or get_background_task_manager()
    completed = collect_completed_background_tasks(tool_metadata, manager=manager)
    if not completed:
        return ""
    message = format_completed_background_task_notifications(completed, manager=manager)
    if message:
        await print_system(message)
    return message
