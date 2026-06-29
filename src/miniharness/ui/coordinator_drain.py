"""Drain completed background tasks back into the runtime.

OpenHarness uses coordinator drain for async agents: once a background worker
finishes, the frontend/backend notices and feeds a structured notification back
to the coordinator.  MiniHarness uses the same primitive for local agents and
keeps a separate notification envelope for generic shell tasks:

- find task IDs remembered in ``tool_metadata``;
- check the process-global background task manager;
- format one deterministic notification;
- mark each terminal task as notified so users are not spammed;
- let the runtime append the payload as a user-role message for the parent
  coordinator.

This module intentionally does not call the LLM.  It only produces the message
payload; UI/runtime code decides whether to display it, persist it, or submit a
follow-up turn.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable
from xml.sax.saxutils import escape

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


def async_agent_task_entries(tool_metadata: dict | None) -> list[dict]:
    """Return mutable OpenHarness-style async agent task entries."""
    if not isinstance(tool_metadata, dict):
        return []
    value = tool_metadata.get("async_agent_tasks")
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def pending_async_agent_entries(tool_metadata: dict | None) -> list[dict]:
    """Return remembered local-agent tasks that still need notification."""
    pending: list[dict] = []
    for entry in async_agent_task_entries(tool_metadata):
        task_id = str(entry.get("task_id") or "").strip()
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


async def wait_for_completed_async_agent_entries(
    tool_metadata: dict | None,
    *,
    manager: BackgroundTaskManager | None = None,
    poll_interval_seconds: float = 0.1,
) -> list[dict]:
    """Block until at least one pending local-agent task reaches terminal state."""
    manager = manager or get_background_task_manager()
    while True:
        pending = pending_async_agent_entries(tool_metadata)
        if not pending:
            return []

        completed: list[dict] = []
        for entry in pending:
            task_id = str(entry.get("task_id") or "").strip()
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

        if completed:
            return completed
        await asyncio.sleep(poll_interval_seconds)


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
        task_id = str(entry.get("task_id") or entry.get("id") or "").strip()
        if not task_id:
            continue
        task = manager.get_task(task_id)
        if task is None:
            continue

        output = manager.read_output(task_id, max_bytes=max_output_bytes).strip()
        status = task.status
        if task.type == "local_agent":
            agent_id = str(entry.get("agent_id") or task.metadata.get("agent_id") or task.id).strip()
            headline = _agent_task_summary(
                agent_id=agent_id,
                description=task.description,
                status=status,
                return_code=task.return_code,
            )
            lines = [
                "<task-notification>",
                f"<task-id>{escape(agent_id)}</task-id>",
                f"<status>{escape(status)}</status>",
                f"<summary>{escape(headline)}</summary>",
            ]
            if output:
                lines.append(f"<result>{escape(output)}</result>")
            lines.append("</task-notification>")
        else:
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


def _agent_task_summary(
    *,
    agent_id: str,
    description: str,
    status: str,
    return_code: int | None,
) -> str:
    label = description.strip() or agent_id
    if status == "completed":
        return f'Agent "{label}" completed'
    if status == "killed":
        return f'Agent "{label}" was stopped'
    if return_code is not None:
        return f'Agent "{label}" failed with exit code {return_code}'
    return f'Agent "{label}" failed'


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
