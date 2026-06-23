"""Session task-list manager.

This is the lightweight counterpart to OpenHarness's task state layer:

- it is session-scoped, not global;
- it stores canonical state in ``AgentLoop.tool_metadata``;
- it uses replace-all semantics so the model must submit the complete list;
- it preserves stable task IDs across updates when possible.

Background subprocess / sub-agent tasks are a separate concern and should not
share this state model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal


TaskStatus = Literal["pending", "in_progress", "completed"]
_VALID_STATUSES: set[str] = {"pending", "in_progress", "completed"}
_STATE_KEY = "task_list_state"


@dataclass(frozen=True)
class TaskItem:
    """Canonical task item stored in session metadata."""

    id: str
    content: str
    status: TaskStatus

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "content": self.content, "status": self.status}


class TaskListManager:
    """Manage the current session's todo/task list."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata
        self._ensure_state()

    def list_tasks(self) -> list[TaskItem]:
        """Return canonical tasks in display order."""
        state = self._ensure_state()
        tasks: list[TaskItem] = []
        for raw in state.get("tasks", []):
            item = _coerce_task(raw)
            if item is not None:
                tasks.append(item)
        return tasks

    def replace_all(self, incoming: list[dict[str, str]]) -> list[TaskItem]:
        """Replace the task list with a validated complete list.

        IDs are preserved by explicit ``id`` first, then by exact content match.
        At most one task may be ``in_progress``.  This mirrors the operational
        discipline expected from coding agents: one active step, explicit
        pending work, and completed items retained for traceability.
        """
        if not isinstance(incoming, list):
            raise ValueError("tasks must be a list")

        existing = self.list_tasks()
        by_id = {item.id: item for item in existing}
        by_content = {item.content: item for item in existing}
        used_ids: set[str] = set()
        cleaned: list[TaskItem] = []
        in_progress_count = 0

        for index, raw in enumerate(incoming):
            if not isinstance(raw, dict):
                raise ValueError(f"Task {index} is not an object")
            content = str(raw.get("content", "")).strip()
            status = str(raw.get("status", "pending")).strip()
            requested_id = str(raw.get("id", "")).strip()

            if not content:
                raise ValueError(f"Task {index} has empty content")
            if status not in _VALID_STATUSES:
                raise ValueError(
                    f"Task {index} has invalid status '{status}'; use pending, in_progress, or completed"
                )
            if status == "in_progress":
                in_progress_count += 1

            task_id = ""
            if requested_id and requested_id in by_id:
                task_id = requested_id
            elif content in by_content:
                task_id = by_content[content].id
            elif requested_id:
                task_id = _normalize_task_id(requested_id)
            if not task_id:
                task_id = _next_task_id(used_ids | set(by_id))
            if task_id in used_ids:
                task_id = _next_task_id(used_ids | set(by_id))
            used_ids.add(task_id)

            cleaned.append(TaskItem(id=task_id, content=content, status=status))  # type: ignore[arg-type]

        if in_progress_count > 1:
            raise ValueError("Only one task may be in_progress at a time")

        state = self._ensure_state()
        state["tasks"] = [item.to_dict() for item in cleaned]
        state["updated_at"] = time.time()
        state["revision"] = int(state.get("revision") or 0) + 1
        return cleaned

    def summary(self) -> dict[str, int]:
        tasks = self.list_tasks()
        return {
            "total": len(tasks),
            "pending": sum(1 for item in tasks if item.status == "pending"),
            "in_progress": sum(1 for item in tasks if item.status == "in_progress"),
            "completed": sum(1 for item in tasks if item.status == "completed"),
        }

    def _ensure_state(self) -> dict[str, Any]:
        state = self._metadata.setdefault(_STATE_KEY, {})
        if not isinstance(state, dict):
            state = {}
            self._metadata[_STATE_KEY] = state
        state.setdefault("tasks", [])
        state.setdefault("revision", 0)
        state.setdefault("updated_at", 0.0)
        return state


def format_task_list(tasks: list[TaskItem]) -> str:
    """Render tasks in a deterministic, frontend-neutral text form."""
    if not tasks:
        return "Tasks: (empty)"
    lines = ["Tasks:"]
    label = {
        "pending": "[ ]",
        "in_progress": "[~]",
        "completed": "[x]",
    }
    for index, item in enumerate(tasks, 1):
        lines.append(f"{index}. {label[item.status]} {item.id} {item.content}")
    total = len(tasks)
    pending = sum(1 for item in tasks if item.status == "pending")
    active = sum(1 for item in tasks if item.status == "in_progress")
    done = sum(1 for item in tasks if item.status == "completed")
    lines.append(f"Summary: {total} tasks ({pending} pending, {active} in progress, {done} completed)")
    return "\n".join(lines)


def _coerce_task(raw: Any) -> TaskItem | None:
    if not isinstance(raw, dict):
        return None
    content = str(raw.get("content", "")).strip()
    status = str(raw.get("status", "pending")).strip()
    task_id = str(raw.get("id", "")).strip()
    if not content or status not in _VALID_STATUSES:
        return None
    if not task_id:
        task_id = "task-unknown"
    return TaskItem(id=task_id, content=content, status=status)  # type: ignore[arg-type]


def _next_task_id(used_ids: set[str]) -> str:
    number = 1
    while True:
        task_id = f"task-{number:03d}"
        if task_id not in used_ids:
            return task_id
        number += 1


def _normalize_task_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    return safe[:40] or "task"
