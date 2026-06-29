"""Tool metadata carryover — structured session state that survives compaction.

This is the single most important architectural difference between a "toy"
agent and a production harness.  The problem:

    Compaction throws away old messages.  Without carryover, the model loses
    everything it knew about *what it was doing* — the goal, the verified work,
    the files it was editing, the errors it encountered.

The solution (mirrors OpenHarness's ``tool_metadata`` on ``QueryEngine``):

    A mutable ``dict`` that lives on ``AgentLoop``, updated after every tool
    execution, and *never destroyed by compaction*.  During full (Tier-4)
    compaction, the dict is read to build ``CompactAttachment`` blocks that
    survive the compaction boundary.

State tracked
-------------

``task_focus_state``
    goal, recent_goals, active_artifacts, verified_state, next_step
    — the agent's "working memory" of what it's doing and why.

``read_file_state``
    Per-file records: path, line span, preview text, timestamp.
    Feeds the ``recent_files`` compact attachment.

``recent_verified_work``
    Chronological log of concrete accomplishments ("Ran bash command X",
    "Inspected file Y").  Feeds the ``verified_work`` attachment.

``recent_work_log``
    Execution checkpoints: every tool invocation leaves a breadcrumb.
    Feeds the ``work_log`` attachment.

``invoked_skills``
    Skill names the agent has loaded via the ``skill`` tool.
    Feeds the ``invoked_skills`` compact attachment so the model
    remembers which skills are active after compaction.

``background_task_state``
    Recent background task IDs, statuses, and output previews.
    Feeds the ``background_tasks`` compact attachment so the model can
    continue polling long-running work after compaction.

``async_agent_tasks``
    OpenHarness-style table of delegated local-agent tasks that still need
    coordinator notifications.

``async_agent_state``
    Compact activity log for agent/send_message actions.
"""

from __future__ import annotations

import re
import time
from typing import Any

from miniharness.tasks import TaskListManager, format_task_list


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def init_tool_metadata() -> dict[str, Any]:
    """Create a fresh ``tool_metadata`` dict for a new session.

    Call once in ``AgentLoop.__init__``.  The returned dict is mutated in
    place for the lifetime of the session — it never gets replaced.
    """
    return {
        "task_focus_state": {
            "goal": "",
            "recent_goals": [],
            "active_artifacts": [],
            "verified_state": [],
            "next_step": "",
        },
        "read_file_state": [],
        "recent_verified_work": [],
        "recent_work_log": [],
        "invoked_skills": [],
        "task_list_state": {
            "tasks": [],
            "revision": 0,
            "updated_at": 0.0,
        },
        "background_task_state": [],
        "async_agent_tasks": [],
        "async_agent_state": [],
    }


# ---------------------------------------------------------------------------
# User goal tracking
# ---------------------------------------------------------------------------

_MAX_GOAL_CHARS = 240
_MAX_RECENT_GOALS = 5


def remember_user_goal(metadata: dict[str, Any], prompt: str) -> None:
    """Record the user's most recent prompt as the current goal.

    Called from ``AgentLoop.run()`` before the turn loop starts, so
    compaction always knows what the user is trying to accomplish.
    """
    task = metadata.setdefault("task_focus_state", {})
    if not isinstance(task, dict):
        metadata["task_focus_state"] = task = {}

    short = prompt.strip()[: _MAX_GOAL_CHARS]
    task["goal"] = short

    # Maintain a deduplicated list of recent goals (most recent last).
    goals: list[str] = task.setdefault("recent_goals", [])
    if short in goals:
        goals.remove(short)
    goals.append(short)
    if len(goals) > _MAX_RECENT_GOALS:
        task["recent_goals"] = goals[-_MAX_RECENT_GOALS:]


# ---------------------------------------------------------------------------
# Per-tool carryover — the heart of the system
# ---------------------------------------------------------------------------

_MAX_ARTIFACTS = 8
_MAX_VERIFIED = 10
_MAX_WORK_LOG = 16
_MAX_READ_FILE_STATE = 12
_MAX_BACKGROUND_TASKS = 8
_MAX_ASYNC_AGENT_TASKS = 12
_MAX_ASYNC_AGENT_EVENTS = 12


def record_tool_carryover(
    metadata: dict[str, Any],
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_output: str,
    is_error: bool,
    result_metadata: dict[str, Any] | None = None,
) -> None:
    """After every successful tool execution, update structured session state.

    This is called from ``AgentLoop._execute_tools()`` for each tool call
    that completes (even if the tool returned an error — we still log it).
    """
    # ---- resolve a file path from the arguments ---------------------------
    file_path = _extract_file_path(tool_name, arguments)
    if file_path and not is_error:
        # Any tool that touches a file leaves a work log entry.
        _remember_work_log(metadata, f"{tool_name} {file_path}")
        # If it's a read or write, it also becomes an active artifact.
        _remember_active_artifact(metadata, file_path)

    # ---- per-tool actions ------------------------------------------------
    if tool_name == "read_file" and not is_error:
        _carryover_read_file(metadata, arguments, result_output)
    elif tool_name in ("write_file", "edit_file") and not is_error:
        _carryover_write_file(metadata, arguments, result_output)
    elif tool_name == "bash" and not is_error:
        _carryover_bash(metadata, arguments, result_output)
    elif tool_name == "grep" and not is_error:
        _carryover_grep(metadata, arguments, result_output)
    elif tool_name == "web_fetch" and not is_error:
        _carryover_web_fetch(metadata, arguments, result_output)
    elif tool_name == "skill" and not is_error:
        _carryover_skill(metadata, arguments)
    elif tool_name in ("memory_add", "memory_log", "memory_search"):
        pass  # memory tools are self-documenting
    elif tool_name == "task" and not is_error:
        _remember_work_log(metadata, "Updated session task list")
    elif tool_name in {"agent", "send_message"}:
        _carryover_async_agent(
            metadata,
            tool_name=tool_name,
            arguments=arguments,
            result_output=result_output,
            is_error=is_error,
            result_metadata=result_metadata,
        )
    elif tool_name in {
        "task_create",
        "task_list",
        "task_get",
        "task_output",
        "task_stop",
        "task_update",
    }:
        _carryover_background_task(
            metadata,
            tool_name=tool_name,
            arguments=arguments,
            result_output=result_output,
            is_error=is_error,
        )

    # ---- always log tool execution (even errors) -------------------------
    if is_error:
        _remember_work_log(metadata, f"{tool_name} ERROR: {result_output[:120]}")


# ---------------------------------------------------------------------------
# Per-tool handlers
# ---------------------------------------------------------------------------


def _carryover_read_file(
    metadata: dict[str, Any],
    arguments: dict[str, Any],
    result_output: str,
) -> None:
    """Record a file read in ``read_file_state``."""
    path = arguments.get("path", "")
    if not path:
        return

    # Estimate the line range from content.
    lines = result_output.split("\n")
    total = len(lines)
    preview = "\n".join(lines[:4])  # first 4 lines as preview

    entry: dict[str, Any] = {
        "path": path,
        "total_lines": total,
        "preview": preview[:240],
        "timestamp": time.time(),
    }

    state: list[dict[str, Any]] = metadata.setdefault("read_file_state", [])
    # Remove existing entry for the same path (deduplicate).
    state = [e for e in state if e.get("path") != path]
    state.append(entry)
    if len(state) > _MAX_READ_FILE_STATE:
        state = state[-_MAX_READ_FILE_STATE:]
    metadata["read_file_state"] = state

    _remember_verified_work(metadata, f"Read file {path}")


def _carryover_write_file(
    metadata: dict[str, Any],
    arguments: dict[str, Any],
    result_output: str,
) -> None:
    """Record a file write."""
    path = arguments.get("path", "")
    _remember_verified_work(metadata, f"Wrote file {path}")


def _carryover_bash(
    metadata: dict[str, Any],
    arguments: dict[str, Any],
    result_output: str,
) -> None:
    """Record a bash command execution."""
    cmd = arguments.get("command", "")
    summary = cmd[:120]
    _remember_verified_work(metadata, f"Ran: {summary}")
    _remember_work_log(metadata, f"bash: {summary}")


def _carryover_grep(
    metadata: dict[str, Any],
    arguments: dict[str, Any],
    result_output: str,
) -> None:
    """Record a grep search."""
    query = arguments.get("query", "")
    _remember_verified_work(metadata, f"Searched for: {query[:120]}")


def _carryover_web_fetch(
    metadata: dict[str, Any],
    arguments: dict[str, Any],
    result_output: str,
) -> None:
    """Record a web fetch."""
    url = arguments.get("url", "")
    _remember_active_artifact(metadata, f"url:{url}")
    _remember_verified_work(metadata, f"Fetched {url[:120]}")


def _carryover_background_task(
    metadata: dict[str, Any],
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_output: str,
    is_error: bool,
) -> None:
    """Record background task state for compaction carryover."""
    if is_error:
        return

    task_id = _extract_background_task_id(arguments, result_output)
    if not task_id:
        if tool_name == "task_list":
            _remember_work_log(metadata, "Listed background tasks")
        return

    state = _background_task_bucket(metadata)
    record = next((item for item in state if item.get("id") == task_id), None)
    if record is None:
        record = {"id": task_id}
        state.append(record)

    record["updated_at"] = time.time()
    record["last_tool"] = tool_name

    if tool_name == "task_create":
        record["type"] = str(arguments.get("type") or "local_bash")
        record["description"] = str(arguments.get("description") or "").strip()[:180]
        record["command"] = str(arguments.get("command") or "").strip()[:240]
        record["status"] = "running"
        _remember_verified_work(
            metadata,
            f"Started background task {task_id}: {record.get('description', '')}",
        )
    elif tool_name == "task_get":
        _merge_background_task_get_output(record, result_output)
        _remember_work_log(metadata, f"Inspected background task {task_id}")
    elif tool_name == "task_output":
        record["last_output_preview"] = result_output.strip()[:240]
        _remember_work_log(metadata, f"Read output for background task {task_id}")
    elif tool_name == "task_stop":
        record["status"] = "killed"
        _remember_verified_work(metadata, f"Stopped background task {task_id}")
    elif tool_name == "task_update":
        progress = arguments.get("progress")
        note = str(arguments.get("status_note") or "").strip()
        if progress is not None:
            record["progress"] = str(progress)
        if note:
            record["status_note"] = note[:180]
        _remember_work_log(metadata, f"Updated background task {task_id}")

    if len(state) > _MAX_BACKGROUND_TASKS:
        del state[: len(state) - _MAX_BACKGROUND_TASKS]


def _carryover_async_agent(
    metadata: dict[str, Any],
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_output: str,
    is_error: bool,
    result_metadata: dict[str, Any] | None = None,
) -> None:
    """Record OpenHarness-style async agent state."""
    if is_error:
        return

    _remember_async_agent_activity(
        metadata,
        tool_name=tool_name,
        arguments=arguments,
        result_output=result_output,
    )
    if tool_name != "agent":
        _remember_work_log(metadata, f"Async agent action via {tool_name}")
        return

    agent_id, task_id = _extract_spawned_agent_identity(
        arguments,
        result_output,
        result_metadata,
    )
    if not task_id or not agent_id:
        return

    bucket = _async_agent_task_bucket(metadata)
    bucket[:] = [
        existing
        for existing in bucket
        if not isinstance(existing, dict)
        or str(existing.get("task_id") or "").strip() != task_id
    ]
    description = str(arguments.get("description") or arguments.get("prompt") or "").strip()
    bucket.append({
        "agent_id": agent_id,
        "task_id": task_id,
        "description": description[:240],
        "status": "spawned",
        "notification_sent": False,
        "spawned_at": time.time(),
    })
    if len(bucket) > _MAX_ASYNC_AGENT_TASKS:
        del bucket[:-_MAX_ASYNC_AGENT_TASKS]

    _remember_verified_work(
        metadata,
        f"Confirmed async-agent activity via agent: {description[:180]}",
    )
    _remember_work_log(metadata, "Async agent action via agent")


def _remember_async_agent_activity(
    metadata: dict[str, Any],
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_output: str,
) -> None:
    bucket = _async_agent_state_bucket(metadata)
    if tool_name == "agent":
        description = str(arguments.get("description") or arguments.get("prompt") or "").strip()
        summary = f"Spawned async agent. {description}".strip()
        if result_output.strip():
            summary = f"{summary} [{result_output.strip()[:180]}]".strip()
    elif tool_name == "send_message":
        target = str(arguments.get("task_id") or "").strip()
        summary = f"Sent follow-up message to async agent {target}".strip()
    else:
        summary = result_output.strip()[:220] or f"Async agent activity via {tool_name}"
    bucket.append(summary)
    if len(bucket) > _MAX_ASYNC_AGENT_EVENTS:
        del bucket[:-_MAX_ASYNC_AGENT_EVENTS]


# ---------------------------------------------------------------------------
# Helpers — artefact / verified-work / work-log management
# ---------------------------------------------------------------------------


def _extract_file_path(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Try to extract a file path from a tool's arguments.

    Different tools use different argument names for the path.
    """
    candidates = ["path", "root"]
    for key in candidates:
        val = arguments.get(key, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _remember_active_artifact(metadata: dict[str, Any], artifact: str) -> None:
    """Add an artifact to ``task_focus_state.active_artifacts``."""
    task = metadata.setdefault("task_focus_state", {})
    arts: list[str] = task.setdefault("active_artifacts", [])
    if artifact in arts:
        arts.remove(artifact)  # move to end (most recent)
    arts.append(artifact)
    if len(arts) > _MAX_ARTIFACTS:
        task["active_artifacts"] = arts[-_MAX_ARTIFACTS:]


def _remember_verified_work(metadata: dict[str, Any], description: str) -> None:
    """Record a verified accomplishment.

    Writes to BOTH ``task_focus_state.verified_state`` (for task-focus
    attachment) AND ``recent_verified_work`` (for its own attachment).
    """
    short = description.strip()[:320]

    # Per-task verified state.
    task = metadata.setdefault("task_focus_state", {})
    vs: list[str] = task.setdefault("verified_state", [])
    vs.append(short)
    if len(vs) > _MAX_VERIFIED:
        task["verified_state"] = vs[-_MAX_VERIFIED:]

    # Global verified work list.
    rvw: list[str] = metadata.setdefault("recent_verified_work", [])
    rvw.append(short)
    if len(rvw) > _MAX_VERIFIED:
        metadata["recent_verified_work"] = rvw[-_MAX_VERIFIED:]


def _remember_work_log(metadata: dict[str, Any], entry: str) -> None:
    """Append a breadcrumb to the work log."""
    wl: list[str] = metadata.setdefault("recent_work_log", [])
    wl.append(f"[{time.strftime('%H:%M:%S')}] {entry.strip()[:240]}")
    if len(wl) > _MAX_WORK_LOG:
        metadata["recent_work_log"] = wl[-_MAX_WORK_LOG:]


def _background_task_bucket(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    bucket = metadata.setdefault("background_task_state", [])
    if not isinstance(bucket, list):
        bucket = []
        metadata["background_task_state"] = bucket
    return bucket


def _async_agent_task_bucket(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    bucket = metadata.setdefault("async_agent_tasks", [])
    if not isinstance(bucket, list):
        bucket = []
        metadata["async_agent_tasks"] = bucket
    return bucket


def _async_agent_state_bucket(metadata: dict[str, Any]) -> list[str]:
    bucket = metadata.setdefault("async_agent_state", [])
    if not isinstance(bucket, list):
        bucket = []
        metadata["async_agent_state"] = bucket
    return bucket


def _extract_background_task_id(arguments: dict[str, Any], output: str) -> str:
    task_id = str(arguments.get("task_id") or "").strip()
    if task_id:
        return _normalize_background_task_id(task_id)
    match = re.search(r"\b(bg-[A-Za-z0-9_-]+)\b", output)
    if match:
        return match.group(1)
    agent_match = re.search(r"\b(agent-[A-Za-z0-9_-]+)\b", output)
    return _normalize_background_task_id(agent_match.group(1)) if agent_match else ""


def _extract_agent_id(output: str) -> str:
    spawned = re.search(r"\bSpawned agent\s+(\S+)\s+\(task_id=", output)
    if spawned:
        return spawned.group(1)
    match = re.search(r"\b(agent-[A-Za-z0-9_-]+)\b", output)
    return match.group(1) if match else ""


def _extract_spawned_agent_identity(
    arguments: dict[str, Any],
    output: str,
    result_metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if isinstance(result_metadata, dict):
        agent_id = str(result_metadata.get("agent_id") or "").strip()
        task_id = str(result_metadata.get("task_id") or "").strip()
        if agent_id and task_id:
            return agent_id, _normalize_background_task_id(task_id)
    return _extract_agent_id(output), _extract_background_task_id(arguments, output)


def _normalize_background_task_id(value: str) -> str:
    if value.startswith("agent-"):
        return "bg-" + value.removeprefix("agent-")
    return value


def _merge_background_task_get_output(record: dict[str, Any], output: str) -> None:
    for line in output.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        normalized = key.strip()
        if normalized in {
            "type",
            "status",
            "description",
            "cwd",
            "command",
            "prompt",
            "argv",
            "output_file",
            "return_code",
        }:
            record[normalized] = value.strip()


# ---------------------------------------------------------------------------
# Readout helpers (used by compact attachment builders)
# ---------------------------------------------------------------------------


def get_task_focus(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a snapshot of the current task focus state."""
    return dict(metadata.get("task_focus_state", {}))


def get_read_file_state(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the read-file state, newest first."""
    state = list(metadata.get("read_file_state", []))
    state.reverse()  # newest first
    return state


def get_recent_verified_work(metadata: dict[str, Any], limit: int = 8) -> list[str]:
    """Return recent verified work entries, newest first."""
    rvw = list(metadata.get("recent_verified_work", []))
    rvw.reverse()
    return rvw[:limit]


def _carryover_skill(
    metadata: dict[str, Any],
    arguments: dict[str, Any],
) -> None:
    """Record a skill invocation."""
    skill_name = str(arguments.get("name", "")).strip()
    if not skill_name:
        return
    skills: list[str] = metadata.setdefault("invoked_skills", [])
    if skill_name in skills:
        skills.remove(skill_name)
    skills.append(skill_name)
    if len(skills) > 8:
        metadata["invoked_skills"] = skills[-8:]

    _remember_active_artifact(metadata, f"skill:{skill_name}")
    _remember_verified_work(metadata, f"Loaded skill {skill_name}")


def get_invoked_skills(metadata: dict[str, Any]) -> list[str]:
    """Return the list of invoked skill names."""
    return list(metadata.get("invoked_skills", []))


def get_recent_work_log(metadata: dict[str, Any], limit: int = 8) -> list[str]:
    """Return recent work log entries, newest first."""
    wl = list(metadata.get("recent_work_log", []))
    wl.reverse()
    return wl[:limit]


def get_background_task_state(metadata: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    """Return recent background task records, newest first."""
    state = [
        item for item in metadata.get("background_task_state", [])
        if isinstance(item, dict)
    ]
    state.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return state[:limit]


def get_async_agent_tasks(metadata: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Return recent delegated agent task records, newest first."""
    state = [
        item for item in metadata.get("async_agent_tasks", [])
        if isinstance(item, dict)
    ]
    state.sort(
        key=lambda item: float(item.get("spawned_at") or item.get("updated_at") or 0),
        reverse=True,
    )
    return state[:limit]


def get_async_agent_state(metadata: dict[str, Any], limit: int = 8) -> list[str]:
    """Return recent async-agent activity entries, newest first."""
    state = [
        str(item)
        for item in metadata.get("async_agent_state", [])
        if str(item).strip()
    ]
    state.reverse()
    return state[:limit]


# ---------------------------------------------------------------------------
# Compact attachments — built from tool_metadata during compaction
# ---------------------------------------------------------------------------


def build_compact_attachments(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Build compact-attachment messages from *metadata*.

    Each attachment is a ``role="user"`` message with ``[Compact attachment: ...]``
    format.  These are injected into the post-compact message list so the
    model retains structured state even after old messages are discarded.

    This is the ONLY function outside ``carryover`` that reads tool_metadata —
    and it lives here so all metadata I/O is in one file.
    """
    attachments: list[dict[str, Any]] = []
    if not metadata:
        return attachments

    builders = [
        _build_task_focus_attachment,
        _build_task_list_attachment,
        _build_async_agent_attachment,
        _build_background_task_attachment,
        _build_recent_files_attachment,
        _build_invoked_skills_attachment,
        _build_verified_work_attachment,
        _build_work_log_attachment,
    ]
    for builder in builders:
        att = builder(metadata)
        if att:
            attachments.append(att)
    return attachments


def _render_attachment(*, kind: str, title: str, body: str) -> dict[str, Any]:
    """Render a compact attachment as a user message dict."""
    return {
        "role": "user",
        "content": f"[Compact attachment: {kind}] {title}\n{body}",
    }


def _build_task_focus_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    tf = get_task_focus(metadata)
    goal = tf.get("goal", "")
    recent_goals = tf.get("recent_goals", [])
    artifacts = tf.get("active_artifacts", [])
    verified = tf.get("verified_state", [])

    if not goal and not artifacts and not verified:
        return None

    lines: list[str] = []
    if goal:
        lines.append(f"Goal: {goal}")
    if recent_goals:
        lines.append(f"Recent goals: {', '.join(recent_goals[-3:])}")
    if artifacts:
        lines.append(f"Active artifacts: {', '.join(artifacts[-5:])}")
    if verified:
        lines.append(f"Verified: {', '.join(verified[-4:])}")
    if not lines:
        return None

    return _render_attachment(kind="task_focus", title="Current Task & Progress", body="\n".join(lines))


def _build_task_list_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    tasks = TaskListManager(metadata).list_tasks()
    if not tasks:
        return None
    active = [task for task in tasks if task.status != "completed"]
    if not active:
        active = tasks[-5:]
    body = format_task_list(active)
    return _render_attachment(kind="task_list", title="Current Task List", body=body)


def _build_background_task_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    tasks = get_background_task_state(metadata)
    if not tasks:
        return None

    lines = ["Recent background tasks:"]
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        agent_id = str(task.get("agent_id") or "").strip()
        status = str(task.get("status") or "unknown").strip()
        description = str(task.get("description") or "").strip()
        note = str(task.get("status_note") or "").strip()
        message = str(task.get("last_message_preview") or "").strip().replace("\n", " ")[:120]
        preview = str(task.get("last_output_preview") or "").strip().replace("\n", " ")[:120]

        label = f"{agent_id} ({task_id})" if agent_id else task_id
        line = f"- {label} [{status}] {description}".strip()
        if note:
            line += f" ({note})"
        lines.append(line)
        if message:
            lines.append(f"  last message: {message}")
        if preview:
            lines.append(f"  output: {preview}")

    return _render_attachment(
        kind="background_tasks",
        title="Background Task State",
        body="\n".join(lines),
    )


def _build_async_agent_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    tasks = get_async_agent_tasks(metadata)
    activity = get_async_agent_state(metadata)
    if not tasks and not activity:
        return None

    lines: list[str] = []
    if tasks:
        lines.append("Delegated agent tasks:")
        for task in tasks:
            agent_id = str(task.get("agent_id") or "").strip()
            task_id = str(task.get("task_id") or "").strip()
            status = str(task.get("status") or "unknown").strip()
            description = str(task.get("description") or "").strip()
            notified = "notified" if bool(task.get("notification_sent")) else "pending notification"
            label = agent_id or task_id
            lines.append(f"- {label} ({task_id}) [{status}, {notified}] {description}".strip())
    if activity:
        lines.append("Recent async-agent activity:")
        lines.extend(f"- {item[:220]}" for item in activity[:5])

    return _render_attachment(
        kind="async_agents",
        title="Delegated Agent State",
        body="\n".join(lines),
    )


def _build_recent_files_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    state = get_read_file_state(metadata)[:4]
    if not state:
        return None
    lines: list[str] = []
    for e in state:
        path = e.get("path", "")
        lines_count = e.get("total_lines", 0)
        preview = e.get("preview", "")[:120].replace("\n", "\\n")
        lines.append(f"  {path} ({lines_count} lines)")
        if preview:
            lines.append(f"    preview: {preview}")
    if not lines:
        return None
    return _render_attachment(kind="recent_files", title="Recently Read Files", body="\n".join(lines))


def _build_invoked_skills_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    skills = get_invoked_skills(metadata)
    if not skills:
        return None
    return _render_attachment(
        kind="invoked_skills",
        title="Skills Used Earlier",
        body=f"The following skills were previously loaded: {', '.join(skills)}.",
    )


def _build_verified_work_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    entries = get_recent_verified_work(metadata)[:8]
    if not entries:
        return None
    lines = [f"  • {e}" for e in entries]
    return _render_attachment(kind="verified_work", title="Recently Verified Work", body="\n".join(lines))


def _build_work_log_attachment(metadata: dict[str, Any]) -> dict[str, Any] | None:
    entries = get_recent_work_log(metadata)[:8]
    if not entries:
        return None
    lines = [f"  {e}" for e in entries]
    return _render_attachment(kind="work_log", title="Recent Work Log", body="\n".join(lines))
