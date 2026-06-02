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
"""

from __future__ import annotations

import time
from typing import Any


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


def record_tool_carryover(
    metadata: dict[str, Any],
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_output: str,
    is_error: bool,
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
    elif tool_name == "task":
        pass  # task tool state is transient

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
