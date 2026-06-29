"""Hook lifecycle events.

Each enum value represents a point in the agent's lifecycle where
external code can intercept and run custom logic.  Think of them as
"extension points" or "event listeners" that fire at specific moments.

Mirrors OpenHarness's ``HookEvent`` enum.
"""

from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    """Events that can trigger hooks.

    Each event carries a *payload* — a dict of contextual data that
    hook commands and prompts receive (via ``$ARGUMENTS`` substitution).
    """

    # ── Session boundaries ──────────────────────────────────────────
    SESSION_START = "session_start"
    """Fires once when an AgentLoop is created / session begins.

    Payload: ``{"cwd": "...", "model": "...", "session_id": "..."}``
    """

    SESSION_END = "session_end"
    """Fires when the session exits (REPL shutdown or single-shot completion).

    Payload: ``{"session_id": "...", "turn_count": N, "message_count": N}``
    """

    # ── User interaction ────────────────────────────────────────────
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    """Fires when the user submits a prompt (before the agent loop starts).

    Payload: ``{"prompt": "...", "session_id": "..."}``
    """

    # ── Tool execution ──────────────────────────────────────────────
    PRE_TOOL_USE = "pre_tool_use"
    """Fires BEFORE each tool is executed.

    Payload: ``{"tool_name": "...", "tool_input": {...}, "session_id": "..."}``

    A blocking hook here can PREVENT the tool from running
    (e.g. a security audit hook that blocks dangerous commands).
    """

    POST_TOOL_USE = "post_tool_use"
    """Fires AFTER each tool completes (success or error).

    Payload: ``{"tool_name": "...", "tool_input": {...}, "output": "...",
    "is_error": bool, "session_id": "..."}``

    Useful for logging, metrics, or post-hoc validation.
    """

    TOOL_FAILED = "tool_failed"
    """Fires when a tool execution FAILS (is_error=True).

    Payload: ``{"tool_name": "...", "tool_input": {...}, "error": "...",
    "session_id": "..."}``

    Separate from POST_TOOL_USE so failure-alert hooks don't fire on
    every normal completion.  Use for: alerting, auto-retry, error
    pattern analysis.
    """

    SUBAGENT_STOP = "subagent_stop"
    """Fires when a delegated subagent task reaches a terminal state.

    Payload: ``{"agent_id": "...", "task_id": "...", "backend_type": "...",
    "status": "completed"|"failed"|"killed", "return_code": int|None,
    "description": "...", "subagent_type": "...", "team": "..."}``
    """

    # ── Compaction ──────────────────────────────────────────────────
    PRE_COMPACT = "pre_compact"
    """Fires BEFORE conversation compaction runs.

    Payload: ``{"trigger": "auto"|"manual"|"reactive", "message_count": N,
    "tokens_used": N, "session_id": "..."}``
    """

    POST_COMPACT = "post_compact"
    """Fires AFTER compaction completes.

    Payload: ``{"trigger": "...", "tiers_run": [...], "messages_before": N,
    "messages_after": N, "session_id": "..."}``
    """

    # ── Notifications ───────────────────────────────────────────────
    NOTIFICATION = "notification"
    """Fires when the harness wants to push a desktop notification.

    Payload: ``{"message": "...", "session_id": "..."}``
    """
