"""Session lifecycle management — switch, save, resolve.

Pure business logic extracted from cli.py.  UI rendering and interactive
input stay in cli.py so this module has no console / rich dependency.
"""

from __future__ import annotations

from miniharness.loop import AgentLoop
from miniharness.sessions.storage import (
    load_session_by_id,
    load_session_by_tag,
    mark_session_latest,
    save_session_snapshot,
)


def save_loop_snapshot(loop: AgentLoop, *, make_latest: bool = True) -> None:
    """Persist the active loop under its own session ID.

    Skips empty sessions that have never been saved — a fresh loop with
    only the system prompt is not worth writing to disk.
    """
    if not loop.session_id:
        return

    messages = loop.export_messages()
    # Only skip when: no user messages, no tag, and never saved before.
    if (
        len(messages) <= 1
        and not loop.tag
        and load_session_by_id(str(loop.cwd), loop.session_id) is None
    ):
        return

    save_session_snapshot(
        cwd=str(loop.cwd),
        model=loop.model,
        messages=messages,
        session_id=loop.session_id,
        tag=loop.tag,
        make_latest=make_latest,
        session_state=loop.export_session_state(),
    )


def switch_session(
    current_loop: AgentLoop,
    target_session_id: str,
    *,
    permission_prompt=None,
    ask_user_prompt=None,
    compact_progress=None,
    event_bus=None,
) -> AgentLoop | None:
    """Save *current_loop* and switch to *target_session_id*.

    Returns a fresh ``AgentLoop`` loaded with the target session's
    messages, or ``None`` if the target session is not found.

    The caller is responsible for:
    - resolving user input to a session ID (+ picker interaction)
    - printing status messages to the console
    """
    # Persist current session before leaving it.
    save_loop_snapshot(current_loop, make_latest=False)

    # Load target from disk.
    data = load_session_by_id(str(current_loop.cwd), target_session_id)
    if data is None:
        data = load_session_by_tag(str(current_loop.cwd), target_session_id)
    if data is None:
        return None

    # Build a clean loop for the target session.
    next_loop = AgentLoop(
        cwd=current_loop.cwd,
        settings=current_loop.settings,
        permission_prompt=permission_prompt,
        ask_user_prompt=ask_user_prompt,
        compact_progress=compact_progress,
        event_bus=event_bus,
    )
    next_loop.restore_messages(data.get("messages", []))
    next_loop.restore_session_state(data.get("session_state"))
    next_loop.session_id = data.get("session_id", target_session_id)
    next_loop.tag = data.get("tag", "")
    mark_session_latest(str(current_loop.cwd), next_loop.session_id)
    return next_loop
