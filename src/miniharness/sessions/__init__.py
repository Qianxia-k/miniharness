"""Session persistence and lifecycle management.

Re-exports the public API from :mod:`.storage` and :mod:`.manager`
so external code can import everything from ``miniharness.sessions``.
"""

from miniharness.sessions.manager import save_loop_snapshot, switch_session
from miniharness.sessions.storage import (
    get_session_dir,
    list_sessions,
    load_latest_session,
    load_session_by_id,
    load_session_by_tag,
    mark_session_latest,
    rename_session,
    save_session_snapshot,
)

__all__ = [
    # storage
    "get_session_dir",
    "list_sessions",
    "load_latest_session",
    "load_session_by_id",
    "load_session_by_tag",
    "mark_session_latest",
    "rename_session",
    "save_session_snapshot",
    # manager
    "save_loop_snapshot",
    "switch_session",
]
