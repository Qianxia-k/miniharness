"""Session persistence — save and restore conversation history.

Mirrors OpenHarness's services/session_storage.py.  Every REPL turn is
saved to disk so ``--continue`` can pick up where you left off.

Storage layout::

    ~/.miniharness/sessions/{project-name}-{sha1-hash}/
        latest.json          ← always the most recent session
        session-{id}.json    ← individual snapshots
"""

from __future__ import annotations

import json
import re
import time
import uuid
from hashlib import sha1
from pathlib import Path
from typing import Any


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _sessions_root() -> Path:
    """Return the MiniHarness sessions directory."""
    path = Path.home() / ".miniharness" / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_slug(cwd: str | Path) -> str:
    """Build a stable directory name for a project directory."""
    resolved = str(Path(cwd).resolve())
    digest = sha1(resolved.encode("utf-8")).hexdigest()[:12]
    name = Path(resolved).name
    return f"{name}-{digest}"


def get_session_dir(cwd: str | Path) -> Path:
    """Return (and create) the session directory for a project."""
    session_dir = _sessions_root() / _project_slug(cwd)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _normalize_session_id(session_id: str | None) -> str:
    """Return a safe session ID suitable for a session filename."""
    sid = session_id or _new_session_id()
    if not _SESSION_ID_RE.fullmatch(sid):
        raise ValueError(
            "session_id must be 1-80 characters containing only letters, numbers, '_' or '-'"
        )
    return sid


def _session_path(session_dir: Path, session_id: str) -> Path:
    sid = _normalize_session_id(session_id)
    return session_dir / f"session-{sid}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(path)


def _try_read_session(session_dir: Path, session_id: str) -> dict[str, Any] | None:
    """Read an existing session file if it exists, without throwing."""
    try:
        path = _session_path(session_dir, session_id)
    except ValueError:
        return None
    return _read_json(path) if path.exists() else None


def save_session_snapshot(
    *,
    cwd: str | Path,
    model: str,
    messages: list[dict[str, Any]],
    session_id: str | None = None,
    tag: str = "",
    make_latest: bool = True,
    session_state: dict[str, Any] | None = None,
) -> Path:
    """Persist a session snapshot.  Saves as both latest.json and session-{id}.json.

    The *tag* parameter is preserved across saves so a ``/tag`` name
    survives subsequent auto-saves.
    """
    session_dir = get_session_dir(cwd)
    sid = _normalize_session_id(session_id)
    now = time.time()
    existing = _try_read_session(session_dir, sid)

    # Preserve existing tag if not explicitly given.
    if not tag and existing:
        tag = existing.get("tag", "")

    summary = ""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content", "").strip():
            summary = msg["content"].strip()[:80]
            break

    payload = {
        "session_id": sid,
        "cwd": str(Path(cwd).resolve()),
        "model": model,
        "messages": messages,
        "created_at": existing.get("created_at", now) if existing else now,
        "updated_at": now,
        "summary": summary,
        "tag": tag,
        "message_count": len(messages),
        "session_state": session_state or {},
    }

    session_path = session_dir / f"session-{sid}.json"
    latest_path = session_dir / "latest.json"

    # Write the durable session file first; latest.json is just the pointer copy.
    _atomic_write_json(session_path, payload)
    if make_latest:
        _atomic_write_json(latest_path, payload)
    return latest_path


def mark_session_latest(cwd: str | Path, session_id: str, *, touch: bool = True) -> bool:
    """Make an existing session the latest session without changing its messages."""
    session_dir = get_session_dir(cwd)
    data = _try_read_session(session_dir, session_id)
    if data is None:
        return False

    if touch:
        data["updated_at"] = time.time()
        _atomic_write_json(_session_path(session_dir, session_id), data)

    _atomic_write_json(session_dir / "latest.json", data)
    return True


def load_latest_session(cwd: str | Path) -> dict[str, Any] | None:
    """Load the most recent session snapshot for a project.

    Returns None when no saved session exists.
    """
    path = get_session_dir(cwd) / "latest.json"
    if not path.exists():
        return None
    return _read_json(path)


def load_session_by_id(cwd: str | Path, session_id: str) -> dict[str, Any] | None:
    """Load a specific session by ID.

    Supports ``session_id="latest"`` as a shortcut for the latest session.
    """
    session_dir = get_session_dir(cwd)
    if session_id == "latest":
        return load_latest_session(cwd)

    try:
        path = _session_path(session_dir, session_id)
    except ValueError:
        return None
    return _read_json(path) if path.exists() else None


def list_sessions(cwd: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    """List saved sessions for a project, newest first."""
    session_dir = get_session_dir(cwd)
    sessions: list[dict[str, Any]] = []

    for path in session_dir.glob("session-*.json"):
        data = _read_json(path)
        if data is None:
            continue

        sessions.append({
            "session_id": data.get("session_id", path.stem.replace("session-", "")),
            "summary": data.get("summary", ""),
            "tag": data.get("tag", ""),
            "message_count": data.get("message_count", 0),
            "model": data.get("model", ""),
            "created_at": data.get("created_at", path.stat().st_mtime),
            "updated_at": data.get("updated_at", data.get("created_at", path.stat().st_mtime)),
        })

    sessions.sort(key=lambda item: item["updated_at"], reverse=True)
    return sessions[:limit]


def rename_session(cwd: str | Path, session_id: str, name: str) -> bool:
    """Add a human-readable tag to a session by writing it into the JSON.

    The tag is stored as a field inside the original ``session-{id}.json``,
    so the session is still listed by ``list_sessions`` and loadable by its
    original ID.  ``load_session_by_tag`` scans all sessions to find a match.

    Returns True on success.
    """
    session_dir = get_session_dir(cwd)
    try:
        session_path = _session_path(session_dir, session_id)
    except ValueError:
        return False
    if not session_path.exists():
        return False

    data = _read_json(session_path)
    if data is None:
        return False
    data["tag"] = name
    data["updated_at"] = time.time()
    _atomic_write_json(session_path, data)

    # Also update latest.json if it points to the same session.
    latest_path = session_dir / "latest.json"
    if latest_path.exists():
        latest_data = _read_json(latest_path)
        if latest_data and latest_data.get("session_id") == session_id:
            latest_data["tag"] = name
            latest_data["updated_at"] = data["updated_at"]
            _atomic_write_json(latest_path, latest_data)
    return True


def load_session_by_tag(cwd: str | Path, tag: str) -> dict[str, Any] | None:
    """Load a session by its tag name.

    Scans all ``session-*.json`` files for one whose ``tag`` field matches.
    """
    session_dir = get_session_dir(cwd)
    for path in sorted(
        session_dir.glob("session-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        data = _read_json(path)
        if data is None:
            continue
        if data.get("tag") == tag:
            return data
    return None
