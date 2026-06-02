"""Audit trail utilities — structured logging for agent activity.

Provides a simple, file-based audit trail that records every tool
execution, session boundary, and failure event as JSONL (one JSON
object per line).

Usage::

    from miniharness.hooks.audit import AuditLogger

    logger = AuditLogger("~/.miniharness/audit")
    logger.log("tool_executed", tool_name="bash", input={"command": "ls"})
    logger.log("tool_failed", tool_name="bash", error="timeout")

    # Read recent entries:
    for entry in logger.tail(20):
        print(entry["tool_name"], entry["event"])
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only JSONL audit logger.

    Each call to :meth:`log` writes one JSON line to the audit file.
    The file is rotated by date (one file per day).
    """

    def __init__(self, log_dir: str = "~/.miniharness/audit") -> None:
        self._log_dir = Path(os.path.expanduser(log_dir))
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: str, /, **fields: Any) -> None:
        """Write one audit entry.

        Parameters
        ----------
        event:
            Event type string (e.g. ``"pre_tool_use"``, ``"tool_failed"``).
        **fields:
            Arbitrary key-value pairs to include in the entry.
            Automatically enriched with ``ts`` (ISO 8601 timestamp).
        """
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": event,
            **fields,
        }
        _append_jsonl(self._today_file(), entry)

    def log_tool_execution(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        output_preview: str = "",
        is_error: bool = False,
        session_id: str = "",
        duration_ms: float = 0,
    ) -> None:
        """Log a complete tool execution (called after POST_TOOL_USE)."""
        self.log(
            "tool_executed",
            tool_name=tool_name,
            input=tool_input,
            output_preview=output_preview[:200],
            is_error=is_error,
            session_id=session_id,
            duration_ms=round(duration_ms, 1),
        )

    def log_tool_failed(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        error: str,
        session_id: str = "",
    ) -> None:
        """Log a tool failure (called on TOOL_FAILED)."""
        self.log(
            "tool_failed",
            tool_name=tool_name,
            input=tool_input,
            error=error[:500],
            session_id=session_id,
        )

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the most recent *n* audit entries."""
        today = self._today_file()
        if not today.exists():
            return []
        entries = _read_jsonl(today)
        return entries[-n:]

    def search(
        self,
        tool_name: str | None = None,
        event: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search recent audit entries by tool_name and/or event type."""
        results: list[dict[str, Any]] = []
        today = self._today_file()
        if not today.exists():
            return []
        for entry in reversed(_read_jsonl(today)):
            if tool_name and entry.get("tool_name") != tool_name:
                continue
            if event and entry.get("event") != event:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _today_file(self) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self._log_dir / f"audit-{date_str}.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    """Atomically append one JSON line to a file."""
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all entries from a JSONL file."""
    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries
