"""Episodic Memory — records of completed tasks and experiences.

Stored as ``episodic.json`` in the per-project memory directory.
Each entry captures what was done, what files were touched, and the outcome.

Agent retrieves past episodes via ``memory_search`` (which searches both
semantic and episodic stores).  It logs new episodes via ``memory_log``.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from miniharness.memory.store import get_memory_dir


class EpisodicStore:
    """Append + search for task traces stored in ``episodic.json``."""

    def __init__(self, cwd: str | Path) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._path = get_memory_dir(self._cwd) / "episodic.json"

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def _write_all(self, entries: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        *,
        task: str,
        summary: str,
        files_touched: list[str] | None = None,
        outcome: str = "",
    ) -> str:
        """Record a completed task episode.  Returns the new entry ID."""
        entries = self._read_all()
        entry_id = uuid.uuid4().hex[:12]
        entries.append({
            "id": entry_id,
            "task": task.strip(),
            "summary": summary.strip(),
            "files_touched": files_touched or [],
            "outcome": outcome.strip(),
            "timestamp": time.time(),
        })
        self._write_all(entries)
        return entry_id

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Keyword search over episode records.  Ranked by match score."""
        keywords = _tokenise(query)
        if not keywords:
            return self.list_all(limit=limit)

        entries = self._read_all()
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for entry in entries:
            # Search across task, summary, outcome, and file names.
            text = (
                f"{entry.get('task', '')} {entry.get('summary', '')} "
                f"{entry.get('outcome', '')} {' '.join(entry.get('files_touched', []))}"
            ).lower()
            score = sum(2 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, entry.get("timestamp", 0), entry))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def list_all(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent episodes, newest first."""
        entries = self._read_all()
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return entries[:limit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    """Split *text* into lowercase keyword tokens."""
    return [t.lower() for t in _WORD_RE.findall(text) if len(t) > 1]
