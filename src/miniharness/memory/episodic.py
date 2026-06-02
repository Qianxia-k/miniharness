"""Episodic Memory — records of completed tasks with TTL and storage limits.

Built on :class:`~miniharness.memory.base.MemoryStore` for production-grade
I/O (atomic writes, auto-pruning, expiry).

Each entry is:
    - ``id`` — unique hex identifier
    - ``task`` — what was done (short title)
    - ``summary`` — longer description of the work
    - ``files_touched`` — list of file paths involved
    - ``outcome`` — result description (e.g. "success", "failed: timeout")
    - ``timestamp`` — epoch seconds

Agent retrieves past episodes via ``memory_search`` and logs new ones via
``memory_log`` (both are tools exposed to the model).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from miniharness.memory.base import MemoryStore


class EpisodicStore(MemoryStore):
    """Production-grade episodic task storage.

    Usage (backward-compatible with old API)::

        store = EpisodicStore("/path/to/project")
        store.log(
            task="Refactored auth module",
            summary="Extracted JWT logic into middleware",
            files_touched=["src/auth.py", "src/middleware.py"],
            outcome="success",
        )
        results = store.search("auth JWT", limit=5)
    """

    _filename = "episodic.json"

    def __init__(
        self,
        cwd: str | Path,
        *,
        max_entries: int | None = 200,
        ttl_seconds: float | None = 1 * 86400,  # x days default
    ) -> None:
        super().__init__(cwd, max_entries=max_entries, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # Public API (backward-compatible)
    # ------------------------------------------------------------------

    def log(
        self,
        *,
        task: str,
        summary: str,
        files_touched: list[str] | None = None,
        outcome: str = "",
    ) -> str:
        """Record a completed task episode.  Returns the new entry ID.

        Automatically prunes expired entries and enforces ``max_entries``
        before writing.  Pruning leaves room for the new entry so the
        final count never exceeds ``max_entries``.
        """
        # Leave room for the new entry.
        effective_max = (
            max(1, self.max_entries - 1)
            if self.max_entries is not None
            else None
        )
        entries = self._read_all()
        saved, self.max_entries = self.max_entries, effective_max
        try:
            entries = self._prune(entries)
        finally:
            self.max_entries = saved

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

    # search() and list_all() are inherited from MemoryStore.
    # The search scoring is: 2× per keyword match in task+summary+outcome+files.

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _entry_search_text(self, entry: dict[str, Any]) -> str:
        """Build searchable text from task, summary, outcome, and file names."""
        parts = [
            entry.get("task", ""),
            entry.get("summary", ""),
            entry.get("outcome", ""),
            " ".join(entry.get("files_touched", [])),
        ]
        return " ".join(parts).lower()
