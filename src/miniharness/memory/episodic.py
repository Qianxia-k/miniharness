"""Episodic Memory — records of completed tasks with TTL and storage limits.

Built on :class:`~miniharness.memory.base.MemoryStore` for production-grade
I/O (atomic writes, auto-pruning, expiry).

Each entry is:
    - ``id`` — unique hex identifier
    - ``task`` — what was done (short title)
    - ``summary`` — longer description of the work
    - ``files_touched`` — list of file paths involved
    - ``outcome`` — result description (e.g. "success", "failed: timeout")
    - ``timestamp`` / ``updated_at`` — epoch seconds
    - ``signature`` — deterministic duplicate key

Agent retrieves past episodes via ``memory_search`` and logs new ones via
``memory_log`` (both are tools exposed to the model).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

from miniharness.memory.base import MemoryStore, normalize_memory_text


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
        source: str = "manual",
    ) -> str:
        """Record or refresh a completed task episode.  Returns its entry ID.

        Automatically prunes expired entries and enforces ``max_entries``
        before writing.  Pruning leaves room for the new entry so the
        final count never exceeds ``max_entries``.
        """
        task = task.strip()
        summary = summary.strip()
        outcome = outcome.strip()
        files = _clean_files(files_touched or [])
        if not task:
            raise ValueError("task is required")

        now = time.time()
        signature = _episode_signature(task, summary, files)
        entries = self._prune(self._read_all())

        existing = _find_by_signature(entries, signature)
        if existing is not None:
            entry_id = str(existing.get("id") or uuid.uuid4().hex[:12])
            existing.update({
                "id": entry_id,
                "task": task,
                "summary": summary,
                "files_touched": files,
                "outcome": outcome,
                "timestamp": now,
                "updated_at": now,
                "signature": signature,
                "status": "active",
                "disabled": False,
                "source": source,
            })
            self._write_all(entries)
            return entry_id

        # Leave room for the new entry.
        effective_max = (
            max(1, self.max_entries - 1)
            if self.max_entries is not None
            else None
        )
        saved, self.max_entries = self.max_entries, effective_max
        try:
            entries = self._prune(entries)
        finally:
            self.max_entries = saved

        entry_id = uuid.uuid4().hex[:12]
        entries.append({
            "id": entry_id,
            "task": task,
            "summary": summary,
            "files_touched": files,
            "outcome": outcome,
            "timestamp": now,
            "created_at": now,
            "updated_at": now,
            "signature": signature,
            "status": "active",
            "disabled": False,
            "source": source,
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


def _episode_signature(task: str, summary: str, files_touched: list[str]) -> str:
    text = "\n".join([task, summary, "\n".join(sorted(files_touched))])
    normalized = normalize_memory_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _find_by_signature(entries: list[dict[str, Any]], signature: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("signature") == signature:
            return entry
    return None


def _clean_files(files: list[Any]) -> list[str]:
    result: list[str] = []
    for item in files:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:50]
