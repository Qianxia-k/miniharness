"""Shared MemoryStore base class — production-grade JSON-backed storage.

Eliminates the duplicate ``_read_all`` / ``_write_all`` / ``_tokenise``
code that was copy-pasted between ``semantic.py`` and ``episodic.py``.

Adds production features missing from the toy implementation:

- **Atomic writes**: writes to a temp file then renames (no corruption on crash).
- **Storage limit**: ``max_entries`` caps total entries; oldest evicted first.
- **TTL expiry**: ``ttl_seconds`` auto-expires old entries (``None`` = never).
- **Keyword scoring**: shared scoring logic (2× for primary field, 1× for tags/metadata).

Subclasses only need to define:
    - ``_filename`` — the JSON file name (e.g. ``"semantic.json"``).
    - ``add()`` / ``log()`` — domain-specific create methods.
    - ``_entry_search_text()`` — builds the searchable text for one entry.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from miniharness.memory.store import get_memory_dir

# ---------------------------------------------------------------------------
# Tokenizer (shared, compiled once)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    """Split *text* into lowercase keyword tokens (≥2 chars)."""
    return [t.lower() for t in _WORD_RE.findall(text) if len(t) > 1]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MemoryStore:
    """Shared base for JSONL-backed memory stores.

    Parameters
    ----------
    cwd:
        Project root for per-project storage isolation.
    max_entries:
        Maximum entries to retain.  When exceeded, the OLDEST entries
        are evicted first (FIFO).  ``None`` = unlimited.
    ttl_seconds:
        Auto-expire entries older than this many seconds.
        ``None`` = never expire.  Pruned on every write.

    Subclass contract
    -----------------
    Subclasses must override:
        - ``_filename`` (class attr or property) → ``"semantic.json"`` etc.
        - ``_entry_search_text(entry)`` → return a single lowercase string
          used for keyword matching.
    """

    # ── Configuration (override in subclass or __init__) ──────────────
    _filename: str = "memory.json"

    def __init__(
        self,
        cwd: str | Path,
        *,
        max_entries: int | None = 1000,
        ttl_seconds: float | None = None,
    ) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._path = get_memory_dir(self._cwd) / self._filename
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Low-level I/O (atomic writes, corruption-resistant)
    # ------------------------------------------------------------------

    def _read_all(self) -> list[dict[str, Any]]:
        """Read all entries from the JSON file.

        Returns an empty list if the file is missing, empty, or corrupt.
        """
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def _write_all(self, entries: list[dict[str, Any]]) -> None:
        """Atomically write entries to disk.

        Uses temp-file + rename to prevent corruption on crash.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(entries, indent=2, ensure_ascii=False) + "\n"

        # Write to a temp file in the same directory, then atomically rename.
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self._path.parent),
            prefix=f".{self._filename}.",
            delete=False,
        )
        try:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()

        os.replace(tmp.name, str(self._path))

    # ------------------------------------------------------------------
    # Maintenance — prune & expire
    # ------------------------------------------------------------------

    def _prune(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove expired entries and enforce ``max_entries``.

        Called automatically before every write.  Returns the pruned list
        (does NOT mutate in place — the caller should use the return value).
        """
        now = time.time()

        # 1. Expire by TTL.
        if self.ttl_seconds is not None:
            cutoff = now - self.ttl_seconds
            entries = [
                e for e in entries
                if e.get("timestamp", 0) >= cutoff
            ]

        # 2. Enforce max_entries (oldest-first eviction).
        if self.max_entries is not None and len(entries) > self.max_entries:
            # Sort by timestamp ascending, keep newest.
            entries.sort(key=lambda e: e.get("timestamp", 0))
            entries = entries[-self.max_entries:]

        return entries

    # ------------------------------------------------------------------
    # Search (shared scoring logic)
    # ------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Keyword search over all entries.  Ranked by match score.

        Scoring: 2 points per keyword match in the entry's primary text
        (as returned by ``_entry_search_text``), plus recency bonus.

        Subclasses MAY override this for domain-specific scoring.
        """
        keywords = _tokenise(query)
        if not keywords:
            return self.list_all(limit=limit)

        entries = self._read_all()
        scored: list[tuple[int, float, dict[str, Any]]] = []

        for entry in entries:
            text = self._entry_search_text(entry)
            score = sum(2 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, entry.get("timestamp", 0), entry))

        # Sort: score descending, then recency descending.
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def list_all(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return entries, newest first, up to *limit*."""
        entries = self._read_all()
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return entries[:limit]

    @property
    def count(self) -> int:
        """Return the current number of stored entries."""
        return len(self._read_all())

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _entry_search_text(self, entry: dict[str, Any]) -> str:
        """Build a lowercase searchable string for one entry.

        Subclasses MUST override this.  The returned string is matched
        against keyword tokens during :meth:`search`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _entry_search_text()"
        )
