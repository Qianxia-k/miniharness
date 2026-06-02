"""Semantic Memory — persistent project facts with TTL and storage limits.

Built on :class:`~miniharness.memory.base.MemoryStore` for production-grade
I/O (atomic writes, auto-pruning, expiry).

Each entry is:
    - ``id`` — unique hex identifier
    - ``fact`` — the fact string (free text)
    - ``tags`` — optional list of tags for categorisation
    - ``created_at`` — epoch timestamp

Agent retrieves facts via ``memory_search`` and adds new ones via
``memory_add`` (both are tools exposed to the model).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from miniharness.memory.base import MemoryStore


class SemanticStore(MemoryStore):
    """Production-grade semantic fact storage.

    Usage (backward-compatible with old API)::

        store = SemanticStore("/path/to/project")
        store.add("The project uses FastAPI for HTTP", tags=["tech-stack"])
        results = store.search("FastAPI", limit=5)
    """

    _filename = "semantic.json"

    def __init__(
        self,
        cwd: str | Path,
        *,
        max_entries: int | None = 500,
        ttl_seconds: float | None = None,
    ) -> None:
        super().__init__(cwd, max_entries=max_entries, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------
    # Public API (backward-compatible)
    # ------------------------------------------------------------------

    def add(self, fact: str, *, tags: list[str] | None = None) -> str:
        """Persist a new fact.  Returns the new entry ID.

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
        # Temporarily lower the cap for this prune pass.
        saved, self.max_entries = self.max_entries, effective_max
        try:
            entries = self._prune(entries)
        finally:
            self.max_entries = saved

        entry_id = uuid.uuid4().hex[:12]
        entries.append({
            "id": entry_id,
            "fact": fact.strip(),
            "tags": tags or [],
            "created_at": time.time(),
        })
        self._write_all(entries)
        return entry_id

    # search() and list_all() are inherited from MemoryStore.
    # The search scoring is: 2× per keyword match in fact + tags.

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _entry_search_text(self, entry: dict[str, Any]) -> str:
        """Build searchable text: fact + space-separated tags."""
        fact = entry.get("fact", "")
        tags = " ".join(entry.get("tags", []))
        return f"{fact} {tags}".lower()
