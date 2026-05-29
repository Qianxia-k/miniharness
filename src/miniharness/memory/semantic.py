"""Semantic Memory — persistent project facts.

Stored as ``semantic.json`` in the per-project memory directory.
Each entry is a small JSON object with a fact string and optional tags.

Agent retrieves facts via ``memory_search`` and adds new ones via
``memory_add``.  Both are tools exposed to the model.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from miniharness.memory.store import get_memory_dir


class SemanticStore:
    """CRUD for project facts stored in ``semantic.json``."""

    def __init__(self, cwd: str | Path) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._path = get_memory_dir(self._cwd) / "semantic.json"

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

    def add(self, fact: str, *, tags: list[str] | None = None) -> str:
        """Persist a new fact.  Returns the new entry ID."""
        entries = self._read_all()
        entry_id = uuid.uuid4().hex[:12]
        entries.append({
            "id": entry_id,
            "fact": fact.strip(),
            "tags": tags or [],
            "created_at": time.time(),
        })
        self._write_all(entries)
        return entry_id

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Keyword search over facts.  Returns entries ranked by match score."""
        keywords = _tokenise(query)
        if not keywords:
            return self._read_all()[-limit:]

        entries = self._read_all()
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for entry in entries:
            # Score: keyword matches in fact text + tag bonus.
            text = entry.get("fact", "")
            tags = entry.get("tags", [])
            score = 0
            for kw in keywords:
                if kw in text.lower():
                    score += 2
                for tag in tags:
                    if kw in tag.lower():
                        score += 1
            if score > 0:
                scored.append((score, entry.get("created_at", 0), entry))

        # Sort by score descending, then recency.
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored[:limit]]

    def list_all(self) -> list[dict[str, Any]]:
        """Return all facts, newest first."""
        entries = self._read_all()
        entries.sort(key=lambda e: e.get("created_at", 0), reverse=True)
        return entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    """Split *text* into lowercase keyword tokens."""
    return [t.lower() for t in _WORD_RE.findall(text) if len(t) > 1]
