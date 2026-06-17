"""Semantic Memory — persistent project facts with TTL and storage limits.

Built on :class:`~miniharness.memory.base.MemoryStore` for production-grade
I/O (atomic writes, auto-pruning, expiry).

Each entry is:
    - ``id`` — unique hex identifier
    - ``fact`` — the fact string (free text)
    - ``tags`` — optional list of tags for categorisation
    - ``created_at`` — epoch timestamp
    - ``updated_at`` — epoch timestamp for refresh/consolidation
    - ``signature`` — deterministic duplicate key for the normalized fact
    - ``status`` / ``disabled`` — active/superseded lifecycle controls

Agent retrieves facts via ``memory_search`` and adds new ones via
``memory_add`` (both are tools exposed to the model).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

from miniharness.memory.base import MemoryStore, normalize_memory_text


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

    def add(
        self,
        fact: str,
        *,
        tags: list[str] | None = None,
        source: str = "manual",
        confidence: float | None = None,
        supersedes: list[str] | None = None,
        contradicts: list[str] | None = None,
    ) -> str:
        """Persist or refresh a fact.  Returns the active entry ID.

        Automatically prunes expired entries and enforces ``max_entries``
        before writing.  Pruning leaves room for the new entry so the
        final count never exceeds ``max_entries``.

        This is intentionally not append-only.  OpenHarness-style memory
        needs lifecycle semantics: duplicate facts refresh the existing
        record, and newer facts can explicitly supersede stale records.
        """
        fact = fact.strip()
        if not fact:
            raise ValueError("fact is required")

        clean_tags = _merge_tags(tags or [])
        now = time.time()
        signature = _fact_signature(fact)
        superseded_ids = _valid_ids([*(supersedes or []), *(contradicts or [])])

        entries = self._prune(self._read_all())

        existing = _find_by_signature(entries, signature)
        if existing is not None:
            entry_id = str(existing.get("id") or uuid.uuid4().hex[:12])
            existing.update({
                "id": entry_id,
                "fact": fact,
                "tags": _merge_tags([*existing.get("tags", []), *clean_tags]),
                "updated_at": now,
                "timestamp": now,
                "signature": signature,
                "status": "active",
                "disabled": False,
                "source": source,
            })
            if confidence is not None:
                existing["confidence"] = _clamp_confidence(confidence)
            if superseded_ids:
                existing["supersedes"] = _merge_ids([
                    *existing.get("supersedes", []),
                    *superseded_ids,
                ])
            _disable_superseded(entries, superseded_ids, entry_id, now)
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
        _disable_superseded(entries, superseded_ids, entry_id, now)
        entries.append({
            "id": entry_id,
            "fact": fact,
            "tags": clean_tags,
            "created_at": now,
            "updated_at": now,
            "timestamp": now,
            "signature": signature,
            "status": "active",
            "disabled": False,
            "source": source,
            "confidence": _clamp_confidence(confidence) if confidence is not None else None,
            "supersedes": superseded_ids,
        })
        self._write_all(entries)
        return entry_id

    def manifest(self, *, limit: int = 50) -> str:
        """Return active facts as a compact manifest for extraction prompts."""
        entries = self.list_all(limit=limit)
        if not entries:
            return "(none)"

        lines: list[str] = []
        for entry in entries:
            tags = ", ".join(entry.get("tags", []))
            tag_text = f" [{tags}]" if tags else ""
            lines.append(f"- {entry.get('id', '?')}: {entry.get('fact', '')}{tag_text}")
        return "\n".join(lines)

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


def _fact_signature(fact: str) -> str:
    normalized = normalize_memory_text(fact)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _find_by_signature(entries: list[dict[str, Any]], signature: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("signature") == signature:
            return entry
    return None


def _disable_superseded(
    entries: list[dict[str, Any]],
    ids: list[str],
    replacement_id: str,
    now: float,
) -> None:
    if not ids:
        return
    id_set = set(ids)
    for entry in entries:
        if entry.get("id") in id_set and entry.get("id") != replacement_id:
            entry["status"] = "superseded"
            entry["disabled"] = True
            entry["superseded_by"] = replacement_id
            entry["updated_at"] = now
            entry["timestamp"] = now


def _merge_tags(tags: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        cleaned = tag.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result[:12]


def _valid_ids(ids: list[Any]) -> list[str]:
    result: list[str] = []
    for item in ids:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
    return result


def _merge_ids(ids: list[Any]) -> list[str]:
    return _valid_ids(ids)


def _clamp_confidence(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
