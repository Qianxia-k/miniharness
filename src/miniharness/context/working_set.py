"""Working Set — tracks files the agent is actively working with.

Engineering-grade implementation with graded protection:

1. **access tracking**  — every read / write / edit records ``last_access``.
2. **auto-protect**     — write / edit operations protect the file for a TTL
   (default 5 min).  After the TTL expires the file becomes evictable.
3. **explicit pin**     — user or agent calls ``pin()``; survives until ``unpin()``.

Eviction order when the set exceeds *max_files*:

    a. unprotected + TTL-expired  (LRU-first)
    b. auto-protected             (LRU-first — last resort)
    c. explicitly pinned          (never evicted)

This behaves like IDE tabs: recently accessed files stay visible, actively
edited files resist closing, and pinned files are permanent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_FILES = 20
AUTO_PROTECT_TTL = 300  # seconds (5 minutes)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass
class ActiveFile:
    """One entry in the working set."""

    path: str
    last_access: float
    explicit_pin: bool = False
    auto_protect_until: float = 0.0  # epoch-seconds; 0 = not protected
    size: int = 0
    page: int = 0
    total_pages: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Working Set
# ---------------------------------------------------------------------------


class WorkingSet:
    """LRU cache of accessed files with IDE-tab semantics.

    Not thread-safe — the agent loop is single-threaded.
    """

    def __init__(
        self,
        *,
        max_files: int = DEFAULT_MAX_FILES,
        auto_protect_ttl: float = AUTO_PROTECT_TTL,
    ) -> None:
        if max_files < 1:
            raise ValueError("max_files must be >= 1")
        self._max = max_files
        self._ttl = auto_protect_ttl
        self._entries: dict[str, ActiveFile] = {}

    # ------------------------------------------------------------------
    # File lifecycle
    # ------------------------------------------------------------------

    def touch(
        self,
        path: str,
        *,
        is_write: bool = False,
        size: int = 0,
        page: int = 0,
        total_pages: int = 0,
    ) -> None:
        """Record that *path* was accessed, creating an entry if needed.

        If *is_write* is True the file gets auto-protected for *self._ttl*
        seconds (a write implies the file is in-progress work).
        """
        if not path or not path.strip():
            return
        path = path.strip()

        now = time.time()
        entry = self._entries.get(path)
        if entry is not None:
            entry.last_access = now
            if is_write:
                entry.auto_protect_until = now + self._ttl
            if size:
                entry.size = size
            if total_pages:
                entry.page = page
                entry.total_pages = total_pages
        else:
            self._entries[path] = ActiveFile(
                path=path,
                last_access=now,
                auto_protect_until=(now + self._ttl) if is_write else 0.0,
                size=size,
                page=page,
                total_pages=total_pages,
            )
            self._evict(protect=path)

    def pin(self, path: str) -> bool:
        """Explicitly pin *path* — it will never be auto-evicted."""
        entry = self._entries.get(path.strip())
        if entry is None:
            return False
        entry.explicit_pin = True
        entry.last_access = time.time()
        return True

    def unpin(self, path: str) -> bool:
        """Remove explicit pin.  The file can again be evicted."""
        entry = self._entries.get(path.strip())
        if entry is None:
            return False
        entry.explicit_pin = False
        return True

    def remove(self, path: str) -> bool:
        """Explicitly remove a file from the working set."""
        return self._entries.pop(path.strip(), None) is not None

    def clear(self) -> None:
        """Remove all files (e.g. on /clear)."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def contains(self, path: str) -> bool:
        return path.strip() in self._entries

    def get(self, path: str) -> ActiveFile | None:
        return self._entries.get(path.strip())

    def list_active(self) -> list[dict[str, Any]]:
        """Return all entries ordered by most-recent access."""
        now = time.time()
        return sorted(
            [
                {
                    "path": e.path,
                    "last_access": e.last_access,
                    "pinned": e.explicit_pin,
                    "protected": e.auto_protect_until > now,
                    "size": e.size,
                    "page": e.page,
                    "total_pages": e.total_pages,
                }
                for e in self._entries.values()
            ],
            key=lambda d: d["last_access"],
            reverse=True,
        )

    @property
    def pinned_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.explicit_pin)

    @property
    def total_count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    def render_for_context(self) -> str:
        """Return a compact block for the system prompt."""
        if not self._entries:
            return ""

        entries = self.list_active()
        max_show = min(len(entries), 10)
        lines = ["[Working Set — recently accessed files]"]

        for entry in entries[:max_show]:
            markers = ""
            if entry["pinned"]:
                markers += "📌"
            elif entry["protected"]:
                markers += "🛡️"
            else:
                markers += "  "
            size_str = _format_size(entry["size"])
            size_info = f" ({size_str})" if size_str else ""
            page_info = ""
            if entry["total_pages"]:
                page_info = f" [page {entry['page']+1}/{entry['total_pages']}]"
            lines.append(f"{markers} {entry['path']}{size_info}{page_info}")

        if len(entries) > max_show:
            lines.append(f"  ... and {len(entries) - max_show} more")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self, *, protect: str | None = None) -> None:
        """Drop entries until within capacity.

        Eviction priority (lowest → highest priority to keep):
        1. Unprotected + TTL expired (safest to evict)
        2. Auto-protected (evict as last resort)
        3. Explicitly pinned (never evicted)
        """
        now = time.time()
        while len(self._entries) > self._max:

            # Tier 1: unprotected & TTL expired (exclude protect + pinned).
            tier1 = [
                (p, e) for p, e in self._entries.items()
                if not e.explicit_pin
                and e.auto_protect_until <= now
                and p != protect
            ]
            if tier1:
                oldest = min(tier1, key=lambda item: item[1].last_access)
                del self._entries[oldest[0]]
                continue

            # Tier 2: auto-protected (still within TTL) — last resort.
            tier2 = [
                (p, e) for p, e in self._entries.items()
                if not e.explicit_pin
                and p != protect
            ]
            if tier2:
                oldest = min(tier2, key=lambda item: item[1].last_access)
                del self._entries[oldest[0]]
                continue

            # Tier 3: everything else is explicitly pinned or protected.
            return


def _format_size(size: int) -> str:
    if size <= 0:
        return ""
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} GB"
