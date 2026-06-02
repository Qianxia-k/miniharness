"""Memory system — persistent knowledge that survives context compaction.

Architecture (production-grade, no duplication)::

    base.py       — MemoryStore base class (atomic I/O, TTL, pruning, search)
    core.py       — CoreMemory (markdown file, always in system prompt)
    semantic.py   — SemanticStore (facts, on MemoryStore)
    episodic.py   — EpisodicStore (task traces, on MemoryStore)
    store.py      — Shared path utilities (project-slug hashing)
"""

from miniharness.memory.base import MemoryStore
from miniharness.memory.core import CoreMemory
from miniharness.memory.store import get_memory_dir

__all__ = ["CoreMemory", "MemoryStore", "get_memory_dir"]
