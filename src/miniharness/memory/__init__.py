"""Memory system — persistent knowledge that survives context compaction.

Core memory (markdown) is always injected into the system prompt.
Semantic and episodic memory (JSON) are queried on-demand via tools.
"""

from miniharness.memory.core import CoreMemory
from miniharness.memory.store import get_memory_dir

__all__ = ["CoreMemory", "get_memory_dir"]
