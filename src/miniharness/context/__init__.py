"""Context management — token budget, compaction, and message-list assembly.

Public API surface (what external code should import from here):

    - ``ContextBudget`` — token counting and budget tracking
    - ``ContextCompiler`` / ``ContextPacket`` — message-list assembly + compaction orchestration
    - ``auto_compact_if_needed`` / ``compact_messages`` — compaction pipeline (4-tier)
    - ``count_tokens`` — standalone token estimation

Internal implementation details (used by ``AgentLoop`` but not part of the
context module's public contract):

    - ``context.carryover`` — tool_metadata state management (import from there directly)
    - ``context.working_set`` — legacy IDE-tab file tracker (deprecated, kept for reference)
"""

from miniharness.context.budget import ContextBudget, count_tokens
from miniharness.context.compactor import auto_compact_if_needed, compact_messages
from miniharness.context.compiler import ContextCompiler, ContextPacket

__all__ = [
    "ContextBudget",
    "ContextCompiler",
    "ContextPacket",
    "auto_compact_if_needed",
    "compact_messages",
    "count_tokens",
]
