"""Context management — token budget, compaction, and message-list assembly.

Architecture (clean separation, no cross-knowledge):

    carryover.py  — OWNS tool_metadata (write + read + build attachments)
    compactor.py — PURE 4-tier compaction (messages → compacted messages)
    compiler.py  — THIN orchestrator (budget check → delegate to compactor)
    budget.py    — Token counting

Data flow::

    loop.py
      ├─ carryover.record_tool_carryover(metadata, ...)     [after each tool]
      ├─ attachments = carryover.build_compact_attachments(metadata)
      └─ compiler.compile(conversation, tools, attachments=attachments)
           └─ auto_compact_if_needed(msgs, attachments=attachments)
                └─ full_llm_compact(msgs, attachments=attachments)
"""

from miniharness.context.budget import ContextBudget, count_tokens
from miniharness.context.compactor import auto_compact_if_needed
from miniharness.context.compiler import ContextCompiler, ContextPacket

__all__ = [
    "auto_compact_if_needed",
    "ContextBudget",
    "ContextCompiler",
    "ContextPacket",
    "count_tokens",
]
