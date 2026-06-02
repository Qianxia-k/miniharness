"""Display helpers — UI rendering for the agent loop.

Separated from ``loop.py`` to keep the agent loop focused on orchestration
(conversation management, tool execution, compaction).  All console output
lives here so the rendering layer can be swapped without touching core logic.

This module holds a module-level ``rich.Console`` instance (the same one
used by ``cli.py``).  Future work: inject the console as a dependency
instead of using a module-level singleton.
"""

from __future__ import annotations

from rich.console import Console

from miniharness.llm import CompactPhase

console = Console()


# ---------------------------------------------------------------------------
# Compaction progress display
# ---------------------------------------------------------------------------


_COMPACT_PHASE_LABELS: dict[CompactPhase, str] = {
    CompactPhase.COMPACT_START: "Compaction started",
    CompactPhase.COMPACT_END: "Compaction complete",
    CompactPhase.COMPACT_FAILED: "Compaction failed",
    CompactPhase.COMPACT_RETRY: "Compaction retry",
    CompactPhase.HOOKS_START: "Pre-compact hooks",
    CompactPhase.CONTEXT_COLLAPSE_START: "Context collapse",
    CompactPhase.CONTEXT_COLLAPSE_END: "Context collapse done",
    CompactPhase.SESSION_MEMORY_START: "Session memory",
    CompactPhase.SESSION_MEMORY_END: "Session memory done",
}


def show_status(message: str) -> None:
    """Print a transient status / info message."""
    console.print(f"  [dim]ℹ {message}[/dim]")


def show_compact_event(phase: CompactPhase, *, trigger: str = "auto") -> None:
    """Print a compaction lifecycle event."""
    label = _COMPACT_PHASE_LABELS.get(phase, phase.value)
    console.print(f"  [dim]🔄 [{trigger}] {label}[/dim]")


def show_compaction_summary(stats: dict) -> None:
    """Print a human-readable summary of what compaction did."""
    parts: list[str] = []

    if stats.get("tier1_microcompact"):
        cleared = stats.get("microcompact_cleared", 0)
        saved = stats.get("microcompact_tokens_saved", 0)
        parts.append(f"microcompact: cleared {cleared} results (~{saved} tok)")

    if stats.get("tier2_context_collapse"):
        blocks = stats.get("context_collapse_blocks", 0)
        saved = stats.get("context_collapse_tokens_saved", 0)
        parts.append(f"context-collapse: {blocks} blocks (~{saved} tok)")

    if stats.get("tier3_session_memory"):
        lines = stats.get("session_memory_lines", 0)
        saved = stats.get("session_memory_tokens_saved", 0)
        parts.append(f"session-memory: {lines} lines (~{saved} tok)")

    if stats.get("tier4_full_llm_compact"):
        summary_chars = stats.get("compacted_summary_chars", 0)
        attachments = stats.get("attachments_built", 0)
        parts.append(
            f"full-LLM: summary={summary_chars} chars, {attachments} attachments"
        )

    tier_list = " → ".join(parts) if parts else "no tiers ran"
    dropped = stats.get("dropped", 0)
    budget_ratio = stats.get("budget_ratio", 0)

    console.print(
        f"  [dim]Compacted: {tier_list} "
        f"({dropped} dropped) "
        f"budget={budget_ratio:.0%}[/dim]"
    )

    if stats.get("tier4_full_llm_compact"):
        summary_chars = stats.get("compacted_summary_chars", 0)
        if summary_chars > 0:
            console.print(
                f"  [bold cyan]📝 Structured compaction summary "
                f"({summary_chars} chars)[/bold cyan]"
            )
