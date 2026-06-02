"""Context Compiler — thin orchestrator between conversation and compaction.

Responsibilities:
    ✅ Export conversation → OpenAI format
    ✅ Check token budget
    ✅ Delegate to 4-tier compaction pipeline when needed
    ✅ Post-compaction system-prompt safety net

What it does NOT do:
    ❌ Build compact attachments → ``carryover.py`` (single owner of tool_metadata)
    ❌ Manage tool_metadata → ``carryover.py``
    ❌ Assemble system prompt text → ``prompts/system.py``

The compiler receives pre-built attachments from the caller (built from
``tool_metadata`` by ``carryover.build_compact_attachments()``).  It passes
them through to the compactor without knowing anything about their contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miniharness.context.budget import ContextBudget
from miniharness.context.compactor import auto_compact_if_needed
from miniharness.messages import Conversation


@dataclass
class ContextPacket:
    """Fully assembled context for one LLM API call."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)


class ContextCompiler:
    """Orchestrate budget checks and compaction.

    Usage::

        compiler = ContextCompiler(budget=budget, llm_stream=llm.stream,
                                   keep_last_n_turns=3)
        packet = await compiler.compile(conversation, tools,
                                        attachments=attachments)
    """

    def __init__(
        self,
        *,
        budget: ContextBudget,
        llm_stream,
        keep_last_n_turns: int = 3,
    ) -> None:
        self.budget = budget
        self.llm_stream = llm_stream
        self.keep_last_n_turns = keep_last_n_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compile(
        self,
        conversation: Conversation,
        tools: list[dict[str, Any]],
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ContextPacket:
        """Assemble the final context for an API call.

        Parameters
        ----------
        conversation:
            Current conversation (first message = complete system prompt).
        tools:
            OpenAI-format tool schemas.
        attachments:
            Pre-built compact attachments from carryover (or None).

        Returns
        -------
        ContextPacket
            Ready-to-use messages + tools + stats.
        """
        msgs = conversation.to_openai()
        msgs = _ensure_system_prompt(msgs)

        stats = self._empty_stats()
        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)

        if self.budget.is_over_budget(msgs):
            msgs, compaction_stats = await auto_compact_if_needed(
                msgs,
                budget=self.budget,
                attachments=attachments,
                llm_stream=self.llm_stream,
                keep_last_n_turns=self.keep_last_n_turns,
            )
            stats.update(compaction_stats)

        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)
        return ContextPacket(messages=msgs, tools=tools, stats=stats)

    async def compact_if_needed(
        self,
        messages: list[dict[str, Any]],
        *,
        attachments: list[dict[str, Any]] | None = None,
        force: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run compaction on raw message list (for PTL reactive compaction)."""
        stats = self._empty_stats()
        stats["token_count"] = self.budget.tokens_used(messages)
        stats["budget_ratio"] = self.budget.usage_ratio(messages)

        if not (force or self.budget.is_over_budget(messages)):
            return messages, stats

        msgs, compaction_stats = await auto_compact_if_needed(
            messages,
            budget=self.budget,
            attachments=attachments,
            llm_stream=self.llm_stream,
            keep_last_n_turns=self.keep_last_n_turns,
        )
        stats.update(compaction_stats)
        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)
        return msgs, stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "token_count": 0,
            "budget_ratio": 0.0,
            "compacted": False,
            "tier1_microcompact": False,
            "tier2_context_collapse": False,
            "tier3_session_memory": False,
            "tier4_full_llm_compact": False,
            "dropped": 0,
            "compacted_summary_chars": 0,
            "attachments_built": 0,
        }


# ---------------------------------------------------------------------------
# Safety net
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PLACEHOLDER = (
    "You are MiniHarness, a coding agent. "
    "(System prompt was lost — please restore from session.)"
)


def _ensure_system_prompt(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure the message list starts with a system prompt."""
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role": "system", "content": _SYSTEM_PROMPT_PLACEHOLDER})
    return msgs
