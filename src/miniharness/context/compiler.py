"""Context Compiler — assembles the final message list for each API call.

Responsibilities (and what it does NOT do):

    ✅ Token budget check + 4-tier compaction orchestration
    ✅ Compact attachment injection post-compaction
    ✅ Post-compaction system-prompt integrity check (safety net)

    ❌ System prompt TEXT assembly — that's ``prompts/system.py``
    ❌ Core Memory I/O — that's ``memory/core.py``

Design principle: the compiler manages the **message list** (what messages
go into the API call and in what order).  It does NOT manage the **content**
of individual messages — that's the caller's responsibility (``AgentLoop``
delegates to ``prompts/system.py`` for the system prompt text).

Mirrors OpenHarness's separation: ``QueryEngine`` owns messages + compaction,
``prompts/context.py`` owns system prompt text assembly.
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
    """Orchestrate budget, compaction, and message-list assembly.

    The compiler receives a fully-formed conversation (including a complete
    system prompt assembled by ``prompts/system.py``), checks the token
    budget, and runs the 4-tier compaction pipeline when needed.

    Usage::

        compiler = ContextCompiler(
            budget=budget,
            llm_stream=loop.llm.stream,
            keep_last_n_turns=3,
        )
        packet = await compiler.compile(conversation, tools, metadata=tool_metadata)

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
        metadata: dict[str, Any] | None = None,
    ) -> ContextPacket:
        """Assemble the final context for an API call.

        Parameters
        ----------
        conversation:
            The current conversation history.  The first message MUST be
            a complete system prompt (assembled by the caller).
        tools:
            OpenAI-format tool schemas.
        metadata:
            ``tool_metadata`` dict for compact attachments during Tier-4
            compaction.  Pass ``None`` for legacy / stateless calls.

        Returns
        -------
        ContextPacket
            Ready-to-use messages + tools + compilation stats.
        """
        stats: dict[str, Any] = {
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

        # 1. Export conversation to OpenAI format.
        msgs = conversation.to_openai()

        # 2. Post-compaction safety net: ensure the system prompt has not
        #    been lost (e.g. if messages were restored from disk without one).
        msgs = _ensure_system_prompt(msgs)

        # 3. Count tokens & record baseline stats.
        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)

        # 4. Run 4-tier compaction if over budget.
        if self.budget.is_over_budget(msgs):
            msgs, compaction_stats = await self._run_compaction(msgs, metadata=metadata)
            stats.update(compaction_stats)

        # 5. Update final token count.
        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)

        return ContextPacket(messages=msgs, tools=tools, stats=stats)

    async def compact_if_needed(
        self,
        messages: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run compaction on raw message list (used for PTL reactive compaction).

        Unlike :meth:`compile`, this works on raw OpenAI-format messages
        (not a ``Conversation`` object) and can be forced to run even if
        the budget check passes.
        """
        stats: dict[str, Any] = {
            "token_count": self.budget.tokens_used(messages),
            "budget_ratio": self.budget.usage_ratio(messages),
            "compacted": False,
            "tier1_microcompact": False,
            "tier2_context_collapse": False,
            "tier3_session_memory": False,
            "tier4_full_llm_compact": False,
            "dropped": 0,
            "compacted_summary_chars": 0,
            "attachments_built": 0,
        }

        should_compact = force or self.budget.is_over_budget(messages)
        if not should_compact:
            return messages, stats

        # 👇 全部交给公共压缩函数，无冗余
        msgs, compaction_stats = await self._run_compaction(messages, metadata=metadata)
        stats.update(compaction_stats)

        # 最终 token 统计
        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)

        return msgs, stats

    # ------------------------------------------------------------------
    # 🔥 公共压缩函数（唯一一份逻辑，彻底去冗余）
    # ------------------------------------------------------------------
    async def _run_compaction(
        self,
        msgs: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Shared compaction logic used by both compile() and compact_if_needed()."""
        msgs, compaction_stats = await auto_compact_if_needed(
            msgs,
            budget=self.budget,
            metadata=metadata,
            llm_stream=self.llm_stream,
            keep_last_n_turns=self.keep_last_n_turns,
        )

        # 统一构建返回 stats
        stats = {
            "compacted": (
                compaction_stats.get("tier1_microcompact", False)
                or compaction_stats.get("tier2_context_collapse", False)
                or compaction_stats.get("tier3_session_memory", False)
                or compaction_stats.get("tier4_full_llm_compact", False)
            ),
            "tier1_microcompact": compaction_stats.get("tier1_microcompact", False),
            "tier2_context_collapse": compaction_stats.get("tier2_context_collapse", False),
            "tier3_session_memory": compaction_stats.get("tier3_session_memory", False),
            "tier4_full_llm_compact": compaction_stats.get("tier4_full_llm_compact", False),
            "dropped": (
                compaction_stats["original_count"] - compaction_stats.get("final_count", len(msgs))
            ),
            "compacted_summary_chars": compaction_stats.get("full_compact_summary_chars", 0),
            "attachments_built": compaction_stats.get("attachments_built", 0),
        }
        return msgs, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PLACEHOLDER = (
    "You are MiniHarness, a coding agent. "
    "(System prompt was lost — please restore from session.)"
)


def _ensure_system_prompt(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Safety net: ensure the message list has a system prompt."""
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role": "system", "content": _SYSTEM_PROMPT_PLACEHOLDER})
    return msgs
