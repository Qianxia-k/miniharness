"""Context Compiler — assembles the final message list for each API call.

This is the single place where every piece of context comes together:

    1. System prompt (static part)
    2. Core Memory (``core.md``)
    3. Conversation history (possibly compacted)
    4. Tool definitions

Before assembly the compiler checks the token budget and runs compaction
when needed.  The result is a :class:`ContextPacket` that is ready to be
passed directly to ``LLMClient.stream()``.

Mirrors OpenHarness's context assembly in ``QueryEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miniharness.context.budget import ContextBudget
from miniharness.context.compactor import compact_messages
from miniharness.messages import Conversation, Message


@dataclass
class ContextPacket:
    """Fully assembled context for one LLM API call."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)


class ContextCompiler:
    """Orchestrate budget, compaction, and message assembly.

    Usage::

        compiler = ContextCompiler(budget, core_memory, llm_stream, keep_last_n=3)
        packet = await compiler.compile(conversation, tools)
        async for event in llm.stream(messages=packet.messages, tools=packet.tools):
            ...

    """

    def __init__(
        self,
        *,
        budget: ContextBudget,
        core_memory,
        llm_stream,
        keep_last_n_turns: int = 3,
        working_set=None,
    ) -> None:
        self.budget = budget
        self.core_memory = core_memory
        self.llm_stream = llm_stream
        self.keep_last_n_turns = keep_last_n_turns
        self.working_set = working_set

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compile(
        self,
        conversation: Conversation,
        tools: list[dict[str, Any]],
    ) -> ContextPacket:
        """Assemble the final context for an API call.

        May run compaction if the conversation is over the token budget.
        """
        stats: dict[str, Any] = {
            "token_count": 0,
            "budget_ratio": 0.0,
            "compacted": False,
            "stage1_truncated": False,
            "stage2_summarised": False,
            "dropped": 0,
            "compacted_summary": None,
        }

        # 1. Export conversation to OpenAI format.
        msgs = conversation.to_openai()

        # 2. Count tokens & record baseline stats.
        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)

        # 3. Compact if over budget.
        if self.budget.is_over_budget(msgs):
            msgs, compaction_stats = await compact_messages(
                msgs,
                budget=self.budget,
                llm_stream=self.llm_stream,
                keep_last_n_turns=self.keep_last_n_turns,
            )
            stats["compacted"] = True
            stats["stage1_truncated"] = compaction_stats.get("stage1_truncated", False)
            stats["stage2_summarised"] = compaction_stats.get("stage2_summarised", False)
            stats["dropped"] = compaction_stats["original_count"] - compaction_stats["final_count"]
            stats["compacted_summary"] = compaction_stats.get("Compacted Summary")

        # 4. Ensure the system prompt includes core memory + working set
        #    (the first message is always the system prompt — refresh it so
        #    changes are picked up even after restore_messages).
        msgs = self._inject_system_context(msgs)

        # 5. Update final token count.
        stats["token_count"] = self.budget.tokens_used(msgs)
        stats["budget_ratio"] = self.budget.usage_ratio(msgs)

        return ContextPacket(messages=msgs, tools=tools, stats=stats)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _inject_system_context(
        self, msgs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Refresh core memory and working set in the system prompt."""
        if not msgs or msgs[0].get("role") != "system":
            return msgs

        content = msgs[0]["content"]

        # Core memory — inject if not already present.
        core_text = self.core_memory.render_for_system_prompt()
        if core_text and "Core Memory" not in content:
            content = f"{content}\n\n{core_text}"

        # Working set — inject if there are active files.
        if self.working_set is not None:
            ws_text = self.working_set.render_for_context()
            if ws_text:
                # Remove any stale working-set section and append fresh.
                if "[Working Set" in content:
                    content = content.split("[Working Set")[0].rstrip()
                content = f"{content}\n\n{ws_text}"

        msgs[0]["content"] = content
        return msgs
