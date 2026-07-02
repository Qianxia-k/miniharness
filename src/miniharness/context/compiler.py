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
from typing import Any, Awaitable, Callable

from miniharness.context.budget import ContextBudget
from miniharness.context.compactor import auto_compact_if_needed
from miniharness.messages import Conversation


@dataclass
class ContextPacket:
    """Fully assembled context for one LLM API call."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextCompileTrace:
    """Inspectable summary of what the compiler sent to the model."""

    message_count: int
    system_prompt_chars: int
    tool_count: int
    attachment_count: int
    attachment_types: list[str]
    message_tokens: int
    tool_tokens: int
    response_reserve_tokens: int
    total_used: int
    available: int
    context_window: int
    soft_limit: int
    tokenizer: str
    compacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_count": self.message_count,
            "system_prompt_chars": self.system_prompt_chars,
            "tool_count": self.tool_count,
            "attachment_count": self.attachment_count,
            "attachment_types": list(self.attachment_types),
            "message_tokens": self.message_tokens,
            "tool_tokens": self.tool_tokens,
            "response_reserve_tokens": self.response_reserve_tokens,
            "total_used": self.total_used,
            "available": self.available,
            "context_window": self.context_window,
            "soft_limit": self.soft_limit,
            "tokenizer": self.tokenizer,
            "compacted": self.compacted,
        }


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
        compact_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.budget = budget
        self.llm_stream = llm_stream
        self.keep_last_n_turns = keep_last_n_turns
        self.compact_progress = compact_progress

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
        stats.update(self.budget.snapshot(msgs, tools=tools))
        stats["token_count"] = stats["total_used"]
        stats["budget_ratio"] = stats["usage_ratio"]
        stats["attachments_built"] = len(attachments or [])

        if self.budget.is_over_budget(msgs, tools=tools):
            msgs, compaction_stats = await auto_compact_if_needed(
                msgs,
                budget=self.budget,
                tools=tools,
                attachments=attachments,
                llm_stream=self.llm_stream,
                keep_last_n_turns=self.keep_last_n_turns,
                progress_callback=self.compact_progress,
            )
            stats.update(compaction_stats)

        stats.update(self.budget.snapshot(msgs, tools=tools))
        stats["token_count"] = stats["total_used"]
        stats["budget_ratio"] = stats["usage_ratio"]
        stats["attachments_built"] = len(attachments or [])
        stats["context_trace"] = build_compile_trace(
            messages=msgs,
            tools=tools,
            attachments=attachments,
            snapshot=stats,
            compacted=bool(stats.get("compacted")),
        ).to_dict()
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
        stats.update(self.budget.snapshot(messages))
        stats["token_count"] = stats["total_used"]
        stats["budget_ratio"] = stats["usage_ratio"]
        stats["attachments_built"] = len(attachments or [])

        if not (force or self.budget.is_over_budget(messages)):
            stats["context_trace"] = build_compile_trace(
                messages=messages,
                tools=[],
                attachments=attachments,
                snapshot=stats,
                compacted=False,
            ).to_dict()
            return messages, stats

        msgs, compaction_stats = await auto_compact_if_needed(
            messages,
            budget=self.budget,
            attachments=attachments,
            llm_stream=self.llm_stream,
            keep_last_n_turns=self.keep_last_n_turns,
            progress_callback=self.compact_progress,
        )
        stats.update(compaction_stats)
        stats.update(self.budget.snapshot(msgs))
        stats["token_count"] = stats["total_used"]
        stats["budget_ratio"] = stats["usage_ratio"]
        stats["attachments_built"] = len(attachments or [])
        stats["context_trace"] = build_compile_trace(
            messages=msgs,
            tools=[],
            attachments=attachments,
            snapshot=stats,
            compacted=bool(stats.get("compacted")),
        ).to_dict()
        return msgs, stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "token_count": 0,
            "budget_ratio": 0.0,
            "message_tokens": 0,
            "tool_tokens": 0,
            "response_reserve_tokens": 0,
            "total_used": 0,
            "available": 0,
            "context_window": 0,
            "soft_limit": 0,
            "tokenizer": "",
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


def build_compile_trace(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    attachments: list[dict[str, Any]] | None,
    snapshot: dict[str, Any],
    compacted: bool,
) -> ContextCompileTrace:
    """Build an OpenHarness-style trace for context observability."""
    system_prompt_chars = 0
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        system_prompt_chars = len(content) if isinstance(content, str) else len(str(content))

    attachment_types: list[str] = []
    for attachment in attachments or []:
        if not isinstance(attachment, dict):
            continue
        raw_type = (
            attachment.get("type")
            or attachment.get("name")
            or attachment.get("title")
            or "attachment"
        )
        attachment_types.append(str(raw_type))

    return ContextCompileTrace(
        message_count=len(messages),
        system_prompt_chars=system_prompt_chars,
        tool_count=len(tools),
        attachment_count=len(attachments or []),
        attachment_types=attachment_types,
        message_tokens=int(snapshot.get("message_tokens") or 0),
        tool_tokens=int(snapshot.get("tool_tokens") or 0),
        response_reserve_tokens=int(snapshot.get("response_reserve_tokens") or 0),
        total_used=int(snapshot.get("total_used") or snapshot.get("token_count") or 0),
        available=int(snapshot.get("available") or 0),
        context_window=int(snapshot.get("context_window") or 0),
        soft_limit=int(snapshot.get("soft_limit") or 0),
        tokenizer=str(snapshot.get("tokenizer") or ""),
        compacted=compacted,
    )
