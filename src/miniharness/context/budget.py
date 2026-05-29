"""Token counting and context budget management.

Mirrors OpenHarness's context budget tracking: before every API call the
harness checks how close the conversation is to the model's context limit
and triggers compaction when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Token estimation (no tiktoken dependency — ~4 chars / token is the
# industry-standard heuristic for mixed English + code text).
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4.0


def count_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the token count of an OpenAI-format message list.

    Counts content strings, tool call JSON, and adds a small per-message
    overhead for role / framing tokens (~4 tokens / message).
    """
    total = 0
    for msg in messages:
        total += 4  # per-message framing overhead
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += max(1, len(content) / _CHARS_PER_TOKEN)
        elif isinstance(content, list):
            # Multi-modal content blocks.
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += max(1, len(block["text"]) / _CHARS_PER_TOKEN)

        # Tool calls in assistant messages.
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += max(1, len(fn.get("name", "")) / _CHARS_PER_TOKEN)
            total += max(1, len(fn.get("arguments", "")) / _CHARS_PER_TOKEN)

    return int(total)


# ---------------------------------------------------------------------------
# Model context windows
# ---------------------------------------------------------------------------

# Default context window sizes (tokens).  Conservative defaults that work
# for most OpenAI-compatible models.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Qwen
    "qwen3.5-397b-a17b": 131072,
    "qwen-max": 32768,
    "qwen-plus": 131072,
    "qwen-turbo": 131072,
    # OpenAI
    "gpt-4.1-mini": 131072,
    "gpt-4.1": 131072,
    "gpt-4o": 131072,
    "gpt-4o-mini": 131072,
    "gpt-4-turbo": 131072,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    # Anthropic (OpenAI-compatible gateways)
    "claude-opus-4-7": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-haiku-4-5-20251001": 200000,
}


def get_context_window(model: str) -> int:
    """Return the context window size for *model*.

    Falls back to 128K for unknown models.
    """
    return _MODEL_CONTEXT_WINDOWS.get(model, 131072)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class ContextBudget:
    """Tracks how much of the model's context window is in use.

    The budget is split into two parts:

    - **fixed overhead**: system prompt + tool definitions + response reserve.
      This is estimated once and doesn't change between turns.
    - **variable conversation**: the message history, which grows over time.

    When the conversation exceeds the available budget, the harness triggers
    compaction (see :mod:`miniharness.context.compactor`).
    """

    model: str
    total: int             # model context window (tokens)
    max_tokens: int        # total × ratio — the "soft cap"
    ratio: float = 0.8     # trigger compaction at 80% usage

    # Estimated fixed overhead per API call.
    system_prompt_tokens: int = 500
    tool_def_tokens: int = 1500
    response_reserve_tokens: int = 4000

    @classmethod
    def for_model(cls, model: str, *, ratio: float = 0.8) -> ContextBudget:
        """Create a budget for *model* using its known context window."""
        total = get_context_window(model)
        return cls(
            model=model,
            total=total,
            max_tokens=int(total * ratio),
            ratio=ratio,
        )

    # ------------------------------------------------------------------
    # Budget queries
    # ------------------------------------------------------------------

    @property
    def fixed_overhead(self) -> int:
        """Tokens consumed by non-conversation content each turn."""
        return self.system_prompt_tokens + self.tool_def_tokens + self.response_reserve_tokens

    @property
    def available(self) -> int:
        """Tokens available for conversation messages."""
        return max(0, self.max_tokens - self.fixed_overhead)

    def tokens_used(self, messages: list[dict[str, Any]]) -> int:
        """Total estimated tokens for the full API call."""
        return self.fixed_overhead + count_tokens(messages)

    def usage_ratio(self, messages: list[dict[str, Any]]) -> float:
        """Fraction of the soft cap currently consumed (0.0 – 1.0+)."""
        return self.tokens_used(messages) / self.max_tokens

    def is_over_budget(self, messages: list[dict[str, Any]]) -> bool:
        """Return True when the conversation should be compacted."""
        return self.tokens_used(messages) > self.available
