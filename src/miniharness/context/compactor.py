"""Conversation compaction — two-stage compression when near context limits.

Stage 1 (low-loss): truncate old tool results to a fixed character cap.
Stage 2 (lossy): ask the model to summarise old messages into one system message.

Mirrors OpenHarness's default session compaction: prune old tool results
first, then fall back to structured summarisation.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Stage 1 — tool-result truncation
# ---------------------------------------------------------------------------

_TOOL_TRUNCATE_CHARS = 500


def _truncate_tool_results(
    messages: list[dict[str, Any]],
    *,
    keep_recent_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Truncate the oldest tool-result messages to *_TOOL_TRUNCATE_CHARS chars.

    Only the first ``keep_recent_ratio`` fraction of tool messages are
    truncated; the most recent tool results are left intact so the model
    still has full context for what happened in the last few turns.
    """
    # Find indices of all tool messages.
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if not tool_indices:
        return messages

    # Only truncate the oldest portion.
    split = max(1, int(len(tool_indices) * keep_recent_ratio))
    to_truncate = set(tool_indices[:split])

    result: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i in to_truncate:
            content = msg.get("content", "") or ""
            if isinstance(content, str) and len(content) > _TOOL_TRUNCATE_CHARS:
                truncated = content[:_TOOL_TRUNCATE_CHARS]
                msg = {
                    **msg,
                    "content": (
                        f"{truncated}\n\n"
                        f"[tool output truncated from {len(content)} chars]"
                    ),
                }
        result.append(msg)

    return result


# ---------------------------------------------------------------------------
# Stage 2 — summarisation
# ---------------------------------------------------------------------------

_SUMMARISE_PROMPT = """Summarise the conversation above.  Keep:
- Every factual claim the user made about the project or their preferences.
- Every concrete decision or file path the agent mentioned.
- The general flow of what was asked and answered.

Be very concise.  Write in the same language the user used."""


async def _summarise_old_messages(
    messages: list[dict[str, Any]],
    *,
    keep_last_n_turns: int,
    llm_stream,
) -> list[dict[str, Any]]:
    """Compress old messages into a single system summary.

    We keep the system prompt + the most recent *keep_last_n_turns*
    user messages (and everything after them), then ask the model to
    summarise the rest.
    """
    # Find the cut point: keep the last N user messages and everything after.
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_last_n_turns:
        return messages  # not enough history to split

    # The cut preserves: messages[0] (system prompt) + messages[cut_idx:].
    cut_idx = user_indices[-keep_last_n_turns]

    old = messages[1:cut_idx]  # everything between system prompt and cut
    recent = messages[cut_idx:]  # the last N turns

    if not old:
        return messages

    # Build a summarisation request.
    summary_input = [
        *old,
        {"role": "user", "content": _SUMMARISE_PROMPT},
    ]

    # Call the model (non-streaming — we just need the text).
    summary_text = ""
    async for event in llm_stream(messages=summary_input, tools=[]):
        from miniharness.llm import StreamComplete, TextDelta

        if isinstance(event, TextDelta):
            summary_text += event.text
        elif isinstance(event, StreamComplete):
            summary_text = event.message.content or summary_text

    if not summary_text.strip():
        return messages  # summarisation failed, return original

    # Insert the summary as a system-level context message, then recent messages.
    summary_msg = {
        "role": "system",
        "content": f"[Previous conversation — auto-summarised]\n\n{summary_text.strip()}",
    }
    return [messages[0], summary_msg, *recent]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def compact_messages(
    messages: list[dict[str, Any]],
    *,
    budget,
    llm_stream,
    keep_last_n_turns: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run two-stage compaction and return (compacted_messages, stats).

    *budget* is a :class:`~miniharness.context.budget.ContextBudget`.
    *llm_stream* is ``loop.llm.stream`` — we call it only during stage 2.
    *keep_last_n_turns* — how many recent user turns to keep intact during stage 2.
    """
    stats: dict[str, Any] = {
        "original_count": len(messages),
        "stage1_truncated": False,
        "stage2_summarised": False,
        "final_count": len(messages),
        "tokens_before": budget.tokens_used(messages),
        "tokens_after": budget.tokens_used(messages),
        "Compacted Summary": None,
    }

    # ---- stage 1 -----------------------------------------------------------
    if budget.is_over_budget(messages):
        messages = _truncate_tool_results(messages)
        stats["stage1_truncated"] = True

    # ---- stage 2 -----------------------------------------------------------
    if budget.is_over_budget(messages):
        messages = await _summarise_old_messages(
            messages,
            keep_last_n_turns=keep_last_n_turns,
            llm_stream=llm_stream,
        )
        stats["Compacted Summary"] = messages[1]["content"] if len(messages) > 1 else None
        stats["stage2_summarised"] = True

    stats["final_count"] = len(messages)
    stats["tokens_after"] = budget.tokens_used(messages)

    return messages, stats
