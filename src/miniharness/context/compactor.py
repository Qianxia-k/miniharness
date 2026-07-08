"""Conversation compaction — 4-tier progressive compression.

Mirrors OpenHarness's ``services/compact/`` pipeline.  Each tier is
progressively more expensive; the pipeline stops early if a cheaper tier
reduces tokens enough.

Tier 1 – *Microcompact*
    Clears old tool-result content.  No LLM call.  Deterministic.
    Only impacts tools whose output is stale after a few turns
    (read_file, bash, grep, ls, web_search, web_fetch, write_file, edit_file).

Tier 2 – *Context Collapse*
    Truncates the middle of oversized text blocks, preserving head + tail.
    No LLM call.  Deterministic.

Tier 3 – *Session Memory*
    Produces a one-line-per-message deterministic summary.  No LLM call.
    Keeps the most recent N messages verbatim.

Tier 4 – *Full LLM Compact*
    Calls the model with a structured 9-section summarisation prompt.
    Before the LLM call we build *compact attachments* from
    ``tool_metadata`` — structured state (task focus, recent files,
    verified work, work log) that survives the compaction boundary
    as explicit user messages injected after the summary.

Usage::

    msgs, stats = await auto_compact_if_needed(
        messages=conversation.to_openai(),
        budget=context_budget,
        metadata=tool_metadata,
        llm_stream=agent_loop.llm.stream,
        keep_last_n_turns=3,
    )
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from miniharness.services.token_estimation import estimate_tokens

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tools whose results become stale quickly — safe to clear.
_COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "bash",
    "grep",
    "ls",
    "web_search",
    "web_fetch",
    "write_file",
    "edit_file",
})

# How many recent tool results to keep intact during microcompact.
_MICROCOMPACT_KEEP_RECENT = 5

# Per-text-block character limit during context collapse.
_CONTEXT_COLLAPSE_CHAR_LIMIT = 2400
_CONTEXT_COLLAPSE_HEAD_CHARS = 900
_CONTEXT_COLLAPSE_TAIL_CHARS = 500

# Session memory compact limits.
_SESSION_MEMORY_KEEP_RECENT = 12
_SESSION_MEMORY_MAX_LINES = 48
_SESSION_MEMORY_MAX_CHARS = 4000

# Full compact output budget.
_COMPACT_MAX_OUTPUT_TOKENS = 4096


# ---------------------------------------------------------------------------
# Tier 1 — Microcompact
# ---------------------------------------------------------------------------

def microcompact_messages(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = _MICROCOMPACT_KEEP_RECENT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clear old tool-result content for compactable tools.

    Finds all ``tool`` messages whose ``tool_call_id`` maps to a
    compactable tool use, keeps the *keep_recent* most recent ones
    intact, and replaces the content of the rest with a placeholder.

    Returns ``(messages, stats)``.  *messages* may be the same list
    object (mutated in place) or a new list.
    """
    # ---- build a mapping: tool_call_id -> tool_name -----------------------
    tool_use_names: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            tool_use_names[tc.get("id", "")] = fn.get("name", "")

    # ---- find all compactable tool-result indices -------------------------
    compactable_indices: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        tc_id = msg.get("tool_call_id", "")
        tool_name = tool_use_names.get(tc_id, "")
        if tool_name in _COMPACTABLE_TOOLS:
            compactable_indices.append(i)

    if not compactable_indices:
        return messages, {"microcompact_cleared": 0, "microcompact_tokens_saved": 0}

    # Keep the most recent *keep_recent* intact.
    protected = set(compactable_indices[-keep_recent:])
    to_clear = [i for i in compactable_indices if i not in protected]

    if not to_clear:
        return messages, {"microcompact_cleared": 0, "microcompact_tokens_saved": 0}

    # Estimate tokens saved (for stats).
    tokens_before = sum(
        estimate_tokens(messages[i].get("content", "") or "")
        for i in to_clear
    )

    # Mutate in place.
    for i in to_clear:
        content = messages[i].get("content", "") or ""
        if isinstance(content, str) and len(content) > 0:
            messages[i] = {
                **messages[i],
                "content": "[Old tool result content cleared]",
            }

    tokens_after = estimate_tokens("[Old tool result content cleared]") * len(to_clear)

    return messages, {
        "microcompact_cleared": len(to_clear),
        "microcompact_tokens_saved": max(0, tokens_before - tokens_after),
    }


# ---------------------------------------------------------------------------
# Tier 2 — Context Collapse
# ---------------------------------------------------------------------------

def context_collapse_messages(
    messages: list[dict[str, Any]],
    *,
    char_limit: int = _CONTEXT_COLLAPSE_CHAR_LIMIT,
    head_chars: int = _CONTEXT_COLLAPSE_HEAD_CHARS,
    tail_chars: int = _CONTEXT_COLLAPSE_TAIL_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Truncate the middle of oversized text blocks.

    For any message with a ``content`` string longer than *char_limit*,
    keep the first *head_chars* and last *tail_chars*, replacing the
    middle with a ``...[collapsed N chars]...`` marker.

    Returns ``(messages, stats)``.
    """
    collapsed = 0
    tokens_saved = 0

    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if len(content) <= char_limit:
            continue

        collapsed_chars = len(content) - head_chars - tail_chars
        if collapsed_chars <= 0:
            continue

        truncated = (
            content[:head_chars]
            + f"\n\n...[collapsed {collapsed_chars} chars]...\n\n"
            + content[-tail_chars:]
        )

        messages[i] = {**msg, "content": truncated}
        collapsed += 1
        tokens_saved += estimate_tokens("x" * collapsed_chars)

    return messages, {
        "context_collapse_blocks": collapsed,
        "context_collapse_tokens_saved": tokens_saved,
    }


# ---------------------------------------------------------------------------
# Tier 3 — Session Memory (deterministic summary)
# ---------------------------------------------------------------------------

def session_memory_compact_messages(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = _SESSION_MEMORY_KEEP_RECENT,
    max_lines: int = _SESSION_MEMORY_MAX_LINES,
    max_chars: int = _SESSION_MEMORY_MAX_CHARS,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce a one-line-per-message summary of older messages.

    Preserves the system prompt + the most recent *keep_recent*
    messages verbatim, and replaces everything in between with a
    compact summary.
    """
    if len(messages) <= keep_recent + 1:
        return messages, {
            "session_memory_summarised": False,
            "session_memory_lines": 0,
        }

    # System prompt always stays.
    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    start = 1 if sys_msg else 0

    old = messages[start:-keep_recent] if keep_recent > 0 else messages[start:]
    recent = messages[-keep_recent:] if keep_recent > 0 else []

    if not old:
        return messages, {"session_memory_summarised": False, "session_memory_lines": 0}

    # Build one-line summaries.
    summary_lines: list[str] = []
    for msg in old:
        line = _summarize_one_message(msg)
        if line:
            summary_lines.append(line)

    if len(summary_lines) > max_lines:
        summary_lines = summary_lines[-max_lines:]

    summary_text = "\n".join(summary_lines)
    if len(summary_text) > max_chars:
        summary_text = summary_text[:max_chars] + "\n...[truncated]"

    # Estimate whether this actually saves tokens.
    old_tokens = sum(_rough_token_count(m.get("content", "") or "") for m in old)
    new_tokens = _rough_token_count(summary_text)
    if new_tokens >= old_tokens and not force:
        return messages, {
            "session_memory_summarised": False,
            "session_memory_lines": 0,
            "session_memory_skipped": "no savings",
        }

    # Assemble: system prompt + summary + recent.
    result: list[dict[str, Any]] = []
    if sys_msg:
        result.append(sys_msg)
    result.append({
        "role": "system",
        "content": f"[Session memory — earlier messages condensed]\n\n{summary_text}",
    })
    result.extend(recent)

    return result, {
        "session_memory_summarised": True,
        "session_memory_lines": len(summary_lines),
        "session_memory_tokens_saved": max(0, old_tokens - new_tokens),
    }


def _summarize_one_message(msg: dict[str, Any]) -> str | None:
    """Reduce a single message to a one-line summary."""
    role = msg.get("role", "")
    if role == "system":
        return None  # system prompt handled separately

    content = msg.get("content", "") or ""

    if role == "user":
        text = _extract_text(content)
        preview = text[:160].replace("\n", " ")
        return f"user: {preview}"

    if role == "assistant":
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            return f"assistant: tool calls → {', '.join(names)}"
        text = _extract_text(content)
        preview = text[:160].replace("\n", " ")
        return f"assistant: {preview}" if preview else None

    if role == "tool":
        tc_id = msg.get("tool_call_id", "")[:8]
        preview = (content[:120] if isinstance(content, str) else str(content)[:120]).replace("\n", " ")
        return f"tool({tc_id}): {preview}"

    return None


def _extract_text(content: Any) -> str:
    """Extract text from content that might be a string or a multimodal list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return " ".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Tier 4 — Full LLM Compact
# ---------------------------------------------------------------------------

# Structured summarisation prompt (mirrors OpenHarness's BASE_COMPACT_PROMPT).
_COMPACT_SYSTEM_PROMPT = """\
You are a conversation summarizer. Your task is to summarize the conversation
so far between an AI agent and a user, paying close attention to the user's
explicit requests and previous actions.
This summary should thoroughly capture technical details, code patterns, and
architectural decisions that are essential for continuing development work
without losing context.

Step 1: Your summary MUST follow the format below and include the written
prompts for each section:

<analysis>
[organize your thoughts and ensure you've covered all necessary points and put
them in this tag. no more than 300 words.]
</analysis>

<summary>
1. Primary Request and Intent: [detailed description of what the user asked for]
2. Key Technical Concepts: [list of important technical concepts, technologies,
   and frameworks mentioned]
3. Files and Code Sections: [enumerate specific files examined, modified, or
   created, with relevant line numbers and code patterns]
4. Errors and fixes: [list of errors that you ran into, and how you fixed them.
   If none, write "None."]
5. Problem Solving: [problems you solved or are currently working on solving]
6. All user messages: [list all messages actually sent by the user, preserving
   their original wording as much as possible]
7. Pending Tasks: [list tasks you have not yet completed from the user's requests]
8. Current Work: [describe in detail what you were doing immediately before
   this summary, including file paths and line numbers]
9. Optional Next Step: [only if you have a clear, concrete next step to take.
   If not, write "None."]
</summary>

Do NOT call any tools or ask any questions. Your ONLY task is to summarize
the conversation. Output the <analysis> and <summary> tags and nothing else."""


async def _call_model_for_summary(
    messages_to_summarize: list[dict[str, Any]],
    llm_stream,
) -> str:
    """Call the LLM to produce a structured summary of old messages.

    Uses a dedicated system prompt that instructs the model NOT to use
    tools and to follow the 9-section format.
    """
    summary_messages = [
        {"role": "system", "content": _COMPACT_SYSTEM_PROMPT},
        *messages_to_summarize,
        {"role": "user", "content": "Please summarize the conversation above following the format exactly."},
    ]

    summary_text = ""
    try:
        async for event in llm_stream(
            messages=summary_messages,
            tools=[],  # no tools — the model must only summarize
        ):
            from miniharness.llm import StreamComplete, TextDelta

            if isinstance(event, TextDelta):
                summary_text += event.text
            elif isinstance(event, StreamComplete):
                summary_text = event.message.content or summary_text
    except Exception:
        # If the summarisation call fails, return empty — caller handles.
        pass

    return summary_text.strip()


async def full_llm_compact(
    messages: list[dict[str, Any]],
    *,
    attachments: list[dict[str, Any]] | None = None,
    llm_stream=None,
    keep_last_n_turns: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full LLM compaction: structured summary + compact attachments.

    *attachments* are pre-built by ``carryover.build_compact_attachments()``.
    The compactor doesn't know about tool_metadata internals — it just
    injects whatever attachments the caller provides.

    Steps:
    1. Run microcompact on the old segment as preparation.
    2. Find the cut point (keep last N user turns + everything after).
    3. Call the LLM to summarize the old segment.
    4. Assemble: boundary marker → summary → recent messages → attachments.
    """
    stats: dict[str, Any] = {
        "full_compact_ran": False,
        "full_compact_summary_chars": 0,
        "attachments_built": 0,
    }

    # Find cut point: keep the last N user messages and everything after.
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_last_n_turns:
        return messages, stats

    cut_idx = user_indices[-keep_last_n_turns]

    # System prompt always stays.
    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    start = 1 if sys_msg else 0

    old = messages[start:cut_idx]
    recent = messages[cut_idx:]

    if not old:
        return messages, stats

    # Pre-compact old messages with microcompact (cheap, reduces summarisation cost).
    old, _ = microcompact_messages(list(old))

    # ---- call model for structured summary -------------------------------
    summary_text = await _call_model_for_summary(old, llm_stream)
    stats["full_compact_ran"] = True
    stats["full_compact_summary_chars"] = len(summary_text)

    if not summary_text:
        # Summarisation failed — fall back to session memory as a safety net.
        result, sm_stats = session_memory_compact_messages(messages, keep_recent=keep_last_n_turns * 2)
        stats["full_compact_fallback"] = "session_memory"
        stats.update(sm_stats)
        return result, stats

    # ---- compact attachments (pre-built by caller from tool_metadata) ----
    attachment_msgs = attachments or []
    stats["attachments_built"] = len(attachment_msgs)

    # ---- assemble post-compact message list ------------------------------
    result: list[dict[str, Any]] = []

    # System prompt.
    if sys_msg:
        result.append(sys_msg)

    # Boundary marker — tells the model compaction happened.
    result.append({
        "role": "user",
        "content": (
            "[CONVERSATION COMPACTED]\n"
            f"The conversation history before this point has been summarized "
            f"to save context space. Below is a structured summary of what "
            f"happened, followed by compact attachments with key state "
            f"that survived compaction.\n"
        ),
    })

    # Structured summary.
    result.append({
        "role": "user",
        "content": (
            "[Compacted conversation summary]\n\n" + summary_text
        ),
    })

    # Compact attachments (task focus, recent files, verified work, work log).
    result.extend(attachment_msgs)

    # Recent messages (preserved verbatim).
    result.extend(recent)

    stats["final_message_count"] = len(result)
    return result, stats


# ---------------------------------------------------------------------------
# Public API — 4-tier pipeline
# ---------------------------------------------------------------------------


async def auto_compact_if_needed(
    messages: list[dict[str, Any]],
    *,
    budget,  # ContextBudget
    tools: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    llm_stream=None,
    keep_last_n_turns: int = 3,
    progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run progressive compaction, stopping when the budget is satisfied.

    Tiers are attempted in order from cheapest to most expensive:
    1. Microcompact (free)
    2. Context Collapse (free)
    3. Session Memory (free)
    4. Full LLM Compact (calls the model)

    *attachments* are pre-built by ``carryover.build_compact_attachments()``
    and injected during Tier 4.  Pass ``None`` if there is no carryover state.

    Returns ``(compacted_messages, stats)``.
    """

    tools = tools or []

    async def emit(phase: str, **payload: Any) -> None:
        if progress_callback is None:
            return
        await progress_callback({
            "phase": phase,
            "token_count": budget.tokens_used(messages, tools=tools),
            "soft_limit": budget.max_tokens,
            "usage_ratio": budget.usage_ratio(messages, tools=tools),
            **payload,
        })

    stats: dict[str, Any] = {
        "original_count": len(messages),
        "tokens_before": budget.tokens_used(messages, tools=tools),
        "compacted": False,
        "tier1_microcompact": False,
        "tier2_context_collapse": False,
        "tier3_session_memory": False,
        "tier4_full_llm_compact": False,
        "final_count": len(messages),
        "tokens_after": budget.tokens_used(messages, tools=tools),
    }

    await emit("start")

    # ---- Tier 1: Microcompact ---------------------------------------------
    if force or budget.is_over_budget(messages, tools=tools):
        await emit("tier_start", tier="microcompact")
        messages, t1_stats = microcompact_messages(messages)
        if t1_stats.get("microcompact_cleared", 0) > 0:
            stats["tier1_microcompact"] = True
            stats["compacted"] = True
        stats.update(t1_stats)
        await emit("tier_end", tier="microcompact", **t1_stats)

    # ---- Tier 2: Context Collapse -----------------------------------------
    if force or budget.is_over_budget(messages, tools=tools):
        await emit("tier_start", tier="context_collapse")
        messages, t2_stats = context_collapse_messages(messages)
        if t2_stats.get("context_collapse_blocks", 0) > 0:
            stats["tier2_context_collapse"] = True
            stats["compacted"] = True
        stats.update(t2_stats)
        await emit("tier_end", tier="context_collapse", **t2_stats)

    # ---- Tier 3: Session Memory -------------------------------------------
    if force or budget.is_over_budget(messages, tools=tools):
        await emit("tier_start", tier="session_memory")
        messages, t3_stats = session_memory_compact_messages(
            messages,
            keep_recent=keep_last_n_turns * 2,
            force=force,
        )
        if t3_stats.get("session_memory_summarised"):
            stats["tier3_session_memory"] = True
            stats["compacted"] = True
        stats.update(t3_stats)
        await emit("tier_end", tier="session_memory", **t3_stats)

    # ---- Tier 4: Full LLM Compact -----------------------------------------
    if budget.is_over_budget(messages, tools=tools):
        await emit("tier_start", tier="full_llm_compact")
        messages, t4_stats = await full_llm_compact(
            messages,
            attachments=attachments,
            llm_stream=llm_stream,
            keep_last_n_turns=keep_last_n_turns,
        )
        if t4_stats.get("full_compact_ran"):
            stats["tier4_full_llm_compact"] = True
            stats["compacted"] = True
        stats.update(t4_stats)
        await emit("tier_end", tier="full_llm_compact", **t4_stats)

    stats["final_count"] = len(messages)
    stats["tokens_after"] = budget.tokens_used(messages, tools=tools)
    await emit("end", compacted=stats["compacted"], tokens_after=stats["tokens_after"])

    return messages, stats


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rough_token_count(text: str) -> int:
    """Backward-compatible wrapper around the shared estimator."""
    return estimate_tokens(text)
