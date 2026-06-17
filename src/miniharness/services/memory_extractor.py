"""Automatic memory extraction — extract facts and task episodes from conversations.

Unlike the manual ``memory_add`` / ``memory_log`` tools (which the model
may forget to call), this service runs automatically after every meaningful
turn.  It uses small, cheap LLM prompts to extract structured data from the
conversation and persists it into the Semantic and Episodic stores.

Architecture::

    AgentLoop.run(prompt) completes
        │
        └─ extract_memories_from_turn(conversation, llm, cwd)
             │
             ├─ _extract_facts() → SemanticStore.add() × N
             │     Prompt includes active memory manifest + lifecycle schema.
             │     Returns: [{"fact": "...", "tags": [...], "supersedes": [...]}]
             │
             └─ _extract_episode() → EpisodicStore.log()
                   Prompt: "What task was just completed?"
                   Returns: {"task": "...", "summary": "...", "outcome": "..."}

Mirrors OpenHarness's ``extract_memories`` pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Fact extraction prompt
# ═══════════════════════════════════════════════════════════════════════════

_FACT_EXTRACTION_PROMPT_TEMPLATE = """\
Analyze the conversation above between a user and a coding agent.  Extract
any FACTS that are worth remembering for future sessions.  Focus on:

- User preferences (coding style, tools, naming conventions)
- Project architecture decisions
- Technologies and frameworks in use
- Known issues or constraints
- Completed milestones

Existing active memory manifest:
__MEMORY_MANIFEST__

Memory lifecycle rules:
- Prefer reusing/superseding existing memory IDs instead of creating duplicates.
- If a new fact replaces, contradicts, or makes an existing fact stale, include
  that old memory ID in "supersedes".
- If a fact is effectively the same as an existing active memory, do not return it.
- Do not store secrets, credentials, raw tokens, or private keys.
- Keep each fact stable and future-useful; avoid transient chat narration.

Return ONLY a JSON object with a "facts" array:

{
    "facts": [
        {
            "fact": "The user prefers pathlib over os.path",
            "tags": ["preference", "python"],
            "confidence": 0.9,
            "supersedes": []
        },
        {
            "fact": "Auth module uses JWT with RS256",
            "tags": ["architecture", "auth"],
            "confidence": 0.8,
            "supersedes": ["old_memory_id"]
        }
    ]
}

If nothing is worth remembering, return {"facts": []}.

Do NOT include any text outside the JSON object."""


# ═══════════════════════════════════════════════════════════════════════════
# Episode extraction prompt
# ═══════════════════════════════════════════════════════════════════════════

_EPISODE_EXTRACTION_PROMPT = """\
Analyze the conversation above.  What task did the agent just complete (or
make significant progress on)?  Summarize it as a task episode.

Return ONLY a JSON object:

{
    "task": "Short task title (max 80 chars)",
    "summary": "What was done, in 1-2 sentences",
    "files_touched": ["list", "of", "file", "paths"],
    "outcome": "success" or "partial" or "failed"
}

If no meaningful task was completed (e.g., just a brief chat), return:
{"task": ""}

Do NOT include any text outside the JSON object."""


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ExtractionResult:
    """Result of automatic memory extraction after a turn."""
    facts: list[dict[str, Any]] = field(default_factory=list)
    episode: dict[str, Any] | None = None
    skipped: bool = False


async def extract_memories_from_turn(
    *,
    messages: list[dict[str, Any]],
    llm,
    cwd: str | Path,
    min_messages: int = 4,
) -> ExtractionResult:
    """Extract and persist memories from a completed agent turn.

    Called automatically after each ``loop.run(prompt)`` returns.
    Skips extraction if the conversation is too short (less than
    *min_messages* total messages).

    Parameters
    ----------
    messages:
        OpenAI-format conversation messages.
    llm:
        The streaming LLM function (``AgentLoop._stream_fn``).
    cwd:
        Project root for per-project memory isolation.
    min_messages:
        Skip extraction if conversation has fewer than this many messages
        (avoids extracting from trivial "hello" turns).

    Returns
    -------
    ExtractionResult
        What was extracted (for display to the user).
    """
    if len(messages) < min_messages:
        return ExtractionResult(skipped=True)

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if len(user_msgs) < 1:
        return ExtractionResult(skipped=True)

    # Run both extractors concurrently.
    import asyncio
    facts_result, episode_result = await asyncio.gather(
        _extract_facts(messages, llm, cwd),
        _extract_episode(messages, llm, cwd),
    )

    return ExtractionResult(facts=facts_result, episode=episode_result)


# ═══════════════════════════════════════════════════════════════════════════
# Fact extraction
# ═══════════════════════════════════════════════════════════════════════════


async def _extract_facts(
    messages: list[dict[str, Any]],
    llm,
    cwd: str | Path,
) -> list[dict[str, Any]]:
    """Extract facts from conversation and persist to SemanticStore.

    Returns the list of extracted facts (for display).
    """
    from miniharness.memory.semantic import SemanticStore

    store = SemanticStore(str(Path(cwd).resolve()))
    prompt = _FACT_EXTRACTION_PROMPT_TEMPLATE.replace(
        "__MEMORY_MANIFEST__",
        store.manifest(limit=60),
    )
    extract_msgs = [*messages, {"role": "user", "content": prompt}]
    response = await _call_llm_for_json(llm, extract_msgs)
    if response is None:
        return []

    facts = response.get("facts", [])
    if not isinstance(facts, list) or not facts:
        return []

    persisted: list[dict[str, Any]] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        fact = (item.get("fact") or "").strip()
        if not fact:
            continue
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        supersedes = item.get("supersedes", [])
        contradicts = item.get("contradicts", [])
        entry_id = store.add(
            fact,
            tags=[t for t in tags if isinstance(t, str)],
            source="auto_extract",
            confidence=item.get("confidence") if isinstance(item.get("confidence"), (int, float)) else None,
            supersedes=supersedes if isinstance(supersedes, list) else [],
            contradicts=contradicts if isinstance(contradicts, list) else [],
        )
        persisted.append({
            "id": entry_id,
            "fact": fact,
            "tags": [t for t in tags if isinstance(t, str)],
            "supersedes": supersedes if isinstance(supersedes, list) else [],
        })
    return persisted


# ═══════════════════════════════════════════════════════════════════════════
# Episode extraction
# ═══════════════════════════════════════════════════════════════════════════


async def _extract_episode(
    messages: list[dict[str, Any]],
    llm,
    cwd: str | Path,
) -> dict[str, Any] | None:
    """Extract task episode from conversation and persist to EpisodicStore.

    Returns the extracted episode dict (for display), or None.
    """
    extract_msgs = [*messages, {"role": "user", "content": _EPISODE_EXTRACTION_PROMPT}]
    response = await _call_llm_for_json(llm, extract_msgs)
    if response is None:
        return None

    task = (response.get("task") or "").strip()
    if not task:
        return None

    from miniharness.memory.episodic import EpisodicStore
    store = EpisodicStore(str(Path(cwd).resolve()))
    files_touched = response.get("files_touched", [])
    if not isinstance(files_touched, list):
        files_touched = []
    store.log(
        task=task,
        summary=(response.get("summary") or "").strip(),
        files_touched=[f for f in files_touched if isinstance(f, str)],
        outcome=(response.get("outcome") or "").strip(),
        source="auto_extract",
    )
    return response


# ═══════════════════════════════════════════════════════════════════════════
# LLM helper
# ═══════════════════════════════════════════════════════════════════════════


async def _call_llm_for_json(
    llm,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
) -> dict[str, Any] | None:
    """Call the LLM with a small prompt, expecting a JSON response.

    Uses ``tools=[]`` to prevent tool calls during extraction.
    Returns the parsed JSON dict, or ``None`` on failure.
    """
    response_text = ""
    try:
        async for event in llm(
            messages=messages,
            tools=[],
            max_tokens_override=max_tokens,
        ):
            from miniharness.llm import StreamComplete, TextDelta
            if isinstance(event, TextDelta):
                response_text += event.text
            elif isinstance(event, StreamComplete):
                response_text = event.message.content or response_text
    except Exception:
        return None

    # Parse JSON — try strict first, then extract from code fences.
    return _parse_json_response(response_text)


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse a JSON response from the model, handling code fences."""
    text = text.strip()
    if not text:
        return None

    # Try direct parse.
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` fence.
    import re
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
