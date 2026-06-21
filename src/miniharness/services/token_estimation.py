"""Model-aware token estimation.

MiniHarness keeps token accounting in one place so CLI, TUI, compaction,
and hooks all reason about the same numbers.  When ``tiktoken`` is available
we use the model's tokenizer where possible; otherwise we fall back to a
conservative character heuristic.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

TOKEN_ESTIMATION_PADDING = 4 / 3
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_IMAGE_TOKEN_ESTIMATE = 3072


def estimate_tokens(text: Any, *, model: str | None = None, padded: bool = False) -> int:
    """Estimate token count for plain text or JSON-like values."""
    raw = _stringify(text)
    if not raw:
        return 0

    tokenizer = _encoding_for_model(model)
    if tokenizer is not None:
        count = len(tokenizer.encode(raw))
    else:
        count = max(1, (len(raw) + DEFAULT_CHARS_PER_TOKEN - 1) // DEFAULT_CHARS_PER_TOKEN)

    if padded:
        count = int(count * TOKEN_ESTIMATION_PADDING)
    return max(1, count)


def estimate_message_tokens(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    include_padding: bool = True,
    image_token_estimate: int = DEFAULT_IMAGE_TOKEN_ESTIMATE,
) -> int:
    """Estimate tokens for OpenAI-format chat messages.

    The shape mirrors OpenHarness's compact accounting: text, tool-use JSON,
    tool-result content, and image blocks are all included.  A small framing
    overhead is added per message because chat APIs serialize role/name/tool
    metadata around the visible content.
    """
    total = 0
    for msg in messages:
        total += 4  # chat message framing overhead
        total += estimate_tokens(msg.get("role", ""), model=model)
        if msg.get("name"):
            total += estimate_tokens(msg["name"], model=model) + 1

        total += _estimate_content_tokens(
            msg.get("content"),
            model=model,
            image_token_estimate=image_token_estimate,
        )

        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            total += estimate_tokens(fn.get("name", ""), model=model)
            total += estimate_tokens(fn.get("arguments", ""), model=model)

        if msg.get("tool_call_id"):
            total += estimate_tokens(msg["tool_call_id"], model=model)

    if include_padding:
        total = int(total * TOKEN_ESTIMATION_PADDING)
    return max(0, total)


def tokenizer_name_for_model(model: str | None) -> str:
    """Return the tokenizer/backend name used for diagnostics."""
    if _encoding_for_model(model) is None:
        return "heuristic"
    try:
        return _encoding_for_model(model).name
    except AttributeError:
        return "tiktoken"


def _estimate_content_tokens(
    content: Any,
    *,
    model: str | None,
    image_token_estimate: int,
) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_tokens(content, model=model)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += estimate_tokens(block, model=model)
                continue
            if not isinstance(block, dict):
                total += estimate_tokens(block, model=model)
                continue
            block_type = str(block.get("type", ""))
            if "image" in block_type or "image_url" in block:
                total += image_token_estimate
            if "text" in block:
                total += estimate_tokens(block["text"], model=model)
            elif "content" in block:
                total += estimate_tokens(block["content"], model=model)
            elif "input" in block:
                total += estimate_tokens(block["input"], model=model)
        return total
    return estimate_tokens(content, model=model)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


@lru_cache(maxsize=64)
def _encoding_for_model(model: str | None):
    try:
        import tiktoken
    except Exception:
        return None

    normalized = (model or "").strip()
    aliases = _candidate_model_names(normalized)
    for candidate in aliases:
        try:
            return tiktoken.encoding_for_model(candidate)
        except Exception:
            continue

    if _looks_openai_compatible(normalized):
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _candidate_model_names(model: str) -> list[str]:
    if not model:
        return []
    lower = model.lower()
    candidates = [model]
    if lower.startswith("qwen"):
        candidates.extend(["gpt-4o", "gpt-4.1"])
    elif lower.startswith("claude"):
        candidates.extend(["gpt-4o", "gpt-4.1"])
    elif lower.startswith("kimi"):
        candidates.append("gpt-4o")
    return candidates


def _looks_openai_compatible(model: str) -> bool:
    lower = model.lower()
    return lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
