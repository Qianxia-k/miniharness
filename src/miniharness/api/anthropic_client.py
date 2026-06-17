"""Anthropic API client with prompt caching.

Prompt caching is an Anthropic API feature that lets you mark portions
of your request as cacheable.  On subsequent requests with the same
content, you get a **90% discount** on input tokens for those portions.

For a coding agent this is transformative — the system prompt (~1000 tokens
per turn) and tool definitions (~1500 tokens) are identical across every
turn.  With caching they cost only 10% — saving ~2250 tokens per turn.

Cache strategy (mirrors best practices for coding agents):

    ┌──────────────────────────────────────┐
    │  System prompt   → cached (ephemeral) │  ~1000 tokens → ~100
    │  Tool definitions → cached (ephemeral) │  ~1500 tokens → ~150
    │  Conversation    → NOT cached          │  changes every turn
    └──────────────────────────────────────┘

Ephemeral cache has a 5-minute TTL.  After 5 minutes of inactivity the
cache expires and the next request pays full price.  In practice this
means the cache stays warm during an active coding session.

Usage::

    from miniharness.api.anthropic_client import AnthropicClient

    client = AnthropicClient(api_key="sk-...", model="claude-sonnet-4-6")
    async for event in client.stream(
        system_prompt="You are a coding agent.",
        messages=conversation,
        tools=tool_schemas,
    ):
        if isinstance(event, TextDelta):
            print(event.text, end="")
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, AsyncIterator

from miniharness.llm import StreamComplete, TextDelta
from miniharness.messages import Message


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic Client
# ═══════════════════════════════════════════════════════════════════════════


class AnthropicClient:
    """Native Anthropic Messages API client with prompt caching.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    model:
        Model ID (e.g. ``"claude-sonnet-4-6"``).
    max_tokens:
        Default max output tokens per request.
    thinking:
        Whether to enable extended thinking (Claude 4+).
    """

    _RETRYABLE = frozenset({429, 500, 502, 503, 529})
    _MAX_RETRIES = 3

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        thinking: bool = False,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.default_max_tokens = max_tokens
        self.thinking = thinking
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        max_tokens_override: int | None = None,
    ) -> AsyncIterator[TextDelta | StreamComplete]:
        """Stream a completion from the Anthropic API with prompt caching.

        Parameters
        ----------
        messages:
            OpenAI-format message list (converted internally to Anthropic format).
        tools:
            OpenAI-format tool schemas (converted internally).
        system_prompt:
            Full system prompt text.  Cached with ``cache_control: ephemeral``.
        max_tokens_override:
            Override the default ``max_tokens`` for this request.
        """
        from anthropic import (
            Anthropic, AsyncAnthropic,
            APIStatusError, APIConnectionError, APITimeoutError,
        )

        params = self._build_params(messages, tools, system_prompt, max_tokens_override)
        client = AsyncAnthropic(api_key=self.api_key, timeout=self.timeout)

        # Retry with exponential backoff.
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                async for event in self._stream_once(client, params):
                    yield event
                return
            except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
                if attempt == self._MAX_RETRIES or not self._is_retryable(exc):
                    raise
                delay = min(1.0 * (2 ** attempt), 30.0) + random.uniform(0, 1)
                await asyncio.sleep(delay)

    async def _stream_once(
        self,
        client,
        params: dict[str, Any],
    ) -> StreamComplete:
        """Execute one streaming API call. Returns the assembled Message."""
        from anthropic import AsyncAnthropic

        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        async with client.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        content_parts.append(event.delta.text)
                        yield TextDelta(text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        idx = event.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tool_calls_acc[idx]["function"]["arguments"] += event.delta.partial_json

                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        idx = event.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": event.content_block.id,
                                "type": "function",
                                "function": {"name": event.content_block.name, "arguments": ""},
                            }

            final = await stream.get_final_message()

        # Assemble tool_calls in index order.
        tool_calls = (
            [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            if tool_calls_acc else None
        )
        content = "".join(content_parts) if content_parts else ""

        yield StreamComplete(
            message=Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
        )

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        max_tokens_override: int | None,
    ) -> dict[str, Any]:
        """Build the Anthropic API request parameters.

        The system prompt and tool definitions are wrapped with
        ``cache_control: {type: "ephemeral"}`` so they benefit from the
        90% token discount on subsequent turns.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens_override or self.default_max_tokens,
        }

        # System prompt — cached.
        if system_prompt:
            params["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Messages — Anthropic format (NOT cached — changes every turn).
        anthropic_msgs = self._convert_messages(messages)
        params["messages"] = anthropic_msgs

        # Tools — cached.  Tools are converted from OpenAI format to
        # Anthropic format, and the LAST tool gets the cache breakpoint
        # (everything up to and including it is cached).
        if tools:
            anthropic_tools = self._convert_tools(tools)
            # Mark last tool as cache breakpoint.
            if anthropic_tools:
                anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            params["tools"] = anthropic_tools

        # Extended thinking (Claude 4+).
        if self.thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": 1024}

        return params

    # ------------------------------------------------------------------
    # Format conversion (OpenAI → Anthropic)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(
        openai_msgs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic format.

        System messages are skipped — they go in the ``system`` parameter.
        """
        result: list[dict[str, Any]] = []
        for msg in openai_msgs:
            role = msg.get("role", "")
            if role == "system":
                continue  # system prompt goes in the system parameter

            if role == "user":
                content = msg.get("content", "")
                result.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content or ""}],
                })

            elif role == "assistant":
                blocks: list[dict] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn.get("name", ""),
                        "input": _try_parse_json(fn.get("arguments", "{}")),
                    })
                result.append({"role": "assistant", "content": blocks})

            elif role == "tool":
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })

        return result

    @staticmethod
    def _convert_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tools to Anthropic format.

        OpenAI:  {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Anthropic: {"name": ..., "description": ..., "input_schema": ...}
        """
        result: list[dict[str, Any]] = []
        for t in openai_tools:
            fn = t.get("function", {})
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    def _is_retryable(self, exc: Exception) -> bool:
        from anthropic import APIStatusError, APIConnectionError, APITimeoutError
        if isinstance(exc, APIStatusError):
            return exc.status_code in self._RETRYABLE
        return isinstance(exc, (APIConnectionError, APITimeoutError))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

import json as _json


def _try_parse_json(s: str) -> dict:
    try:
        return _json.loads(s)
    except _json.JSONDecodeError:
        return {}
