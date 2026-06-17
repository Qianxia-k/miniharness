"""Conversation message models.

OpenHarness has rich content blocks for text, images, tool_use, and tool_result.
MiniHarness starts simpler and uses OpenAI-style message dictionaries.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """One item in the conversation history."""

    role: Role
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_openai(self) -> dict[str, Any]:
        """Convert to the format expected by OpenAI-compatible chat APIs."""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = normalize_tool_calls(self.tool_calls)
        return data


class Conversation(BaseModel):
    """Mutable conversation history."""

    messages: list[Message] = Field(default_factory=list)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def to_openai(self) -> list[dict[str, Any]]:
        return [message.to_openai() for message in self.messages]


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool calls whose ``function.arguments`` are valid JSON objects.

    Some OpenAI-compatible providers reject the whole request when historical
    assistant tool calls contain malformed JSON arguments.  The model may still
    emit invalid JSON during a turn; MiniHarness records a tool error for that,
    but the saved assistant message must be provider-safe for future turns.
    """
    normalized = copy.deepcopy(tool_calls)
    for tool_call in normalized:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        function["arguments"] = normalize_tool_arguments(function.get("arguments"))
    return normalized


def normalize_tool_arguments(arguments: Any) -> str:
    """Return a JSON object string suitable for function-call history."""
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)

    if arguments is None or arguments == "":
        return "{}"

    if not isinstance(arguments, str):
        return json.dumps({"_invalid_arguments": str(arguments)}, ensure_ascii=False)

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"_invalid_arguments": arguments}, ensure_ascii=False)

    if not isinstance(parsed, dict):
        return json.dumps({"_invalid_arguments": arguments}, ensure_ascii=False)

    return json.dumps(parsed, ensure_ascii=False)
