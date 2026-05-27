"""Conversation message models.

OpenHarness has rich content blocks for text, images, tool_use, and tool_result.
MiniHarness starts simpler and uses OpenAI-style message dictionaries.
"""

from __future__ import annotations

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
            data["tool_calls"] = self.tool_calls
        return data


class Conversation(BaseModel):
    """Mutable conversation history."""

    messages: list[Message] = Field(default_factory=list)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def to_openai(self) -> list[dict[str, Any]]:
        return [message.to_openai() for message in self.messages]

