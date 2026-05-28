"""The MiniHarness agent loop.

This is the heart of the project. It mirrors the OpenHarness idea:

model response (async streamed) -> optional tool calls -> tool results -> model again
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from miniharness.config.settings import Settings
from miniharness.llm import LLMClient, StreamComplete, TextDelta
from miniharness.messages import Conversation, Message
from miniharness.permissions import PermissionChecker
from miniharness.providers import get_profile
from miniharness.tool_registry import create_default_registry


SYSTEM_PROMPT = """You are MiniHarness, a small coding agent.

You can explain code and use tools when needed. Be concise and practical.
"""

console = Console()


class AgentLoop:
    """Run one user request through the async agent loop.

    Owns the conversation history so multiple prompts can share context
    across calls — the foundation for interactive multi-turn sessions.
    Mirrors OpenHarness's QueryEngine which owns ``_messages`` the same way.
    """

    def __init__(self, *, cwd: Path, settings: Settings) -> None:
        self.cwd = cwd
        self.settings = settings

        # Resolve provider profile + overrides from settings.
        provider_profile = get_profile(settings.provider.name)
        model = settings.provider.model or provider_profile.default_model
        base_url = settings.provider.base_url or provider_profile.base_url

        self.model = model
        self.session_id: str | None = None  # set by CLI/REPL for persistence
        self.tag: str = ""  # human-readable tag, set by /tag command
        self.llm = LLMClient(profile=provider_profile, model=model, base_url=base_url)
        self.permissions = PermissionChecker(cwd=cwd)
        self.tools = create_default_registry(cwd=cwd, permissions=self.permissions)

        # The conversation lives on the AgentLoop so it survives across
        # multiple run() calls.  This is the key refactor for step 1a.
        self.conversation = Conversation()
        self.conversation.append(Message(role="system", content=SYSTEM_PROMPT))

    async def run(self, prompt: str) -> str:
        """Run one turn of the agent loop and return the final assistant text.

        Appends the user message to the existing conversation, so history
        from previous calls is preserved automatically.
        """
        self.conversation.append(Message(role="user", content=prompt))

        for turn in range(1, self.settings.max_turns + 1):
            response_message = None

            async for event in self.llm.stream(
                messages=self.conversation.to_openai(),
                tools=self.tools.to_openai_tools(),
            ):
                if isinstance(event, TextDelta):
                    console.print(event.text, end="")

                elif isinstance(event, StreamComplete):
                    response_message = event.message

            if response_message is None:
                return "No response from model."

            self.conversation.append(response_message)

            if response_message.tool_calls:
                console.print()
                await self._execute_tools(response_message.tool_calls)
                continue

            console.print()
            return response_message.content or ""

        return "Reached maximum turns without a final answer."

    def export_messages(self) -> list[dict]:
        """Export all messages as JSON-serializable dicts.

        Used by session persistence to save conversation history to disk.
        """
        return [msg.model_dump() for msg in self.conversation.messages]

    def restore_messages(self, messages_data: list[dict]) -> None:
        """Replace the entire conversation with previously-saved messages.

        Mirrors OpenHarness's QueryEngine.load_messages().
        """
        self.conversation = Conversation()
        for data in messages_data:
            self.conversation.append(Message(**data))

    def clear(self) -> None:
        """Reset the conversation, keeping only the system prompt.

        Mirrors OpenHarness's QueryEngine.clear().
        """
        self.conversation = Conversation()
        self.conversation.append(Message(role="system", content=SYSTEM_PROMPT))

    async def _execute_tools(self, tool_calls: list[dict]) -> None:
        """Execute each tool call and append results to the conversation."""
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                self.conversation.append(
                    Message(
                        role="tool",
                        content=f"Invalid JSON arguments: {raw_args}",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            console.print(
                f"  [dim]→ {tool_name}({json.dumps(arguments)})[/dim]"
            )

            result = await self.tools.execute(tool_name, arguments)
            if result.is_error:
                console.print(f"  [yellow]! {result.output[:120]}[/yellow]")
            else:
                preview = result.output[:80].replace("\n", " ")
                console.print(
                    f"  [dim]← {preview}...[/dim]" if len(result.output) > 80
                    else f"  [dim]← {preview}[/dim]"
                )

            self.conversation.append(
                Message(
                    role="tool",
                    content=result.output,
                    tool_call_id=tool_call_id,
                )
            )
