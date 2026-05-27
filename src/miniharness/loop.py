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

    Accepts a single Settings object (mirrors OpenHarness's pattern where the
    runtime passes a resolved config bundle to the engine).
    """

    def __init__(self, *, cwd: Path, settings: Settings) -> None:
        self.cwd = cwd
        self.settings = settings

        # Resolve provider profile + overrides from settings.
        provider_profile = get_profile(settings.provider.name)
        model = settings.provider.model or provider_profile.default_model
        base_url = settings.provider.base_url or provider_profile.base_url

        self.llm = LLMClient(profile=provider_profile, model=model, base_url=base_url)
        self.permissions = PermissionChecker(cwd=cwd)
        self.tools = create_default_registry(cwd=cwd, permissions=self.permissions)

    async def run(self, prompt: str) -> str:
        """Run the async loop and return the final assistant text."""
        conversation = Conversation()
        conversation.append(Message(role="system", content=SYSTEM_PROMPT))
        conversation.append(Message(role="user", content=prompt))

        for turn in range(1, self.settings.max_turns + 1):
            response_message = None

            async for event in self.llm.stream(
                messages=conversation.to_openai(),
                tools=self.tools.to_openai_tools(),
            ):
                if isinstance(event, TextDelta):
                    console.print(event.text, end="")

                elif isinstance(event, StreamComplete):
                    response_message = event.message

            if response_message is None:
                return "No response from model."

            conversation.append(response_message)

            if response_message.tool_calls:
                console.print()
                await self._execute_tools(response_message.tool_calls, conversation)
                continue

            console.print()
            return response_message.content or ""

        return "Reached maximum turns without a final answer."

    async def _execute_tools(
        self,
        tool_calls: list[dict],
        conversation: Conversation,
    ) -> None:
        """Execute each tool call and append results to the conversation."""
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                conversation.append(
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

            conversation.append(
                Message(
                    role="tool",
                    content=result.output,
                    tool_call_id=tool_call_id,
                )
            )
