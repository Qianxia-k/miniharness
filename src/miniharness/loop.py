"""The MiniHarness agent loop.

This is the heart of the project. It mirrors the OpenHarness idea:

model response (async streamed) -> optional tool calls -> tool results -> model again
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from miniharness.config.settings import Settings
from miniharness.context.budget import ContextBudget
from miniharness.context.compactor import compact_messages
from miniharness.llm import LLMClient, StreamComplete, TextDelta
from miniharness.memory.core import CoreMemory
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
        self.llm = LLMClient(
            profile=provider_profile,
            model=model,
            base_url=base_url,
            agent_settings=settings.agent,
        )
        self.permissions = PermissionChecker(cwd=cwd)
        self.tools = create_default_registry(cwd=cwd, permissions=self.permissions)
        self.budget = ContextBudget.for_model(
            model, ratio=settings.context_budget_ratio
        )
        self.core_memory = CoreMemory(cwd)

        # The conversation lives on the AgentLoop so it survives across
        # multiple run() calls.  This is the key refactor for step 1a.
        self.conversation = Conversation()
        self.conversation.append(
            Message(role="system", content=self._build_system_prompt())
        )

    async def run(self, prompt: str) -> str:
        """Run one turn of the agent loop and return the final assistant text.

        Appends the user message to the existing conversation, so history
        from previous calls is preserved automatically.
        """
        self.conversation.append(Message(role="user", content=prompt))

        # ---- context budget check -------------------------------------------
        msgs = self.conversation.to_openai()
        if self.budget.is_over_budget(msgs):
            console.print(
                f"  [dim]Context budget {self.budget.usage_ratio(msgs):.0%} used, "
                f"compacting...[/dim]"
            )
            msgs, stats = await compact_messages(
                msgs,
                budget=self.budget,
                llm_stream=self.llm.stream,
                keep_last_n_turns=self.settings.keep_last_n_turns,
            )
            # Replace the conversation with compacted messages.
            self.conversation = Conversation()
            for m in msgs:
                self.conversation.append(Message(**m))

            dropped = stats["original_count"] - stats["final_count"]
            summary = stats.get("Compacted Summary")
            summary_info = ""
            if summary:
                summary_info = f" summary={len(summary)} chars"
            console.print(
                f"  [dim]Compacted: {stats['original_count']} → "
                f"{stats['final_count']} messages "
                f"({dropped} dropped, "
                f"stage1={stats['stage1_truncated']}, "
                f"stage2={stats['stage2_summarised']}{summary_info})[/dim]"
            )
            # 新增：漂亮打印完整的总结内容（像大模型思考过程）
            if summary:
                console.print(f"  [bold cyan]📝 LLM 压缩总结：[/bold cyan]")
                console.print(f"  [dim]{summary}[/dim]")  # 输出完整内容，不截断
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

    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt from the static part + core memory."""
        core = self.core_memory.render_for_system_prompt()
        if core:
            return f"{SYSTEM_PROMPT}\n\n{core}"
        return SYSTEM_PROMPT

    def set_model(self, model: str) -> None:
        """Switch the model for subsequent turns.

        Updates the AgentLoop-level attribute, LLM client, and context budget
        so compaction thresholds stay accurate for the new model.
        """
        self.model = model
        self.llm.model = model
        self.budget = ContextBudget.for_model(
            model, ratio=self.settings.context_budget_ratio
        )

    def clear(self) -> None:
        """Reset the conversation, keeping only the system prompt + core memory.

        Mirrors OpenHarness's QueryEngine.clear().
        """
        self.conversation = Conversation()
        self.conversation.append(
            Message(role="system", content=self._build_system_prompt())
        )

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
                f"  [dim]→ {tool_name}({json.dumps(arguments, ensure_ascii=False)})[/dim]"
            )

            result = await self.tools.execute(tool_name, arguments)
            if result.is_error:
                console.print(f"  [yellow]! {result.output[:120]}[/yellow]")
            elif tool_name.startswith("memory_"):
                # Memory writes — show the full result prominently.
                console.print(f"  [bold cyan]memory[/bold cyan] {result.output}")
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
