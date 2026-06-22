"""Shared UI runtime controller.

This is the boundary between frontends (TUI, future web UI) and the agent
engine.  Frontends should send lines and render events; this controller owns
the same session, command, sandbox, and persistence semantics that the CLI REPL
uses.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from miniharness.commands import CommandContext, CommandRegistry
from miniharness.commands.builtin import (
    cmd_clear,
    cmd_exit,
    cmd_help,
    cmd_history,
    cmd_hooks,
    cmd_max_tokens,
    cmd_mcp,
    cmd_memory,
    cmd_model,
    cmd_permissions,
    cmd_plugins,
    cmd_project,
    cmd_skills,
    cmd_temperature,
    cmd_tokens,
    cmd_tools,
    cmd_top_p,
    cmd_turns,
)
from miniharness.commands.types import CommandResult
from miniharness.config.settings import Settings
from miniharness.loop import AgentLoop
from miniharness.runtime import RuntimeEventBus
from miniharness.sessions import (
    list_sessions,
    load_session_by_id,
    load_session_by_tag,
    rename_session,
    save_loop_snapshot,
    switch_session,
)


SystemPrinter = Callable[[str], Awaitable[None]]
AgentRunner = Callable[[AgentLoop, str], Awaitable[str]]
ClearHandler = Callable[[], Awaitable[None]]
PermissionPrompt = Callable[[str, str], Awaitable[bool]]
CompactProgressHandler = Callable[[dict], Awaitable[None]]


@dataclass
class RuntimeController:
    """One interactive MiniHarness runtime session."""

    cwd: Path
    settings: Settings
    permission_prompt: PermissionPrompt | None = None
    compact_progress: CompactProgressHandler | None = None
    event_bus: RuntimeEventBus | None = None
    loop: AgentLoop = field(init=False)
    commands: CommandRegistry = field(init=False)
    _sandbox_started: bool = field(default=False, init=False)
    _background_tasks: set[asyncio.Task] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.cwd = self.cwd.expanduser().resolve()
        self.loop = AgentLoop(
            cwd=self.cwd,
            settings=self.settings,
            permission_prompt=self.permission_prompt,
            compact_progress=self.compact_progress,
            event_bus=self.event_bus,
        )
        self.loop.session_id = uuid.uuid4().hex[:12]
        self.commands = self._build_command_registry()

    async def start(self) -> None:
        """Start runtime-owned resources."""
        if self.settings.sandbox.enabled:
            from miniharness.sandbox import start_sandbox

            await start_sandbox(cwd=self.cwd, image=self.settings.sandbox.image)
            self._sandbox_started = True

    async def close(self) -> None:
        """Close runtime-owned resources."""
        await self.drain_background_tasks(timeout=1.0)

        mcp = getattr(self.loop, "_mcp_manager", None)
        if mcp is not None:
            try:
                await asyncio.shield(asyncio.wait_for(mcp.close(), timeout=5.0))
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

        if self._sandbox_started:
            from miniharness.sandbox import stop_sandbox

            try:
                await asyncio.shield(asyncio.wait_for(stop_sandbox(), timeout=3.0))
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            self._sandbox_started = False

    async def drain_background_tasks(self, *, timeout: float | None = None) -> None:
        """Wait briefly for best-effort background work, then cancel leftovers."""
        if not self._background_tasks:
            return
        tasks = set(self._background_tasks)
        try:
            if timeout is None:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                done, pending = await asyncio.wait(tasks, timeout=timeout)
                await asyncio.gather(*done, return_exceptions=True)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        finally:
            self._background_tasks.difference_update(tasks)

    async def handle_line(
        self,
        line: str,
        *,
        run_agent: AgentRunner,
        print_system: SystemPrinter,
        clear_output: ClearHandler | None = None,
    ) -> bool:
        """Handle one frontend-submitted line.

        Returns ``False`` when the frontend should shut down.
        """
        stripped = line.strip()
        if not stripped:
            return True

        if stripped.startswith("/"):
            result = self.commands.dispatch(stripped, self._make_context())
            await self._render_command_result(
                result,
                run_agent=run_agent,
                print_system=print_system,
                clear_output=clear_output,
            )

            new_loop = getattr(self._last_context, "_new_loop", None)
            if new_loop is not None:
                await self._replace_loop(new_loop)
                return True

            if result.should_save and not result.exit:
                save_loop_snapshot(self.loop)
            return not result.exit

        await run_agent(self.loop, stripped)
        save_loop_snapshot(self.loop)
        self._schedule_memory_extraction(print_system)
        return True

    async def _render_command_result(
        self,
        result: CommandResult,
        *,
        run_agent: AgentRunner,
        print_system: SystemPrinter,
        clear_output: ClearHandler | None,
    ) -> None:
        if result.message:
            await print_system(result.message)
        if result.exit:
            return
        if result.refresh_runtime:
            self.commands = self._build_command_registry()
        if result.submit_prompt:
            await run_agent(self.loop, result.submit_prompt)
            save_loop_snapshot(self.loop)
            self._schedule_memory_extraction(print_system)
        if result.refresh_runtime and clear_output is not None:
            # cmd_clear already mutated the loop; the frontend transcript should
            # match that cleared conversation.
            if result.message and "cleared" in result.message.lower():
                await clear_output()

    def _schedule_memory_extraction(self, print_system: SystemPrinter) -> None:
        """Start best-effort post-turn memory extraction without blocking input."""
        messages = self.loop.conversation.to_openai()
        stream_fn = self.loop.stream_fn
        task = asyncio.create_task(
            self._extract_memories(messages, stream_fn, print_system)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _extract_memories(
        self,
        messages: list[dict],
        stream_fn,
        print_system: SystemPrinter,
    ) -> None:
        """Run post-turn memory extraction in the background."""
        try:
            from miniharness.services.memory_extractor import extract_memories_from_turn

            result = await asyncio.wait_for(
                extract_memories_from_turn(
                    messages=messages,
                    llm=stream_fn,
                    cwd=self.cwd,
                ),
                timeout=20.0,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return
        except Exception:
            return

        message = _format_memory_extraction_result(result)
        if message:
            await print_system(message)

    async def _replace_loop(self, new_loop: AgentLoop) -> None:
        old_mcp = getattr(self.loop, "_mcp_manager", None)
        if old_mcp is not None:
            try:
                await asyncio.shield(asyncio.wait_for(old_mcp.close(), timeout=5))
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        self.loop = new_loop
        self.commands = self._build_command_registry()

    def _make_context(self) -> CommandContext:
        ctx = CommandContext(
            loop=self.loop,
            console=None,
            cwd=self.loop.cwd,
            skill_registry=getattr(self.loop, "skill_registry", None),
            hook_registry=getattr(self.loop, "hook_registry", None),
            tool_registry=getattr(self.loop, "tools", None),
        )
        self._last_context = ctx
        return ctx

    def _build_command_registry(self) -> CommandRegistry:
        reg = CommandRegistry()
        reg.register("exit", cmd_exit, description="Exit MiniHarness", aliases=["quit", "q"], source="builtin")
        reg.register("clear", cmd_clear, description="Clear conversation history", source="builtin")
        reg.register("help", cmd_help, description="Show available commands", source="builtin")
        reg.register("history", cmd_history, description="Show message count", source="builtin")
        reg.register("tokens", cmd_tokens, description="Show current context token budget", source="builtin")
        reg.register("model", cmd_model, description="Show or switch the model", source="builtin")
        reg.register("turns", cmd_turns, description="Show or set max turns", source="builtin")
        reg.register("permissions", cmd_permissions, description="Show or set permission mode", source="builtin")
        reg.register("temperature", cmd_temperature, description="Show or set LLM temperature", source="builtin")
        reg.register("top-p", cmd_top_p, description="Show or set LLM top_p", source="builtin")
        reg.register("max-tokens", cmd_max_tokens, description="Show or set max output tokens", source="builtin")
        reg.register("memory", cmd_memory, description="Show core/semantic/episodic memory", source="builtin")
        reg.register("project", cmd_project, description="Show project instructions", source="builtin")
        reg.register("hooks", cmd_hooks, description="Show hook configuration", source="builtin")
        reg.register("skills", cmd_skills, description="List available skills", source="builtin")
        reg.register("plugins", cmd_plugins, description="List, inspect, or toggle plugins", source="builtin")
        reg.register("tools", cmd_tools, description="List, describe, or execute tools", source="builtin")
        reg.register("mcp", cmd_mcp, description="Show MCP server connection status", source="builtin")
        reg.register("sessions", self._cmd_sessions, description="List saved sessions", source="builtin")
        reg.register("resume", self._cmd_resume, description="Resume a saved session", source="builtin")
        reg.register("tag", self._cmd_tag, description="Tag current session", source="builtin")
        if self.loop.skill_registry is not None:
            reg.register_from_skills(self.loop.skill_registry)
        return reg

    def _cmd_sessions(self, args: str, ctx: CommandContext) -> CommandResult:
        sessions = list_sessions(str(self.cwd))
        if not sessions:
            return CommandResult.ok("No saved sessions for this project.")
        lines = ["Saved sessions (newest first):"]
        for s in sessions:
            marker = " ← current" if s["session_id"] == self.loop.session_id else ""
            summary = s.get("summary") or "(empty)"
            lines.append(
                f"  {s['session_id']}{marker}  {s['message_count']} msgs  "
                f"{s.get('model', '?')}  {summary[:80]}"
            )
        return CommandResult.ok("\n".join(lines))

    def _cmd_resume(self, args: str, ctx: CommandContext) -> CommandResult:
        target = args.strip()
        if not target:
            latest = list_sessions(str(self.cwd))
            if not latest:
                return CommandResult.ok("No saved sessions for this project.")
            target = latest[0]["session_id"]

        data = load_session_by_id(str(self.cwd), target) or load_session_by_tag(str(self.cwd), target)
        if data is None:
            return CommandResult.ok(f"Session '{target}' not found.")
        target_id = data.get("session_id", target)
        if self.loop.session_id == target_id:
            return CommandResult.ok(f"Already on session {target_id}.")

        new_loop = switch_session(
            self.loop,
            target_id,
            permission_prompt=self.permission_prompt,
            compact_progress=self.compact_progress,
            event_bus=self.event_bus,
        )
        if new_loop is None:
            return CommandResult.ok(f"Session '{target}' not found.")
        ctx._new_loop = new_loop
        return CommandResult.ok(
            f"Restored session {new_loop.session_id}"
            + (f" @{new_loop.tag}" if new_loop.tag else "")
        )

    def _cmd_tag(self, args: str, ctx: CommandContext) -> CommandResult:
        tag = args.strip()
        if not tag:
            return CommandResult.ok("Usage: /tag <name>")
        if not self.loop.session_id:
            return CommandResult.ok("No active session to tag.")
        if rename_session(str(self.cwd), self.loop.session_id, tag):
            self.loop.tag = tag
            return CommandResult.ok(f"Session {self.loop.session_id} tagged as '{tag}'.", should_save=True)
        return CommandResult.ok(f"Could not tag session '{tag}'.")


def _format_memory_extraction_result(result) -> str:
    """Format memory extraction output for any frontend renderer."""
    if getattr(result, "skipped", False):
        return ""

    facts = getattr(result, "facts", []) or []
    episode = getattr(result, "episode", None)
    if not facts and not episode:
        return ""

    lines = ["Memory updated:"]
    if episode:
        task = str(episode.get("task", "")).strip()
        outcome = str(episode.get("outcome", "")).strip()
        if task:
            suffix = f" [{outcome}]" if outcome else ""
            lines.append(f"- episode: {task}{suffix}")
        summary = str(episode.get("summary", "")).strip()
        if summary:
            lines.append(f"  {summary[:160]}")

    if facts:
        lines.append(f"- facts remembered: {len(facts)}")
        for item in facts[:3]:
            if isinstance(item, dict):
                fact = str(item.get("fact", "")).strip()
                if fact:
                    lines.append(f"  - {fact[:120]}")

    return "\n".join(lines)
