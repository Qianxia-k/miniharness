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
from typing import Any, Awaitable, Callable

from miniharness.commands import CommandContext, CommandRegistry
from miniharness.commands.builtin import (
    cmd_clear,
    cmd_agents,
    cmd_diff,
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
    cmd_tasks,
    cmd_temperature,
    cmd_tokens,
    cmd_tools,
    cmd_top_p,
    cmd_turns,
)
from miniharness.commands.types import CommandResult
from miniharness.config.settings import Settings
from miniharness.loop import AgentLoop
from miniharness.messages import Message
from miniharness.runtime import RuntimeEventBus
from miniharness.sessions import (
    list_sessions,
    load_session_by_id,
    load_session_by_tag,
    rename_session,
    save_loop_snapshot,
    switch_session,
)
from miniharness.state import AppState, AppStateStore
from miniharness.ui.protocol import TaskSnapshot


SystemPrinter = Callable[[str], Awaitable[None]]
AgentRunner = Callable[[AgentLoop, str], Awaitable[str]]
ClearHandler = Callable[[], Awaitable[None]]
PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]
CompactProgressHandler = Callable[[dict], Awaitable[None]]


@dataclass
class RuntimeController:
    """One interactive MiniHarness runtime session."""

    cwd: Path
    settings: Settings
    permission_prompt: PermissionPrompt | None = None
    ask_user_prompt: AskUserPrompt | None = None
    compact_progress: CompactProgressHandler | None = None
    event_bus: RuntimeEventBus | None = None
    system_prompt_override: str | None = None
    system_prompt_mode: str | None = None
    session_hooks: dict[str, Any] | None = None
    tool_policy: dict[str, Any] | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    loop: AgentLoop = field(init=False)
    commands: CommandRegistry = field(init=False)
    state_store: AppStateStore = field(init=False)
    _sandbox_started: bool = field(default=False, init=False)
    _background_tasks: set[asyncio.Task] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.cwd = self.cwd.expanduser().resolve()
        self.loop = AgentLoop(
            cwd=self.cwd,
            settings=self.settings,
            permission_prompt=self.permission_prompt,
            ask_user_prompt=self.ask_user_prompt,
            compact_progress=self.compact_progress,
            event_bus=self.event_bus,
            system_prompt_override=self.system_prompt_override,
            system_prompt_mode=self.system_prompt_mode,
            session_hooks=self.session_hooks,
            tool_policy=self.tool_policy,
            permission_mode=self.permission_mode,
            max_turns=self.max_turns,
        )
        self.loop.session_id = uuid.uuid4().hex[:12]
        self.commands = self._build_command_registry()
        self.state_store = AppStateStore(self._build_state())

    async def start(self) -> None:
        """Start runtime-owned resources."""
        if self.settings.sandbox.enabled:
            from miniharness.sandbox import start_sandbox

            await start_sandbox(cwd=self.cwd, image=self.settings.sandbox.image)
            self._sandbox_started = True
        self.sync_state()

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

        try:
            from miniharness.tasks import get_background_task_manager

            await asyncio.shield(asyncio.wait_for(
                get_background_task_manager().close(),
                timeout=3.0,
            ))
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

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

            notification = await self._drain_coordinator_notifications(
                run_agent=run_agent,
                print_system=print_system,
            )
            permissions = await self._drain_swarm_permission_requests(print_system)
            if (result.should_save or notification or permissions) and not result.exit:
                save_loop_snapshot(self.loop)
            self.sync_state()
            return not result.exit

        await run_agent(self.loop, stripped)
        await self._drain_coordinator_notifications(
            run_agent=run_agent,
            print_system=print_system,
        )
        await self._drain_swarm_permission_requests(print_system)
        save_loop_snapshot(self.loop)
        self._schedule_memory_extraction(print_system)
        self.sync_state()
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
            self.sync_state()
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

    async def _drain_coordinator_notifications(
        self,
        *,
        run_agent: AgentRunner,
        print_system: SystemPrinter,
    ) -> str:
        """Drain completed task notifications and auto-submit agent results.

        OpenHarness feeds completed async-agent results back to the coordinator
        as user-role ``<task-notification>`` messages.  MiniHarness follows the
        same contract by submitting agent notifications through the same
        ``run_agent`` adapter that CLI/TUI/Web use for normal user prompts.

        Generic shell task notifications are displayed and persisted, but are
        not automatically submitted as coordinator turns.
        """
        payloads: list[str] = []

        first_message = await self._notify_completed_background_tasks(print_system)
        if first_message:
            payloads.append(first_message)
            if "<task-notification>" in first_message:
                await print_system("Submitting background agent result to coordinator...")
                await run_agent(self.loop, first_message)
            else:
                self.loop.conversation.append(Message(role="user", content=first_message))

        try:
            from miniharness.ui.coordinator_drain import (
                format_completed_background_task_notifications,
                pending_async_agent_entries,
                wait_for_completed_async_agent_entries,
            )
        except Exception:
            return "\n\n".join(payloads)

        while pending_async_agent_entries(self.loop.tool_metadata):
            pending = pending_async_agent_entries(self.loop.tool_metadata)
            await print_system(
                f"Waiting for {len(pending)} background agent task(s) to finish..."
            )
            completed = await self._wait_for_async_agent_batch(
                wait_for_completed_async_agent_entries,
                print_system,
            )
            message = format_completed_background_task_notifications(completed)
            if not message.strip():
                return "\n\n".join(payloads)
            payloads.append(message)
            await print_system(message)
            await print_system("Submitting background agent result to coordinator...")
            await run_agent(self.loop, message)

        return "\n\n".join(payloads)

    async def _wait_for_async_agent_batch(
        self,
        wait_for_completed_async_agent_entries: Callable[..., Awaitable[list[dict]]],
        print_system: SystemPrinter,
    ) -> list[dict]:
        """Wait for one agent completion batch while servicing worker prompts.

        Worker agents may block on a permission request that only the parent
        runtime can ask the user to resolve.  The coordinator wait loop must
        therefore keep draining pending permission files instead of sleeping in
        a single uninterruptible wait.
        """
        while True:
            await self._drain_swarm_permission_requests(print_system)
            completed = await wait_for_completed_async_agent_entries(
                self.loop.tool_metadata,
                timeout_seconds=0.25,
            )
            if completed:
                return completed
            try:
                from miniharness.ui.coordinator_drain import pending_async_agent_entries
            except Exception:
                return []
            if not pending_async_agent_entries(self.loop.tool_metadata):
                return []

    async def _notify_completed_background_tasks(self, print_system: SystemPrinter) -> str:
        """Return completed background task notifications, if any."""
        try:
            from miniharness.ui.coordinator_drain import drain_completed_background_tasks

            message = await drain_completed_background_tasks(
                self.loop.tool_metadata,
                print_system=print_system,
            )
            return message
        except Exception:
            return ""

    async def _drain_swarm_permission_requests(self, print_system: SystemPrinter) -> int:
        """Resolve pending delegated-agent permission requests."""
        try:
            from miniharness.swarm.permission_sync import (
                PermissionResolution,
                evaluate_permission_request,
                read_pending_permissions,
                resolve_permission,
            )
        except Exception:
            return 0

        requests = await read_pending_permissions()
        resolved = 0
        for request in requests:
            decision = evaluate_permission_request(request, self.loop.permissions)
            prompt = _format_swarm_permission_prompt(request)
            if decision.allowed:
                allowed = True
            elif decision.requires_confirmation and self.permission_prompt is not None:
                allowed = await self.permission_prompt(request.tool_name, prompt)
            elif decision.requires_confirmation:
                resolved_decision = self.loop.permissions.resolve_interactive(
                    decision,
                    prompt,
                )
                allowed = resolved_decision.allowed
            else:
                allowed = False
            await resolve_permission(
                request.id,
                PermissionResolution(
                    decision="approved" if allowed else "rejected",
                    feedback=None if allowed else (decision.reason or "User denied."),
                ),
                request.team_name,
            )
            resolved += 1
            await print_system(
                f"Permission {'approved' if allowed else 'denied'} for "
                f"{request.worker_id}: {request.tool_name}"
            )
        return resolved

    async def _replace_loop(self, new_loop: AgentLoop) -> None:
        old_mcp = getattr(self.loop, "_mcp_manager", None)
        if old_mcp is not None:
            try:
                await asyncio.shield(asyncio.wait_for(old_mcp.close(), timeout=5))
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        self.loop = new_loop
        self.commands = self._build_command_registry()
        self.sync_state()

    def sync_state(self) -> AppState:
        """Refresh and publish the observable runtime state."""
        return self.state_store.set(**self._build_state().__dict__)

    def _build_state(self) -> AppState:
        mcp_connected = 0
        mcp_failed = 0
        mcp = getattr(self.loop, "_mcp_manager", None)
        if mcp is not None:
            try:
                statuses = mcp.list_statuses()
                mcp_connected = sum(1 for item in statuses if item.state == "connected")
                mcp_failed = sum(1 for item in statuses if item.state == "failed")
            except Exception:
                mcp_connected = 0
                mcp_failed = 0

        provider = self.settings.provider.name or "unknown"
        base_url = self.settings.provider.base_url or ""
        return AppState(
            model=self.loop.model,
            permission_mode=self.loop.permissions.mode,
            theme="default",
            cwd=str(self.cwd),
            session_id=self.loop.session_id or "",
            provider=provider,
            base_url=base_url,
            mcp_connected=mcp_connected,
            mcp_failed=mcp_failed,
        )

    def task_snapshots(self) -> list[TaskSnapshot]:
        """Return UI-safe snapshots for session and background tasks."""
        snapshots: list[TaskSnapshot] = []

        try:
            for item in self.loop.task_manager.list_tasks():
                snapshots.append(TaskSnapshot(
                    id=item.id,
                    type="session_task",
                    status=item.status,
                    description=item.content,
                    metadata={},
                ))
        except Exception:
            pass

        try:
            from miniharness.tasks import get_background_task_manager

            for record in get_background_task_manager().list_tasks():
                metadata = dict(record.metadata)
                if record.return_code is not None:
                    metadata["return_code"] = str(record.return_code)
                snapshots.append(TaskSnapshot(
                    id=record.id,
                    type=record.type,
                    status=record.status,
                    description=record.description,
                    metadata=metadata,
                ))
        except Exception:
            pass

        return snapshots

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
        reg.register("tasks", cmd_tasks, description="Show current task list", source="builtin")
        reg.register("agents", cmd_agents, description="List or inspect delegated agent definitions", source="builtin")
        reg.register("tokens", cmd_tokens, description="Show current context token budget", source="builtin")
        reg.register("diff", cmd_diff, description="Show git diff output", source="builtin")
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
            ask_user_prompt=self.ask_user_prompt,
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


def _format_swarm_permission_prompt(request) -> str:
    worker = request.worker_id or request.worker_name or "worker"
    detail = request.description or f"Allow {request.tool_name}?"
    return f"Allow delegated agent {worker} to run {request.tool_name}? {detail}"
