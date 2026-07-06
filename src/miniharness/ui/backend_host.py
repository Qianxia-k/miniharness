"""Backend host — stdin/stdout JSON-lines ↔ shared runtime bridge.

Each line of stdin is a JSON request.  Each response / streaming event
is a ``MHJSON:`` line on stdout.

This mirrors the production OpenHarness shape: the frontend does not call the
agent loop directly.  It talks to a backend host, and the backend host delegates
all line handling to the shared runtime controller used by interactive modes.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import uuid
from typing import Any, Coroutine
from pathlib import Path

from miniharness.config.settings import Settings
from miniharness.loop import AgentLoop
from miniharness.runtime import (
    AssistantCompleteEvent,
    AssistantDeltaEvent,
    CompactProgressRuntimeEvent,
    ErrorRuntimeEvent,
    LineCompleteEvent,
    PermissionRequestEvent,
    ReadyRuntimeEvent,
    RuntimeEvent,
    RuntimeEventBus,
    ShutdownRuntimeEvent,
    StatusRuntimeEvent,
    SystemRuntimeEvent,
    TokenUsageRuntimeEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    UserQuestionRequestEvent,
)
from miniharness.ui.protocol import (
    AssistantComplete,
    AssistantDelta,
    CompactProgressEvent,
    ErrorEvent,
    LineComplete,
    PermissionRequest,
    ReadyEvent,
    ShutdownEvent,
    StateSnapshot,
    StatusEvent,
    SystemMessage,
    TasksSnapshot,
    TokenUsageEvent,
    ToolCompleted,
    ToolStarted,
    UserQuestionRequest,
    decode_message,
    encode_event,
)
from miniharness.ui.runtime import RuntimeController
from miniharness.state import AppState


class BackendHost:
    """Run AgentLoop, speak JSON-lines on stdin/stdout."""

    def __init__(self, *, cwd: Path, settings: Settings) -> None:
        self.cwd = cwd
        self.settings = settings
        self._runtime: RuntimeController | None = None
        self._pending_perm: dict[str, asyncio.Future[bool]] = {}
        self._pending_questions: dict[str, asyncio.Future[str]] = {}
        self._request_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._active_request_task: asyncio.Task[bool] | None = None
        self.event_bus = RuntimeEventBus()
        self.event_bus.subscribe(self._emit_protocol_event)
        self._busy = False
        self._running = True
        self._state_unsubscribe = None

    async def run(self) -> None:
        self._runtime = RuntimeController(
            cwd=self.cwd,
            settings=self.settings,
            permission_prompt=self._ask_permission,
            ask_user_prompt=self._ask_user_question,
            compact_progress=self._emit_compact_progress,
            event_bus=self.event_bus,
        )
        self._state_unsubscribe = self._runtime.state_store.subscribe(
            self._emit_state_snapshot
        )
        await self._runtime.start()

        await self._publish(ReadyRuntimeEvent(
            model=self._runtime.loop.model,
            cwd=str(self.cwd),
            session_id=self._runtime.loop.session_id or "",
        ))
        self._emit_state_snapshot(self._runtime.state_store.get())
        self._emit_tasks_snapshot()

        self._start_reader_thread(asyncio.get_running_loop())
        try:
            while self._running:
                msg = await self._request_queue.get()
                request_type = msg.get("type", "")

                if request_type == "shutdown":
                    await self._publish(ShutdownRuntimeEvent())
                    break
                if request_type == "interrupt":
                    await self._interrupt_active_request()
                    continue
                if request_type != "submit_line":
                    await self._publish(ErrorRuntimeEvent(message=f"Unknown request type: {request_type}"))
                    continue

                if self._busy:
                    await self._publish(ErrorRuntimeEvent(message="Session is busy"))
                    continue

                line = msg.get("line", "")
                if not line.strip():
                    continue

                self._busy = True
                try:
                    should_continue = await self._run_active_request(self._handle_line(line))
                finally:
                    self._busy = False

                if not should_continue:
                    await self._publish(ShutdownRuntimeEvent())
                    break
        finally:
            self._running = False
            await self._shutdown()

    def _start_reader_thread(self, loop: asyncio.AbstractEventLoop) -> threading.Thread:
        """Read frontend requests without tying process shutdown to stdin EOF."""
        thread = threading.Thread(
            target=self._read_requests_sync,
            args=(loop,),
            name="miniharness-ui-stdin",
            daemon=True,
        )
        thread.start()
        return thread

    def _read_requests_sync(self, loop: asyncio.AbstractEventLoop) -> None:
        while self._running:
            raw = sys.stdin.readline()
            if raw == "":
                loop.call_soon_threadsafe(self._request_queue.put_nowait, {"type": "shutdown"})
                return
            msg = decode_message(raw.strip())
            if msg is None:
                continue

            request_type = msg.get("type", "")
            if request_type == "permission_response":
                loop.call_soon_threadsafe(
                    self._handle_permission_response,
                    msg.get("request_id", ""),
                    msg.get("allowed", False),
                )
                continue
            if request_type == "user_question_response":
                loop.call_soon_threadsafe(
                    self._handle_user_question_response,
                    msg.get("request_id", ""),
                    msg.get("answer", ""),
                )
                continue
            if request_type == "interrupt":
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._interrupt_active_request())
                )
                continue
            if request_type == "shutdown":
                loop.call_soon_threadsafe(self._request_queue.put_nowait, msg)
                return

            loop.call_soon_threadsafe(self._request_queue.put_nowait, msg)

    async def _run_active_request(self, awaitable: Coroutine[Any, Any, bool]) -> bool:
        task = asyncio.create_task(awaitable)
        self._active_request_task = task
        try:
            return await task
        except asyncio.CancelledError:
            await self._publish(SystemRuntimeEvent(message="Interrupted by user."))
            await self._publish(LineCompleteEvent())
            return True
        finally:
            if self._active_request_task is task:
                self._active_request_task = None

    async def _interrupt_active_request(self) -> None:
        task = self._active_request_task
        if task is None or task.done():
            return
        task.cancel()

    async def _handle_line(self, line: str) -> bool:
        if not line.strip():
            return True
        runtime = self._runtime
        if runtime is None:
            return False
        try:
            should_continue = await runtime.handle_line(
                line,
                run_agent=self._run_agent,
                print_system=self._print_system,
                clear_output=self._clear_output,
            )
            await self._publish(StatusRuntimeEvent(
                message=_format_runtime_status(runtime)
            ))
            self._emit_tasks_snapshot()
            return should_continue
        except Exception as exc:
            await self._publish(ErrorRuntimeEvent(message=str(exc)))
            return True
        finally:
            await self._publish(LineCompleteEvent())

    async def _run_agent(self, loop: AgentLoop, prompt: str) -> str:
        """Run one agent turn and translate loop side effects into UI events."""
        try:
            return await self._run_agent_impl(loop, prompt)
        except asyncio.CancelledError:
            await self._publish(SystemRuntimeEvent(message="Cancelled."))
            return "Cancelled."

    async def _run_agent_impl(self, loop: AgentLoop, prompt: str) -> str:
        result = await loop.run(prompt)

        await self._emit_token_usage(loop)

        if _is_error_result(result):
            await self._publish(ErrorRuntimeEvent(message=result))
            return result

        return result

    async def _print_system(self, message: str) -> None:
        await self._publish(SystemRuntimeEvent(message=message))

    async def _clear_output(self) -> None:
        await self._publish(SystemRuntimeEvent(message="Conversation cleared."))

    async def _emit_compact_progress(self, event: dict[str, Any]) -> None:
        detail = {
            key: value for key, value in event.items()
            if key not in {
                "phase",
                "tier",
                "token_count",
                "soft_limit",
                "usage_ratio",
                "compacted",
                "tokens_after",
            }
        }
        await self._publish(CompactProgressRuntimeEvent(
            phase=str(event.get("phase") or ""),
            tier=str(event.get("tier") or ""),
            token_count=int(event.get("token_count") or 0),
            soft_limit=int(event.get("soft_limit") or 0),
            usage_ratio=float(event.get("usage_ratio") or 0.0),
            compacted=bool(event.get("compacted", False)),
            tokens_after=int(event.get("tokens_after") or 0),
            detail=detail,
        ))

    async def _emit_token_usage(self, loop: AgentLoop) -> None:
        stats = getattr(loop, "last_context_stats", {}) or {}
        if not stats:
            return
        await self._publish(TokenUsageRuntimeEvent(
            token_count=int(stats.get("token_count") or stats.get("total_used") or 0),
            context_window=int(stats.get("context_window") or loop.budget.total),
            soft_limit=int(stats.get("soft_limit") or loop.budget.max_tokens),
            usage_ratio=float(stats.get("usage_ratio") or stats.get("budget_ratio") or 0.0),
            message_tokens=int(stats.get("message_tokens") or 0),
            tool_tokens=int(stats.get("tool_tokens") or 0),
            response_reserve_tokens=int(stats.get("response_reserve_tokens") or 0),
            available=int(stats.get("available") or 0),
            tokenizer=str(stats.get("tokenizer") or ""),
            model=str(stats.get("model") or loop.model),
        ))

    def _handle_permission_response(self, request_id: str, allowed: bool) -> None:
        fut = self._pending_perm.pop(request_id, None)
        if fut and not fut.done():
            fut.set_result(allowed)

    async def _ask_permission(self, tool_name: str, prompt: str) -> bool:
        """Request permission from the frontend and wait for response."""
        req_id = uuid.uuid4().hex[:8]
        fut: asyncio.Future[bool] = asyncio.Future()
        self._pending_perm[req_id] = fut
        await self._publish(PermissionRequestEvent(
            request_id=req_id,
            tool_name=tool_name,
            prompt=prompt,
        ))
        try:
            return await asyncio.wait_for(fut, timeout=300)
        except asyncio.TimeoutError:
            return False

    def _handle_user_question_response(self, request_id: str, answer: str) -> None:
        fut = self._pending_questions.pop(request_id, None)
        if fut and not fut.done():
            fut.set_result(str(answer))

    async def _ask_user_question(self, question: str) -> str:
        """Request an answer from the frontend and wait for response."""
        req_id = uuid.uuid4().hex[:8]
        fut: asyncio.Future[str] = asyncio.Future()
        self._pending_questions[req_id] = fut
        await self._publish(UserQuestionRequestEvent(
            request_id=req_id,
            question=question,
        ))
        try:
            return await asyncio.wait_for(fut, timeout=300)
        except asyncio.TimeoutError:
            return ""

    async def _shutdown(self) -> None:
        if self._state_unsubscribe is not None:
            self._state_unsubscribe()
            self._state_unsubscribe = None
        if self._runtime is not None:
            await self._runtime.close()
            self._runtime = None

    async def _publish(self, event: RuntimeEvent) -> None:
        await self.event_bus.emit(event)

    def _emit_protocol_event(self, event: RuntimeEvent) -> None:
        protocol_event = _runtime_to_protocol_event(event)
        if protocol_event is not None:
            self._emit(protocol_event)

    def _emit(self, event) -> None:
        sys.stdout.write(encode_event(event) + "\n")
        sys.stdout.flush()

    def _emit_state_snapshot(self, state: AppState) -> None:
        self._emit(StateSnapshot(state=_state_payload(state)))

    def _emit_tasks_snapshot(self) -> None:
        if self._runtime is None:
            return
        self._emit(TasksSnapshot(tasks=self._runtime.task_snapshots()))


def _is_error_result(result: str) -> bool:
    """
    检查结果字符串是否为已知的错误类型。
    
    Args:
        result (str): LLM 或工具返回的结果字符串
    
    Returns:
        bool: 若结果以已知错误前缀开头则返回 True，否则返回 False
    """
    return result.startswith((
        "API error:",
        "Network error:",
        "Error:",
        "Hook blocked:",
        "No response from model.",
        "Reached maximum turns",
    ))


def _format_runtime_status(runtime: RuntimeController) -> str:
    state = runtime.state_store.get()
    parts = [
        f"Model: {state.model}",
        f"Session: {state.session_id[:12]}",
        f"Mode: {state.permission_mode}",
    ]
    if state.mcp_connected or state.mcp_failed:
        parts.append(f"MCP: {state.mcp_connected} connected/{state.mcp_failed} failed")
    return "  |  ".join(parts)


def _state_payload(state: AppState) -> dict[str, Any]:
    return {
        "model": state.model,
        "cwd": state.cwd,
        "session_id": state.session_id,
        "provider": state.provider,
        "auth_status": state.auth_status,
        "base_url": state.base_url,
        "permission_mode": state.permission_mode,
        "theme": state.theme,
        "vim_enabled": state.vim_enabled,
        "voice_enabled": state.voice_enabled,
        "voice_available": state.voice_available,
        "voice_reason": state.voice_reason,
        "fast_mode": state.fast_mode,
        "effort": state.effort,
        "passes": state.passes,
        "mcp_connected": state.mcp_connected,
        "mcp_failed": state.mcp_failed,
        "bridge_sessions": state.bridge_sessions,
        "output_style": state.output_style,
        "keybindings": dict(state.keybindings),
    }


def _runtime_to_protocol_event(event: RuntimeEvent):
    if isinstance(event, ReadyRuntimeEvent):
        return ReadyEvent(model=event.model, cwd=event.cwd, session_id=event.session_id)
    if isinstance(event, AssistantDeltaEvent):
        return AssistantDelta(text=event.text)
    if isinstance(event, AssistantCompleteEvent):
        return AssistantComplete(text=event.text)
    if isinstance(event, ToolStartedEvent):
        return ToolStarted(tool_name=event.tool_name, tool_input=event.tool_input)
    if isinstance(event, ToolCompletedEvent):
        return ToolCompleted(
            tool_name=event.tool_name,
            output=event.output,
            is_error=event.is_error,
        )
    if isinstance(event, PermissionRequestEvent):
        return PermissionRequest(
            request_id=event.request_id,
            tool_name=event.tool_name,
            prompt=event.prompt,
        )
    if isinstance(event, UserQuestionRequestEvent):
        return UserQuestionRequest(
            request_id=event.request_id,
            question=event.question,
        )
    if isinstance(event, ErrorRuntimeEvent):
        return ErrorEvent(message=event.message)
    if isinstance(event, StatusRuntimeEvent):
        return StatusEvent(message=event.message)
    if isinstance(event, SystemRuntimeEvent):
        return SystemMessage(message=event.message)
    if isinstance(event, TokenUsageRuntimeEvent):
        return TokenUsageEvent(
            token_count=event.token_count,
            context_window=event.context_window,
            soft_limit=event.soft_limit,
            usage_ratio=event.usage_ratio,
            message_tokens=event.message_tokens,
            tool_tokens=event.tool_tokens,
            response_reserve_tokens=event.response_reserve_tokens,
            available=event.available,
            tokenizer=event.tokenizer,
            model=event.model,
        )
    if isinstance(event, CompactProgressRuntimeEvent):
        return CompactProgressEvent(
            phase=event.phase,
            tier=event.tier,
            token_count=event.token_count,
            soft_limit=event.soft_limit,
            usage_ratio=event.usage_ratio,
            compacted=event.compacted,
            tokens_after=event.tokens_after,
            detail=event.detail,
        )
    if isinstance(event, LineCompleteEvent):
        return LineComplete()
    if isinstance(event, ShutdownRuntimeEvent):
        return ShutdownEvent()
    return None
