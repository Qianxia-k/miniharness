"""Backend host — stdin/stdout JSON-lines ↔ shared runtime bridge.

Each line of stdin is a JSON request.  Each response / streaming event
is a ``MHJSON:`` line on stdout.

This mirrors the production OpenHarness shape: the frontend does not call the
agent loop directly.  It talks to a backend host, and the backend host delegates
all line handling to the shared runtime controller used by interactive modes.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import uuid
from typing import Any, Coroutine
from pathlib import Path

from miniharness.config.settings import Settings
from miniharness.loop import AgentLoop
from miniharness.llm import StreamComplete, TextDelta
from miniharness.ui.protocol import (
    AssistantComplete,
    AssistantDelta,
    ErrorEvent,
    LineComplete,
    PermissionRequest,
    ReadyEvent,
    ShutdownEvent,
    StatusEvent,
    SystemMessage,
    ToolCompleted,
    ToolStarted,
    decode_message,
    encode_event,
)
from miniharness.ui.runtime import RuntimeController


class BackendHost:
    """Run AgentLoop, speak JSON-lines on stdin/stdout."""

    def __init__(self, *, cwd: Path, settings: Settings) -> None:
        self.cwd = cwd
        self.settings = settings
        self._runtime: RuntimeController | None = None
        self._pending_perm: dict[str, asyncio.Future[bool]] = {}
        self._request_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._active_request_task: asyncio.Task[bool] | None = None
        self._busy = False
        self._running = True

    async def run(self) -> None:
        self._runtime = RuntimeController(
            cwd=self.cwd,
            settings=self.settings,
            permission_prompt=self._ask_permission,
        )
        await self._runtime.start()

        self._emit(ReadyEvent(
            model=self._runtime.loop.model,
            cwd=str(self.cwd),
            session_id=self._runtime.loop.session_id or "",
        ))

        self._start_reader_thread(asyncio.get_running_loop())
        try:
            while self._running:
                msg = await self._request_queue.get()
                request_type = msg.get("type", "")

                if request_type == "shutdown":
                    self._emit(ShutdownEvent())
                    break
                if request_type == "interrupt":
                    await self._interrupt_active_request()
                    continue
                if request_type != "submit_line":
                    self._emit(ErrorEvent(message=f"Unknown request type: {request_type}"))
                    continue

                if self._busy:
                    self._emit(ErrorEvent(message="Session is busy"))
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
                    self._emit(ShutdownEvent())
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
            self._emit(SystemMessage(message="Interrupted by user."))
            self._emit(LineComplete())
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
            self._emit(StatusEvent(
                message=(
                    f"Model: {runtime.loop.model}  |  "
                    f"Session: {(runtime.loop.session_id or '')[:12]}"
                )
            ))
            return should_continue
        except Exception as exc:
            self._emit(ErrorEvent(message=str(exc)))
            return True
        finally:
            self._emit(LineComplete())

    async def _run_agent(self, loop: AgentLoop, prompt: str) -> str:
        """Run one agent turn and translate loop side effects into UI events."""
        try:
            return await self._run_agent_impl(loop, prompt)
        except asyncio.CancelledError:
            self._emit(SystemMessage(message="Cancelled."))
            return "Cancelled."

    async def _run_agent_impl(self, loop: AgentLoop, prompt: str) -> str:

        # ── Hook: LLM stream → assistant_delta events ──────────────
        original_stream = loop.llm.stream

        async def _stream_wrapper(*a, **kw):
            async for evt in original_stream(*a, **kw):
                if isinstance(evt, TextDelta):
                    self._emit(AssistantDelta(text=evt.text))
                    continue  # suppress — don't let _call_llm console.print() raw text
                yield evt

        loop.llm.stream = _stream_wrapper

        # ── Hook: tool execution → tool_started/tool_completed ────
        original_exec = loop._execute_tools

        async def _exec_wrapper(tool_calls):
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                self._emit(ToolStarted(tool_name=name, tool_input=args))

            await original_exec(tool_calls)

            msgs = loop.conversation.messages
            for tc in tool_calls:
                tc_id = tc["id"]
                name = tc["function"]["name"]
                for m in reversed(msgs):
                    if m.tool_call_id == tc_id:
                        self._emit(ToolCompleted(
                            tool_name=name,
                            output=m.content or "",
                            is_error="Error" in (m.content or ""),
                        ))
                        break

        loop._execute_tools = _exec_wrapper

        try:
            result = await loop.run(prompt)
        finally:
            loop.llm.stream = original_stream
            loop._execute_tools = original_exec

        self._emit(AssistantComplete(text=result))
        return result

    async def _print_system(self, message: str) -> None:
        self._emit(SystemMessage(message=message))

    async def _clear_output(self) -> None:
        self._emit(SystemMessage(message="Conversation cleared."))

    def _handle_permission_response(self, request_id: str, allowed: bool) -> None:
        fut = self._pending_perm.pop(request_id, None)
        if fut and not fut.done():
            fut.set_result(allowed)

    async def _ask_permission(self, tool_name: str, prompt: str) -> bool:
        """Request permission from the frontend and wait for response."""
        req_id = uuid.uuid4().hex[:8]
        fut: asyncio.Future[bool] = asyncio.Future()
        self._pending_perm[req_id] = fut
        self._emit(PermissionRequest(request_id=req_id, tool_name=tool_name, prompt=prompt))
        try:
            return await asyncio.wait_for(fut, timeout=300)
        except asyncio.TimeoutError:
            return False

    async def _shutdown(self) -> None:
        if self._runtime is not None:
            await self._runtime.close()
            self._runtime = None

    def _emit(self, event) -> None:
        sys.stdout.write(encode_event(event) + "\n")
        sys.stdout.flush()
