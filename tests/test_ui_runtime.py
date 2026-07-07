import asyncio
from pathlib import Path

import pytest

from miniharness.config.settings import Settings
from miniharness.messages import Message
from miniharness.runtime import AssistantCompleteEvent, AssistantDeltaEvent
from miniharness.sessions.storage import load_latest_session, save_session_snapshot
from miniharness.ui.backend_host import BackendHost
from miniharness.ui.protocol import AssistantComplete, LineComplete, SystemMessage, TokenUsageEvent
from miniharness.ui.runtime import RuntimeController
from miniharness.ui.tui import MiniHarnessTUI, PermissionModal


@pytest.mark.asyncio
async def test_runtime_slash_command_does_not_call_agent(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    system_messages: list[str] = []
    agent_calls: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        agent_calls.append(prompt)
        return "should not run"

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        should_continue = await runtime.handle_line(
            "/history",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert should_continue is True
    assert agent_calls == []
    assert any("Conversation has" in msg for msg in system_messages)


@pytest.mark.asyncio
async def test_runtime_tokens_command_reports_budget_without_agent_call(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    system_messages: list[str] = []
    agent_calls: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        agent_calls.append(prompt)
        return "should not run"

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        should_continue = await runtime.handle_line(
            "/tokens",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert should_continue is True
    assert agent_calls == []
    assert any("Context Token Budget" in msg for msg in system_messages)
    assert any("tokenizer:" in msg for msg in system_messages)


@pytest.mark.asyncio
async def test_runtime_diff_command_uses_shared_git_diff_without_agent_call(tmp_path: Path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    target = tmp_path / "tracked.txt"
    target.write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.email=a@example.com", "-c", "user.name=A", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    target.write_text("hello\nworld\n", encoding="utf-8")

    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    system_messages: list[str] = []
    agent_calls: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        agent_calls.append(prompt)
        return "should not run"

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        should_continue = await runtime.handle_line(
            "/diff full",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert should_continue is True
    assert agent_calls == []
    assert any("diff --git" in msg for msg in system_messages)
    assert any("+world" in msg for msg in system_messages)


@pytest.mark.asyncio
async def test_runtime_agents_command_lists_agent_definitions(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    system_messages: list[str] = []
    agent_calls: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        agent_calls.append(prompt)
        return "should not run"

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        should_continue = await runtime.handle_line(
            "/agents verification",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert should_continue is True
    assert agent_calls == []
    assert any("Agent: verification" in msg for msg in system_messages)
    assert any("VERDICT:" in msg for msg in system_messages)


@pytest.mark.asyncio
async def test_runtime_prompt_calls_agent_and_saves_session(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    prompts: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        prompts.append(prompt)
        loop.conversation.append(Message(role="user", content=prompt))
        return "ok"

    async def print_system(message: str) -> None:
        pass

    try:
        await runtime.handle_line(
            "hello",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    latest = load_latest_session(str(tmp_path))
    assert prompts == ["hello"]
    assert latest is not None
    assert latest["session_id"] == runtime.loop.session_id


@pytest.mark.asyncio
async def test_runtime_memory_extraction_reports_via_system_message(tmp_path: Path, monkeypatch):
    from miniharness.services.memory_extractor import ExtractionResult

    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    system_messages: list[str] = []

    async def fake_extract_memories_from_turn(**kwargs):
        return ExtractionResult(
            facts=[{"fact": "MiniHarness TUI uses shared RuntimeController", "tags": ["ui"]}],
            episode={
                "task": "Unified UI runtime",
                "summary": "Moved memory extraction into the shared runtime pipeline.",
                "outcome": "success",
            },
        )

    monkeypatch.setattr(
        "miniharness.services.memory_extractor.extract_memories_from_turn",
        fake_extract_memories_from_turn,
    )

    async def run_agent(loop, prompt: str) -> str:
        loop.conversation.append(Message(role="user", content=prompt))
        loop.conversation.append(Message(role="assistant", content="done"))
        return "done"

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        await runtime.handle_line(
            "build the tui",
            run_agent=run_agent,
            print_system=print_system,
        )
        await runtime.drain_background_tasks()
    finally:
        await runtime.close()

    assert any("Memory updated:" in message for message in system_messages)
    assert any("Unified UI runtime" in message for message in system_messages)


@pytest.mark.asyncio
async def test_runtime_resume_switches_loop_without_agent_call(tmp_path: Path):
    save_session_snapshot(
        cwd=str(tmp_path),
        model="gpt",
        messages=[{"role": "user", "content": "target session"}],
        session_id="target-session",
    )
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    original_session = runtime.loop.session_id
    agent_calls: list[str] = []
    system_messages: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        agent_calls.append(prompt)
        return "should not run"

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        await runtime.handle_line(
            "/resume target-session",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert agent_calls == []
    assert runtime.loop.session_id == "target-session"
    assert runtime.loop.session_id != original_session
    assert any("Restored session target-session" in msg for msg in system_messages)


def test_tui_stylesheets_parse():
    from textual.css.stylesheet import Stylesheet

    for css in (MiniHarnessTUI.CSS, PermissionModal.DEFAULT_CSS):
        sheet = Stylesheet()
        sheet.add_source(css)
        sheet.parse()


def test_tui_startup_requests_resume_before_prompt(tmp_path: Path):
    app = MiniHarnessTUI(cwd=tmp_path, prompt="continue work", resume_session_id="latest")

    assert app._startup_requests() == [
        {"type": "submit_line", "line": "/resume"},
        {"type": "submit_line", "line": "continue work"},
    ]


def test_tui_startup_requests_named_resume(tmp_path: Path):
    app = MiniHarnessTUI(cwd=tmp_path, resume_session_id="abc123")

    assert app._startup_requests() == [
        {"type": "submit_line", "line": "/resume abc123"},
    ]


@pytest.mark.asyncio
async def test_backend_permission_prompt_round_trip(tmp_path: Path):
    host = BackendHost(cwd=tmp_path, settings=Settings())
    emitted: list[object] = []
    host._emit = emitted.append  # type: ignore[method-assign]

    task = asyncio.create_task(host._ask_permission("write_file", "Allow write?"))
    await asyncio.sleep(0)

    assert emitted
    event = emitted[0]
    request_id = event.request_id
    assert event.type == "permission_request"
    assert event.tool_name == "write_file"

    host._handle_permission_response(request_id, True)

    assert await task is True


@pytest.mark.asyncio
async def test_backend_uses_runtime_for_slash_commands(tmp_path: Path):
    host = BackendHost(cwd=tmp_path, settings=Settings())
    emitted: list[object] = []
    host._emit = emitted.append  # type: ignore[method-assign]
    host._runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    try:
        assert await host._handle_line("/history") is True
    finally:
        await host._shutdown()

    assert any(isinstance(evt, SystemMessage) and "Conversation has" in evt.message for evt in emitted)
    assert any(isinstance(evt, LineComplete) for evt in emitted)


@pytest.mark.asyncio
async def test_backend_run_agent_relays_runtime_events(tmp_path: Path):
    host = BackendHost(cwd=tmp_path, settings=Settings())
    emitted: list[object] = []
    host._emit = emitted.append  # type: ignore[method-assign]
    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        event_bus=host.event_bus,
    )

    async def fake_run(prompt: str) -> str:
        runtime.loop.conversation.append(Message(role="user", content=prompt))
        await runtime.loop._emit_event(AssistantDeltaEvent(text="ok"))
        response = Message(role="assistant", content="ok")
        runtime.loop.conversation.append(response)
        runtime.loop.last_context_stats = runtime.loop.budget.snapshot(
            runtime.loop.conversation.to_openai(),
            tools=[],
        )
        await runtime.loop._emit_event(AssistantCompleteEvent(text="ok"))
        return response.content or ""

    runtime.loop.run = fake_run  # type: ignore[method-assign]

    try:
        result = await host._run_agent(runtime.loop, "hello")
    finally:
        await runtime.close()

    assert result == "ok"
    assert any(getattr(evt, "type", "") == "assistant_delta" and evt.text == "ok" for evt in emitted)
    assert any(isinstance(evt, AssistantComplete) and evt.text == "ok" for evt in emitted)
    assert any(isinstance(evt, TokenUsageEvent) and evt.token_count > 0 for evt in emitted)


@pytest.mark.asyncio
async def test_backend_agent_error_result_emits_error_event(tmp_path: Path):
    host = BackendHost(cwd=tmp_path, settings=Settings())
    emitted: list[object] = []
    host._emit = emitted.append  # type: ignore[method-assign]
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    async def fake_run(prompt: str) -> str:
        runtime.loop.conversation.append(Message(role="user", content=prompt))
        return "API error: boom"

    runtime.loop.run = fake_run  # type: ignore[method-assign]

    try:
        result = await host._run_agent(runtime.loop, "hello")
    finally:
        await runtime.close()

    assert result == "API error: boom"
    assert any(getattr(evt, "type", "") == "error" and "boom" in evt.message for evt in emitted)
    assert not any(isinstance(evt, AssistantComplete) for evt in emitted)
