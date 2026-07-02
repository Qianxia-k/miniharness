from pathlib import Path

import pytest
from pydantic import BaseModel

from miniharness.config.settings import Settings
from miniharness.hooks import HookEvent, HookExecutionContext, HookExecutor, HookRegistry
from miniharness.hooks.schemas import CommandHookDefinition
from miniharness.permissions import PermissionChecker
from miniharness.runtime import RuntimeEventBus, ToolCompletedEvent
from miniharness.tools.base import BaseTool, ToolResult
from miniharness.tools.offload import offload_if_needed
from miniharness.tool_registry import create_default_registry
from miniharness.ui.runtime import RuntimeController


def test_default_registry_has_all_tools(tmp_path: Path):
    registry = create_default_registry(cwd=tmp_path, permissions=PermissionChecker(cwd=tmp_path))

    assert registry.get("read_file") is not None
    assert registry.get("ls") is not None
    assert registry.get("grep") is not None
    assert registry.get("glob") is not None
    assert registry.get("write_file") is not None
    assert registry.get("edit_file") is not None
    assert registry.get("todo_write") is not None
    assert registry.get("bash") is not None
    assert registry.get("web_fetch") is not None
    assert registry.get("task") is not None
    assert registry.get("agent") is not None
    assert registry.get("agent_list") is not None
    assert registry.get("send_message") is not None
    assert registry.get("team_create") is not None
    assert registry.get("team_list") is not None
    assert registry.get("team_delete") is not None
    assert registry.get("task_create") is not None
    assert registry.get("task_list") is not None
    assert registry.get("task_get") is not None
    assert registry.get("task_output") is not None
    assert registry.get("task_stop") is not None
    assert registry.get("task_update") is not None
    assert registry.get("memory_search") is not None
    assert registry.get("memory_add") is not None
    assert registry.get("memory_log") is not None


def test_default_registry_exposes_agent_messaging_tools_to_model(tmp_path: Path):
    registry = create_default_registry(cwd=tmp_path, permissions=PermissionChecker(cwd=tmp_path))
    names = {
        tool["function"]["name"]
        for tool in registry.to_openai_tools()
    }

    assert "agent" in names
    assert "agent_list" in names
    assert "send_message" in names
    assert "team_create" in names
    assert "team_list" in names
    assert "team_delete" in names


def test_unknown_tool(tmp_path: Path):
    """Executing an unknown tool returns an error."""
    import asyncio

    registry = create_default_registry(cwd=tmp_path, permissions=PermissionChecker(cwd=tmp_path))
    result = asyncio.run(registry.execute("nonexistent", {}))
    assert result.is_error is True
    assert "Unknown tool" in result.output


def test_bash_rejects_mermaid_edge_before_shell_creates_redirect_file(tmp_path: Path):
    import asyncio

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = asyncio.run(registry.execute("bash", {
        "command": "C[CLI] --> L[AgentLoop]",
    }))

    assert result.is_error is True
    assert "probable Markdown/Mermaid" in result.output
    assert not (tmp_path / "L[AgentLoop]").exists()


def test_bash_allows_quoted_arrow_text(tmp_path: Path):
    import asyncio

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = asyncio.run(registry.execute("bash", {
        "command": "printf '%s\\n' 'C[CLI] --> L[AgentLoop]'",
    }))

    assert result.is_error is False
    assert "C[CLI] --> L[AgentLoop]" in result.output


def test_bash_allows_normal_command(tmp_path: Path):
    import asyncio

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = asyncio.run(registry.execute("bash", {"command": "printf ok"}))

    assert result.is_error is False
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_write_file_uses_async_permission_prompt(tmp_path: Path):
    prompts: list[tuple[str, str]] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append((tool_name, prompt))
        return True

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("write_file", {
        "path": "allowed.txt",
        "content": "ok",
    })

    assert result.is_error is False
    assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "ok"
    assert prompts
    assert prompts[0][0] == "write_file"
    assert "allowed.txt" in prompts[0][1]


@pytest.mark.asyncio
async def test_write_file_async_permission_denial_blocks_write(tmp_path: Path):
    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        return False

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("write_file", {
        "path": "denied.txt",
        "content": "nope",
    })

    assert result.is_error is True
    assert "User denied" in result.output
    assert not (tmp_path / "denied.txt").exists()


@pytest.mark.asyncio
async def test_glob_lists_matching_files_without_permission_prompt(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "src" / "app.txt").write_text("ok\n", encoding="utf-8")
    prompts: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append(tool_name)
        return False

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("glob", {"pattern": "src/**/*.py"})

    assert result.is_error is False
    assert result.output == "src/app.py"
    assert prompts == []


@pytest.mark.asyncio
async def test_todo_write_updates_markdown_checklist_with_permission(tmp_path: Path):
    prompts: list[tuple[str, str]] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append((tool_name, prompt))
        return True

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    created = await registry.execute("todo_write", {"item": "wire glob tool"})
    checked = await registry.execute("todo_write", {
        "item": "wire glob tool",
        "checked": True,
    })

    assert created.is_error is False
    assert checked.is_error is False
    assert (tmp_path / "TODO.md").read_text(encoding="utf-8") == "# TODO\n- [x] wire glob tool\n"
    assert [item[0] for item in prompts] == ["todo_write", "todo_write"]


@pytest.mark.asyncio
async def test_todo_write_respects_plan_mode(tmp_path: Path):
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="plan"),
    )

    result = await registry.execute("todo_write", {"item": "should not write"})

    assert result.is_error is True
    assert "Read-only mode" in result.output
    assert not (tmp_path / "TODO.md").exists()


@pytest.mark.asyncio
async def test_permission_prompt_emits_notification_hook(tmp_path: Path):
    payload_path = tmp_path / "permission-hook.json"
    command = (
        f"python3 -c "
        f"\"import os, pathlib; pathlib.Path({str(payload_path)!r}).write_text("
        "os.environ['MINIHARNESS_HOOK_PAYLOAD'], encoding='utf-8')\""
    )
    hook_registry = HookRegistry()
    hook_registry.register(
        HookEvent.NOTIFICATION,
        CommandHookDefinition(command=command, matcher="notification"),
    )
    hook_executor = HookExecutor(
        hook_registry,
        HookExecutionContext(cwd=tmp_path),
    )

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        return True

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
        hook_executor=hook_executor,
    )

    result = await registry.execute("write_file", {
        "path": "allowed.txt",
        "content": "ok",
    })

    assert result.is_error is False
    payload = payload_path.read_text(encoding="utf-8")
    assert '"notification_type": "permission_prompt"' in payload
    assert '"tool_name": "write_file"' in payload


class LargeOutputInput(BaseModel):
    size: int = 13000


class LargeOutputTool(BaseTool):
    name = "large_output"
    description = "Return a large output for pipeline tests."
    input_model = LargeOutputInput

    async def execute(self, arguments: LargeOutputInput) -> ToolResult:
        return ToolResult("x" * arguments.size)


@pytest.mark.asyncio
async def test_agent_loop_records_offloaded_tool_artifact(tmp_path: Path):
    bus = RuntimeEventBus()
    events: list[object] = []
    bus.subscribe(events.append)
    runtime = RuntimeController(cwd=tmp_path, settings=Settings(), event_bus=bus)
    runtime.loop.tools.register(
        LargeOutputTool(cwd=tmp_path, permissions=runtime.loop.permissions)
    )

    try:
        await runtime.loop._execute_tools([
            {
                "id": "call-large",
                "type": "function",
                "function": {
                    "name": "large_output",
                    "arguments": '{"size": 13000}',
                },
            }
        ])
    finally:
        await runtime.close()

    artifacts = runtime.loop.tool_metadata["task_focus_state"]["active_artifacts"]
    assert artifacts
    assert any("large_output" in artifact for artifact in artifacts)
    assert runtime.loop.conversation.messages[-1].content.startswith("[Tool output truncated]")
    completed = [event for event in events if isinstance(event, ToolCompletedEvent)]
    assert completed[-1].artifact_path
    assert completed[-1].original_output_chars == 13000


def test_tool_output_offload_thresholds_are_env_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MINIHARNESS_TOOL_OUTPUT_INLINE_CHARS", "20")
    monkeypatch.setenv("MINIHARNESS_TOOL_OUTPUT_PREVIEW_CHARS", "5")

    inline, artifact = offload_if_needed(tool_name="bash", output="x" * 300)

    assert artifact is not None
    assert artifact.exists()
    assert "Original size: 300 chars" in inline
    assert "Inline preview (first 128 chars" in inline
