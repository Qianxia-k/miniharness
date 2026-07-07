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
    assert registry.get("git_status") is not None
    assert registry.get("git_diff") is not None
    assert registry.get("enter_worktree") is not None
    assert registry.get("exit_worktree") is not None
    assert registry.get("lsp") is not None
    assert registry.get("sleep") is not None
    assert registry.get("ask_user_question") is not None
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
    assert registry.get("enter_plan_mode") is not None
    assert registry.get("exit_plan_mode") is not None
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
    assert "+ok" in prompts[0][1]
    assert "(+1 -0)" in prompts[0][1]


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
async def test_write_file_can_refuse_missing_parent_when_requested(tmp_path: Path):
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = await registry.execute("write_file", {
        "path": "missing/child.txt",
        "content": "nope",
        "create_directories": False,
    })

    assert result.is_error is True
    assert "Parent directory does not exist" in result.output
    assert not (tmp_path / "missing").exists()


@pytest.mark.asyncio
async def test_write_file_rejects_overwriting_binary_file(tmp_path: Path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"abc\x00def")
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = await registry.execute("write_file", {
        "path": "data.bin",
        "content": "text",
    })

    assert result.is_error is True
    assert "Binary file" in result.output
    assert target.read_bytes() == b"abc\x00def"


@pytest.mark.asyncio
async def test_read_file_returns_numbered_limited_range(tmp_path: Path):
    (tmp_path / "sample.py").write_text(
        "one\n"
        "two\n"
        "three\n"
        "four\n",
        encoding="utf-8",
    )
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
    )

    result = await registry.execute("read_file", {
        "path": "sample.py",
        "offset": 1,
        "limit": 2,
    })

    assert result.is_error is False
    assert result.output == "     2\ttwo\n     3\tthree"


@pytest.mark.asyncio
async def test_read_file_rejects_binary_files(tmp_path: Path):
    (tmp_path / "data.bin").write_bytes(b"abc\x00def")
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
    )

    result = await registry.execute("read_file", {"path": "data.bin"})

    assert result.is_error is True
    assert "Binary file" in result.output


@pytest.mark.asyncio
async def test_edit_file_permission_prompt_includes_diff(tmp_path: Path):
    target = tmp_path / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")
    prompts: list[tuple[str, str]] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append((tool_name, prompt))
        return True

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("edit_file", {
        "path": "app.py",
        "old_str": "print('old')",
        "new_str": "print('new')",
    })

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "print('new')\n"
    assert prompts[0][0] == "edit_file"
    assert "Allow edit_file to update" in prompts[0][1]
    assert "+print('new')" in prompts[0][1]
    assert "-print('old')" in prompts[0][1]
    assert "(+1 -1)" in prompts[0][1]


@pytest.mark.asyncio
async def test_edit_file_rejects_binary_files(tmp_path: Path):
    target = tmp_path / "data.bin"
    target.write_bytes(b"abc\x00def")
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = await registry.execute("edit_file", {
        "path": "data.bin",
        "old_str": "abc",
        "new_str": "xyz",
    })

    assert result.is_error is True
    assert "Binary file" in result.output
    assert target.read_bytes() == b"abc\x00def"


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
async def test_git_status_reports_repository_state_without_permission_prompt(tmp_path: Path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "tracked.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "tracked.txt").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
    prompts: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append(tool_name)
        return False

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("git_status", {
        "include_diff_stat": True,
        "max_entries": 10,
    })

    assert result.is_error is False
    assert "Repository:" in result.output
    assert "Branch:" in result.output
    assert "tracked.txt" in result.output
    assert "untracked.txt" in result.output
    assert "Unstaged diff stat:" in result.output
    assert prompts == []


@pytest.mark.asyncio
async def test_git_status_rejects_non_repository(tmp_path: Path):
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
    )

    result = await registry.execute("git_status", {})

    assert result.is_error is True
    assert "requires a git repository" in result.output


@pytest.mark.asyncio
async def test_git_diff_reports_unstaged_changes_without_permission_prompt(tmp_path: Path):
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
    prompts: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append(tool_name)
        return False

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("git_diff", {
        "scope": "unstaged",
        "stat_only": False,
    })

    assert result.is_error is False
    assert "diff --git" in result.output
    assert "+world" in result.output
    assert prompts == []


@pytest.mark.asyncio
async def test_git_diff_rejects_non_repository(tmp_path: Path):
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
    )

    result = await registry.execute("git_diff", {})

    assert result.is_error is True
    assert "requires a git repository" in result.output


@pytest.mark.asyncio
async def test_enter_worktree_permission_denial_blocks_creation(tmp_path: Path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "tracked.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.email=a@example.com", "-c", "user.name=A", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        return False

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("enter_worktree", {
        "branch": "feature/demo",
    })

    assert result.is_error is True
    assert "User denied" in result.output
    assert not (tmp_path / ".miniharness" / "worktrees" / "feature-demo").exists()


@pytest.mark.asyncio
async def test_enter_and_exit_worktree_manage_git_worktree(tmp_path: Path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "tracked.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.email=a@example.com", "-c", "user.name=A", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    create = await registry.execute("enter_worktree", {
        "branch": "feature/demo",
    })

    worktree_path = tmp_path / ".miniharness" / "worktrees" / "feature-demo"
    assert create.is_error is False
    assert f"Path: {worktree_path}" in create.output
    assert worktree_path.exists()
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert git_dir.stdout.strip()

    remove = await registry.execute("exit_worktree", {
        "path": str(worktree_path),
    })

    assert remove.is_error is False
    assert not worktree_path.exists()


@pytest.mark.asyncio
async def test_lsp_inspects_python_symbols_without_permission_prompt(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text(
        "def greet(name: str) -> str:\n"
        "    \"\"\"Return a greeting.\"\"\"\n"
        "    return f'Hello {name}'\n"
        "\n"
        "class Greeter:\n"
        "    def speak(self, name: str) -> str:\n"
        "        return greet(name)\n",
        encoding="utf-8",
    )
    (pkg / "app.py").write_text(
        "from pkg.utils import greet\n"
        "\n"
        "def run() -> str:\n"
        "    return greet('Ada')\n",
        encoding="utf-8",
    )
    prompts: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append(tool_name)
        return False

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    symbols = await registry.execute("lsp", {
        "operation": "document_symbol",
        "file_path": "pkg/utils.py",
    })
    workspace = await registry.execute("lsp", {
        "operation": "workspace_symbol",
        "query": "greet",
    })
    definition = await registry.execute("lsp", {
        "operation": "go_to_definition",
        "file_path": "pkg/app.py",
        "symbol": "greet",
    })
    references = await registry.execute("lsp", {
        "operation": "find_references",
        "file_path": "pkg/app.py",
        "symbol": "greet",
    })
    hover = await registry.execute("lsp", {
        "operation": "hover",
        "file_path": "pkg/app.py",
        "symbol": "greet",
    })

    assert symbols.is_error is False
    assert "function greet - pkg/utils.py:1:1" in symbols.output
    assert "class Greeter - pkg/utils.py:5:1" in symbols.output
    assert workspace.is_error is False
    assert "function greet - pkg/utils.py:1:1" in workspace.output
    assert definition.is_error is False
    assert "function greet - pkg/utils.py:1:1" in definition.output
    assert references.is_error is False
    assert "pkg/app.py:4:return greet('Ada')" in references.output
    assert hover.is_error is False
    assert "signature: def greet(name)" in hover.output
    assert "docstring: Return a greeting." in hover.output
    assert prompts == []


@pytest.mark.asyncio
async def test_lsp_records_carryover_when_executed_by_agent_loop(tmp_path: Path):
    (tmp_path / "app.py").write_text(
        "def main() -> str:\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    try:
        await runtime.loop._execute_tools([
            {
                "id": "call-lsp",
                "type": "function",
                "function": {
                    "name": "lsp",
                    "arguments": '{"operation":"document_symbol","file_path":"app.py"}',
                },
            }
        ])
    finally:
        await runtime.close()

    assert "app.py" in runtime.loop.tool_metadata["task_focus_state"]["active_artifacts"]
    assert any(
        "Ran lsp document_symbol for app.py" in entry
        for entry in runtime.loop.tool_metadata["recent_verified_work"]
    )


@pytest.mark.asyncio
async def test_sleep_tool_is_read_only_and_does_not_prompt(tmp_path: Path):
    prompts: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append(tool_name)
        return False

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        permission_prompt=permission_prompt,
    )

    result = await registry.execute("sleep", {"seconds": 0.0})

    assert result.is_error is False
    assert result.output == "Slept for 0.0 seconds"
    assert prompts == []


@pytest.mark.asyncio
async def test_ask_user_question_returns_unavailable_without_frontend_callback(tmp_path: Path):
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
    )

    result = await registry.execute("ask_user_question", {"question": "Which branch?"})

    assert result.is_error is True
    assert "unavailable" in result.output


@pytest.mark.asyncio
async def test_ask_user_question_uses_frontend_callback(tmp_path: Path):
    questions: list[str] = []

    async def ask_user_prompt(question: str) -> str:
        questions.append(question)
        return "feature/login"

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
        ask_user_prompt=ask_user_prompt,
    )

    result = await registry.execute("ask_user_question", {"question": "Which branch?"})

    assert result.is_error is False
    assert result.output == "feature/login"
    assert questions == ["Which branch?"]


@pytest.mark.asyncio
async def test_runtime_controller_wires_ask_user_question_into_agent_loop(tmp_path: Path):
    questions: list[str] = []

    async def ask_user_prompt(question: str) -> str:
        questions.append(question)
        return "pytest"

    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        ask_user_prompt=ask_user_prompt,
    )

    try:
        await runtime.loop._execute_tools([
            {
                "id": "call-question",
                "type": "function",
                "function": {
                    "name": "ask_user_question",
                    "arguments": '{"question":"Which test should I run?"}',
                },
            }
        ])
    finally:
        await runtime.close()

    assert questions == ["Which test should I run?"]
    assert runtime.loop.conversation.messages[-1].content == "pytest"
    assert any(
        "Asked user a follow-up question" in entry
        for entry in runtime.loop.tool_metadata["recent_work_log"]
    )


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


@pytest.mark.asyncio
async def test_plan_mode_tools_switch_permission_mode_and_gate_writes(tmp_path: Path):
    prompts: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        prompts.append(tool_name)
        return True

    permissions = PermissionChecker(cwd=tmp_path, mode="default")
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=permissions,
        permission_prompt=permission_prompt,
    )

    entered = await registry.execute("enter_plan_mode", {})
    blocked = await registry.execute("write_file", {
        "path": "blocked.txt",
        "content": "no",
    })
    exited = await registry.execute("exit_plan_mode", {})
    allowed = await registry.execute("write_file", {
        "path": "allowed.txt",
        "content": "ok",
    })

    assert entered.is_error is False
    assert entered.output == "Permission mode set to plan"
    assert permissions.mode == "default"
    assert blocked.is_error is True
    assert "Read-only mode" in blocked.output
    assert not (tmp_path / "blocked.txt").exists()
    assert exited.output == "Permission mode set to default"
    assert allowed.is_error is False
    assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "ok"
    assert prompts == ["write_file"]


@pytest.mark.asyncio
async def test_plan_mode_tools_record_carryover_metadata(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    try:
        await runtime.loop._execute_tools([
            {
                "id": "call-plan",
                "type": "function",
                "function": {
                    "name": "enter_plan_mode",
                    "arguments": "{}",
                },
            }
        ])
    finally:
        await runtime.close()

    assert runtime.loop.permissions.mode == "plan"
    assert runtime.loop.tool_metadata["permission_mode"] == "plan"
    assert any(
        "Entered plan mode" in entry
        for entry in runtime.loop.tool_metadata["recent_work_log"]
    )
