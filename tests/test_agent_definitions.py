from pathlib import Path

import pytest

from miniharness.coordinator.agent_definitions import (
    get_agent_definition,
    load_agents_dir,
)
from miniharness.config.settings import Settings
from miniharness.permissions import PermissionChecker
from miniharness.plugins.loader import load_plugins
from miniharness.swarm.spawn_utils import build_teammate_argv
from miniharness.tasks import (
    BackgroundTaskManager,
    reset_agent_registry_for_tests,
    reset_background_task_manager_for_tests,
    reset_team_registry_for_tests,
)
from miniharness.tasks.background import BackgroundTaskRecord
from miniharness.tool_registry import create_default_registry
from miniharness.prompts.system import assemble_system_prompt
from miniharness.swarm.spawn_utils import (
    AGENT_HOOKS_ENV_VAR,
    AGENT_ID_ENV_VAR,
    AGENT_MAX_TURNS_ENV_VAR,
    AGENT_NAME_ENV_VAR,
    AGENT_PERMISSION_MODE_ENV_VAR,
    AGENT_TEAM_ENV_VAR,
    AGENT_TOOL_POLICY_ENV_VAR,
)
from miniharness.ui.runtime import RuntimeController


def test_load_agents_dir_reads_frontmatter_and_body(tmp_path: Path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: Reviews code changes carefully.
model: inherit
disallowed_tools: write_file, edit_file
permission_mode: plan
maxTurns: 3
initialPrompt: Always inspect the diff before answering.
hooks:
  subagent_stop:
    - type: command
      command: "printf stopped"
---

You are a careful reviewer. Inspect diffs and report risks.
""",
        encoding="utf-8",
    )

    agents = load_agents_dir(agents_dir, source="project")

    assert len(agents) == 1
    agent = agents[0]
    assert agent.name == "reviewer"
    assert agent.description == "Reviews code changes carefully."
    assert agent.model == "inherit"
    assert agent.disallowed_tools == ["write_file", "edit_file"]
    assert agent.permission_mode == "plan"
    assert agent.max_turns == 3
    assert agent.initial_prompt == "Always inspect the diff before answering."
    assert agent.hooks == {
        "subagent_stop": [{"type": "command", "command": "printf stopped"}]
    }
    assert agent.system_prompt.startswith("You are a careful reviewer.")
    assert agent.source == "project"


def test_project_agent_definition_overrides_builtin(tmp_path: Path):
    agents_dir = tmp_path / ".miniharness" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "verification.md").write_text(
        """---
name: verification
description: Project-specific verifier.
---

Verify this project with its custom checks.
""",
        encoding="utf-8",
    )

    agent = get_agent_definition("verification", cwd=tmp_path)

    assert agent is not None
    assert agent.source == "project"
    assert agent.description == "Project-specific verifier."
    assert "custom checks" in agent.system_prompt


def test_system_prompt_lists_available_agent_definitions(tmp_path: Path):
    agents_dir = tmp_path / ".miniharness" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: Project reviewer for local conventions.
---

Review code using project conventions.
""",
        encoding="utf-8",
    )

    prompt = assemble_system_prompt(
        base_prompt="BASE",
        cwd=tmp_path,
        tool_count=3,
    )

    assert "# Delegation And Subagents" in prompt
    assert 'subagent_type="worker"' in prompt
    assert "Workers cannot see your parent conversation" in prompt
    assert "Do not invent worker results" in prompt
    assert "Spawn a fresh verification worker" in prompt
    assert "**worker**" in prompt
    assert "**verification**" in prompt
    assert "**reviewer**: Project reviewer for local conventions." in prompt


def test_system_prompt_lists_plugin_agent_definitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    plugin_dir = home / ".miniharness" / "plugins" / "review-pack"
    agents_dir = plugin_dir / "agents"
    agents_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        '{"name": "review-pack", "description": "Review plugin"}',
        encoding="utf-8",
    )
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: Plugin reviewer for framework rules.
---

Review code using plugin framework rules.
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    plugins = load_plugins(Settings(), cwd=tmp_path)
    plugin_index = [{
        "name": plugins[0].name,
        "description": plugins[0].description,
        "active": False,
        "skills": plugins[0].skills,
        "_plugin": plugins[0],
    }]

    prompt = assemble_system_prompt(
        base_prompt="BASE",
        cwd=tmp_path,
        tool_count=3,
        plugin_index=plugin_index,
    )

    assert "**review-pack:reviewer**: Plugin reviewer for framework rules." in prompt


def test_teammate_argv_forwards_agent_system_prompt(tmp_path: Path):
    argv = build_teammate_argv(
        cwd=tmp_path,
        model="inherit",
        system_prompt="Role line one\nRole line two",
        system_prompt_mode="append",
    )

    assert "--model" not in argv
    assert "--append-system-prompt" in argv
    assert argv[argv.index("--append-system-prompt") + 1] == "Role line one\nRole line two"


@pytest.mark.asyncio
async def test_agent_tool_uses_project_agent_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    agents_dir = tmp_path / ".miniharness" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: Project reviewer.
model: inherit
permission_mode: plan
disallowed_tools: write_file, edit_file
maxTurns: 2
initialPrompt: Start with a read-only review checklist.
---

Review code and return only findings with file references.
""",
        encoding="utf-8",
    )
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    reset_team_registry_for_tests()
    captured: dict[str, object] = {}

    async def fake_create_agent_task(self, **kwargs):
        captured.update(kwargs)
        task_id = "bg-agentdef"
        output_file = self.tasks_dir / f"{task_id}.log"
        output_file.write_text("", encoding="utf-8")
        record = BackgroundTaskRecord(
            id=task_id,
            type="local_agent",
            status="running",
            description=str(kwargs["description"]),
            cwd=str(kwargs["cwd"]),
            output_file=output_file,
            command=kwargs.get("command"),
            prompt=str(kwargs["prompt"]),
            argv=list(kwargs.get("argv") or []),
            created_at=1.0,
            started_at=1.0,
        )
        self._tasks[task_id] = record
        return record

    monkeypatch.setattr(BackgroundTaskManager, "create_agent_task", fake_create_agent_task)

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    created = await registry.execute("agent", {
        "description": "review current diff",
        "prompt": "Please review.",
        "subagent_type": "reviewer",
    })

    assert created.is_error is False
    assert "Spawned agent reviewer@default" in created.output
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "--append-system-prompt" in argv
    assert "Review code and return only findings" in argv[argv.index("--append-system-prompt") + 1]
    extra_env = captured["extra_env"]
    assert isinstance(extra_env, dict)
    assert extra_env[AGENT_ID_ENV_VAR] == "reviewer@default"
    assert extra_env[AGENT_NAME_ENV_VAR] == "reviewer"
    assert extra_env[AGENT_TEAM_ENV_VAR] == "default"
    assert AGENT_HOOKS_ENV_VAR not in extra_env
    assert extra_env[AGENT_MAX_TURNS_ENV_VAR] == "2"
    assert extra_env[AGENT_PERMISSION_MODE_ENV_VAR] == "plan"
    assert AGENT_TOOL_POLICY_ENV_VAR in extra_env
    assert "write_file" in extra_env[AGENT_TOOL_POLICY_ENV_VAR]
    assert "edit_file" in extra_env[AGENT_TOOL_POLICY_ENV_VAR]
    assert captured["prompt"].startswith("Start with a read-only review checklist.\n\nPlease review.")
    task = manager.get_task("bg-agentdef")
    assert task is not None
    assert task.metadata["agent_definition"] == "reviewer"
    assert task.metadata["agent_definition_source"] == "project"
    assert task.metadata["agent_permission_mode"] == "plan"


@pytest.mark.asyncio
async def test_agent_tool_forwards_project_agent_hooks_to_worker_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    agents_dir = tmp_path / ".miniharness" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: Project reviewer with session hooks.
hooks:
  subagent_stop:
    - type: command
      command: "printf '$TASK_ID' >> hook.log"
---

Review code and report findings.
""",
        encoding="utf-8",
    )
    reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    reset_team_registry_for_tests()
    captured: dict[str, object] = {}

    async def fake_create_agent_task(self, **kwargs):
        captured.update(kwargs)
        task_id = "bg-agent-hooks"
        output_file = self.tasks_dir / f"{task_id}.log"
        output_file.write_text("", encoding="utf-8")
        record = BackgroundTaskRecord(
            id=task_id,
            type="local_agent",
            status="running",
            description=str(kwargs["description"]),
            cwd=str(kwargs["cwd"]),
            output_file=output_file,
            command=kwargs.get("command"),
            prompt=str(kwargs["prompt"]),
            argv=list(kwargs.get("argv") or []),
            created_at=1.0,
            started_at=1.0,
        )
        self._tasks[task_id] = record
        return record

    monkeypatch.setattr(BackgroundTaskManager, "create_agent_task", fake_create_agent_task)

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    created = await registry.execute("agent", {
        "description": "review current diff",
        "prompt": "Please review.",
        "subagent_type": "reviewer",
    })

    assert created.is_error is False
    extra_env = captured["extra_env"]
    assert isinstance(extra_env, dict)
    assert AGENT_HOOKS_ENV_VAR in extra_env
    assert "subagent_stop" in extra_env[AGENT_HOOKS_ENV_VAR]
    assert "printf '$TASK_ID'" in extra_env[AGENT_HOOKS_ENV_VAR]


@pytest.mark.asyncio
async def test_agent_tool_policy_hides_and_blocks_disallowed_tools(tmp_path: Path):
    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        tool_policy={"disallowed_tools": ["write_file", "edit_file", "agent"]},
    )

    tool_names = {
        tool["function"]["name"]
        for tool in runtime.loop.tools.to_openai_tools()
    }

    assert "read_file" in tool_names
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    assert "agent" not in tool_names

    result = await runtime.loop.tools.execute("write_file", {
        "path": "blocked.txt",
        "content": "nope",
    })

    assert result.is_error is True
    assert "not active" in result.output
    assert not (tmp_path / "blocked.txt").exists()


def test_agent_tool_policy_allowlist_supports_patterns(tmp_path: Path):
    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        tool_policy={"tools": ["read_*", "ls"]},
    )

    tool_names = {
        tool["function"]["name"]
        for tool in runtime.loop.tools.to_openai_tools()
    }

    assert "read_file" in tool_names
    assert "ls" in tool_names
    assert "grep" not in tool_names
    assert "bash" not in tool_names


@pytest.mark.asyncio
async def test_agent_permission_mode_is_enforced_by_worker_runtime(tmp_path: Path):
    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        permission_mode="plan",
    )

    assert runtime.loop.permissions.mode == "plan"
    result = await runtime.loop.tools.execute("write_file", {
        "path": "blocked-by-plan.txt",
        "content": "nope",
    })

    assert result.is_error is True
    assert "Read-only mode" in result.output
    assert not (tmp_path / "blocked-by-plan.txt").exists()


def test_agent_max_turns_overrides_worker_runtime_limit(tmp_path: Path):
    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(max_turns=8),
        max_turns=2,
    )

    assert runtime.loop.max_turns == 2
    assert runtime.loop.settings.max_turns == 8


@pytest.mark.asyncio
async def test_agent_tool_uses_plugin_agent_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    plugin_dir = home / ".miniharness" / "plugins" / "review-pack"
    agents_dir = plugin_dir / "agents"
    agents_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        '{"name": "review-pack", "description": "Review plugin"}',
        encoding="utf-8",
    )
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: Plugin reviewer.
model: inherit
---

Review code using plugin-specific rules.
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    plugins = load_plugins(Settings(), cwd=tmp_path)
    plugin_index = [{
        "name": plugins[0].name,
        "description": plugins[0].description,
        "active": False,
        "skills": plugins[0].skills,
        "_plugin": plugins[0],
    }]
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    reset_team_registry_for_tests()
    captured: dict[str, object] = {}

    async def fake_create_agent_task(self, **kwargs):
        captured.update(kwargs)
        task_id = "bg-plugin-agent"
        output_file = self.tasks_dir / f"{task_id}.log"
        output_file.write_text("", encoding="utf-8")
        record = BackgroundTaskRecord(
            id=task_id,
            type="local_agent",
            status="running",
            description=str(kwargs["description"]),
            cwd=str(kwargs["cwd"]),
            output_file=output_file,
            command=kwargs.get("command"),
            prompt=str(kwargs["prompt"]),
            argv=list(kwargs.get("argv") or []),
            created_at=1.0,
            started_at=1.0,
        )
        self._tasks[task_id] = record
        return record

    monkeypatch.setattr(BackgroundTaskManager, "create_agent_task", fake_create_agent_task)

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
        plugin_index=plugin_index,
    )
    created = await registry.execute("agent", {
        "description": "plugin review",
        "prompt": "Please review.",
        "subagent_type": "review-pack:reviewer",
    })

    assert created.is_error is False
    assert "Spawned agent review-pack-reviewer@default" in created.output
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "--append-system-prompt" in argv
    assert "plugin-specific rules" in argv[argv.index("--append-system-prompt") + 1]
    task = manager.get_task("bg-plugin-agent")
    assert task is not None
    assert task.metadata["agent_definition"] == "review-pack:reviewer"
    assert task.metadata["agent_definition_source"] == "plugin"
