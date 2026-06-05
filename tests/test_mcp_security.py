import asyncio
from pathlib import Path

from miniharness.config.settings import Settings
from miniharness.mcp.client import McpClientManager
from miniharness.mcp.config import load_mcp_server_configs
from miniharness.mcp.tool_adapter import McpToolAdapter
from miniharness.mcp.types import McpConnectionStatus, McpToolInfo
from miniharness.permissions import PermissionChecker
from miniharness.plugins.gating import is_tool_visible
from miniharness.tool_registry import ToolRegistry


def test_filesystem_allowed_directories_are_added_to_stdio_args(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    settings.mcp_servers = {
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "allowed_directories": ["."],
        }
    }

    config = load_mcp_server_configs(settings)["filesystem"]

    assert str(tmp_path.resolve()) in config.args
    assert config.allowed_directories == [str(tmp_path.resolve())]


def test_filesystem_allowed_directories_support_workspace_templates(tmp_path: Path):
    workspace = tmp_path / "target-project"
    workspace.mkdir()
    settings = Settings()
    settings.mcp_servers = {
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "allowed_directories": ["${cwd}", "${workspace}/data"],
        }
    }

    config = load_mcp_server_configs(settings, cwd=workspace)["filesystem"]

    assert str(workspace.resolve()) in config.args
    assert str((workspace / "data").resolve()) in config.args
    assert config.allowed_directories == [
        str(workspace.resolve()),
        str((workspace / "data").resolve()),
    ]


def test_filesystem_allowed_directories_use_agent_cwd_not_process_cwd(
    tmp_path: Path,
    monkeypatch,
):
    process_cwd = tmp_path / "process"
    workspace = tmp_path / "workspace"
    process_cwd.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(process_cwd)

    settings = Settings()
    settings.mcp_servers = {
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "allowed_directories": ["${cwd}"],
        }
    }

    config = load_mcp_server_configs(settings, cwd=workspace)["filesystem"]

    assert str(workspace.resolve()) in config.args
    assert str(process_cwd.resolve()) not in config.args


def test_disabled_mcp_server_is_not_connected(tmp_path: Path):
    settings = Settings()
    settings.mcp_servers = {
        "optional_github": {
            "type": "stdio",
            "enabled": False,
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
        }
    }

    configs = load_mcp_server_configs(settings, cwd=tmp_path)
    manager = McpClientManager(configs)

    asyncio.run(manager.connect_all())

    status = manager.get_status("optional_github")
    assert status is not None
    assert status.state == "disabled"
    assert status.detail == "Disabled by configuration"


def test_mcp_read_sensitive_path_is_blocked(tmp_path: Path):
    registry = ToolRegistry(permissions=PermissionChecker(cwd=tmp_path, mode="bypass"))
    manager = _FakeMcpManager()
    registry.register(McpToolAdapter(
        manager=manager,
        tool_info=McpToolInfo(
            server_name="filesystem",
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    ))

    result = asyncio.run(registry.execute(
        "mcp__filesystem__read_file",
        {"path": str(Path.home() / ".ssh/id_rsa")},
    ))

    assert result.is_error
    assert "protected" in result.output or "sensitive" in result.output
    assert manager.calls == []


def test_unknown_mcp_tool_requires_permission_by_default(tmp_path: Path, monkeypatch):
    from miniharness import permissions as permissions_module

    monkeypatch.setattr(permissions_module, "_ask_confirmation", lambda prompt: False)

    registry = ToolRegistry(permissions=PermissionChecker(cwd=tmp_path, mode="default"))
    manager = _FakeMcpManager()
    registry.register(McpToolAdapter(
        manager=manager,
        tool_info=McpToolInfo(
            server_name="custom",
            name="transform",
            description="Unknown side effects",
            input_schema={"type": "object", "properties": {}},
        ),
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
    ))

    result = asyncio.run(registry.execute(
        "mcp__custom__transform",
        {},
    ))

    assert result.is_error
    assert "User denied" in result.output
    assert manager.calls == []


def test_inactive_plugin_mcp_tool_is_hidden_and_blocked(tmp_path: Path):
    plugin_index = [{"name": "demo-plugin", "active": False}]
    registry = ToolRegistry(
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
        is_tool_enabled=lambda _name, tool: is_tool_visible(tool, plugin_index),
    )
    manager = _FakeMcpManager()
    adapter = McpToolAdapter(
        manager=manager,
        tool_info=McpToolInfo(
            server_name="demo-plugin:filesystem",
            name="read_file",
            description="Read a plugin file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    registry.register(adapter)

    exposed_names = {
        t["function"]["name"]
        for t in registry.to_openai_tools()
    }
    assert adapter.name not in exposed_names

    result = asyncio.run(registry.execute(adapter.name, {"path": str(tmp_path / "a.txt")}))

    assert result.is_error
    assert "not active" in result.output
    assert manager.calls == []


def test_active_plugin_mcp_tool_is_exposed_and_executable(tmp_path: Path):
    plugin_index = [{"name": "demo-plugin", "active": True}]
    registry = ToolRegistry(
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
        is_tool_enabled=lambda _name, tool: is_tool_visible(tool, plugin_index),
    )
    manager = _FakeMcpManager()
    adapter = McpToolAdapter(
        manager=manager,
        tool_info=McpToolInfo(
            server_name="demo-plugin:filesystem",
            name="read_file",
            description="Read a plugin file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    registry.register(adapter)

    exposed_names = {
        t["function"]["name"]
        for t in registry.to_openai_tools()
    }
    assert adapter.name in exposed_names

    result = asyncio.run(registry.execute(adapter.name, {"path": str(tmp_path / "a.txt")}))

    assert not result.is_error
    assert manager.calls == [("demo-plugin:filesystem", "read_file", {"path": str(tmp_path / "a.txt")})]


def test_system_prompt_hides_inactive_plugin_mcp_servers(tmp_path: Path):
    from miniharness.prompts.system import assemble_system_prompt

    manager = _FakeStatusManager([
        McpConnectionStatus(
            name="filesystem",
            state="connected",
            transport="stdio",
            tools=[McpToolInfo(server_name="filesystem", name="read_file")],
        ),
        McpConnectionStatus(
            name="demo-plugin:filesystem",
            state="connected",
            transport="stdio",
            tools=[McpToolInfo(server_name="demo-plugin:filesystem", name="plugin_read")],
        ),
    ])
    plugin_index = [{"name": "demo-plugin", "active": False}]

    prompt = assemble_system_prompt(
        base_prompt="You are an agent.",
        cwd=tmp_path,
        mcp_manager=manager,
        plugin_index=plugin_index,
    )

    assert "**filesystem**" in prompt
    assert "demo-plugin:filesystem" not in prompt

    plugin_index[0]["active"] = True
    prompt = assemble_system_prompt(
        base_prompt="You are an agent.",
        cwd=tmp_path,
        mcp_manager=manager,
        plugin_index=plugin_index,
    )

    assert "demo-plugin:filesystem" in prompt


class _FakeMcpManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        self.calls.append((server_name, tool_name, arguments))
        return "ok"


class _FakeStatusManager:
    def __init__(self, statuses: list[McpConnectionStatus]) -> None:
        self._statuses = statuses

    def list_statuses(self) -> list[McpConnectionStatus]:
        return self._statuses
