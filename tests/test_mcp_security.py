import asyncio
from pathlib import Path

from miniharness.config.settings import Settings
from miniharness.mcp.client import McpClientManager
from miniharness.mcp.config import load_mcp_server_configs
from miniharness.mcp.tool_adapter import McpToolAdapter
from miniharness.mcp.types import McpToolInfo
from miniharness.permissions import PermissionChecker
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


class _FakeMcpManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        self.calls.append((server_name, tool_name, arguments))
        return "ok"
