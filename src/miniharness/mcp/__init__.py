"""MCP (Model Context Protocol) integration.

Module map::

    types.py        — McpServerConfig, McpToolInfo, McpConnectionStatus
    config.py       — load_mcp_server_configs()
    client.py       — McpClientManager (connect, discover, call, reconnect)
    tool_adapter.py — McpToolAdapter (wraps MCP tool → BaseTool)
    auth_tool.py    — McpAuthTool (runtime credential updates)
    resource_tools.py — ListMcpResourcesTool, ReadMcpResourceTool

Quick start::

    from miniharness.mcp import McpClientManager, load_mcp_server_configs
    from miniharness.config.settings import Settings, McpStdioServerConfig

    settings = Settings()
    settings.mcp_servers = {
        "filesystem": McpStdioServerConfig(
            command="npx", args=["-y", "@mcp/server-filesystem", "/tmp"]
        ),
    }

    manager = McpClientManager(load_mcp_server_configs(settings))
    await manager.connect_all()
    # Tools are auto-registered via create_default_registry(mcp_manager=manager).
"""

from miniharness.mcp.auth_tool import McpAuthTool
from miniharness.mcp.client import McpClientManager
from miniharness.mcp.config import load_mcp_server_configs
from miniharness.mcp.resource_tools import (
    ListMcpResourcesTool,
    ReadMcpResourceTool,
)
from miniharness.mcp.tool_adapter import McpToolAdapter
from miniharness.mcp.types import (
    McpConnectionStatus,
    McpHttpServerConfig,
    McpServerConfig,
    McpStdioServerConfig,
    McpToolInfo,
)

__all__ = [
    "ListMcpResourcesTool",
    "McpAuthTool",
    "McpClientManager",
    "McpConnectionStatus",
    "McpHttpServerConfig",
    "McpServerConfig",
    "McpStdioServerConfig",
    "McpToolAdapter",
    "McpToolInfo",
    "ReadMcpResourceTool",
    "load_mcp_server_configs",
]
