"""Runtime visibility rules for plugin-contributed capabilities."""

from __future__ import annotations

from typing import Any


def active_plugin_names(plugin_index: list[dict] | None) -> set[str]:
    """Return plugin names currently active in the runtime context."""
    return {
        str(entry.get("name", ""))
        for entry in (plugin_index or [])
        if entry.get("active") and entry.get("name")
    }


def plugin_name_for_mcp_server(server_name: str) -> str | None:
    """Return the plugin name for a namespaced MCP server.

    Plugin MCP servers are configured as ``<plugin>:<server>``. Direct MCP
    servers have no plugin namespace and are always visible.
    """
    if ":" not in server_name:
        return None
    plugin_name, _ = server_name.split(":", 1)
    return plugin_name or None


def is_mcp_server_visible(server_name: str, plugin_index: list[dict] | None) -> bool:
    """Return whether an MCP server should be visible to the model."""
    plugin_name = plugin_name_for_mcp_server(server_name)
    if plugin_name is None:
        return True
    return plugin_name in active_plugin_names(plugin_index)


def is_tool_visible(tool: Any, plugin_index: list[dict] | None) -> bool:
    """Return whether a registered tool should be exposed/executable now."""
    server_name = getattr(tool, "_server_name", "")
    if not server_name:
        return True
    return is_mcp_server_visible(str(server_name), plugin_index)
