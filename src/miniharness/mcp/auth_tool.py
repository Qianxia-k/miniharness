"""MCP auth tool — lets the model update MCP server credentials at runtime.

When an MCP server requires dynamic credentials (e.g. OAuth tokens that
expire), the model can call this tool to update them and reconnect.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class McpAuthToolInput(BaseModel):
    server: str = Field(description="MCP server name to update credentials for")
    credential: str = Field(description="Credential value (API key, token, etc.)")
    credential_type: str = Field(
        default="header",
        description="How to send the credential: 'header' or 'env'"
    )


class McpAuthTool(BaseTool):
    """Update credentials for an MCP server and reconnect."""

    name = "mcp_auth"
    description = (
        "Update authentication credentials for an MCP server. "
        "Use this when an MCP tool call fails with an auth error."
    )
    input_model = McpAuthToolInput

    def __init__(self, *, cwd: Path, manager=None, permissions=None) -> None:
        super().__init__(cwd=cwd, permissions=permissions)
        self._manager = manager

    async def execute(self, arguments: McpAuthToolInput) -> ToolResult:
        if self._manager is None:
            return ToolResult("MCP manager not available.", is_error=True)

        server_name = arguments.server.strip()
        status = self._manager.get_status(server_name)
        if status is None:
            return ToolResult(f"MCP server '{server_name}' not configured.", is_error=True)

        config = self._manager._configs.get(server_name)
        if config is None:
            return ToolResult(f"No config found for server '{server_name}'.", is_error=True)

        # Update the config in-memory.
        if arguments.credential_type == "header":
            if hasattr(config, "headers"):
                config.headers = dict(getattr(config, "headers", {}) or {})
                config.headers["Authorization"] = f"Bearer {arguments.credential.strip()}"
        elif arguments.credential_type == "env":
            if hasattr(config, "env"):
                config.env = dict(getattr(config, "env", {}) or {})
                config.env["API_KEY"] = arguments.credential.strip()

        # Reconnect this specific server.
        try:
            await self._manager.reconnect_all()
            new_status = self._manager.get_status(server_name)
            state = new_status.state if new_status else "unknown"
            return ToolResult(
                f"Credentials updated and reconnected. "
                f"Server '{server_name}' is now: {state}."
            )
        except Exception as exc:
            return ToolResult(f"Reconnect failed: {exc}", is_error=True)
