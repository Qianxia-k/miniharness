"""MCP (Model Context Protocol) types — server configs and tool info.

MCP is a standard protocol for connecting AI agents to external tools
and data sources.  An MCP server exposes:

    - **tools** — functions the model can call
    - **resources** — data the model can read (files, DB rows, API responses)

This module defines the configuration and runtime types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Server configuration (what users put in settings.json)
# ---------------------------------------------------------------------------


class McpStdioServerConfig(BaseModel):
    """An MCP server launched as a subprocess.

    Example::

        {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
    """

    type: Literal["stdio"] = "stdio"
    enabled: bool = Field(default=True, description="Whether this server should be started")
    command: str = Field(description="Executable to launch")
    args: list[str] = Field(default_factory=list, description="CLI arguments")
    env: dict[str, str] | None = Field(default=None, description="Extra env vars")
    cwd: str | None = Field(default=None, description="Working directory")
    allowed_directories: list[str] = Field(
        default_factory=list,
        description=(
            "Filesystem roots to pass to filesystem MCP servers as allowed directories. "
            "This is host-side configuration, not a protocol-level access control."
        ),
    )
    roots: list[str] = Field(
        default_factory=list,
        description="Alias for allowed_directories for MCP Roots-style configuration.",
    )


class McpHttpServerConfig(BaseModel):
    """An MCP server accessible via HTTP (SSE transport).

    Example::

        {
            "type": "http",
            "url": "http://localhost:8000/mcp"
        }
    """

    type: Literal["http"] = "http"
    enabled: bool = Field(default=True, description="Whether this server should be connected")
    url: str = Field(description="MCP endpoint URL")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")


# Union type for settings validation.
McpServerConfig = McpStdioServerConfig | McpHttpServerConfig


# ---------------------------------------------------------------------------
# Runtime types (populated after connection)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpToolInfo:
    """Metadata about one tool exposed by an MCP server."""

    server_name: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpResourceInfo:
    """Metadata about one resource exposed by an MCP server."""

    server_name: str
    name: str
    uri: str
    description: str = ""


@dataclass
class McpConnectionStatus:
    """Runtime status of one MCP server connection."""

    name: str
    state: Literal["connected", "failed", "pending", "disabled"] = "pending"
    transport: str = "unknown"
    detail: str = ""  # error message if failed
    tools: list[McpToolInfo] = field(default_factory=list)
    resources: list[McpResourceInfo] = field(default_factory=list)
