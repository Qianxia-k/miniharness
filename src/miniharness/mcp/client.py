"""MCP Client Manager — connects to servers, discovers tools, routes calls.

Production-grade implementation.  Handles:
- Connection lifecycle (start / reconnect / shutdown)
- Tool discovery and registration
- Tool call routing (delegates to the correct server session)
- Graceful degradation (MCP library not installed → all servers "disabled")

Usage::

    manager = McpClientManager(server_configs)
    await manager.connect_all()
    tools = manager.list_tools()  # → list[McpToolInfo]
    result = await manager.call_tool("server_name", "tool_name", {"arg": "val"})
    await manager.close()
"""

from __future__ import annotations

from typing import Any

from miniharness.mcp.types import (
    McpConnectionStatus,
    McpHttpServerConfig,
    McpServerConfig,
    McpStdioServerConfig,
    McpToolInfo,
)


class McpClientManager:
    """Manage connections to MCP servers.

    One manager instance handles all configured servers.  Each server
    starts in "pending" state and transitions to "connected" or "failed"
    during :meth:`connect_all`.

    Parameters
    ----------
    server_configs:
        Dict mapping server name → server config.
    """

    def __init__(self, server_configs: dict[str, McpServerConfig]) -> None:
        self._configs = server_configs
        self._statuses: dict[str, McpConnectionStatus] = {
            name: McpConnectionStatus(
                name=name,
                state="pending" if getattr(cfg, "enabled", True) else "disabled",
                transport=cfg.type if hasattr(cfg, "type") else "unknown",
                detail="" if getattr(cfg, "enabled", True) else "Disabled by configuration",
            )
            for name, cfg in server_configs.items()
        }
        # Populated after successful connection.
        self._sessions: dict[str, Any] = {}  # name → ClientSession
        self._stacks: dict[str, Any] = {}    # name → AsyncExitStack

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect_all(self) -> None:
        """Connect to all configured servers.

        Servers marked "disabled" are skipped.  Each connection runs
        independently — one failure doesn't affect others.
        """
        for name, config in self._configs.items():
            status = self._statuses[name]
            if status.state == "disabled":
                continue
            try:
                if isinstance(config, McpStdioServerConfig):
                    await self._connect_stdio(name, config)
                elif isinstance(config, McpHttpServerConfig):
                    await self._connect_http(name, config)
                else:
                    self._statuses[name].state = "failed"
                    self._statuses[name].detail = f"Unknown transport: {type(config).__name__}"
            except Exception as exc:
                self._statuses[name].state = "failed"
                self._statuses[name].detail = str(exc)

    async def close(self) -> None:
        """Close all connections and kill subprocesses.

        During asyncio shutdown, task cancellation can interrupt graceful
        cleanup.  We catch everything to prevent noise on ``/q``.
        """
        for name, stack in list(self._stacks.items()):
            # 1. Try graceful close via the async exit stack.
            try:
                await stack.aclose()
            except Exception:
                pass
            # 2. Fallback: force-kill any lingering subprocesses that the
            #    mcp library's anyio task group may have left behind.
            try:
                await _kill_stdio_children()
            except Exception:
                pass

        self._stacks.clear()
        self._sessions.clear()

    async def reconnect_all(self) -> None:
        """Close all connections and reconnect from scratch."""
        await self.close()
        for name, cfg in self._configs.items():
            self._statuses[name] = McpConnectionStatus(
                name=name,
                state="pending" if getattr(cfg, "enabled", True) else "disabled",
                transport=cfg.type if hasattr(cfg, "type") else "unknown",
                detail="" if getattr(cfg, "enabled", True) else "Disabled by configuration",
            )
        await self.connect_all()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_statuses(self) -> list[McpConnectionStatus]:
        """Return connection status for all servers."""
        return list(self._statuses.values())

    def list_tools(self) -> list[McpToolInfo]:
        """Return all tools from all connected servers."""
        tools: list[McpToolInfo] = []
        for status in self._statuses.values():
            tools.extend(status.tools)
        return tools

    def get_status(self, name: str) -> McpConnectionStatus | None:
        """Return status for one server."""
        return self._statuses.get(name)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call a tool on a connected MCP server.

        Returns the tool's text output, or an error string.
        """
        session = self._sessions.get(server_name)
        if session is None:
            status = self._statuses.get(server_name)
            detail = status.detail if status else "not configured"
            return f"MCP server '{server_name}' is not connected: {detail}"

        try:
            result = await session.call_tool(tool_name, arguments)
        except Exception as exc:
            return f"MCP tool call failed: {exc}"

        # Assemble text output.
        parts: list[str] = []
        for item in getattr(result, "content", []) or []:
            if getattr(item, "type", "") == "text":
                parts.append(getattr(item, "text", ""))
            else:
                try:
                    parts.append(item.model_dump_json())
                except Exception:
                    parts.append(str(item))

        if not parts:
            sc = getattr(result, "structuredContent", None)
            if sc is not None:
                parts.append(str(sc))

        return "\n".join(parts) if parts else "(no output)"

    # ------------------------------------------------------------------
    # Internal — connection handlers
    # ------------------------------------------------------------------

    async def _connect_stdio(self, name: str, config: McpStdioServerConfig) -> None:
        """Connect to a stdio-based MCP server."""
        _check_mcp_available()

        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters

        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )

        # stdio_client is an async context manager yielding (read, write).
        # We manage the exit stack ourselves so we can close later.
        stack = _AsyncExitStackCompat()
        transport = await stack.enter_async_context(stdio_client(params))
        read_stream, write_stream = transport

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        await self._on_connected(name, config, stack, session)

    async def _connect_http(self, name: str, config: McpHttpServerConfig) -> None:
        """Connect to an HTTP (SSE) MCP server."""
        _check_mcp_available()

        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        stack = _AsyncExitStackCompat()
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(headers=config.headers or None)
        )
        transport = await stack.enter_async_context(
            streamable_http_client(config.url, http_client=http_client)
        )
        read_stream, write_stream, _ = transport

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        await self._on_connected(name, config, stack, session)

    async def _on_connected(
        self,
        name: str,
        config: McpServerConfig,
        stack: Any,
        session: Any,
    ) -> None:
        """Handle post-connection setup: discover tools + resources."""
        from miniharness.mcp.types import McpResourceInfo

        # Discover tools.
        tool_result = await session.list_tools()
        tools = [
            McpToolInfo(
                server_name=name,
                name=t.name,
                description=getattr(t, "description", "") or "",
                input_schema=dict(getattr(t, "inputSchema", {}) or {}),
            )
            for t in (tool_result.tools if hasattr(tool_result, "tools") else [])
        ]

        # Discover resources (best-effort — not all servers support this).
        resources: list[McpResourceInfo] = []
        try:
            resource_result = await session.list_resources()
            for r in (resource_result.resources if hasattr(resource_result, "resources") else []):
                resources.append(McpResourceInfo(
                    server_name=name,
                    name=getattr(r, "name", "") or str(getattr(r, "uri", "")),
                    uri=str(getattr(r, "uri", "")),
                    description=getattr(r, "description", "") or "",
                ))
        except Exception:
            # "Method not found" is expected for servers that don't
            # implement resources — silently ignore.
            pass

        self._sessions[name] = session
        self._stacks[name] = stack
        self._statuses[name] = McpConnectionStatus(
            name=name,
            state="connected",
            transport=config.type if hasattr(config, "type") else "stdio",
            tools=tools,
            resources=resources,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MCP_CHECKED = False
_MCP_AVAILABLE = False


def _check_mcp_available() -> None:
    """Raise RuntimeError if the 'mcp' package is not installed."""
    global _MCP_CHECKED, _MCP_AVAILABLE
    if _MCP_CHECKED:
        if not _MCP_AVAILABLE:
            raise RuntimeError("MCP library not installed. Run: pip install mcp")
        return

    _MCP_CHECKED = True
    try:
        import mcp  # noqa: F401
        _MCP_AVAILABLE = True
    except ImportError:
        _MCP_AVAILABLE = False
        raise RuntimeError("MCP library not installed. Run: pip install mcp")


class _AsyncExitStackCompat:
    """Thin wrapper that adapts non-standard async context managers.

    Some MCP transport functions return tuples instead of proper
    async context managers.  This wrapper handles both cases.
    """

    def __init__(self) -> None:
        self._stack = _create_async_exit_stack()

    async def enter_async_context(self, cm):
        try:
            return await self._stack.enter_async_context(cm)
        except (TypeError, AttributeError):
            # If cm is not an async context manager, treat it as a raw
            # tuple and push a no-op closer.
            self._stack.push_async_callback(_noop)
            return cm

    async def aclose(self) -> None:
        await self._stack.aclose()

    def push_async_callback(self, cb):
        self._stack.push_async_callback(cb)


async def _noop() -> None:
    pass


async def _kill_stdio_children() -> None:
    """Send SIGTERM to any lingering child processes of the current process.

    This is a last-resort cleanup for MCP stdio subprocesses that the
    ``mcp`` library's anyio task group may have left running during
    asyncio shutdown.
    """
    import signal
    import psutil
    try:
        current = psutil.Process()
        children = current.children(recursive=True)
        for child in children:
            try:
                child.send_signal(signal.SIGTERM)
            except psutil.NoSuchProcess:
                pass
    except Exception:
        # psutil may not be installed — that's fine.
        pass


def _create_async_exit_stack():
    """Create an AsyncExitStack, with fallback for older Python."""
    from contextlib import AsyncExitStack
    return AsyncExitStack()
