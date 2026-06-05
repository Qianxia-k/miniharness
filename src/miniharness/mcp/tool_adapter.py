"""MCP tool adapter — wraps an MCP tool as a MiniHarness BaseTool.

When the model calls ``mcp__<server>__<tool>(...)``, this adapter:
1. Extracts the server name and tool name from the adapter name.
2. Delegates to ``McpClientManager.call_tool(server, tool, args)``.
3. Returns the result as a ``ToolResult``.

Dynamically creates a Pydantic input model from the tool's JSON Schema
so the OpenAI tool-calling API receives properly typed parameters.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, create_model

from miniharness.mcp.types import McpToolInfo
from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


class McpToolAdapter(BaseTool):
    """Wraps one MCP tool so the model can call it via the OpenAI API.

    The adapter's ``name`` is ``mcp__<server>__<tool>`` (double underscore
    delimiters).  The input schema is derived from the MCP tool's
    ``input_schema``.
    """

    def __init__(
        self,
        *,
        manager,  # McpClientManager
        tool_info: McpToolInfo,
        cwd: Path | None = None,
        permissions=None,
    ) -> None:
        self._manager = manager
        self._tool_info = tool_info
        self._server_name = tool_info.server_name
        self._mcp_tool_name = tool_info.name

        # Build the adapter name: mcp__<server>__<tool>
        server_seg = _sanitize(tool_info.server_name)
        tool_seg = _sanitize(tool_info.name)
        self.name = f"mcp__{server_seg}__{tool_seg}"
        self.description = tool_info.description or f"MCP tool {tool_info.name} (server: {tool_info.server_name})"

        # Dynamically create a Pydantic model from the JSON Schema.
        self.input_model = _schema_to_model(self.name, tool_info.input_schema)

        # Call BaseTool.__init__ for cwd/permissions.
        super().__init__(cwd=cwd or Path.cwd(), permissions=permissions)

    def to_openai_tool(self) -> dict:
        """Override to use the instance-level input_model."""
        from miniharness.tools.base import _pydantic_to_openai
        return _pydantic_to_openai(self.input_model, self.name, self.description)

    async def execute(self, arguments: BaseModel) -> ToolResult:
        """Execute the MCP tool via the manager."""
        args_dict = (
            arguments.model_dump(mode="json", exclude_none=True)
            if hasattr(arguments, "model_dump")
            else dict(arguments)
        )
        output = await self._manager.call_tool(
            self._server_name,
            self._mcp_tool_name,
            args_dict,
        )
        is_error = output.startswith("MCP server") or output.startswith("MCP tool call failed")
        return ToolResult(output=output, is_error=is_error)

    def permission_requests(self, arguments: BaseModel) -> list[ToolPermissionRequest]:
        args_dict = (
            arguments.model_dump(mode="json", exclude_none=True)
            if hasattr(arguments, "model_dump")
            else dict(arguments)
        )
        is_read_only = _is_read_only_mcp_tool(self._mcp_tool_name)
        requests: list[ToolPermissionRequest] = []

        for path in _extract_paths(args_dict):
            requests.append(ToolPermissionRequest(
                is_read_only=is_read_only,
                file_path=path,
                reason=(
                    f"Allow MCP tool {self.name} to "
                    f"{'read' if is_read_only else 'access/change'} {path}?"
                ),
            ))

        for command in _extract_commands(args_dict):
            requests.append(ToolPermissionRequest(
                is_read_only=False,
                command=command,
                reason=f"Allow MCP tool {self.name} to run command: {command[:120]}?",
            ))

        if not requests:
            requests.append(ToolPermissionRequest(
                is_read_only=is_read_only,
                reason=(
                    f"Allow read-only MCP tool {self.name}?"
                    if is_read_only
                    else f"Allow mutating/unknown MCP tool {self.name}?"
                ),
            ))

        return requests


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_READ_ONLY_PREFIXES = (
    "read", "list", "ls", "get", "search", "find", "grep", "stat",
    "describe", "inspect", "fetch", "query",
)
_MUTATING_PREFIXES = (
    "write", "create", "delete", "remove", "move", "rename", "edit",
    "update", "patch", "append", "copy", "run", "exec", "execute",
    "shell", "bash",
)
_PATH_KEYS = {
    "path", "file_path", "filepath", "root", "directory", "dir",
    "source", "destination", "target", "from", "to",
}
_COMMAND_KEYS = {"command", "cmd", "script", "shell", "bash"}


def _is_read_only_mcp_tool(tool_name: str) -> bool:
    normalized = tool_name.lower().replace("-", "_")
    if normalized.startswith(_MUTATING_PREFIXES):
        return False
    if normalized.startswith(_READ_ONLY_PREFIXES):
        return True
    return False


def _extract_paths(value) -> list[str]:
    paths: list[str] = []

    def walk(item, key: str | None = None) -> None:
        if isinstance(item, dict):
            for child_key, child_value in item.items():
                walk(child_value, str(child_key).lower())
            return
        if isinstance(item, list):
            for child in item:
                walk(child, key)
            return
        if key in _PATH_KEYS and isinstance(item, str) and item.strip():
            paths.append(item.strip())

    walk(value)
    return _dedupe(paths)


def _extract_commands(value) -> list[str]:
    commands: list[str] = []

    def walk(item, key: str | None = None) -> None:
        if isinstance(item, dict):
            for child_key, child_value in item.items():
                walk(child_value, str(child_key).lower())
            return
        if isinstance(item, list):
            for child in item:
                walk(child, key)
            return
        if key in _COMMAND_KEYS and isinstance(item, str) and item.strip():
            commands.append(item.strip())

    walk(value)
    return _dedupe(commands)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _sanitize(name: str) -> str:
    """Sanitize a name segment for use in a tool name.

    Replaces non-alphanumeric characters with ``_``.  If the result
    is empty or doesn't start with a letter, prepends ``"mcp_"``.
    """
    result = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    if not result:
        return "mcp_tool"
    if not result[0].isalpha():
        result = "mcp_" + result
    return result


def _schema_to_model(tool_name: str, schema: dict) -> type[BaseModel]:
    """Create a Pydantic model from a JSON Schema object.

    Parameters
    ----------
    tool_name:
        Used to generate a unique model name.
    schema:
        JSON Schema dict with ``properties`` and optional ``required``.
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    required: set[str] = set(schema.get("required", []))
    if not isinstance(required, (list, set)):
        required = set()

    fields: dict[str, tuple[type, Field]] = {}
    for key, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        json_type = prop.get("type", "string")
        desc = prop.get("description", "")

        py_type = _JSON_TYPE_MAP.get(str(json_type), str)
        if key in required:
            fields[key] = (py_type, Field(description=desc))
        else:
            fields[key] = (py_type | None, Field(default=None, description=desc))

    # Sanitize model name for create_model.
    model_name = tool_name.replace("-", "_").replace(".", "_").title().replace("_", "")
    if not model_name[0].isalpha():
        model_name = "Mcp" + model_name

    return create_model(model_name, **fields) if fields else _empty_model()


class _EmptyInput(BaseModel):
    """Fallback input model for tools with no parameters."""
    pass


def _empty_model() -> type[BaseModel]:
    return _EmptyInput
