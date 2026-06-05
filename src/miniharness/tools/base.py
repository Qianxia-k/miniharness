"""Base classes for MiniHarness tools.

Mirrors OpenHarness's BaseTool with Pydantic input models:
    - Each tool defines a Pydantic BaseModel for its arguments.
    - to_openai_tool() is auto-generated from the model.
    - execute() receives a validated model instance, not a raw dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from miniharness.permissions import PermissionChecker


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolPermissionRequest:
    """Permission facts extracted from one tool invocation.

    Tools that need registry-level policy checks return one or more of these
    before execution.  Built-in tools can keep their existing internal checks;
    adapters for external tools, especially MCP, should use this so they do not
    bypass MiniHarness permission modes.
    """

    is_read_only: bool
    file_path: str | None = None
    command: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Pydantic → OpenAI schema converter
# ---------------------------------------------------------------------------


def _pydantic_to_openai(
    model_class: type[BaseModel],
    tool_name: str,
    tool_description: str,
) -> dict[str, Any]:
    """Generate an OpenAI function-calling schema from a Pydantic model."""
    json_schema = model_class.model_json_schema()
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field_name, field_info in model_class.model_fields.items():
        field_schema = json_schema.get("properties", {}).get(field_name, {})
        prop: dict[str, Any] = {}

        # Map JSON Schema type to OpenAI parameter type.
        json_type = field_schema.get("type", "string")
        if json_type in ("integer", "number", "boolean"):
            prop["type"] = json_type
        elif json_type == "array":
            prop["type"] = "array"
            if "items" in field_schema:
                prop["items"] = field_schema["items"]
        else:
            prop["type"] = "string"

        if "description" in field_schema:
            prop["description"] = field_schema["description"]

        properties[field_name] = prop

        if field_info.is_required():
            required.append(field_name)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Base tool
# ---------------------------------------------------------------------------


class BaseTool:
    """Base interface every MiniHarness tool follows.

    Subclasses define:
        name: str
        description: str
        input_model: type[BaseModel]
    """

    name: str
    description: str
    input_model: type[BaseModel]

    def __init__(self, *, cwd: Path, permissions: PermissionChecker) -> None:
        self.cwd = cwd
        self.permissions = permissions

    @classmethod
    def to_openai_tool(cls) -> dict[str, Any]:
        """Return this tool's model-facing schema, generated from input_model."""
        return _pydantic_to_openai(cls.input_model, cls.name, cls.description)

    async def execute(self, arguments: BaseModel) -> ToolResult:
        """Execute this tool with a validated Pydantic model instance."""
        raise NotImplementedError

    def permission_requests(self, arguments: BaseModel) -> list[ToolPermissionRequest]:
        """Return registry-level permission checks required before execution.

        The default is empty for backward compatibility with built-in tools
        that already perform their own permission checks inside ``execute``.
        External tool adapters should override this.
        """
        del arguments
        return []
