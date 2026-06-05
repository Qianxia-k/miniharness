"""Run a shell command in the workspace.

When the sandbox is active, commands are executed inside an isolated Docker
container (mirrors OpenHarness's bash_tool + shell.py integration).
"""

from __future__ import annotations

import re
import subprocess

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class BashInput(BaseModel):
    """Arguments for bash."""

    command: str = Field(description="The shell command to execute")


class BashTool(BaseTool):
    name = "bash"
    description = "Run a shell command in the workspace."
    input_model = BashInput

    async def execute(self, arguments: BashInput) -> ToolResult:
        command = arguments.command.strip()
        if not command:
            return ToolResult("command is required", is_error=True)

        markup_error = _reject_probable_markup(command)
        if markup_error:
            return ToolResult(markup_error, is_error=True)

        decision = self.permissions.can_run_command(command)
        if not decision.allowed:
            return ToolResult(decision.reason, is_error=True)

        # ---- sandbox path ----
        from miniharness.sandbox import is_sandbox_active, get_sandbox

        if is_sandbox_active():
            sandbox = get_sandbox()
            try:
                output = await sandbox.exec_command(command)
                return ToolResult(output)
            except Exception as exc:
                return ToolResult(f"Sandbox exec failed: {exc}", is_error=True)

        # ---- direct path (no sandbox) ----
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.cwd),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Command timed out after 30s: {command}", is_error=True)

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return ToolResult(output.strip() or f"(exit code {result.returncode})")


_MERMAID_DIRECTIVE_RE = re.compile(
    r"^\s*(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|"
    r"erDiagram|gantt|pie|mindmap|journey|gitGraph)\b",
    re.IGNORECASE,
)
_MERMAID_EDGE_RE = re.compile(
    r"^\s*[\w.-]+(?:\[[^\]]+\]|\([^)]+\)|\{[^}]+\})?\s*(?:-->|---|==>)"
)


def _reject_probable_markup(command: str) -> str | None:
    """Reject Markdown/Mermaid snippets before shell redirection can create files."""
    lines = [line.strip() for line in command.splitlines() if line.strip()]
    if not lines:
        return None

    if any(line.startswith("```") for line in lines):
        return _markup_error("markdown code fence")

    if any(_MERMAID_DIRECTIVE_RE.match(line) for line in lines):
        return _markup_error("mermaid diagram directive")

    if any(_MERMAID_EDGE_RE.match(line) for line in lines):
        return _markup_error("mermaid diagram edge")

    if _contains_unquoted_mermaid_arrow(command):
        return _markup_error("unquoted '-->' sequence")

    return None


def _contains_unquoted_mermaid_arrow(command: str) -> bool:
    in_single = False
    in_double = False
    escaped = False

    for idx, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if (
            not in_single
            and not in_double
            and command.startswith("-->", idx)
        ):
            return True
    return False


def _markup_error(reason: str) -> str:
    return (
        f"Refusing to run probable Markdown/Mermaid content as shell ({reason}). "
        "Use write_file/edit_file to create documentation instead of bash."
    )
