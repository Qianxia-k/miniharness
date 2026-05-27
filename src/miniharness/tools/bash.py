"""Run a shell command in the workspace.

When the sandbox is active, commands are executed inside an isolated Docker
container (mirrors OpenHarness's bash_tool + shell.py integration).
"""

from __future__ import annotations

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
