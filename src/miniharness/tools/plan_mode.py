"""Tools for switching MiniHarness permission plan mode."""

from __future__ import annotations

from pydantic import BaseModel

from miniharness.tools.base import BaseTool, ToolResult


class EnterPlanModeInput(BaseModel):
    """No-argument input for entering plan mode."""


class ExitPlanModeInput(BaseModel):
    """No-argument input for leaving plan mode."""


class EnterPlanModeTool(BaseTool):
    """Switch the current session permission mode to plan."""

    name = "enter_plan_mode"
    description = "Switch permission mode to plan."
    input_model = EnterPlanModeInput

    async def execute(self, arguments: EnterPlanModeInput) -> ToolResult:
        del arguments
        self.permissions.mode = "plan"
        return ToolResult(output="Permission mode set to plan")


class ExitPlanModeTool(BaseTool):
    """Switch the current session permission mode back to default."""

    name = "exit_plan_mode"
    description = "Switch permission mode back to default."
    input_model = ExitPlanModeInput

    async def execute(self, arguments: ExitPlanModeInput) -> ToolResult:
        del arguments
        self.permissions.mode = "default"
        return ToolResult(output="Permission mode set to default")
