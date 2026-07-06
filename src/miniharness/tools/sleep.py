"""Sleep tool."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class SleepInput(BaseModel):
    """Arguments for sleep."""

    seconds: float = Field(default=1.0, ge=0.0, le=30.0)


class SleepTool(BaseTool):
    """Pause execution briefly."""

    name = "sleep"
    description = "Sleep for a short duration."
    input_model = SleepInput

    def is_read_only(self, arguments: SleepInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: SleepInput) -> ToolResult:
        await asyncio.sleep(arguments.seconds)
        return ToolResult(output=f"Slept for {arguments.seconds} seconds")
