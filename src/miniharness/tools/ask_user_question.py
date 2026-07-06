"""Tool for asking the interactive user a follow-up question."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


AskUserPrompt = Callable[[str], Awaitable[str]]


class AskUserQuestionInput(BaseModel):
    """Arguments for asking the user a question."""

    question: str = Field(description="The exact question to ask the user")


class AskUserQuestionTool(BaseTool):
    """Ask the interactive user a question and return the answer."""

    name = "ask_user_question"
    description = "Ask the interactive user a follow-up question and return the answer."
    input_model = AskUserQuestionInput

    def __init__(self, *, ask_user_prompt: AskUserPrompt | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ask_user_prompt = ask_user_prompt

    def is_read_only(self, arguments: AskUserQuestionInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: AskUserQuestionInput) -> ToolResult:
        if self._ask_user_prompt is None:
            return ToolResult(
                output="ask_user_question is unavailable in this session",
                is_error=True,
            )
        answer = str(await self._ask_user_prompt(arguments.question)).strip()
        if not answer:
            return ToolResult(output="(no response)")
        return ToolResult(output=answer)
