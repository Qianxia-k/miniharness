"""Hook definition schemas.

Each hook type has its own Pydantic model.  The ``type`` field acts as a
discriminant so a list of mixed hook types can be validated.

Supported types:

    ``command``
        Runs a shell command.  Good for logging / notifications / file ops.

    ``prompt``
        Asks a model to validate.  Good for security auditing / policy.

    ``confirm``
        Asks the HUMAN user for interactive approval.  Good for high-risk
        operations (rm -rf, database drops, production deploys).

Mirrors OpenHarness's hook types, with ``confirm`` added for interactive
human-in-the-loop approval workflows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CommandHookDefinition(BaseModel):
    """A hook that executes a shell command.

    Example::

        {
            "type": "command",
            "command": "echo '$TOOL_NAME called' >> /tmp/audit.log",
            "matcher": "bash:*rm*",
            "block_on_failure": false
        }

    ``$ARGUMENTS`` → full JSON payload.
    ``$KEY_NAME``   → individual payload fields (e.g. ``$TOOL_NAME``).
    """

    type: Literal["command"] = "command"
    command: str = Field(description="Shell command to execute")
    timeout_seconds: int = Field(default=30, ge=1, le=600,
                                  description="Max execution time in seconds")
    matcher: str | None = Field(
        default=None,
        description=(
            "fnmatch pattern. Matched against tool_name, tool_input values, "
            "prompt text, and event name.  E.g. 'bash:*rm*' matches bash "
            "commands starting with rm; '*:/etc/*' matches any tool touching /etc."
        ),
    )
    block_on_failure: bool = Field(
        default=False,
        description="If True, a failed hook blocks the triggering action"
    )


class PromptHookDefinition(BaseModel):
    """A hook that asks a model to validate an event.

    Example::

        {
            "type": "prompt",
            "prompt": "Is this command safe? $ARGUMENTS",
            "matcher": "bash:*pip install*",
            "block_on_failure": true
        }

    The model receives a system prompt instructing it to return
    ``{"ok": true}`` or ``{"ok": false, "reason": "why"}``.
    """

    type: Literal["prompt"] = "prompt"
    prompt: str = Field(description="Prompt to send to the model for validation")
    timeout_seconds: int = Field(default=30, ge=1, le=600,
                                  description="Max model response time in seconds")
    matcher: str | None = Field(
        default=None,
        description="Same multi-field fnmatch as command hooks.",
    )
    block_on_failure: bool = Field(
        default=True,
        description="If True (default for prompt), a failed hook blocks the action"
    )


class ConfirmHookDefinition(BaseModel):
    """A hook that asks the HUMAN USER for interactive approval.

    Example::

        {
            "type": "confirm",
            "message": "Allow: $TOOL_NAME to run '$COMMAND'?",
            "matcher": "bash:*rm -rf*",
            "block_on_failure": true
        }

    The user sees a ``[bold red]⚠ APPROVAL REQUIRED[/]`` prompt with the
    rendered message and must type ``yes`` to proceed.  Any other response
    (or timeout) denies the operation.

    This is for HIGH-RISK operations where you want a human in the loop.
    """

    type: Literal["confirm"] = "confirm"
    message: str = Field(
        default="Approve execution of $TOOL_NAME?",
        description="Confirmation message shown to the user.  Supports $KEY_NAME substitution.",
    )
    timeout_seconds: int = Field(
        default=120, ge=10, le=3600,
        description="How long to wait for human response before auto-denying"
    )
    matcher: str | None = Field(
        default=None,
        description="Only prompt for confirmation when the matcher fires.",
    )
    block_on_failure: bool = Field(
        default=True,
        description="Always True for confirm — timeout/deny blocks the action"
    )


# Union type — a hook list can contain mixed types.
HookDefinition = CommandHookDefinition | PromptHookDefinition | ConfirmHookDefinition
