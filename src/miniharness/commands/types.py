"""Command system types.

A slash command is a user-facing ``/<name>`` handler registered with
the ``CommandRegistry``.  When the user types ``/<name> [args]``, the
registry dispatches to the handler and returns a ``CommandResult``
telling the REPL what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Command handler protocol
# ---------------------------------------------------------------------------


class CommandHandler(Protocol):
    """A callable that handles a slash command.

    Receives the raw argument string (everything after the command name)
    and a context object, and returns a ``CommandResult``.
    """

    def __call__(self, args: str, ctx: CommandContext) -> CommandResult: ...


# ---------------------------------------------------------------------------
# Context passed to every handler
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    """Context available to every slash-command handler.

    Attributes
    ----------
    loop:
        The active ``AgentLoop`` instance.  Handlers can modify the loop
        (e.g. change model, clear conversation, switch sessions).
    console:
        The ``rich.Console`` instance for output.
    cwd:
        Working directory.
    skill_registry:
        The skill registry (for skill-related commands).
    hook_registry:
        The hook registry (for hook-related commands).
    tool_registry:
        The tool registry (for tool-related commands — /tools, /tool).
    """

    loop: Any  # AgentLoop (avoid circular import)
    console: Any
    cwd: Path
    skill_registry: Any = None
    hook_registry: Any = None
    tool_registry: Any = None


# ---------------------------------------------------------------------------
# Result returned by every handler
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """What the REPL should do after executing a command.

    Attributes
    ----------
    message:
        Text to display to the user.  ``None`` = nothing to say.
    submit_prompt:
        If set, submit this text as a user prompt to the agent loop.
        (Used by skill slash-commands to trigger the skill.)
    exit:
        ``True`` = exit the REPL.
    should_save:
        ``True`` = save the session to disk after this command.
    refresh_runtime:
        ``True`` = the command modified runtime state (model, permissions,
        etc.) and the REPL should refresh any cached state.
    """

    message: str | None = None
    submit_prompt: str | None = None
    exit: bool = False
    should_save: bool = False
    refresh_runtime: bool = False

    # ── Factory helpers ───────────────────────────────────────────

    @classmethod
    def ok(cls, message: str = "", *, should_save: bool = False) -> CommandResult:
        """Command succeeded with an optional message."""
        return cls(message=message or None, should_save=should_save)

    @classmethod
    def done(cls, message: str = "") -> CommandResult:
        """Exit the REPL."""
        return cls(message=message or None, exit=True)

    @classmethod
    def prompt(cls, prompt_text: str) -> CommandResult:
        """Submit text as a user prompt to the agent loop."""
        return cls(submit_prompt=prompt_text, should_save=True)

    @classmethod
    def refreshed(cls, message: str = "") -> CommandResult:
        """Runtime state was changed; REPL should refresh."""
        return cls(message=message or None, refresh_runtime=True, should_save=True)
