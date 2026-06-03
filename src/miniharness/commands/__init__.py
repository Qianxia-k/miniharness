"""Command system — extensible slash-command registry.

Replaces the hardcoded if/elif chain in ``cli.py`` with a pluggable
registry.  Commands can be registered from built-in handlers, skills,
and (future) hooks/plugins.

Module map::

    types.py      — CommandResult, CommandContext, CommandHandler protocol
    registry.py   — CommandRegistry (register, dispatch, lookup)
    builtin.py    — All built-in command handlers
"""

from miniharness.commands.registry import CommandRegistry
from miniharness.commands.types import CommandContext, CommandHandler, CommandResult

__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandRegistry",
    "CommandResult",
]
