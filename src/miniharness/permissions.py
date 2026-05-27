"""Permission checks for tool execution.

Read operations are always allowed. Write and shell operations prompt the user
for confirmation before proceeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.prompt import Confirm


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str = ""


class PermissionChecker:
    """Interactive permission checker using terminal prompts."""

    def __init__(self, *, cwd: Path) -> None:
        self.cwd = cwd

    def can_read(self, path: Path) -> PermissionDecision:
        """Reading files is always safe."""
        return PermissionDecision(True)

    def can_write(self, path: Path) -> PermissionDecision:
        """Ask the user before writing to a file."""
        rel = self._relative(path)
        if Confirm.ask(f"Allow write to [bold]{rel}[/bold]?", default=False):
            return PermissionDecision(True)
        return PermissionDecision(False, f"User denied write to {rel}")

    def can_run_command(self, command: str) -> PermissionDecision:
        """Ask the user before running a shell command."""
        preview = command[:120] + "..." if len(command) > 120 else command
        if Confirm.ask(f"Allow command: [bold]{preview}[/bold]?", default=False):
            return PermissionDecision(True)
        return PermissionDecision(False, f"User denied command: {preview}")

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)
