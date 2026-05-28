"""Permission checks for tool execution.

Mirrors OpenHarness's permission modes:

    default       — ask before write and shell (interactive)
    accept-edits  — auto-allow writes, ask before shell
    bypass        — allow everything without prompting
    plan          — deny all writes and shell (read-only)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.prompt import Confirm


PermissionMode = Literal["default", "accept-edits", "bypass", "plan"]

_MODE_ORDER: list[PermissionMode] = ["default", "accept-edits", "bypass", "plan"]


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str = ""


class PermissionChecker:
    """Permission checker with four operating modes."""

    def __init__(self, *, cwd: Path, mode: PermissionMode = "default") -> None:
        self.cwd = cwd
        self.mode: PermissionMode = mode

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def cycle_mode(self) -> str:
        """Cycle to the next permission mode and return its name."""
        idx = _MODE_ORDER.index(self.mode)
        self.mode = _MODE_ORDER[(idx + 1) % len(_MODE_ORDER)]
        return self.mode

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    def can_read(self, path: Path) -> PermissionDecision:
        """Reading files is always safe in every mode."""
        return PermissionDecision(True)

    def can_write(self, path: Path) -> PermissionDecision:
        """Check write permission according to the current mode."""
        if self.mode == "bypass":
            return PermissionDecision(True)
        if self.mode == "plan":
            return PermissionDecision(False, "Read-only mode (plan)")

        # accept-edits and default: both allow writes without asking for
        # edits (the tool-level distinction is handled by can_run_command).
        if self.mode == "accept-edits":
            return PermissionDecision(True)

        # default: ask
        rel = self._relative(path)
        if Confirm.ask(f"Allow write to [bold]{rel}[/bold]?", default=False):
            return PermissionDecision(True)
        return PermissionDecision(False, f"User denied write to {rel}")

    def can_run_command(self, command: str) -> PermissionDecision:
        """Check shell command permission according to the current mode."""
        if self.mode == "bypass":
            return PermissionDecision(True)
        if self.mode == "plan":
            return PermissionDecision(False, "Read-only mode (plan)")

        # default and accept-edits: always ask before running shell.
        preview = command[:120] + "..." if len(command) > 120 else command
        if Confirm.ask(f"Allow command: [bold]{preview}[/bold]?", default=False):
            return PermissionDecision(True)
        return PermissionDecision(False, f"User denied command: {preview}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)
