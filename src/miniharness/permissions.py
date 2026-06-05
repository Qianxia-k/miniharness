"""Permission checks for tool execution.

Architecture — Hook vs Permission (clear division of labor)::

    ┌─────────────────────────────────────────────────────────┐
    │  Hook System (hooks/)                                    │
    │                                                         │
    │  "I KNOW this pattern is dangerous — block it."         │
    │                                                         │
    │  • Pattern-driven (43 dangerous commands, 26 files)     │
    │  • Configurable via presets (enable/disable per group)  │
    │  • Can BLOCK, CONFIRM, REVIEW, or LOG                   │
    │  • Runs BEFORE the tool executes                        │
    │  • If blocked → tool never runs, permission never asked │
    └──────────────────────────┬──────────────────────────────┘
                               │ hook passed / no match
                               ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Permission System (this file)                           │
    │                                                         │
    │  "I DON'T know this — should I ask the user?"           │
    │                                                         │
    │  • Mode-driven (default / accept-edits / bypass / plan) │
    │  • Coarse-grained: all writes, all commands, or none    │
    │  • Defense-in-depth: sensitive paths always denied       │
    │  • Runs INSIDE each tool's execute() method             │
    └─────────────────────────────────────────────────────────┘

Key principle: pattern-based blocking belongs in hooks.
Mode-based confirmation belongs in permissions.
They complement, not duplicate.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console


_console = Console()


PermissionMode = Literal["default", "accept-edits", "bypass", "plan"]

_MODE_ORDER: list[PermissionMode] = ["default", "accept-edits", "bypass", "plan"]


# ---------------------------------------------------------------------------
# Defense-in-depth: sensitive paths ALWAYS denied, even if hooks are disabled.
# These are a SUBSET of hooks/presets.py SENSITIVE_FILE_PATTERNS — the hook
# system handles the full set; these are the last-resort safety net.
# ---------------------------------------------------------------------------

_CRITICAL_PATH_PATTERNS: tuple[str, ...] = (
    # SSH keys
    "*/.ssh/id_*",
    "*/.ssh/*_key",
    # Cloud credentials
    "*/.aws/credentials",
    "*/.config/gcloud/credentials*",
    # System security files
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/sudoers.d/*",
    # Harness credential stores
    "*/.miniharness/credentials.*",
    "*/.openharness/credentials.*",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str = ""
    requires_confirmation: bool = False


# ---------------------------------------------------------------------------
# PermissionChecker
# ---------------------------------------------------------------------------


class PermissionChecker:
    """Mode-based permission checker.

    Permissions are the SECOND line of defense (hooks are first).
    They decide based on OPERATING MODE, not pattern matching.

    Parameters
    ----------
    cwd:
        Working directory.
    mode:
        ``"default"`` — ask before writes and shell commands.
        ``"accept-edits"`` — auto-allow file writes, ask before shell.
        ``"bypass"`` — allow everything.
        ``"plan"`` — read-only, deny all writes and shell.
    path_rules:
        User-configured ``(pattern, allow)`` tuples for path control.
    denied_commands:
        User-configured command patterns to deny.
    allowed_tools:
        If set, ONLY these tools may execute (whitelist).
    denied_tools:
        Tools that are always denied.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        mode: PermissionMode = "default",
        path_rules: list[tuple[str, bool]] | None = None,
        denied_commands: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
    ) -> None:
        self.cwd = cwd
        self.mode: PermissionMode = mode

        self._path_rules: list[tuple[str, bool]] = list(path_rules or [])
        self._denied_commands: list[str] = list(denied_commands or [])
        self._allowed_tools: frozenset[str] | None = (
            frozenset(allowed_tools) if allowed_tools else None
        )
        self._denied_tools: frozenset[str] = frozenset(denied_tools or [])

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def cycle_mode(self) -> str:
        idx = _MODE_ORDER.index(self.mode)
        self.mode = _MODE_ORDER[(idx + 1) % len(_MODE_ORDER)]
        return self.mode

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        tool_name: str,
        file_path: str | None = None,
        command: str | None = None,
        is_read_only: bool = False,
    ) -> PermissionDecision:
        """Full permission evaluation.

        Order:
        1. Tool deny/allow lists (user-configured)
        2. Critical path check (defense-in-depth, unoverridable)
        3. User path rules
        4. User command deny patterns
        5. Mode-based decision
        """
        # 1. Tool deny/allow lists.
        if tool_name in self._denied_tools:
            return PermissionDecision(False, reason=f"Tool '{tool_name}' is denied.")
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            return PermissionDecision(False, reason=f"Tool '{tool_name}' is not in allowed list.")

        # 2. Critical path check — last-resort safety net.
        if file_path:
            resolved = str(Path(file_path).expanduser().resolve())
            for pattern in _CRITICAL_PATH_PATTERNS:
                if fnmatch.fnmatch(resolved, pattern):
                    return PermissionDecision(
                        False,
                        reason=f"Path '{file_path}' is protected — denied.",
                    )

        # 3. User-configured path rules.
        if file_path and self._path_rules:
            for pattern, allow in self._path_rules:
                if fnmatch.fnmatch(file_path, pattern):
                    if not allow:
                        return PermissionDecision(
                            False,
                            reason=f"Path '{file_path}' matches deny rule '{pattern}'.",
                        )

        # 4. User-configured command deny patterns.
        if command:
            for pattern in self._denied_commands:
                if fnmatch.fnmatch(command, pattern):
                    return PermissionDecision(
                        False,
                        reason=f"Command matches deny pattern '{pattern}' — denied.",
                    )

        # 5. Mode-based decision.
        return self._mode_decision(is_read_only, command)

    def _mode_decision(
        self,
        is_read_only: bool,
        command: str | None,
    ) -> PermissionDecision:
        """Mode-based logic — the core of the permission system."""
        if self.mode == "bypass":
            return PermissionDecision(True)

        if self.mode == "plan" and not is_read_only:
            return PermissionDecision(False, reason="Read-only mode (plan)")

        if is_read_only:
            return PermissionDecision(True)

        if self.mode == "accept-edits":
            if command:
                return PermissionDecision(
                    False, requires_confirmation=True,
                    reason="Shell commands require confirmation."
                )
            return PermissionDecision(True)

        # default: confirm all mutations.
        return PermissionDecision(
            False, requires_confirmation=True,
            reason="Write / shell operations require confirmation."
        )

    # ------------------------------------------------------------------
    # Convenience methods (called by tools)
    # ------------------------------------------------------------------

    def can_read(self, path: Path) -> PermissionDecision:
        """Check read permission.

        Reads are always allowed UNLESS the path matches a critical
        pattern or a user-configured deny rule.
        """
        return self.evaluate(
            tool_name="read_file",
            file_path=str(path),
            is_read_only=True,
        )

    def can_write(self, path: Path) -> PermissionDecision:
        """Check write permission with interactive confirmation.

        Also rejects garbage filenames that look like shell artifacts
        (e.g. ``C[CLI`` from broken markdown link parsing).
        """
        # Reject paths with clearly garbage characters.
        name = path.name
        garbage_chars = set("[]{}()$`\\")
        if any(c in name for c in garbage_chars):
            return PermissionDecision(
                False,
                reason=f"Filename contains invalid characters: {name!r}. "
                       f"Remove brackets/braces/dollar/backtick from the path.",
            )

        result = self.evaluate(
            tool_name="write_file",
            file_path=str(path),
            is_read_only=False,
        )
        return self._resolve_interactive(result, f"Allow write to {self._relative(path)}?")

    def can_run_command(self, command: str) -> PermissionDecision:
        """Check shell command permission with interactive confirmation."""
        result = self.evaluate(
            tool_name="bash",
            command=command,
            is_read_only=False,
        )
        preview = command[:120] + "..." if len(command) > 120 else command
        return self._resolve_interactive(result, f"Allow: {preview}?")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_interactive(
        self, result: PermissionDecision, prompt: str
    ) -> PermissionDecision:
        """If the result requires confirmation, ask the user interactively."""
        if not result.requires_confirmation:
            return result
        if _ask_confirmation(prompt):
            return PermissionDecision(True)
        return PermissionDecision(False, reason="User denied.")

    def resolve_interactive(
        self, result: PermissionDecision, prompt: str
    ) -> PermissionDecision:
        """Public wrapper for registry-level permission prompts."""
        return self._resolve_interactive(result, prompt)

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)


def _ask_confirmation(prompt: str) -> bool:
    """Read a yes/no permission response from stdin with a deny-by-default policy.

    Rich's Confirm helper is convenient, but in a streaming REPL it can leave
    residual input behind when stdout is busy. A direct line read keeps the
    permission prompt atomic: exactly one stdin line is consumed per decision.
    """
    while True:
        _console.print(f"  [bold yellow]?[/] {prompt} [y/n] (n): ", end="")
        try:
            answer = input()
        except (EOFError, KeyboardInterrupt):
            _console.print()
            return False

        normalized = answer.strip().lower()
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"", "n", "no"}:
            return False
        _console.print("[yellow]Please answer y or n.[/yellow]")
