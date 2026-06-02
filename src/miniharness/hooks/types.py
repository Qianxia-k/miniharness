"""Hook result types.

When a hook runs, it produces a ``HookResult``.  When all hooks for an
event have run, their results are bundled into an ``AggregatedHookResult``
which the engine checks to decide whether to continue or block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HookResult:
    """Result from a single hook execution.

    Attributes
    ----------
    hook_type:
        The hook's ``type`` field (``"command"`` or ``"prompt"``).
    success:
        ``True`` if the hook passed (exit code 0, or parsed ``{"ok": true}``).
    output:
        Text output from the hook execution (stdout+stderr for commands,
        raw model response for prompt hooks).
    blocked:
        ``True`` when ``block_on_failure`` is set AND the hook failed.
        The engine checks this to decide whether to continue.
    reason:
        Human-readable explanation of the result.
    metadata:
        Extra structured data (e.g. ``{"returncode": 0}`` for commands).
    """

    hook_type: str
    success: bool
    output: str = ""
    blocked: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregatedHookResult:
    """Aggregated result for all hooks triggered by one event.

    Attributes
    ----------
    results:
        Individual results from each hook that ran for this event.
    blocked:
        ``True`` if **any** hook blocked continuation.
    reason:
        The ``reason`` from the first blocking hook (or empty string).
    """

    results: list[HookResult] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """Return whether any hook blocked continuation."""
        return any(r.blocked for r in self.results)

    @property
    def reason(self) -> str:
        """Return the first blocking reason, if any."""
        for r in self.results:
            if r.blocked:
                return r.reason or r.output
        return ""

    @property
    def all_passed(self) -> bool:
        """Return True if every hook succeeded."""
        return all(r.success for r in self.results) if self.results else True
