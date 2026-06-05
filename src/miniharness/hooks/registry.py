"""Hook registry — stores hooks grouped by lifecycle event.

The registry is a simple in-memory container that maps each
:class:`~miniharness.hooks.events.HookEvent` to a list of
:class:`~miniharness.hooks.schemas.HookDefinition` objects.

Hooks can be registered from two sources:

1. **Settings** — the ``hooks`` field in ``Settings`` (ultimately from
   ``settings.json`` or programmatic configuration).
2. **Plugins** — future: plugins will contribute hooks via a ``.hooks``
   property.

Mirrors OpenHarness's ``HookRegistry`` + ``load_hook_registry()``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from miniharness.hooks.events import HookEvent
from miniharness.hooks.schemas import HookDefinition


class HookRegistry:
    """In-memory store for hooks, grouped by event.

    Usage::

        registry = HookRegistry()
        registry.register(HookEvent.PRE_TOOL_USE, CommandHookDefinition(
            command="echo 'audit: $TOOL_NAME'", matcher="bash"
        ))
        hooks = registry.get(HookEvent.PRE_TOOL_USE)
    """

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookDefinition]] = defaultdict(list)

    def register(self, event: HookEvent, hook: HookDefinition) -> None:
        """Register one hook for an event."""
        self._hooks[event].append(hook)

    def register_many(self, event: HookEvent, hooks: list[HookDefinition]) -> None:
        """Register multiple hooks for an event."""
        self._hooks[event].extend(hooks)

    def register_from_dict(self, event_name: str, hook_dict: dict) -> None:
        """Register a hook from a raw dict (used by plugin loading).

        The dict must have a ``"type"`` field.  Invalid dicts are silently
        skipped.  Unknown event names are also skipped.
        """
        from miniharness.hooks.schemas import (
            CommandHookDefinition,
            PromptHookDefinition,
        )
        try:
            event = HookEvent(event_name)
        except ValueError:
            return

        hook_type = hook_dict.get("type", "")
        try:
            if hook_type == "command":
                hook = CommandHookDefinition(**hook_dict)
            elif hook_type == "prompt":
                hook = PromptHookDefinition(**hook_dict)
            else:
                return
        except Exception:
            return

        self._hooks[event].append(hook)

    def get(self, event: HookEvent) -> list[HookDefinition]:
        """Return all hooks registered for *event*.

        Returns a copy so callers cannot mutate the registry.
        """
        return list(self._hooks.get(event, []))

    @property
    def total_count(self) -> int:
        """Total number of registered hooks across all events."""
        return sum(len(hooks) for hooks in self._hooks.values())

    def summary(self) -> str:
        """Return a human-readable summary of all registered hooks."""
        lines: list[str] = []
        for event in HookEvent:
            hooks = self.get(event)
            if not hooks:
                continue
            lines.append(f"  {event.value}:")
            for hook in hooks:
                matcher = getattr(hook, "matcher", None)
                detail = (
                    getattr(hook, "command", None)
                    or getattr(hook, "prompt", None)
                    or ""
                )
                suffix = f" matcher={matcher}" if matcher else ""
                # Truncate detail for display.
                if len(detail) > 80:
                    detail = detail[:77] + "..."
                lines.append(f"    - {hook.type}{suffix}: {detail}")
        return "\n".join(lines) if lines else "  (no hooks registered)"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_hook_registry(
    hooks_config: dict[str, list[dict[str, Any]]] | None = None,
    *,
    plugins: list | None = None,  # list[LoadedPlugin] | None
) -> HookRegistry:
    """Build a ``HookRegistry`` from settings + plugins.

    Parameters
    ----------
    hooks_config:
        Dict from ``settings.hooks`` or ``_build_hooks_config()``.
    plugins:
        Optional list of ``LoadedPlugin`` objects.  Plugin hooks are
        registered AFTER settings hooks (settings can override).

    Returns
    -------
    HookRegistry
    """
    from miniharness.hooks.schemas import (
        CommandHookDefinition,
        ConfirmHookDefinition,
        PromptHookDefinition,
    )

    _HOOK_TYPE_MAP: dict[str, type] = {
        "command": CommandHookDefinition,
        "prompt": PromptHookDefinition,
        "confirm": ConfirmHookDefinition,
    }

    registry = HookRegistry()

    # Settings hooks.
    if hooks_config:
        for raw_event, hook_dicts in hooks_config.items():
            try:
                event = HookEvent(raw_event)
            except ValueError:
                continue
            for hd in hook_dicts:
                hook_type = hd.get("type", "")
                model_cls = _HOOK_TYPE_MAP.get(hook_type)
                if model_cls is None:
                    continue
                try:
                    hook = model_cls(**hd)
                    registry.register(event, hook)
                except Exception:
                    continue

    # Plugin hooks.
    for plugin in (plugins or []):
        if not getattr(plugin, "enabled", True):
            continue
        for event_name, hooks in getattr(plugin, "hooks", {}).items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                continue
            for h in hooks:
                try:
                    registry.register_from_dict(event_name, h)
                except Exception:
                    continue

    return registry
