"""Small async runtime event bus.

The bus keeps engine code decoupled from frontend protocols.  Producers emit
typed runtime events; subscribers decide whether to render them, serialize
them, log them, or assert on them in tests.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from miniharness.runtime.events import RuntimeEvent

RuntimeEventHandler = Callable[[RuntimeEvent], Awaitable[None] | None]


class RuntimeEventBus:
    """Fan-out runtime events to async or sync subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[RuntimeEventHandler] = []

    def subscribe(self, handler: RuntimeEventHandler) -> Callable[[], None]:
        """Subscribe a handler and return an unsubscribe function."""
        self._subscribers.append(handler)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    async def emit(self, event: RuntimeEvent) -> None:
        """Emit one event to all subscribers in registration order."""
        for handler in list(self._subscribers):
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
