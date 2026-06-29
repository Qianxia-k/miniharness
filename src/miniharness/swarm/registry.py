"""Backend registry for MiniHarness delegated-agent execution."""

from __future__ import annotations

import logging
import os

from miniharness.swarm.subprocess_backend import SubprocessBackend
from miniharness.swarm.types import BackendStatus, TeammateExecutor


logger = logging.getLogger(__name__)

TEAMMATE_MODE_ENV_VAR = "MINIHARNESS_TEAMMATE_MODE"
AUTO_BACKEND = "auto"


class BackendRegistry:
    """Register, select, and health-check teammate execution backends.

    The registry mirrors the OpenHarness shape: tool implementations ask for a
    teammate executor, while backend selection and availability checks stay in
    this layer.  MiniHarness currently ships only the subprocess backend, but
    the selection contract is intentionally ready for tmux, in-process, or
    remote backends without rewriting model-facing tools.
    """

    def __init__(self, *, preferred_backend: str | None = None) -> None:
        self._executors: dict[str, TeammateExecutor] = {}
        self._preferred_backend = preferred_backend or os.environ.get(
            TEAMMATE_MODE_ENV_VAR,
            AUTO_BACKEND,
        )
        self._detected_backend: str | None = None
        self._register_defaults()

    def register(self, executor: TeammateExecutor) -> None:
        """Register a teammate executor under its declared backend type."""
        self._executors[executor.backend_type] = executor
        self._detected_backend = None
        logger.debug("Registered teammate backend: %s", executor.backend_type)

    def get_executor(self, backend_type: str | None = None) -> TeammateExecutor:
        """Return a teammate executor for an explicit or auto-selected backend."""
        resolved = self.resolve_backend_type(backend_type)
        return self._executors[resolved]

    def resolve_backend_type(self, backend_type: str | None = None) -> str:
        """Resolve ``auto`` or an explicit backend name to a registered backend."""
        requested = (backend_type or self._preferred_backend or AUTO_BACKEND).strip()
        if not requested or requested == AUTO_BACKEND:
            return self.detect_backend()

        executor = self._executors.get(requested)
        if executor is None:
            available = ", ".join(sorted(self._executors)) or "(none)"
            raise ValueError(f"Unknown teammate backend: {requested}. Available: {available}")
        if not executor.is_available():
            raise ValueError(f"Teammate backend is not available: {requested}")
        return requested

    def detect_backend(self) -> str:
        """Detect and cache the best currently available backend.

        MiniHarness currently has one built-in executor, so this selects the
        first available backend in registration order.  Keeping this as an
        explicit method avoids baking subprocess assumptions into tools.
        """
        if self._detected_backend is not None:
            return self._detected_backend

        for backend_type, executor in self._executors.items():
            if executor.is_available():
                self._detected_backend = backend_type
                logger.debug("Detected teammate backend: %s", backend_type)
                return backend_type
        raise RuntimeError("No available teammate backend registered")

    def set_preferred_backend(self, backend_type: str | None) -> None:
        """Set the process-local preferred backend and clear detection cache."""
        self._preferred_backend = backend_type or AUTO_BACKEND
        self._detected_backend = None

    def available_backends(self) -> list[str]:
        """Return registered backend type names."""
        return sorted(self._executors)

    def list_backends(self) -> list[BackendStatus]:
        """Return availability and active-selection status for all backends."""
        try:
            active = self.resolve_backend_type(None)
        except Exception:
            active = ""

        statuses: list[BackendStatus] = []
        for backend_type in sorted(self._executors):
            executor = self._executors[backend_type]
            available = executor.is_available()
            statuses.append(
                BackendStatus(
                    backend_type=backend_type,
                    available=available,
                    active=backend_type == active,
                    reason="" if available else "backend reported unavailable",
                )
            )
        return statuses

    def health_check(self) -> dict[str, object]:
        """Return a serializable health summary for registered backends."""
        statuses = self.list_backends()
        return {
            "backends": {
                status.backend_type: {
                    "available": status.available,
                    "active": status.active,
                    "reason": status.reason,
                }
                for status in statuses
            },
            "total_count": sum(1 for status in statuses if status.available),
        }

    def reset(self) -> None:
        """Reset registry state and re-register built-in backends."""
        self._executors.clear()
        self._detected_backend = None
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(SubprocessBackend())


_GLOBAL_BACKEND_REGISTRY: BackendRegistry | None = None


def get_backend_registry() -> BackendRegistry:
    global _GLOBAL_BACKEND_REGISTRY
    if _GLOBAL_BACKEND_REGISTRY is None:
        _GLOBAL_BACKEND_REGISTRY = BackendRegistry()
    return _GLOBAL_BACKEND_REGISTRY


def reset_backend_registry_for_tests(registry: BackendRegistry | None = None) -> BackendRegistry:
    global _GLOBAL_BACKEND_REGISTRY
    _GLOBAL_BACKEND_REGISTRY = registry or BackendRegistry()
    return _GLOBAL_BACKEND_REGISTRY
