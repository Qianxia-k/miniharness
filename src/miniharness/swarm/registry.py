"""Registry for MiniHarness delegated-agent backends."""

from __future__ import annotations

from miniharness.swarm.subprocess_backend import SubprocessBackend
from miniharness.swarm.types import TeammateExecutor


class BackendRegistry:
    """Store available teammate backends by type."""

    def __init__(self) -> None:
        self._executors: dict[str, TeammateExecutor] = {}
        self.register(SubprocessBackend())

    def register(self, executor: TeammateExecutor) -> None:
        self._executors[executor.backend_type] = executor

    def get_executor(self, backend_type: str = "subprocess") -> TeammateExecutor:
        executor = self._executors.get(backend_type)
        if executor is None:
            raise ValueError(f"Unknown teammate backend: {backend_type}")
        return executor


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
