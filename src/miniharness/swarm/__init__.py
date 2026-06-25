"""Swarm backend primitives for delegated MiniHarness agents."""

from miniharness.swarm.registry import BackendRegistry, get_backend_registry, reset_backend_registry_for_tests
from miniharness.swarm.subprocess_backend import SubprocessBackend
from miniharness.swarm.types import SpawnResult, TeammateMessage, TeammateSpawnConfig

__all__ = [
    "BackendRegistry",
    "SpawnResult",
    "SubprocessBackend",
    "TeammateMessage",
    "TeammateSpawnConfig",
    "get_backend_registry",
    "reset_backend_registry_for_tests",
]
