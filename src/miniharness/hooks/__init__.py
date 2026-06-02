"""Hook system — lifecycle event interception and extension.

Module map::

    events.py    — HookEvent enum (10 lifecycle event types)
    schemas.py   — HookDefinition models (command / prompt / confirm)
    types.py     — HookResult + AggregatedHookResult
    registry.py  — HookRegistry + load_hook_registry()
    executor.py  — HookExecutor + HookExecutionContext
    presets.py   — Production preset configurations (no hardcoding)
    audit.py     — AuditLogger for structured JSONL audit trails
"""

from miniharness.hooks.audit import AuditLogger
from miniharness.hooks.events import HookEvent
from miniharness.hooks.executor import HookExecutionContext, HookExecutor
from miniharness.hooks.registry import HookRegistry, load_hook_registry
from miniharness.hooks.schemas import (
    CommandHookDefinition,
    ConfirmHookDefinition,
    PromptHookDefinition,
)
from miniharness.hooks.types import AggregatedHookResult, HookResult

__all__ = [
    # Events
    "HookEvent",
    # Registry
    "HookRegistry",
    "load_hook_registry",
    # Executor
    "HookExecutor",
    "HookExecutionContext",
    # Schemas
    "CommandHookDefinition",
    "PromptHookDefinition",
    "ConfirmHookDefinition",
    # Types
    "HookResult",
    "AggregatedHookResult",
    # Infrastructure
    "AuditLogger",
]
