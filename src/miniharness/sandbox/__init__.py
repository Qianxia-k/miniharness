"""MiniHarness sandbox module.

Mirrors OpenHarness's sandbox architecture in a teaching-friendly size:
    - path_validator: enforce filesystem boundaries (symlink-safe).
    - session: manage sandbox lifecycle (start / stop / is_active).
    - docker: run commands in an isolated Docker container.

Usage:
    from miniharness.sandbox import is_sandbox_active, validate_sandbox_path
"""

from miniharness.sandbox.path_validator import validate_sandbox_path
from miniharness.sandbox.session import (
    get_sandbox,
    is_sandbox_active,
    start_sandbox,
    stop_sandbox,
)

__all__ = [
    "get_sandbox",
    "is_sandbox_active",
    "start_sandbox",
    "stop_sandbox",
    "validate_sandbox_path",
]
