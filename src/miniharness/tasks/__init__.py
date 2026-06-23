"""Session-scoped task state."""

from miniharness.tasks.background import (
    BackgroundTaskManager,
    BackgroundTaskRecord,
    get_background_task_manager,
    reset_background_task_manager_for_tests,
)
from miniharness.tasks.manager import (
    TaskItem,
    TaskListManager,
    format_task_list,
)

__all__ = [
    "BackgroundTaskManager",
    "BackgroundTaskRecord",
    "TaskItem",
    "TaskListManager",
    "format_task_list",
    "get_background_task_manager",
    "reset_background_task_manager_for_tests",
]
