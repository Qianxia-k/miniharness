"""Path resolution for MiniHarness configuration and data directories."""

from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_BASE_DIR = ".miniharness"
_CONFIG_FILE_NAME = "settings.json"


def get_config_dir() -> Path:
    """Return the configuration directory, creating it if needed."""
    env_dir = os.environ.get("MINIHARNESS_CONFIG_DIR")
    config_dir = Path(env_dir).expanduser() if env_dir else Path.home() / _DEFAULT_BASE_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file_path() -> Path:
    """Return the path to the main settings file."""
    return get_config_dir() / _CONFIG_FILE_NAME


def get_data_dir() -> Path:
    """Return the data directory for persistent runtime state."""
    env_dir = os.environ.get("MINIHARNESS_DATA_DIR")
    data_dir = Path(env_dir).expanduser() if env_dir else get_config_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_logs_dir() -> Path:
    """Return the logs directory."""
    env_dir = os.environ.get("MINIHARNESS_LOGS_DIR")
    logs_dir = Path(env_dir).expanduser() if env_dir else get_config_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_sessions_dir() -> Path:
    """Return the session storage directory."""
    path = get_data_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_tasks_dir() -> Path:
    """Return the background task storage directory."""
    path = get_data_dir() / "tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_config_dir(cwd: str | Path) -> Path:
    """Return the per-project .miniharness directory."""
    path = Path(cwd).expanduser().resolve() / ".miniharness"
    path.mkdir(parents=True, exist_ok=True)
    return path
