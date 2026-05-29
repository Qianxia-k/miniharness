"""Core Memory — persistent knowledge injected into every system prompt.

Stored as ``core.md`` in the per-project memory directory.  The file uses
simple markdown headings to organise three sections:

- ``## 项目上下文`` (or ``## Project Context``)
- ``## 用户偏好`` (or ``## User Preferences``)
- ``## 重要决策`` (or ``## Key Decisions``)

The file is human-editable and survives context compaction — it is never
trimmed or summarised away.
"""

from __future__ import annotations

from pathlib import Path

from miniharness.memory.store import get_memory_dir

# Default template for new core.md files.
_DEFAULT_CORE_MD = """## 项目上下文

<!-- 项目描述、技术栈、架构约定等 -->

## 用户偏好

<!-- 用户的工作方式、编码风格偏好等 -->

## 重要决策

<!-- 关键的架构决策和原因 -->
"""


class CoreMemory:
    """Load, render, and persist the core memory block.

    The rendered text is prepended to the system prompt so the model
    always has access to it regardless of how much compaction has run.
    """

    def __init__(self, cwd: str | Path) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._path = get_memory_dir(self._cwd) / "core.md"

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def read(self) -> str:
        """Return the raw markdown content, creating a default file if needed."""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(_DEFAULT_CORE_MD, encoding="utf-8")
        return self._path.read_text(encoding="utf-8")

    def write(self, content: str) -> None:
        """Overwrite the core.md file with new content."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Render for system prompt
    # ------------------------------------------------------------------

    def render_for_system_prompt(self) -> str:
        """Return a compact block suitable for injection into the system prompt.

        Strips HTML comments and excessive blank lines so the result is
        as short as possible while preserving all user-written content.
        """
        raw = self.read()
        # Remove HTML comments (<!-- ... -->).
        import re

        cleaned = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
        # Collapse 3+ blank lines into 2.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        # Strip leading / trailing whitespace.
        cleaned = cleaned.strip()
        if not cleaned:
            return ""

        return f"[Core Memory]\n{cleaned}"
