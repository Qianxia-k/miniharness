"""Command registry — stores and dispatches slash commands.

The registry is the central lookup table for all ``/<name>`` commands.
Handlers are registered by name and dispatched when the user types
a line starting with ``/``.

Commands come from:
    - **Built-in** — always registered (``/exit``, ``/help``, etc.)
    - **Skills** — auto-generated from user-invocable skills in the skill registry
    - **Extensions** — future: hooks and plugins can register commands
"""

from __future__ import annotations

import inspect
from typing import Any

from miniharness.commands.types import CommandContext, CommandHandler, CommandResult
from miniharness.plugins.gating import is_plugin_active


class CommandRegistry:
    """In-memory registry mapping command names → handlers.

    Usage::

        registry = CommandRegistry()
        registry.register("help", help_handler, description="Show help")
        registry.register("exit", exit_handler, aliases=["quit", "q"])

        result = registry.dispatch("/exit", context)
        if result.exit:
            break
    """

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}
        self._descriptions: dict[str, str] = {}
        self._sources: dict[str, str] = {}  # name → "builtin" | "skill" | "extension"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        description: str = "",
        aliases: list[str] | None = None,
        source: str = "builtin",
    ) -> None:
        """Register a command.

        Parameters
        ----------
        name:
            Command name without the ``/`` prefix.
        handler:
            Callable ``(args: str, ctx: CommandContext) -> CommandResult``.
        description:
            One-line description shown in ``/help``.
        aliases:
            Additional names that invoke the same handler.
        source:
            Where this command came from (``"builtin"``, ``"skill"``, ``"extension"``).
            Used for deduplication — built-in commands take precedence.
        """
        # Built-in commands are never overridden by skills or extensions.
        existing_source = self._sources.get(name)
        if existing_source == "builtin":
            return
        if existing_source == "skill" and source == "extension":
            # Extensions can override skills.
            pass

        self._handlers[name] = handler
        self._descriptions[name] = description
        self._sources[name] = source

        for alias in (aliases or []):
            if alias not in self._handlers or self._sources.get(alias) != "builtin":
                self._handlers[alias] = handler

    def register_single_skill(self, skill: Any) -> None:
        """Register a single ``SkillDefinition`` as a slash command."""
        if not getattr(skill, "user_invocable", True):
            return
        cmd_name = getattr(skill, "command_name", None)
        if not cmd_name or not _is_valid_command_name(cmd_name):
            return
        self.register(
            cmd_name,
            _make_skill_handler(skill),
            description=getattr(skill, "description", ""),
            aliases=_skill_aliases(skill, cmd_name),
            source="skill",
        )

    def register_from_skills(self, skill_registry: Any) -> None:
        """Auto-register a ``/<name>`` command for each user-invocable skill."""
        for skill in skill_registry.list_skills():
            if not skill.user_invocable:
                continue
            cmd_name = getattr(skill, "command_name", skill.name)
            if not cmd_name or not _is_valid_command_name(cmd_name):
                continue

            self.register(
                cmd_name,
                _make_skill_handler(skill),
                description=skill.description,
                aliases=_skill_aliases(skill, cmd_name),
                source="skill",
            )
    def register_from_tools(self, tool_registry: Any) -> None:
        """Auto-register commands for tools exposed by the model.

        Future: this will allow models to dynamically add commands as they
        discover new tools.
        """
        for tool in tool_registry.to_openai_tools():
            cmd_name = tool["name"]
            if not _is_valid_command_name(cmd_name):
                continue

            self.register(
                cmd_name,
                _make_tool_handler(tool),
                description=tool.get("description", ""),
                source="tool",
            )
    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, line: str, ctx: CommandContext) -> CommandResult:
        """Parse and execute a slash command.

        Parameters
        ----------
        line:
            Raw input line starting with ``/`` (e.g. ``"/model gpt-4"``).
        ctx:
            The current command context.

        Returns
        -------
        CommandResult
            Telling the REPL what to do next.
        """
        if not line.startswith("/"):
            return CommandResult.ok()

        # Parse: "/model gpt-4" → ("model", "gpt-4")
        parts = line[1:].split(maxsplit=1)
        name = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = self.lookup(name)
        if handler is None:
            return CommandResult.ok(
                f"Unknown command: /{name}. Type /help for available commands."
            )

        try:
            result = handler(args, ctx)
            if inspect.isawaitable(result):
                return CommandResult.ok(
                    f"Command /{name} must be run through the async runtime dispatcher."
                )
            return result
        except Exception as exc:
            return CommandResult.ok(f"Command error: {exc}")

    async def dispatch_async(self, line: str, ctx: CommandContext) -> CommandResult:
        """Parse and execute a slash command, awaiting async handlers."""
        if not line.startswith("/"):
            return CommandResult.ok()

        parts = line[1:].split(maxsplit=1)
        name = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = self.lookup(name)
        if handler is None:
            return CommandResult.ok(
                f"Unknown command: /{name}. Type /help for available commands."
            )

        try:
            result = handler(args, ctx)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            return CommandResult.ok(f"Command error: {exc}")

    def lookup(self, name: str) -> CommandHandler | None:
        """Find a handler by command name (without ``/``)."""
        return self._handlers.get(name.lower())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_commands(self) -> list[dict[str, Any]]:
        """Return all registered commands with metadata."""
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for name, handler in self._handlers.items():
            if name in seen:
                continue
            seen.add(name)
            result.append({
                "name": name,
                "description": self._descriptions.get(name, ""),
                "source": self._sources.get(name, "builtin"),
            })
        result.sort(key=lambda c: (c["source"] != "builtin", c["name"]))
        return result

    @property
    def count(self) -> int:
        return len({n for n in self._handlers})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_valid_command_name(name: str) -> bool:
    """Check that a command name is safe to register.

    Must be non-empty, alphanumeric + hyphens, not purely numeric.
    """
    if not name or not name.strip():
        return False
    return all(c.isalnum() or c in "-_:" for c in name)


def _skill_aliases(skill: Any, cmd_name: str) -> list[str] | None:
    """Return safe aliases for a skill slash command.

    Plugin skills must remain namespaced, so they never get a bare local-name
    alias such as ``/hello-world``.
    """
    if getattr(skill, "plugin_name", None):
        return None
    return [skill.name] if skill.name != cmd_name else None


def _make_skill_handler(skill) -> CommandHandler:
    """Create a handler that submits a skill as a user prompt.

    When the user types ``/<skill-name> [args]``, the skill content
    is rendered (with ``${ARGUMENTS}`` and ``${SKILL_DIR}`` substitution)
    and submitted as the next user prompt.
    """

    def handler(args: str, ctx: CommandContext) -> CommandResult:
        plugin_name = getattr(skill, "plugin_name", None)
        if plugin_name:
            plugin_index = getattr(getattr(ctx, "loop", None), "_plugin_index", None)
            if not is_plugin_active(plugin_name, plugin_index):
                return CommandResult.ok(
                    f"Plugin skill '/{skill.invocation_name}' is inactive. "
                    f"Run /plugins {plugin_name} on first."
                )

        # Build the prompt from skill content.
        content = skill.content
        if skill.base_dir:
            content = content.replace("${SKILL_DIR}", skill.base_dir)
        content = content.replace("${ARGUMENTS}", args)
        content = content.replace("$ARGUMENTS", args)

        # If args were provided and neither placeholder was in the content,
        # append them at the end.
        if args and "${ARGUMENTS}" not in skill.content and "$ARGUMENTS" not in skill.content:
            content = f"{content}\n\nUser input: {args}"

        prompt = (
            f"[Skill invoked: /{skill.invocation_name}]\n"
            f"Follow these instructions:\n\n{content}"
        )
        return CommandResult.prompt(prompt)

    return handler
