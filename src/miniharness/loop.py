"""The MiniHarness agent loop — conversation orchestration.

This module owns the **orchestration** of a coding-agent session:

    - Conversation lifecycle (create, append, compact, clear)
    - Turn loop (LLM call → tool execution → repeat)
    - Error recovery (PTL → reactive compaction, completion-token renegotiation)
    - Structured state carryover (tool_metadata updated after each tool)
    - Hook dispatch at lifecycle points (via ``hooks/``)

It does NOT own:

    - System prompt TEXT assembly → ``prompts/system.py``
    - Token budget / compaction MECHANICS → ``context/compiler.py``
    - Permission logic → ``permissions.py``
    - Runtime event rendering → ``runtime/events.py`` + frontend adapters
    - LLM wire protocol → ``llm.py``
    - Hook execution ENGINE → ``hooks/executor.py``

Architecture::

    ┌─────────────────────────────────────────────────────┐
    │  AgentLoop (orchestrator)                            │
    │                                                     │
    │  Dependencies (injected at __init__):                 │
    │    llm: LLMClient        ← provider wire protocol    │
    │    tools: ToolRegistry    ← tool execution            │
    │    permissions: Perm...   ← access control            │
    │    budget: ContextBudget  ← token accounting          │
    │    compiler: Context...   ← compaction orchestration  │
    │    core_memory: Core...   ← persistent project context│
    │    tool_metadata: dict    ← structured session state  │
    │    hook_executor: Hook... ← lifecycle hook dispatch   │
    │                                                     │
    │  Collaborators (called per-turn):                    │
    │    prompts.system.assemble_system_prompt()           │
    │    runtime event emission                             │
    │    context.carryover.*                                │
    │    tools.offload.offload_if_needed()                  │
    └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Awaitable, Callable

from miniharness.config.settings import Settings
from miniharness.context.budget import ContextBudget
from miniharness.context.carryover import (
    build_compact_attachments,
    init_tool_metadata,
    record_tool_carryover,
    remember_active_artifact,
    remember_user_goal,
)
from miniharness.context.compiler import ContextCompiler
from miniharness.hooks import (
    AggregatedHookResult,
    HookEvent,
    HookExecutionContext,
    HookExecutor,
    HookResult,
    load_hook_registry,
)
from miniharness.config.settings import HookSettings
from miniharness.hooks.presets import (
    approval_preset,
    audit_log_preset,
    code_security_preset,
    dangerous_command_preset,
    sensitive_file_preset,
    merge_config
)
from miniharness.llm import (
    CompactPhase,
    CompletionTokenLimitError,
    LLMClient,
    PromptTooLongError,
    StreamComplete,
    TextDelta,
)
from miniharness.mcp import McpClientManager, load_mcp_server_configs
from miniharness.memory.core import CoreMemory
from miniharness.messages import (
    Conversation,
    Message,
    normalize_tool_arguments,
    normalize_tool_calls,
)
from miniharness.permissions import PermissionChecker
from miniharness.plugins import load_plugins
from miniharness.plugins.gating import is_tool_visible
from miniharness.plugins.tool import PluginTool
from miniharness.prompts.project_instructions import load_project_instructions
from miniharness.prompts.system import assemble_system_prompt
from miniharness.providers import get_profile
from miniharness.runtime import (
    AssistantCompleteEvent,
    AssistantDeltaEvent,
    CompactProgressRuntimeEvent,
    RuntimeEvent,
    RuntimeEventBus,
    StatusRuntimeEvent,
    SystemRuntimeEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
)
from miniharness.skills import SkillTool, load_skill_registry
from miniharness.tasks import TaskListManager
from miniharness.tool_registry import create_default_registry
from miniharness.tools.offload import offload_if_needed


# ---------------------------------------------------------------------------
# Hook config assembly — bridges Settings → presets → registry
# ---------------------------------------------------------------------------


def _build_hooks_config(
    hook_settings,
    *,
    session_hooks: dict[str, Any] | None = None,
) -> dict:
    """Build the full hooks configuration dict from ``HookSettings``.

    This is the bridge between user configuration (Settings) and the
    hook system (load_hook_registry).  It:

    1. Loads enabled presets from ``hooks/presets.py``.
    2. Merges user-provided ``custom_hooks`` on top.
    3. Returns a dict ready for ``load_hook_registry()``.

    Users never need to call this directly — it runs automatically
    when ``AgentLoop`` is created.
    """

    hs = hook_settings if isinstance(hook_settings, HookSettings) else HookSettings()
    config: dict = {}

    if hs.dangerous_commands:
        merge_config(config, dangerous_command_preset())
    if hs.sensitive_files:
        merge_config(config, sensitive_file_preset())
    if hs.human_approval:
        merge_config(config, approval_preset())
    if hs.audit_log:
        merge_config(config, audit_log_preset(log_dir=hs.audit_log_dir))
    if hs.code_security_review:
        merge_config(config, code_security_preset())

    # User custom hooks are merged LAST — they can override presets.
    if hs.custom_hooks:
        merge_config(config, hs.custom_hooks)

    normalized_session_hooks = _normalize_hooks_config(session_hooks)
    if normalized_session_hooks:
        merge_config(config, normalized_session_hooks)

    return config


def _normalize_hooks_config(raw: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Normalize agent/session hook config into registry-ready shape."""
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, list[dict[str, Any]]] = {}
    for event, hooks in raw.items():
        event_name = str(event).strip()
        if not event_name:
            continue
        if isinstance(hooks, dict):
            normalized[event_name] = [hooks]
        elif isinstance(hooks, list):
            hook_list = [hook for hook in hooks if isinstance(hook, dict)]
            if hook_list:
                normalized[event_name] = hook_list
    return normalized


def _normalize_tool_policy(raw: dict[str, Any] | None) -> dict[str, list[str]]:
    """Normalize session-scoped agent tool policy."""
    if not isinstance(raw, dict):
        return {}
    policy: dict[str, list[str]] = {}
    tools = _string_list(raw.get("tools"))
    disallowed = _string_list(
        raw.get("disallowed_tools", raw.get("disallowedTools"))
    )
    if tools is not None:
        policy["tools"] = tools
    if disallowed:
        policy["disallowed_tools"] = disallowed
    return policy


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.replace("\n", ",").split(",")]
        return [item for item in items if item]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return None


def _tool_allowed_by_policy(name: str, policy: dict[str, list[str]]) -> bool:
    if not policy:
        return True
    lowered_name = name.lower()
    disallowed = policy.get("disallowed_tools") or []
    if any(_tool_pattern_matches(lowered_name, pattern) for pattern in disallowed):
        return False
    allowed = policy.get("tools")
    if allowed is None or any(pattern.strip() == "*" for pattern in allowed):
        return True
    return any(_tool_pattern_matches(lowered_name, pattern) for pattern in allowed)


def _tool_pattern_matches(lowered_name: str, pattern: str) -> bool:
    normalized = pattern.strip().lower()
    if not normalized:
        return False
    return lowered_name == normalized or fnmatchcase(lowered_name, normalized)


def _normalize_permission_mode(raw: str | None) -> str:
    if not raw:
        return "default"
    aliases = {
        "default": "default",
        "accept-edits": "accept-edits",
        "accept_edits": "accept-edits",
        "acceptEdits": "accept-edits",
        "bypass": "bypass",
        "bypassPermissions": "bypass",
        "dangerously-skip-permissions": "bypass",
        "full_auto": "bypass",
        "dontAsk": "bypass",
        "plan": "plan",
    }
    return aliases.get(raw.strip(), "default")


def _normalize_max_turns(raw: int | None, *, fallback: int) -> int:
    if raw is None:
        return max(1, int(fallback))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(fallback))


# ---------------------------------------------------------------------------
# Static system prompt — the unchanging part of the instructions.
# Dynamic parts (env info, core memory, on-demand memories) are assembled
# by ``prompts/system.py`` each turn.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are MiniHarness, a small coding agent.

You can explain code and use tools when needed. Be concise and practical.
"""


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Orchestrate a multi-turn coding-agent session.

    Owns the conversation history and structured session state
    (``tool_metadata``).  Each call to :meth:`run` executes one
    user prompt through the full agent loop.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        settings: Settings,
        permission_prompt: Callable[[str, str], Awaitable[bool]] | None = None,
        compact_progress: Callable[[dict], Awaitable[None]] | None = None,
        event_bus: RuntimeEventBus | None = None,
        system_prompt_override: str | None = None,
        system_prompt_mode: str | None = None,
        session_hooks: dict[str, Any] | None = None,
        tool_policy: dict[str, Any] | None = None,
        permission_mode: str | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.cwd = cwd
        self.settings = settings
        self._permission_prompt = permission_prompt
        self._compact_progress = compact_progress
        self._event_bus = event_bus
        self._system_prompt_override = (system_prompt_override or "").strip()
        self._system_prompt_mode = system_prompt_mode or "append"
        self._tool_policy = _normalize_tool_policy(tool_policy)
        self.max_turns = _normalize_max_turns(max_turns, fallback=settings.max_turns)

        # ── Provider & model ──────────────────────────────────────────
        provider_profile = get_profile(settings.provider.name)
        model = settings.provider.model or provider_profile.default_model
        base_url = settings.provider.base_url or provider_profile.base_url
        self.model = model

        # ── Collaborators (each owns one clear responsibility) ─────────
        if provider_profile.api_format == "anthropic":
            from miniharness.api.anthropic_client import AnthropicClient
            self._anthropic = AnthropicClient(
                api_key=provider_profile.resolve_api_key(),
                model=model,
                max_tokens=settings.agent.max_tokens or 4096,
                timeout=settings.agent.request_timeout,
            )
            self.llm = None
            self._stream_fn = self._anthropic.stream
        else:
            self._anthropic = None
            self.llm = LLMClient(
                profile=provider_profile,
                model=model,
                base_url=base_url,
                agent_settings=settings.agent,
            )
            self._stream_fn = self.llm.stream
        self.permissions = PermissionChecker(
            cwd=cwd,
            mode=_normalize_permission_mode(permission_mode),
        )

        # ── Plugins: load first — all downstream loaders consume them ──
        self._plugins = load_plugins(settings, cwd=cwd)

        # ── Plugin visibility index (for prompt/tool gating) ─────────
        self._plugin_index: list[dict] = []
        for plugin in self._plugins:
            entry = {
                "name": plugin.name,
                "description": plugin.description or "",
                "active": False,
                "skills": plugin.skills,
                "_plugin": plugin,  # full LoadedPlugin for hooks/MCP introspection
            }
            self._plugin_index.append(entry)

        # ── Structured session state ──────────────────────────────────
        self.tool_metadata = init_tool_metadata()
        self.task_manager = TaskListManager(self.tool_metadata)
        self.last_context_stats: dict[str, object] = {}

        # ── Hooks: settings + plugins → registry → executor ──────────
        self.hook_registry = load_hook_registry(
            _build_hooks_config(settings.hooks, session_hooks=session_hooks), plugins=self._plugins,
        )
        self._hook_executor = HookExecutor(
            registry=self.hook_registry,
            context=HookExecutionContext(
                cwd=cwd,
                llm_stream=self._stream_fn,
            ),
        )

        # ── MCP: settings + plugins → manager → tool adapters ────────
        self._mcp_manager = McpClientManager(
            load_mcp_server_configs(settings, cwd=cwd, plugins=self._plugins)
        )
        self.tools = create_default_registry(
            cwd=cwd, permissions=self.permissions,
            mcp_manager=self._mcp_manager,
            is_tool_enabled=self._is_tool_enabled,
            plugin_index=self._plugin_index,
            permission_prompt=self._permission_prompt,
            task_manager=self.task_manager,
            hook_executor=self._hook_executor,
        )

        # ── Skills: bundled + project + user + plugins → registry ────
        self.skill_registry = load_skill_registry(cwd=cwd, plugins=self._plugins)
        self.tools.register(SkillTool(
            cwd=cwd,
            registry=self.skill_registry,
            permissions=self.permissions,
            plugin_index=self._plugin_index,
        ))

        self.tools.register(PluginTool(
            cwd=cwd,
            permissions=self.permissions,
            plugin_index=self._plugin_index,
        ))

        self.budget = ContextBudget.for_model(
            model, ratio=settings.context_budget_ratio
        )
        self.core_memory = CoreMemory(cwd)
        self.compiler = ContextCompiler(
            budget=self.budget,
            llm_stream=self._stream_fn,
            keep_last_n_turns=settings.keep_last_n_turns,
            compact_progress=self._compact_progress,
        )

        # ── Session identity ──────────────────────────────────────────
        self.session_id: str | None = None
        self.tag: str = ""
        self._session_started = False  # lazy fire of SESSION_START hook
        self._mcp_connected = False  # lazy connect MCP on first run()

        # ── Conversation (first message = system prompt) ──────────────
        system_content = self._build_system_prompt(user_query="")
        self.conversation = Conversation()
        self.conversation.append(Message(role="system", content=system_content))

    # ------------------------------------------------------------------
    # Public: run one user prompt
    # ------------------------------------------------------------------

    async def run(self, prompt: str) -> str:
        """Execute one user prompt through the agent loop.

        Returns the final assistant text, or an error description.
        """
        # ── Lazy MCP connect (first run only) ─────────────────────────
        if not self._mcp_connected:
            self._mcp_connected = True
            if self._mcp_manager._configs:
                try:
                    await self._mcp_manager.connect_all()
                except Exception as exc:
                    await self._emit_event(SystemRuntimeEvent(
                        message=f"! MCP connection failed: {exc}"
                    ))

                await self._emit_mcp_diagnostics()

                for tool_info in self._mcp_manager.list_tools():
                    from miniharness.mcp.tool_adapter import McpToolAdapter
                    try:
                        self.tools.register(McpToolAdapter(
                            manager=self._mcp_manager,
                            tool_info=tool_info,
                            cwd=self.cwd,
                            permissions=self.permissions,
                        ))
                    except Exception:
                        pass

        # ── Per-turn setup ────────────────────────────────────────────
        self._refresh_system_prompt(user_query=prompt)
        remember_user_goal(self.tool_metadata, prompt)

        # ── Hook: session_start (lazy — fires on first run()) ──────────
        if not self._session_started:
            self._session_started = True
            await self._fire_hook(HookEvent.SESSION_START, {
                "cwd": str(self.cwd),
                "model": self.model,
                "session_id": self.session_id or "",
            })

        # ── Hook: user_prompt_submit ──────────────────────────────────
        hook_result = await self._fire_hook(HookEvent.USER_PROMPT_SUBMIT, {
            "prompt": prompt,
            "session_id": self.session_id or "",
        })
        if hook_result.blocked:
            return f"Hook blocked: {hook_result.reason}"

        self.conversation.append(Message(role="user", content=prompt))

        # ── Compile context (budget check + compaction) ────────────────
        # ── Hook: pre_compact (fires before any compaction) ────────────
        await self._fire_hook(HookEvent.PRE_COMPACT, {
            "trigger": "auto",
            "message_count": len(self.conversation.messages),
            "tokens_used": self.budget.tokens_used(self.conversation.to_openai()),
            "session_id": self.session_id or "",
        })

        tools_openai = self.tools.to_openai_tools()
        attachments = build_compact_attachments(self.tool_metadata)
        packet = await self.compiler.compile(
            self.conversation, tools_openai, attachments=attachments,
        )
        self.last_context_stats = dict(packet.stats)
        if packet.stats.get("compacted"):
            self._replace_conversation(packet.messages)
            await self._emit_event(SystemRuntimeEvent(
                message=_format_compaction_summary(packet.stats)
            ))

            # ── Hook: post_compact ────────────────────────────────────
            await self._fire_hook(HookEvent.POST_COMPACT, {
                "trigger": "auto",
                "tiers_run": [
                    t for t in ["tier1_microcompact", "tier2_context_collapse",
                                 "tier3_session_memory", "tier4_full_llm_compact"]
                    if packet.stats.get(t)
                ],
                "messages_before": packet.stats.get("dropped", 0) + len(packet.messages),
                "messages_after": len(packet.messages),
                "session_id": self.session_id or "",
            })

        # ── Turn loop ─────────────────────────────────────────────────
        max_tokens_override: int | None = None
        reactive_compact_attempted = False

        for turn in range(1, self.max_turns + 1):
            turn_messages = (
                packet.messages if turn == 1
                else self.conversation.to_openai()
            )

            try:
                response_message = await self._call_llm(
                    turn_messages, packet.tools, max_tokens_override
                )
            except CompletionTokenLimitError as exc:
                if exc.supported_limit is not None:
                    max_tokens_override = exc.supported_limit
                    await self._emit_event(StatusRuntimeEvent(
                        message=(
                            f"Model rejected max_tokens; retrying with "
                            f"limit {max_tokens_override}."
                        )
                    ))
                    continue
                return f"Error: {exc}"

            except PromptTooLongError:
                if not reactive_compact_attempted:
                    reactive_compact_attempted = True
                    await self._emit_event(StatusRuntimeEvent(
                        message="Prompt too long; running reactive compaction..."
                    ))
                    await self._emit_event(CompactProgressRuntimeEvent(
                        phase=str(CompactPhase.COMPACT_START.value),
                        detail={"trigger": "reactive"},
                    ))

                    msgs = self.conversation.to_openai()
                    attachments = build_compact_attachments(self.tool_metadata)
                    msgs, cstats = await self.compiler.compact_if_needed(
                        msgs, attachments=attachments,
                    )
                    if cstats.get("compacted"):
                        self._replace_conversation(msgs)
                        await self._emit_event(SystemRuntimeEvent(
                            message=_format_compaction_summary(cstats)
                        ))
                        await self._emit_event(CompactProgressRuntimeEvent(
                            phase=str(CompactPhase.COMPACT_END.value),
                            compacted=True,
                            tokens_after=int(cstats.get("token_count") or 0),
                            detail={"trigger": "reactive"},
                        ))
                        continue

                    await self._emit_event(CompactProgressRuntimeEvent(
                        phase=str(CompactPhase.COMPACT_FAILED.value),
                        detail={"trigger": "reactive"},
                    ))
                return "Error: prompt too long even after compaction."

            except Exception as exc:
                error_msg = str(exc)
                if "connect" in error_msg.lower() or "timeout" in error_msg.lower():
                    return f"Network error: {error_msg}"
                return f"API error: {error_msg}"

            if response_message is None:
                return "No response from model."

            self.conversation.append(response_message)

            if response_message.tool_calls:
                await self._execute_tools(response_message.tool_calls)
                continue

            await self._emit_event(AssistantCompleteEvent(
                text=response_message.content or ""
            ))
            return response_message.content or ""

        return "Reached maximum turns without a final answer."

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tools(self, tool_calls: list[dict]) -> None:
        """Execute tool calls: validate → check permissions → run →
        offload → record carryover → append to conversation."""
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                tool_call["function"]["arguments"] = normalize_tool_arguments(raw_args)
                self.conversation.append(Message(
                    role="tool",
                    content=f"Invalid JSON arguments: {raw_args}",
                    tool_call_id=tool_call_id,
                ))
                await self._emit_event(ToolCompletedEvent(
                    tool_name=tool_name,
                    output=f"Invalid JSON arguments: {raw_args}",
                    is_error=True,
                ))
                continue

            await self._emit_event(ToolStartedEvent(
                tool_name=tool_name,
                tool_input=arguments,
            ))

            # ── Hook: pre_tool_use (can block tool execution) ────────
            pre_result = await self._fire_hook(HookEvent.PRE_TOOL_USE, {
                "tool_name": tool_name,
                "tool_input": arguments,
                "session_id": self.session_id or "",
            })
            if pre_result.blocked:
                output = f"Hook blocked: {pre_result.reason}"
                self.conversation.append(Message(
                    role="tool",
                    content=output,
                    tool_call_id=tool_call_id,
                ))
                await self._emit_event(ToolCompletedEvent(
                    tool_name=tool_name,
                    output=output,
                    is_error=True,
                ))
                continue

            # Execute.
            result = await self.tools.execute(tool_name, arguments)

            # Offload large output.
            inline_text, artifact_path = offload_if_needed(
                tool_name=tool_name, output=result.output
            )
            if artifact_path is not None:
                remember_active_artifact(self.tool_metadata, str(artifact_path))
                await self._emit_event(SystemRuntimeEvent(
                    message=f"Output offloaded -> {artifact_path}"
                ))

            # ── Hook: post_tool_use (logging / audit / post-validation) ──
            await self._fire_hook(HookEvent.POST_TOOL_USE, {
                "tool_name": tool_name,
                "tool_input": arguments,
                "output": result.output[:500],  # first 500 chars for hooks
                "is_error": result.is_error,
                "session_id": self.session_id or "",
            })

            # ── Hook: tool_failed (alerting / retry on failure) ────────
            if result.is_error:
                await self._fire_hook(HookEvent.TOOL_FAILED, {
                    "tool_name": tool_name,
                    "tool_input": arguments,
                    "error": result.output[:500],
                    "session_id": self.session_id or "",
                })

            # Record structured state.
            record_tool_carryover(
                self.tool_metadata,
                tool_name=tool_name,
                arguments=arguments,
                result_output=result.output,  # use full output for carryover
                is_error=result.is_error,
                result_metadata=result.metadata,
            )

            # Append to conversation (use offloaded text if applicable).
            self.conversation.append(Message(
                role="tool",
                content=inline_text,
                tool_call_id=tool_call_id,
            ))
            await self._emit_event(ToolCompletedEvent(
                tool_name=tool_name,
                output=inline_text,
                is_error=result.is_error,
                artifact_path=str(artifact_path) if artifact_path is not None else "",
                original_output_chars=len(result.output or ""),
            ))

    # ------------------------------------------------------------------
    # System prompt management
    # ------------------------------------------------------------------

    def _build_system_prompt(self, *, user_query: str = "") -> str:
        """Assemble the full system prompt for a turn."""
        core_text = self.core_memory.render_for_system_prompt()
        instructions = load_project_instructions(self.cwd) or ""
        prompt = assemble_system_prompt(
            base_prompt=SYSTEM_PROMPT,
            cwd=self.cwd,
            core_memory_text=core_text,
            user_query=user_query,
            tool_count=len(self.tools._tools),
            skill_registry=self.skill_registry,
            mcp_manager=self._mcp_manager,
            plugin_index=self._plugin_index,
            project_instructions=instructions,
        )
        if self._system_prompt_override:
            if self._system_prompt_mode == "replace":
                return self._system_prompt_override
            return (
                f"{prompt}\n\n"
                "## Delegated Agent Role\n"
                f"{self._system_prompt_override}"
            )
        return prompt

    def _is_tool_enabled(self, name: str, tool) -> bool:
        """Runtime tool gating for plugin-contributed capabilities."""
        return (
            _tool_allowed_by_policy(name, self._tool_policy)
            and is_tool_visible(tool, self._plugin_index)
        )

    def _refresh_system_prompt(self, *, user_query: str = "") -> None:
        """Update the system prompt in-place for a new turn.

        Called at the top of :meth:`run` so the model sees fresh
        environment info and on-demand memories each turn.
        """
        if self.conversation.messages and self.conversation.messages[0].role == "system":
            self.conversation.messages[0].content = self._build_system_prompt(
                user_query=user_query
            )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def export_messages(self) -> list[dict]:
        """Export all messages as JSON-serializable dicts."""
        return [msg.to_openai() for msg in self.conversation.messages]

    def export_session_state(self) -> dict:
        """Export non-message session state."""
        import copy

        return {
            "tool_metadata": copy.deepcopy(self.tool_metadata),
        }

    def restore_messages(self, messages_data: list[dict]) -> None:
        """Replace the conversation with previously-saved messages.

        Message history and session state are restored separately so callers can
        handle older snapshots that do not contain structured metadata.
        """
        self.conversation = Conversation()
        for data in messages_data:
            message = Message(**data)
            if message.tool_calls:
                message.tool_calls = normalize_tool_calls(message.tool_calls)
            self.conversation.append(message)

    def restore_session_state(self, state: dict | None) -> None:
        """Restore structured session state in place.

        The dict object is mutated instead of replaced because tools hold
        references to it through managers such as ``TaskListManager``.
        """
        self.tool_metadata.clear()
        self.tool_metadata.update(init_tool_metadata())
        if isinstance(state, dict):
            metadata = state.get("tool_metadata")
            if isinstance(metadata, dict):
                self.tool_metadata.update(metadata)

    def set_model(self, model: str) -> None:
        """Switch the model and update dependent components."""
        self.model = model
        if self.llm is not None:
            self.llm.model = model
        if self._anthropic is not None:
            self._anthropic.model = model
        self.budget = ContextBudget.for_model(
            model, ratio=self.settings.context_budget_ratio
        )
        self.compiler.budget = self.budget

    def set_max_turns(self, max_turns: int) -> None:
        """Set the effective turn limit for this loop."""
        self.max_turns = max(1, int(max_turns))
        self.settings.max_turns = self.max_turns

    def clear(self) -> None:
        """Reset conversation and session state."""
        self.conversation = Conversation()
        self.conversation.append(
            Message(role="system", content=self._build_system_prompt())
        )
        self.tool_metadata.clear()
        self.tool_metadata.update(init_tool_metadata())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens_override: int | None,
    ) -> Message | None:
        """Stream an LLM call and emit frontend-neutral text deltas.

        Returns the assembled assistant Message, or None if streaming failed.
        """
        response_message = None
        # Anthropic needs system_prompt as a separate parameter.
        stream_kwargs: dict = {"messages": messages, "tools": tools}
        if max_tokens_override is not None:
            stream_kwargs["max_tokens_override"] = max_tokens_override
        if self._anthropic is not None:
            stream_kwargs["system_prompt"] = self.conversation.messages[0].content if self.conversation.messages else ""

        async for event in self._stream_fn(**stream_kwargs):
            if isinstance(event, TextDelta):
                await self._emit_event(AssistantDeltaEvent(text=event.text))
            elif isinstance(event, StreamComplete):
                response_message = event.message
        return response_message

    async def _emit_event(self, event: RuntimeEvent) -> None:
        """Emit a runtime event if this loop was wired to an event bus."""
        if self._event_bus is not None:
            await self._event_bus.emit(event)

    async def _emit_mcp_diagnostics(self) -> None:
        """Emit MCP connection diagnostics as frontend-neutral system events."""
        statuses = self._mcp_manager.list_statuses()
        tools_count = len(self._mcp_manager.list_tools())
        connected = sum(1 for s in statuses if s.state == "connected")

        if not statuses:
            return

        if connected > 0:
            await self._emit_event(SystemRuntimeEvent(
                message=(
                    f"MCP: {connected} server(s) connected, "
                    f"{tools_count} tool(s) available"
                )
            ))

        for s in statuses:
            if s.state == "failed":
                await self._emit_event(SystemRuntimeEvent(
                    message=f"! MCP server '{s.name}': {s.detail[:150]}"
                ))
            elif s.state == "pending":
                await self._emit_event(SystemRuntimeEvent(
                    message=f"MCP server '{s.name}' pending (will retry next run)"
                ))

    async def _fire_hook(
        self, event: HookEvent, payload: dict
    ) -> AggregatedHookResult:
        """Fire hooks for a lifecycle event.

        Returns the aggregated result.  Callers should check
        ``result.blocked`` to decide whether to stop.
        """
        try:
            return await self._hook_executor.execute(event, payload)
        except Exception:
            # Hooks are best-effort — a buggy hook must not crash the agent.
            return AggregatedHookResult(results=[
                HookResult(
                    hook_type="internal",
                    success=False,
                    output="Hook execution raised an exception",
                    blocked=False,  # never block on internal hook errors
                    reason="Internal hook error",
                )
            ])

    @property
    def mcp_manager(self):
        """Expose the MCP manager for /mcp command introspection."""
        return self._mcp_manager

    @property
    def stream_fn(self):
        """Return the active model stream function for runtime services."""
        return self._stream_fn

    def _replace_conversation(self, messages: list[dict]) -> None:
        """Replace the live conversation with a compacted message list."""
        self.conversation = Conversation()
        for m in messages:
            self.conversation.append(Message(**m))

def _format_compaction_summary(stats: dict) -> str:
    before = stats.get("token_count") or stats.get("total_used") or 0
    after = stats.get("tokens_after") or stats.get("token_count_after") or before
    dropped = stats.get("dropped")
    parts = [f"Context compacted: {before} -> {after} tokens"]
    if dropped:
        parts.append(f"dropped {dropped} message(s)")
    return ", ".join(parts) + "."
