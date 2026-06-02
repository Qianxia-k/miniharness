"""Hook executor — runs hooks sequentially for a given event.

The executor is the engine that:
1. Looks up all hooks registered for an event.
2. Filters them by **matcher** (optional fnmatch pattern).
3. Dispatches each hook to the appropriate runner (command or prompt).
4. Collects results into an ``AggregatedHookResult``.

Mirrors OpenHarness's ``HookExecutor``, simplified for teaching clarity.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miniharness.hooks.events import HookEvent
from miniharness.hooks.registry import HookRegistry
from miniharness.hooks.schemas import (
    CommandHookDefinition,
    ConfirmHookDefinition,
    HookDefinition,
    PromptHookDefinition,
)
from miniharness.hooks.types import AggregatedHookResult, HookResult


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


@dataclass
class HookExecutionContext:
    """Context passed to every hook execution.

    Attributes
    ----------
    cwd:
        Working directory for command hooks.
    llm_stream:
        Async streaming function for prompt hooks (same signature as
        ``LLMClient.stream``).
    """

    cwd: Path
    llm_stream: Any = None  # async callable for LLM validation


# ---------------------------------------------------------------------------
# Argument injection
# ---------------------------------------------------------------------------


def _inject_arguments(template: str, payload: dict[str, Any], *, shell_escape: bool = False) -> str:
    """Replace ``$KEY_NAME`` placeholders in *template* with payload values.

    ``$ARGUMENTS`` is always replaced with the full JSON payload.

    If *shell_escape* is True, values are passed through ``shlex.quote()``
    (for command hooks).  Otherwise they are inserted as-is (for prompt hooks).
    """
    # Replace the full-payload placeholder first.
    payload_json = json.dumps(payload, ensure_ascii=False)
    if shell_escape:
        template = template.replace("$ARGUMENTS", shlex.quote(payload_json))
    else:
        template = template.replace("$ARGUMENTS", payload_json)

    # Replace individual key placeholders (e.g. $TOOL_NAME → payload["tool_name"]).
    for key, value in payload.items():
        placeholder = f"${key.upper()}"
        if placeholder in template:
            str_value = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            if shell_escape:
                str_value = shlex.quote(str_value)
            template = template.replace(placeholder, str_value)

    return template


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def _matches_hook(hook: HookDefinition, payload: dict[str, Any]) -> bool:
    """Return True if *hook* should fire for this *payload*.

    If the hook has no ``matcher``, it always fires.

    If a matcher is set, we build a list of **candidate subjects** from
    the payload and match against ALL of them — the hook fires if ANY
    candidate matches.  Candidates include:

    1. ``tool_name`` — e.g. ``"bash"``, ``"read_file"``
    2. ``tool_name:input_value`` — e.g. ``"bash:rm -rf /tmp"``,
       ``"write_file:/etc/hostname"``, ``"grep:password"``.
       Extracted from ``tool_input.command``, ``.path``, ``.root``,
       ``.url``, or ``.query``.
    3. ``prompt`` — the user's prompt text
    4. ``event`` — the event name (e.g. ``"pre_tool_use"``)

    This means you can write matchers like:

    * ``"bash:rm*"`` — block bash commands starting with ``rm``
    * ``"*:pip install*"`` — block any tool containing ``pip install``
    * ``"read_file:/etc/*"`` — block read_file on ``/etc`` paths
    * ``"*password*"`` — block anything mentioning ``password``
    * ``"bash"`` — match ALL bash invocations (broad audit)
    """
    matcher = getattr(hook, "matcher", None)
    if not matcher:
        return True

    subjects = _extract_matchable_subjects(payload)
    return any(fnmatch.fnmatch(s, matcher) for s in subjects)


def _extract_matchable_subjects(payload: dict[str, Any]) -> list[str]:
    """Build a list of strings from the payload for matcher comparison.

    Each string represents one "angle" the matcher can test against.
    """
    subjects: list[str] = []

    # 1. Tool name (always present in tool hooks).
    tool_name = str(payload.get("tool_name") or "")
    if tool_name:
        subjects.append(tool_name)

    # 2. Tool input values — extract actionable strings.
    #    For "bash", the input is {"command": "rm -rf /"}.
    #    For "write_file", the input is {"path": "/etc/hostname", "content": "..."}.
    #    We create "tool_name:value" pairs so you can write matchers like
    #    "bash:rm*" or "write_file:/etc/*".
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        for key in ("command", "path", "root", "url", "query"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                # Combined form: "bash:rm -rf /tmp"
                if tool_name:
                    subjects.append(f"{tool_name}:{val}")
                # Raw value: "rm -rf /tmp" (matches *rm* patterns directly)
                subjects.append(val)

    # 3. Prompt text (for user_prompt_submit).
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        subjects.append(prompt)

    # 4. Event name (always present — injected by HookExecutor.execute).
    event = payload.get("event")
    if isinstance(event, str) and event.strip():
        subjects.append(event)

    return subjects


# ---------------------------------------------------------------------------
# JSON parser for prompt hooks
# ---------------------------------------------------------------------------


def _parse_hook_response(text: str) -> dict[str, Any]:
    """Parse a model's response from a prompt hook.

    Tries to extract ``{"ok": true/false, "reason": "..."}``.
    Falls back to treating the raw text as a reason string.
    """
    # Try strict JSON parse first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from within markdown code fences.
    import re
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: treat simple affirmative responses as ok.
    lowered = text.strip().lower()
    if lowered in {"ok", "true", "yes", "safe", "allow"}:
        return {"ok": True}
    if lowered.startswith("ok"):
        return {"ok": True}

    # Anything else: not ok, use the text as the reason.
    return {"ok": False, "reason": text.strip() or "hook returned invalid response"}


# ---------------------------------------------------------------------------
# HookExecutor
# ---------------------------------------------------------------------------


class HookExecutor:
    """Execute hooks sequentially for a lifecycle event.

    Usage::

        registry = load_hook_registry(hooks_config)
        ctx = HookExecutionContext(cwd=Path.cwd(), llm_stream=llm.stream)
        executor = HookExecutor(registry, ctx)

        result = await executor.execute(HookEvent.PRE_TOOL_USE, {
            "tool_name": "bash",
            "tool_input": {"command": "rm file.txt"},
            "session_id": "abc123",
        })

        if result.blocked:
            print(f"Blocked: {result.reason}")
    """

    def __init__(
        self,
        registry: HookRegistry,
        context: HookExecutionContext,
    ) -> None:
        self._registry = registry
        self._context = context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> AggregatedHookResult:
        """Run all matching hooks for *event*.

        Hooks run sequentially — each one completes before the next starts.
        A ``matcher`` on the hook filters by tool_name/prompt/event.
        """
        # Inject the event name into the payload so $EVENT works in templates.
        payload = {**payload, "event": event.value}

        results: list[HookResult] = []
        for hook in self._registry.get(event):
            if not _matches_hook(hook, payload):
                continue

            if isinstance(hook, CommandHookDefinition):
                result = await self._run_command_hook(hook, payload)
            elif isinstance(hook, PromptHookDefinition):
                result = await self._run_prompt_hook(hook, payload)
            elif isinstance(hook, ConfirmHookDefinition):
                result = await self._run_confirm_hook(hook, payload)
            else:
                continue

            results.append(result)

        return AggregatedHookResult(results=results)

    # ------------------------------------------------------------------
    # Hook runners
    # ------------------------------------------------------------------

    async def _run_command_hook(
        self,
        hook: CommandHookDefinition,
        payload: dict[str, Any],
    ) -> HookResult:
        """Run a shell command as a subprocess."""
        command = _inject_arguments(hook.command, payload, shell_escape=True)

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self._context.cwd),
                timeout=hook.timeout_seconds,
                env={
                    **os.environ,
                    "MINIHARNESS_HOOK_PAYLOAD": json.dumps(payload, ensure_ascii=False),
                },
            )
        except subprocess.TimeoutExpired:
            return HookResult(
                hook_type="command",
                success=False,
                output=f"Timeout after {hook.timeout_seconds}s",
                blocked=hook.block_on_failure,
                reason=f"Hook timed out after {hook.timeout_seconds}s",
            )
        except Exception as exc:
            return HookResult(
                hook_type="command",
                success=False,
                output=str(exc),
                blocked=hook.block_on_failure,
                reason=f"Hook execution failed: {exc}",
            )

        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"

        success = proc.returncode == 0
        return HookResult(
            hook_type="command",
            success=success,
            output=output.strip(),
            blocked=hook.block_on_failure and not success,
            reason=output.strip() if not success else "",
            metadata={"returncode": proc.returncode},
        )

    async def _run_prompt_hook(
        self,
        hook: PromptHookDefinition,
        payload: dict[str, Any],
    ) -> HookResult:
        """Ask a model to validate the event by returning structured JSON."""
        if self._context.llm_stream is None:
            return HookResult(
                hook_type="prompt",
                success=False,
                output="No LLM stream available for prompt hook",
                blocked=hook.block_on_failure,
                reason="Prompt hook requires llm_stream in HookExecutionContext",
            )

        prompt_text = _inject_arguments(hook.prompt, payload, shell_escape=False)

        # Build a strict validation prompt.
        system_prompt = (
            "You are a safety validator.  Analyze the event payload and "
            "decide whether it should be allowed.  "
            "You MUST respond with ONLY a JSON object and nothing else.\n\n"
            'Return {"ok": true} if everything is safe.\n'
            'Return {"ok": false, "reason": "explain why"} if something is wrong.\n\n'
            "Do NOT include any text outside the JSON object."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        # Collect model response.
        response_text = ""
        try:
            from miniharness.llm import StreamComplete, TextDelta

            async for event in self._context.llm_stream(
                messages=messages,
                tools=[],  # no tools — the model must only judge
            ):
                if isinstance(event, TextDelta):
                    response_text += event.text
                elif isinstance(event, StreamComplete):
                    response_text = event.message.content or response_text
        except Exception as exc:
            return HookResult(
                hook_type="prompt",
                success=False,
                output=str(exc),
                blocked=hook.block_on_failure,
                reason=f"LLM call failed: {exc}",
            )

        # Parse the response.
        parsed = _parse_hook_response(response_text)
        success = parsed.get("ok", False)
        reason = parsed.get("reason", "") if not success else ""

        return HookResult(
            hook_type="prompt",
            success=success,
            output=response_text.strip(),
            blocked=hook.block_on_failure and not success,
            reason=reason,
        )

    async def _run_confirm_hook(
        self,
        hook: ConfirmHookDefinition,
        payload: dict[str, Any],
    ) -> HookResult:
        """Ask the HUMAN USER for interactive approval.

        Renders the hook's message (with ``$KEY_NAME`` substitution),
        shows a prominent warning, and waits for the user to type ``yes``.

        Timeout or any response other than ``yes`` → denied.
        """
        message = _inject_arguments(hook.message, payload, shell_escape=False)

        try:
            from rich.console import Console
            from rich.prompt import Prompt

            console = Console()
            console.print()
            console.print(
                f"[bold red]⚠ APPROVAL REQUIRED[/bold red] "
                f"[dim]({hook.timeout_seconds}s timeout)[/dim]"
            )
            console.print(f"  {message}")
            console.print(
                "  [dim]Type [bold green]yes[/bold green] to approve, "
                "anything else to deny.[/dim]"
            )

            # Use asyncio.wait_for for timeout.
            async def _prompt() -> str:
                return await asyncio.to_thread(
                    Prompt.ask, "  [bold]Approve?[/bold]", default="no"
                )

            response = await asyncio.wait_for(
                _prompt(), timeout=hook.timeout_seconds
            )
        except asyncio.TimeoutError:
            return HookResult(
                hook_type="confirm",
                success=False,
                output="Timed out waiting for approval",
                blocked=True,
                reason=f"Approval timed out after {hook.timeout_seconds}s",
            )

        approved = response.strip().lower() == "yes"
        return HookResult(
            hook_type="confirm",
            success=approved,
            output=f"User responded: {response}",
            blocked=not approved,
            reason="" if approved else f"User denied: {response}",
        )
