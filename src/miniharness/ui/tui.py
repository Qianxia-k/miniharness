"""Textual frontend for MiniHarness.

The TUI is a renderer over the backend protocol.  It does not own agent
business logic; prompts, slash commands, sessions, tools, permissions, and
memory extraction all flow through ``BackendHost`` and ``RuntimeController``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.panel import Panel
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from miniharness.ui.protocol import decode_event


class PermissionModal(ModalScreen[bool]):
    """Permission prompt overlay for mutating tools."""

    DEFAULT_CSS = """
    PermissionModal {
        align: center middle;
    }
    #permission-dialog {
        width: 64;
        height: auto;
        padding: 1 2;
        border: round green;
    }
    #permission-actions {
        align: center middle;
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, tool_name: str, prompt: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                Panel.fit(
                    f"Allow tool [bold]{self._tool_name}[/bold]?\n\n{self._prompt[:500]}",
                    title="Permission Required",
                )
            ),
            Horizontal(
                Button("Allow", id="allow", variant="success"),
                Button("Deny", id="deny", variant="error"),
                id="permission-actions",
            ),
            id="permission-dialog",
        )

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_key(self, event: events.Key) -> None:
        if event.key == "y":
            self.dismiss(True)
        elif event.key in ("n", "escape"):
            self.dismiss(False)


def _startup_requests_tui(resume_id: str | None, initial_prompt: str | None) -> list[dict]:
    """Build initial backend requests in runtime order."""
    requests: list[dict] = []
    if resume_id:
        line = "/resume" if resume_id == "latest" else f"/resume {resume_id}"
        requests.append({"type": "submit_line", "line": line})
    if initial_prompt:
        requests.append({"type": "submit_line", "line": initial_prompt})
    return requests


class MiniHarnessTUI(App[None]):
    """Production-oriented terminal UI for MiniHarness."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-row {
        height: 1fr;
    }
    #transcript-column {
        width: 3fr;
        min-width: 60;
    }
    #side-column {
        width: 1fr;
        min-width: 30;
    }
    #transcript {
        height: 1fr;
        border: solid green;
    }
    #current-response {
        min-height: 3;
        max-height: 8;
        border: round green;
        padding: 0 1;
    }
    #composer {
        dock: bottom;
        height: 3;
        border: solid green;
    }
    #status-panel, #session-panel, #memory-panel, #tool-panel {
        border: round grey;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("ctrl+l", "clear_transcript", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "focus_composer", "Focus"),
    ]

    def __init__(
        self,
        *,
        cwd: Path,
        prompt: str | None = None,
        resume_session_id: str | None = None,
    ) -> None:
        super().__init__()
        self._cwd = cwd
        self._initial_prompt = prompt
        self._resume_session_id = resume_session_id
        self._proc: subprocess.Popen | None = None
        self._reader_task: asyncio.Task | None = None
        self._assistant_buffer = ""
        self._busy = False
        self._model = "?"
        self._session_id = ""
        self._last_tool = "No tool calls yet."
        self._last_memory = "No memory updates yet."

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-row"):
            with Vertical(id="transcript-column"):
                yield RichLog(id="transcript", wrap=True, highlight=True, markup=True)
                yield Static("Ready.", id="current-response")
                yield Input(placeholder="Ask MiniHarness or enter a /command", id="composer")
            with Vertical(id="side-column"):
                yield Static("Starting...", id="status-panel")
                yield Static("Session pending.", id="session-panel")
                yield Static("No memory updates yet.", id="memory-panel")
                yield Static("No tool calls yet.", id="tool-panel")
        yield Footer()

    def on_mount(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "miniharness",
            "--backend-only",
            "--cwd",
            str(self._cwd),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader_task = asyncio.create_task(self._read_backend_events())
        for request in self._startup_requests():
            self._send(request)
        self.query_one("#composer", Input).focus()

    def on_unmount(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._proc is not None:
            try:
                self._send({"type": "shutdown"})
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()

    def _startup_requests(self) -> list[dict]:
        return _startup_requests_tui(self._resume_session_id, self._initial_prompt)

    @on(Input.Submitted, "#composer")
    def handle_submit(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""
        if not line or self._busy:
            return
        if line in ("/exit", "/quit", "/q"):
            self.exit()
            return
        self._submit_line(line)

    def _submit_line(self, line: str) -> None:
        self._busy = True
        composer = self.query_one("#composer", Input)
        composer.disabled = True
        self._append(f"[bold cyan]user>[/bold cyan] {line}")
        self._set_current("[dim]Working...[/dim]")
        self._send({"type": "submit_line", "line": line})

    async def _read_backend_events(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        loop = asyncio.get_event_loop()
        while True:
            try:
                raw = await loop.run_in_executor(None, self._proc.stdout.readline)
            except Exception:
                break
            if not raw:
                break
            event = decode_event(raw.strip())
            if event is None:
                continue
            self._dispatch_event(event)

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")

        if event_type == "ready":
            self._model = event.get("model", "?")
            self._session_id = event.get("session_id", "")
            self._append("[dim]system> backend ready[/dim]")
            self._refresh_sidebars()
            return

        if event_type == "assistant_delta":
            self._assistant_buffer += event.get("text", "")
            self._set_current(f"[bold]assistant>[/bold] {self._assistant_buffer}")
            return

        if event_type == "assistant_complete":
            text = self._assistant_buffer or event.get("text", "") or "(empty response)"
            self._append(f"[bold green]assistant>[/bold green] {text}")
            self._assistant_buffer = ""
            self._set_current("Ready.")
            return

        if event_type == "tool_started":
            name = event.get("tool_name", "?")
            tool_input = event.get("tool_input", {})
            payload = json.dumps(tool_input, ensure_ascii=False)[:240]
            self._last_tool = f"{name}\n{payload}"
            self._append(f"[dim]tool> {name} {payload}[/dim]")
            self._refresh_sidebars()
            return

        if event_type == "tool_completed":
            name = event.get("tool_name", "?")
            output = (event.get("output", "") or "").replace("\n", " ")
            is_error = bool(event.get("is_error", False))
            prefix = "tool-error>" if is_error else "tool-result>"
            self._last_tool = f"{name}\n{output[:500]}"
            style = "red" if is_error else "grey50"
            self._append(f"[{style}]{prefix} {name}: {output[:500]}[/{style}]")
            self._refresh_sidebars()
            return

        if event_type == "system_message":
            message = event.get("message", "")
            if message:
                if message.startswith("Memory updated:"):
                    self._last_memory = message
                    self._refresh_sidebars()
                self._append(f"[dim]system> {message}[/dim]")
            return

        if event_type == "status":
            message = event.get("message", "")
            if message:
                self._set_current("[dim]Ready.[/dim]")
                self._refresh_sidebars(extra_status=message)
            return

        if event_type == "permission_request":
            self._handle_permission_request(event)
            return

        if event_type == "error":
            self._append(f"[red]error> {event.get('message', '')}[/red]")
            self._finish_line()
            return

        if event_type == "line_complete":
            self._finish_line()
            return

        if event_type == "shutdown":
            self._append("[dim]system> backend shutdown[/dim]")
            self.exit()

    def _handle_permission_request(self, event: dict[str, Any]) -> None:
        request_id = event.get("request_id", "")
        tool_name = event.get("tool_name", "")
        prompt = event.get("prompt", "")

        def _done(allowed: bool | None) -> None:
            self._send({
                "type": "permission_response",
                "request_id": request_id,
                "allowed": bool(allowed),
            })
            label = "allowed" if allowed else "denied"
            self._append(f"[dim]permission> {tool_name}: {label}[/dim]")

        self.push_screen(PermissionModal(tool_name, prompt), _done)

    def _finish_line(self) -> None:
        self._busy = False
        composer = self.query_one("#composer", Input)
        composer.disabled = False
        composer.focus()
        if not self._assistant_buffer:
            self._set_current("Ready.")
        self._refresh_sidebars()

    def action_interrupt(self) -> None:
        if self._busy:
            self._send({"type": "interrupt"})

    def action_clear_transcript(self) -> None:
        self.query_one("#transcript", RichLog).clear()
        self._set_current("Transcript cleared.")

    def action_focus_composer(self) -> None:
        self.query_one("#composer", Input).focus()

    def action_quit(self) -> None:
        self.exit()

    def _append(self, message: str) -> None:
        self.query_one("#transcript", RichLog).write(message)

    def _set_current(self, message: str) -> None:
        self.query_one("#current-response", Static).update(message)

    def _refresh_sidebars(self, *, extra_status: str = "") -> None:
        status_lines = [
            "[b]Status[/b]",
            f"model: {self._model}",
            f"cwd: {self._cwd}",
        ]
        if extra_status:
            status_lines.append(extra_status)
        self.query_one("#status-panel", Static).update("\n".join(status_lines))
        self.query_one("#session-panel", Static).update(
            "\n".join(["[b]Session[/b]", self._session_id or "(pending)"])
        )
        self.query_one("#memory-panel", Static).update(
            "\n".join(["[b]Memory[/b]", self._last_memory])
        )
        self.query_one("#tool-panel", Static).update(
            "\n".join(["[b]Last Tool[/b]", self._last_tool])
        )

    def _send(self, message: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass


def run_tui(
    *,
    cwd: Path,
    prompt: str | None = None,
    resume_session_id: str | None = None,
) -> None:
    MiniHarnessTUI(cwd=cwd, prompt=prompt, resume_session_id=resume_session_id).run()
