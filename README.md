# MiniHarness

MiniHarness is a local-first coding agent harness for learning, dogfooding, and
iterating on production-style agent infrastructure. It is intentionally compact,
but it is not a prompt-only demo: the project has a shared runtime, structured
tool schemas, permission checks, hooks, MCP integration, sessions, memory,
context compaction, plugins, skills, background tasks, and both CLI and TUI
frontends over the same agent pipeline.

中文文档: [README.zh-CN.md](./README.zh-CN.md)

## What It Does

MiniHarness runs an agent loop like this:

```text
user input
  -> shared RuntimeController
  -> slash command handling or AgentLoop
  -> dynamic system prompt + compacted conversation context
  -> OpenAI-compatible streaming model call
  -> ToolRegistry dispatch
  -> permission and hook checks
  -> tool result / task notification / memory update
  -> session save + UI state snapshot
```

The CLI and TUI are frontends. They submit user input to the same runtime
controller and consume the same runtime events, so features such as sessions,
permissions, tools, memory, compaction, tasks, and plugin state are owned by the
backend pipeline rather than duplicated in the UI.

## Current Capabilities

- OpenAI-compatible streaming chat completions.
- Provider profiles for Qwen/DashScope, OpenAI, and compatible endpoints.
- Shared CLI/TUI runtime with structured events, state snapshots, task
  snapshots, permission prompts, and user-question prompts.
- Pydantic-validated built-in tools exposed through model tool schemas.
- File, search, shell, LSP, web fetch, todo, task, memory, plan-mode, and
  multi-agent tools.
- MCP client support for stdio and HTTP servers, including project-aware
  filesystem roots.
- Registry-level tool gating for inactive plugin tools and MCP adapters.
- Permission modes, sensitive-path protection, hook presets, and audit logging.
- Sessions with save, list, resume, tag, and isolated switching.
- Token estimation with `tiktoken`, context budget reporting, and tiered
  compaction.
- Core, semantic, episodic, and session-memory support.
- Plugin and skill discovery from bundled, project, user, and plugin sources.
- Background shell tasks and delegated agent tasks with coordinator result
  draining.
- Optional Docker sandbox for shell execution.

## Quick Start

```bash
git clone <repo-url>
cd miniharness
uv sync --extra dev
cp .env.example .env
```

Set one API credential:

```text
DASHSCOPE_API_KEY
OPENAI_API_KEY
MINIHARNESS_API_KEY
```

Run:

```bash
uv run mh "inspect this project"
uv run mh
uv run mh --tui
uv run mh --cwd /path/to/project "explain the codebase"
uv run mh --continue
uv run mh --resume <session-id-or-tag>
uv run mh --sessions
uv run mh --dry-run "check config"
```

## CLI Options

```text
uv run mh [PROMPT] [OPTIONS]

--cwd                    tool workspace and `${cwd}` expansion root
--profile                provider profile
--model, -m              override model name
--base-url               override API base URL
--dry-run                print resolved settings and exit
--max-turns              maximum agent loop turns
--context-budget-ratio   soft context budget ratio before compaction
--temperature            sampling temperature
--top-p                  nucleus sampling threshold
--max-tokens             maximum output tokens
--sandbox / --no-sandbox enable or disable Docker sandboxing
--sandbox-image          Docker image for sandbox execution
--continue, -c           resume the most recent session
--resume                 resume a session by ID or tag
--sessions               list saved sessions and exit
--tui                    launch the Textual frontend
```

## REPL Commands

```text
/help                 show commands
/exit, /quit, /q      exit
/clear                clear conversation history
/history              show message count
/project              show project instructions
/diff [full|staged]   show git diff output
/model                show or switch model
/turns                show or set max turns
/tokens               show context token budget
/permissions          cycle or inspect permission mode
/temperature          show or set temperature
/top-p                show or set top_p
/max-tokens           show or set max output tokens
/memory               inspect memory
/hooks                show hook configuration
/skills               list skills
/plugins [name]       list, inspect, or toggle plugins
/tools [name] [json]  list, inspect, or execute tools
/agents [name]        list or inspect delegated agent definitions
/tasks                show task snapshots
/mcp                  show MCP server status
/sessions             list sessions
/resume [id|tag]      resume session
/tag <name>           tag current session
```

During an active model/tool turn, slash commands are processed after the turn
returns. Press `Ctrl-C` in the CLI to interrupt the current turn.

## Built-in Tools

Core tools include:

```text
read_file       read text files with optional offset/limit
ls              list directory entries
grep            search literal text
glob            match files by glob pattern
git_status      inspect git root, branch, dirty files, and diff stats
git_diff        inspect unstaged, staged, or HEAD diff output
enter_worktree  create a git worktree for isolated coding work
exit_worktree   remove a git worktree by path
lsp             inspect Python symbols, definitions, references, and hover text
write_file      create or overwrite a file
edit_file       exact-string replacement with permission diff preview
bash            run shell commands in the workspace or sandbox
web_fetch       fetch a URL and convert HTML to text
todo_write      maintain the current todo list
task            maintain a session-scoped task list
task_*          create/list/get/output/stop/update background tasks
agent           start a delegated agent task
agent_list      list delegated agent definitions
send_message    send messages to delegated agent tasks
team_*          create/list/delete agent teams
ask_user_question ask the frontend/user for missing information
sleep           wait without blocking the process
enter_plan_mode / exit_plan_mode
memory_search / memory_add / memory_log
list_mcp_resources / read_mcp_resource / mcp_auth
```

Connected MCP tools are exposed as:

```text
mcp__<server>__<tool>
```

## Configuration

Settings are resolved in this order:

```text
defaults
-> user MCP config
-> project MCP config
-> MINIHARNESS_MCP_SERVERS
-> environment variables
-> provider auto-detection
-> CLI overrides
```

Common environment variables:

```text
MINIHARNESS_PROFILE
MINIHARNESS_MODEL
MINIHARNESS_BASE_URL
MINIHARNESS_MAX_TURNS
MINIHARNESS_CONTEXT_BUDGET_RATIO
MINIHARNESS_TEMPERATURE
MINIHARNESS_TOP_P
MINIHARNESS_MAX_TOKENS
MINIHARNESS_SANDBOX_ENABLED
MINIHARNESS_SANDBOX_IMAGE
MINIHARNESS_ALLOW_PROJECT_PLUGINS
DASHSCOPE_API_KEY
OPENAI_API_KEY
MINIHARNESS_API_KEY
```

## MCP

MCP server configuration is loaded from:

```text
~/.miniharness/mcp.json
<project>/.miniharness/mcp.json
MINIHARNESS_MCP_SERVERS
plugin mcp.json files
```

Project config overrides user config for servers with the same name. The common
workflow is to `cd` into the target project before launching `mh`; `${cwd}`,
`${workspace}`, `${project}`, and `${home}` templates are expanded at runtime.

Example:

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "allowed_directories": ["${cwd}"]
    }
  }
}
```

`allowed_directories` and `roots` are treated as filesystem roots. MCP schemas
come from external servers, so MiniHarness still routes MCP execution through
tool permissions and hooks.

## Skills And Plugins

Skills are markdown instruction files. MiniHarness injects a compact skill
index into context and lets the model load full instructions on demand through
the `skill` tool.

Skill sources:

```text
bundled skills
project .miniharness/skills/<name>/SKILL.md
project .claude/skills/<name>/SKILL.md
user ~/.miniharness/skills/<name>/SKILL.md
plugin skills
```

Plugins are discovered from:

```text
~/.miniharness/plugins/<name>/
<project>/.miniharness/plugins/<name>/
```

A plugin may contribute:

```text
plugin.json      manifest
skills/          skill definitions
hooks.json       hook definitions
mcp.json         MCP server definitions
agents/          delegated agent definitions
```

Use `/plugins` to inspect, enable, or disable plugin contributions.

## Sessions, Memory, And Context

Sessions are stored under:

```text
~/.miniharness/sessions/<project-slug>/
```

Session switching creates a fresh `AgentLoop` for the target conversation, which
keeps session IDs, histories, tool metadata, and save targets isolated.

The context system rebuilds the prompt each turn from static instructions,
runtime facts, project instructions, connected MCP status, enabled skills,
memory, conversation history, and carryover attachments. Token usage is
estimated with `tiktoken`; when the soft budget is exceeded, MiniHarness applies
tiered compaction and emits progress events for frontends.

## Permissions And Hooks

Permission modes:

```text
default       confirm writes, shell, and unknown mutating operations
accept-edits allow file edits, confirm shell
bypass        allow most actions except hard-denied critical paths
plan          read-only mode
```

Hooks provide a second safety layer for dangerous commands, sensitive paths,
human approval, and audit logs. Audit records are written under
`~/.miniharness/audit/` by default.

## Project Layout

```text
src/miniharness/cli.py              CLI entrypoint
src/miniharness/ui/                 TUI protocol, backend host, shared runtime
src/miniharness/loop.py             AgentLoop orchestration
src/miniharness/runtime/            runtime events
src/miniharness/state/              observable app state snapshots
src/miniharness/tool_registry.py    tool schemas, gating, execution
src/miniharness/tools/              built-in tools
src/miniharness/context/            token budget, carryover, compaction
src/miniharness/sessions/           session persistence and switching
src/miniharness/services/           LSP, memory extraction, session memory
src/miniharness/mcp/                MCP config, clients, adapters, resources
src/miniharness/skills/             skill discovery and loading
src/miniharness/plugins/            plugin discovery and contributions
src/miniharness/hooks/              hook events, presets, executor
src/miniharness/swarm/              delegated-agent coordination
src/miniharness/tasks/              background task runtime
src/miniharness/config/             settings and path helpers
```

## Verification

```bash
uv run pytest
python3 -m compileall src/miniharness
uv run ruff check .
```

The test suite covers permissions, MCP security behavior, hooks, sessions,
memory, token estimation, compaction events, runtime events, TUI runtime
behavior, state snapshots, task snapshots, background tasks, delegated-agent
coordination, tool registry behavior, skills, plugins, sandbox path validation,
and provider defaults.

## Known Limits

- MiniHarness is production-oriented, but still compact and under active
  hardening.
- Git workflow tools, patch application, richer LSP backends, provider hot
  switching, release packaging, and end-to-end dogfood tests are still important
  next steps.
- Direct MCP tools are exposed after connection. Plugin-contributed MCP tools
  are gated by plugin activation; large direct tool sets should add semantic
  per-turn tool selection.
- `edit_file` currently uses exact string replacement, not full patch
  application.
- Docker sandboxing requires Docker to be installed and available on `PATH`.

## License

MIT
