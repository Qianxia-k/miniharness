
# MiniHarness

A teaching-first mini coding agent harness — a compact, readable reimplementation of the core ideas，Updating continuously 

Chinese: [README.zh-CN.md](./README.zh-CN.md)

Architecture summary

user input → message history → LLM call (streaming) → tool execution → loop until final answer

Key features

- Async streaming agent loop — real-time token output with tool call interception
- Multi-provider support — DashScope (Qwen), OpenAI, and OpenAI-compatible endpoints with provider auto-detection
- Built-in tools — `read_file`, `write_file`, `edit_file`, `grep`, `bash` (Pydantic input validation)
- Optional Docker sandbox for isolated bash execution
- Layered configuration (defaults → env → auto-detect → CLI overrides)
- Interactive permission prompts for write/bash operations
- Robust LLM retry with exponential backoff and jitter

Quick start

1. Clone and install

```bash
git clone <repo-url> && cd miniharness
uv sync --extra dev
```

2. Prepare credentials

```bash
cp .env.example .env
# Edit .env to set DASHSCOPE_API_KEY or OPENAI_API_KEY
```

3. Run the agent

```bash
uv run mh "explain this project"
uv run mh --dry-run "test"          # show resolved settings
uv run mh --sandbox "list files"    # run commands inside Docker sandbox
uv run mh -m gpt-4.1-mini "..."     # override model
```

Configuration

Settings are resolved in layers (lower → higher priority): defaults, environment variables, provider auto-detection, CLI args.

Common environment variables

- `DASHSCOPE_API_KEY` — Qwen (DashScope)
- `OPENAI_API_KEY` — OpenAI
- `MINIHARNESS_PROFILE` — force provider profile
- `MINIHARNESS_MODEL` — override model name
- `MINIHARNESS_MAX_TURNS` — max agent loop turns (default: 8)
- `MINIHARNESS_SANDBOX_ENABLED` — enable Docker sandbox
- `MINIHARNESS_SANDBOX_IMAGE` — Docker image for sandbox

CLI snapshot

```text
uv run mh [PROMPT] [OPTIONS]

	--profile       Provider profile (qwen, openai, openai-compatible)
	--model, -m     Override model name
	--base-url      Override API base URL
	--max-turns     Maximum agent loop turns
	--sandbox       Enable Docker sandbox
	--sandbox-image Docker image for sandbox
	--dry-run       Show resolved settings and exit
	--cwd           Working directory (default: current directory)
```

Project layout (high-level)

```
src/miniharness/     # application
tests/               # pytest tests
docs/                # architecture docs
.env.example         # environment template
pyproject.toml       # dependencies & build
```

Running tests

```bash
uv run pytest -v
```

Design notes

- Settings object, not raw env reads — modules read from a shared `Settings` instance.
- Pydantic for tool input validation and schema generation.
- Workspace boundary enforced for file operations; sandbox adds container isolation when enabled.

License

MIT

