# MiniHarness

A teaching-first mini coding agent harness — rebuild the core of [OpenHarness](https://github.com/anthropics/OpenHarness) in a small, readable codebase (~1000 lines).

```text
user input → message history → LLM call (streaming) → tool execution → loop until final answer
```

## Features

- **Async streaming agent loop** — real-time token output with tool call interception
- **Multi-provider** — DashScope (Qwen), OpenAI, and OpenAI-compatible APIs with auto-detection
- **5 built-in tools** — `read_file`, `write_file`, `edit_file` (old_str/new_str), `grep`, `bash`
- **Pydantic tool inputs** — each tool has its own `BaseModel` for parameter validation and auto-generated JSON Schema
- **Docker sandbox** — optional container isolation for bash commands (`--network none`)
- **Layered config** — defaults → env vars → provider auto-detect → CLI overrides
- **Permission system** — interactive prompts for write/bash operations
- **LLM retry** — exponential backoff + jitter for transient API errors

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd miniharness
uv sync --extra dev

# 2. Set your API key
cp .env.example .env
# Edit .env with your DASHSCOPE_API_KEY or OPENAI_API_KEY

# 3. Run
uv run mh "explain this project"
uv run mh --dry-run "test"          # check resolved settings
uv run mh --sandbox "list files"    # enable Docker sandbox
uv run mh -m gpt-4.1-mini "..."     # override model
```

## Configuration

Settings are resolved in layers (earlier = lower priority):

| Layer | Source |
|-------|--------|
| Defaults | `config/settings.py` dataclass defaults |
| Env vars | `MINIHARNESS_MODEL`, `MINIHARNESS_PROFILE`, etc. |
| Auto-detect | Detect provider from `DASHSCOPE_API_KEY` / `OPENAI_API_KEY` |
| CLI args | `--profile`, `--model`, `--sandbox`, etc. (highest priority) |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DASHSCOPE_API_KEY` | Qwen (DashScope) API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `MINIHARNESS_PROFILE` | Force a specific provider profile |
| `MINIHARNESS_MODEL` | Override the default model |
| `MINIHARNESS_MAX_TURNS` | Max agent loop turns (default: 8) |
| `MINIHARNESS_SANDBOX_ENABLED` | Enable Docker sandbox (`true`/`1`) |
| `MINIHARNESS_SANDBOX_IMAGE` | Docker image for sandbox |

### CLI Options

```
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

## Architecture

```
cli.py                   # CLI entrypoint, config loading, sandbox lifecycle
├── config/
│   ├── settings.py      # Settings dataclass (ProviderSettings, SandboxSettings)
│   └── __init__.py      # load_settings(), apply_cli_overrides()
├── loop.py              # Agent loop: model ↔ tools, message history
├── llm.py               # AsyncOpenAI streaming + retry logic
├── providers.py         # Provider registry (Qwen, OpenAI, compatible)
├── messages.py          # Conversation / Message models
├── permissions.py       # Interactive permission prompts (rich)
├── tool_registry.py     # Tool registry with Pydantic validation
├── tools/
│   ├── base.py          # BaseTool, Pydantic → OpenAI function schema
│   ├── read_file.py     # Read file with sandbox path validation
│   ├── write_file.py    # Write/create files
│   ├── edit_file.py     # old_str → new_str replacement
│   ├── grep.py          # Content search with ripgrep or Python fallback
│   └── bash.py          # Shell commands (sandbox or direct)
└── sandbox/
    ├── docker.py        # DockerSandbox: start / exec_command / stop
    ├── path_validator.py # Symlink-safe workspace boundary checks
    └── session.py        # Module-level sandbox singleton
```

## Project Structure

```
miniharness/
├── src/miniharness/     # Application code
├── tests/               # Test suite (pytest)
├── docs/                 # Architecture docs
├── .env.example          # Environment template
├── pyproject.toml        # Build config and dependencies
└── README.md
```

## Running Tests

```bash
uv run pytest -v
```

## Design Principles

- **Settings object, not env vars** — every module reads from `Settings`, never from `os.environ`
- **Frozen provider profiles** — immutable dataclass registry, not scattered if/elif chains
- **Pydantic at tool boundary** — `BaseModel.__init__` validates and converts before `execute()` runs
- **Events over callbacks** — the LLM client yields `TextDelta | StreamComplete`, the loop drives the UI from a single place
- **Workspace as boundary** — all file paths resolve inside `cwd`; the sandbox adds Docker isolation

## License

MIT
