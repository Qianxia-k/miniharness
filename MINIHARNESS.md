# Project Instructions

<!--
  This file tells MiniHarness (and compatible agents) how to work in
  this project.  Edit it freely — it's your project's constitution.

  What to include:
  - Build / test commands
  - Code style rules
  - Architecture constraints
  - File / directory conventions
  - Security rules (e.g. "never log credentials")

  See also:
  - Core Memory: ~/.miniharness/memory/core.md (agent-accumulated knowledge)
  - Skills: .miniharness/skills/ (task-specific instruction packs)
-->

## Build & Test
- Build: `uv build`
- Test: `uv run pytest tests/`
- Lint: `uv run ruff check src/`

## Code Style
- Use type hints on all public functions.
- Use `pathlib.Path` for file paths.
- Max line length: 100 characters.

## Project Structure
- Source code: `src/`
- Tests: `tests/`
- Configuration: `config/`

## Security
- Never commit secrets (.env, credentials, API keys).
- Use environment variables for sensitive configuration.
