# MiniHarness Architecture

MiniHarness is modeled after the core path of OpenHarness, but with fewer layers.

## OpenHarness Comparison

```text
OpenHarness:
  cli.py
   -> ui/app.py
   -> ui/runtime.py
   -> engine/query_engine.py
   -> engine/query.py
   -> api/client.py
   -> tools/*
   -> permissions/checker.py

MiniHarness:
  cli.py
   -> loop.py
   -> llm.py
   -> tool_registry.py
   -> tools/*
   -> permissions.py
```

## Core Concepts

### Message

A message is one item in the conversation history. It can be from:

- user
- assistant
- tool

### Tool

A tool is a function the model can request. The harness, not the model, executes the tool.

### Tool Registry

The registry knows all available tools. It can:

- list tool schemas for the model
- find a tool by name
- execute the selected tool

### Agent Loop

The loop is the heart of the harness:

1. Send messages and tool schemas to the model.
2. If the model returns text only, print it and stop.
3. If the model returns tool calls, execute them.
4. Append tool results to messages.
5. Repeat.

### Permission Checker

Before executing a risky tool, MiniHarness asks whether the action is allowed.

The first version will be simple:

- reading files is allowed
- writing files asks for confirmation
- shell commands ask for confirmation

