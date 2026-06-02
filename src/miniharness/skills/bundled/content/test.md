---
name: test
description: Write and run unit tests for existing or new code.
---

# test

Write comprehensive unit tests for the specified code.

## When to Use

Use when the user asks to:
- Write tests for a module or function
- Add test coverage
- Run the test suite
- Fix failing tests

## Workflow

1. Read the source file to understand the interfaces.
2. Check if a test file already exists (e.g., `tests/test_<module>.py`).
3. If it exists, read it to understand existing patterns.
4. Write tests covering:
   - **Happy path**: normal inputs produce expected outputs.
   - **Edge cases**: empty input, None values, boundary values.
   - **Error cases**: invalid input raises appropriate exceptions.
5. Use the project's existing test framework (pytest, unittest).
6. Run the tests to verify they pass.
7. If fixing a bug, write a test that reproduces the bug FIRST.

## Style Guidelines

- One assertion per test when practical.
- Use descriptive test names: `test_<function>_<scenario>_<expected>`.
- Use fixtures for shared setup (pytest).
- Mock external dependencies (APIs, databases, filesystem).
