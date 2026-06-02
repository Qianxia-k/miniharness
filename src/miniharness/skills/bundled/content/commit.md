# commit

Create clean, well-structured git commits with conventional commit messages.

## When to Use

Use when the user asks to:
- Commit changes
- Create a PR or prepare code for review
- Write a commit message

## Workflow

1. Run `git status` and `git diff --staged` to understand what changed.
2. If nothing is staged, ask the user what to include.
3. Write a commit message following conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `refactor:` for code restructuring
   - `test:` for test changes
   - `docs:` for documentation
   - `chore:` for build/deps/tooling
4. The body should explain WHY, not just WHAT.
5. Run `git commit -m "..."` with the message.
6. Confirm the commit succeeded.

## Example

```
feat: add JWT authentication middleware

Implement token verification as a FastAPI middleware that runs before
every protected endpoint.  Tokens are validated against the configured
secret and user info is attached to the request scope.
```
