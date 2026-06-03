---
name: code-review
description: Review code changes for bugs, security issues, and code quality problems.
---

# code-review

Perform a thorough code review of staged or specified changes.

## When to Use

Use when the user asks to:
- Review code changes
- Check for bugs
- Audit for security issues
- Evaluate code quality

## Review Checklist

### Correctness
- Does the logic handle edge cases (empty inputs, null values, large data)?
- Are there off-by-one errors?
- Is error handling complete (not swallowing exceptions silently)?

### Security
- Are there hardcoded secrets (API keys, passwords, tokens)?
- Is user input properly validated and sanitized?
- Any SQL injection, XSS, or path traversal vulnerabilities?
- Are dependencies pinned to specific versions?

### Code Quality
- Are variable/function names clear and descriptive?
- Is the code DRY (no copy-pasted logic)?
- Are functions small and single-purpose?
- Is there adequate test coverage for the changes?

### Performance
- Are there N+1 queries or unnecessary loops?
- Could large data operations benefit from streaming/pagination?
- Are expensive operations cached where appropriate?

## Output Format

For each finding, report:
1. **Severity**: critical / high / medium / low
2. **File & line**: where the issue is
3. **Description**: what's wrong
4. **Fix**: how to resolve it

If no issues found, confirm the code looks good.
