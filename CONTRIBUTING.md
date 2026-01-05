# Contributing to Sandbox

This document uses [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt) keywords: MUST, SHOULD, MAY, etc.

---

## Code

Code MUST be correct, minimal, and readable. Code SHOULD match existing conventions.

**Heuristic**: If you can't justify why a line exists, remove it.

Red flags:
- Abstractions used only once
- Comments describing what code does
- Speculative features

---

## Comments

Comments SHOULD explain *why*, not *what*. Comments MUST NOT duplicate what code already conveys.

**Heuristic**: If code is unclear without a comment, improve the code first. Use comments only when the reasoning isn't obvious from well-written code.

Red flags:
- `# Get the user` before `user = get_user()`
- `# Loop through items` before `for item in items:`
- Describing what code does rather than why it exists

Good comments explain:
- Non-obvious design decisions
- External constraints and API quirks
- Performance tradeoffs
- Business logic that can't be expressed in code

---

## Tests

Tests MUST be included with implementation. Tests MUST exercise YOUR code, not library behavior.

**Heuristic**: Each test should cover a unique code path. If two tests exercise the same logic, one is redundant.

Red flags:
- Multiple tests for the same error condition
- Tests that pass/fail based on library behavior
- Unit tests for trivial wrappers

Structure:
- Unit tests: `tests/unit/` (no external dependencies)
- Integration tests: `tests/integration/` (requires services)

---

## Documentation

Public APIs MUST have Google-style docstrings. Documentation MUST be updated with code changes.

**Heuristic**: Source of truth belongs close to the code. Docstrings document APIs; `docs/` provides examples and guides.

Red flags:
- Same information in docstring and docs/
- Missing docstring on public function

---

## Commits

Commits MUST follow [COMMIT_GUIDELINES.md](./COMMIT_GUIDELINES.md).

**Heuristic**: One logical change per commit. If you use "and" in the message, consider splitting.

Each commit MUST:
- Include implementation, tests, and docs together
- Leave codebase in working state
- Pass all tests
