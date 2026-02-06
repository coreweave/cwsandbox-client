# Contributing to Sandbox

This document uses [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt) keywords: MUST, SHOULD, MAY, etc.

For development setup and workflow, see [DEVELOPMENT.md](DEVELOPMENT.md).

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

---

## License headers

Source code should contain an SPDX-style license header, reflecting:
- Year & Copyright owner
- SPDX License identifier `SPDX-License-Identifier: Apache-2.0` or
  `SPDX-License-Identifier: BSD-3-Clause` for examples.
- Package Name: `SPDX-PackageName: aviato-client`

This can be partially automated with [FSFe REUSE](https://reuse.software/dev/#tool)
```shell
reuse annotate --license Apache-2.0 --copyright 'CoreWeave, Inc.'  --year 2025 --skip-existing $FILE
```

Blindly adding the headers to every file without review risks assigning the
wrong copyright owner! You should endeavor to understand who owns
contributions!

- The Aviato Client library source & testcases are licensed under the Apache-2.0 license
  to protect the rights of all parties.
- The Aviato Client usage examples (`examples/` directory) are licensed with
  the [BSD-3-Clause license](https://spdx.org/licenses/BSD-3-Clause.html) to encourage usage of the Aviato Client, while
  protecting CoreWeave's trademarks & name.
