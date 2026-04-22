<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: cwsandbox-client
-->

# Tests

## Structure

```
tests/
├── unit/cwsandbox/        # Mock-based tests, no network calls (284 tests)
│   ├── conftest.py     # Auth env var clearing (autouse=True)
│   └── test_*.py       # 10 test files (one per source module + utilities)
└── integration/           # Real backend operations (31 tests)
    ├── conftest.py     # Integration-root CLI options (e.g., --cwsandbox-runner-ids)
    └── cwsandbox/
        ├── conftest.py # Shared fixtures (auth, sandbox_defaults, runner-ID resolution)
        └── test_*.py   # 3 test files (auth, sandbox, session)
```

## Running Tests

```bash
mise run test                              # Unit tests (default)
mise run test:e2e                          # Integration tests (requires auth)
mise run test:e2e:parallel                 # Integration tests in parallel
mise run test:all                          # All tests

# Direct pytest for specific tests
uv run pytest tests/unit/cwsandbox/test_sandbox.py  # Single file
uv run pytest -k "test_create"                   # By name pattern
```

## Fixtures

### Unit Test Fixtures (`tests/unit/cwsandbox/conftest.py`)

| Fixture | Scope | Autouse | Description |
|---------|-------|---------|-------------|
| `clean_auth_env` | function | Yes | Clears all auth env vars before each test |
| `mock_api_key` | function | No | Sets `CWSANDBOX_API_KEY` to `"test-api-key"`, returns value |
| `mock_base_url` | function | No | Sets `CWSANDBOX_BASE_URL` to `"http://test-api.example.com"`, returns value |

### Integration Test Fixtures (`tests/integration/cwsandbox/conftest.py`)

| Fixture | Scope | Autouse | Description |
|---------|-------|---------|-------------|
| `require_auth` | module | Yes | Skips tests if no auth configured |
| `_validate_runner_ids` | session | Yes | Fails fast (pytest.UsageError) when `--cwsandbox-runner-ids` or `CWSANDBOX_TEST_RUNNER_IDS` names a runner the discovery service does not know. Zero-cost when no runner targeting is configured. |
| `configured_runner_ids` | session | No | Returns `tuple[str, ...] \| None` resolved from `--cwsandbox-runner-ids` / `CWSANDBOX_TEST_RUNNER_IDS` (CLI wins). Consume this in tests that construct their own `SandboxDefaults` or call `Sandbox.run()` without defaults, so the runner pin still applies. |
| `sandbox_defaults` | module | No | Returns `SandboxDefaults` with `python:3.11`, 60s lifetime, `("integration-test", <session-tag>)` tags. Inherits `runner_ids` from `configured_runner_ids` when set. |
| `discovered_infrastructure` | module | No | Returns `(runner_id, profile_name)` for pin-targeting tests. Selects the first healthy runner with profiles from `cwsandbox.list_runners()`, honoring `configured_runner_ids` allowlist when set. Fails fast with a clear message if no candidates match. |

### Integration CLI flags (`tests/integration/conftest.py`)

| Flag | Env var | Description |
|------|---------|-------------|
| `--cwsandbox-runner-ids=<csv>` | `CWSANDBOX_TEST_RUNNER_IDS` | Comma-separated runner IDs to pin e2e sandboxes to. CLI wins over env. Pass empty (`--cwsandbox-runner-ids=`) to clear the env and auto-schedule. |

Set environment variables before running integration tests. A `.env` file in the project root is automatically loaded via python-dotenv.

## Unit Test Patterns

**Auth isolation**: The `clean_auth_env` fixture (autouse=True) clears all auth env vars before each test. Use `mock_api_key` and similar fixtures to set specific values.

**Sandbox tests**: Use sync patterns to block for results: `.result()` for both `OperationRef` (file ops, stop) and `Process` (from exec). Process inherits from OperationRef. The hybrid API tests use mocked `_LoopManager.run_async()` to control coroutine execution. Use `check=True` with exec to test `SandboxExecutionError` handling on non-zero returncodes.

**Session tests**: Support both sync and async patterns. Sync context manager tests use regular pytest; async context manager tests use `@pytest.mark.asyncio`. Internal async methods (like `_close_async`) require async tests.

**Function tests**: Test `RemoteFunction` class methods (`.remote()`, `.map()`, `.local()`), async function detection (should raise `AsyncFunctionError`), decorator removal via AST, and global/closure variable extraction from bytecode.

**_LoopManager tests**: Use `_LoopManager._reset_for_testing()` in teardown to reset the singleton between tests.

**_cleanup tests**: Use `_reset_for_testing()` from `cwsandbox._cleanup` in teardown to restore original signal handlers between tests.

**Wandb/reporter tests**: Session reporter tests use `report_to=["wandb"]` for explicit opt-in or `report_to=[]` for disabled. To test wandb logging, set `session._reporter._run = mock_run` directly rather than mocking `wandb.log`, since the reporter calls `run.log()` on its cached run instance.

## Integration Tests

Run against real backend - use sparingly to avoid resource consumption.

### Quick Start

```bash
cp .env.example .env
# Edit .env with your credentials
mise run test:e2e
```

### Authentication

Set environment variables before running tests (in priority order):

1. `CWSANDBOX_API_KEY` environment variable (takes priority)
2. `WANDB_API_KEY` only if running live W&B metrics checks in `test_wandb.py`

A `.env` file in the project root is automatically loaded via python-dotenv.

Tests skip gracefully with clear messages when no auth is configured. The `require_auth` fixture in conftest.py validates auth upfront rather than letting tests fail with opaque RPC errors.

### Targeting specific runners

When rolling out a change to a specific runner, pin every e2e sandbox to that runner rather than letting the backend auto-schedule. The `sandbox_defaults` fixture reads the target list from either the pytest CLI flag or an env var; the CLI flag wins when both are set.

```bash
# CLI: pin to one runner for a single test
mise run test:e2e -- --cwsandbox-runner-ids=runner-a

# CLI: pin to a set
mise run test:e2e -- --cwsandbox-runner-ids=runner-a,runner-b

# Env: convenient when iterating in a shell
CWSANDBOX_TEST_RUNNER_IDS=runner-a mise run test:e2e

# Clear the env for one invocation (auto-schedule)
CWSANDBOX_TEST_RUNNER_IDS=runner-a mise run test:e2e -- --cwsandbox-runner-ids=
```

Parsing normalises the list: whitespace is stripped, empty tokens are dropped, and duplicates are collapsed preserving first-seen order. If any resolved ID is unknown to the discovery service, pytest stops immediately with `pytest.UsageError` naming the missing IDs and their source (CLI vs env). When no targeting is configured, no discovery call is made - the default path is byte-identical to pre-change.

Scope: this constrains runner placement only. Profile selection remains backend-driven; `container_image`, `resources`, `tags`, `max_lifetime_seconds` are unchanged.

### Contract for new e2e tests

Any test path that may create a sandbox - directly via `Sandbox.run()`, indirectly via `Session`, or via a `@session.function()` - MUST honor the configured runner pin. To comply, either:

1. Consume `sandbox_defaults` (or `sandbox_defaults.with_overrides(...)`) - the pin is inherited automatically, OR
2. Accept the `configured_runner_ids` fixture and forward it: pass `runner_ids=list(configured_runner_ids)` to `Sandbox.run()` when non-None.

**Opt-out:** tests that are specifically validating `runner_ids` or `profile_ids` semantics (e.g., `test_sandbox_with_runway_and_runner_ids`) may opt out. Document the opt-out with an in-test comment explaining why the pin is not forwarded. Note that an explicit `runner_ids=[]` intentionally clears any default.

## Test File Reference

### Unit Test Files (`tests/unit/cwsandbox/`)

| File | Coverage |
|------|----------|
| `test_auth.py` | Built-in auth behavior and active auth mode overrides |
| `test_cleanup.py` | atexit handlers, signal handlers, re-entrancy guard |
| `test_defaults.py` | SandboxDefaults configuration, merge_tags, with_overrides |
| `test_exceptions.py` | Exception hierarchy, custom attributes |
| `test_function.py` | RemoteFunction class, decorator, .remote(), .map(), .local() |
| `test_loop_manager.py` | _LoopManager singleton, run_sync, run_async |
| `test_sandbox.py` | Sandbox class, status handling, exec, file ops |
| `test_session.py` | Session class, sandbox management, context managers, reporter lifecycle |
| `test_wandb.py` | WandbReporter metrics, per-sandbox tracking, lazy run detection |
| `test_types.py` | OperationRef, ProcessResult, Process, StreamReader |
| `test_utilities.py` | cwsandbox.results(), cwsandbox.wait() utilities |

### Integration Test Files (`tests/integration/cwsandbox/`)

| File | Coverage |
|------|----------|
| `test_sandbox.py` | Sandbox lifecycle, file ops, exec. Uses `require_auth`. |
| `test_session.py` | Session management, function execution. Uses `require_auth`. |
| `test_wandb.py` | W&B metrics logging. Uses `require_auth`; live checks also need `WANDB_API_KEY`. |

## Parallel Execution

Integration tests can run in parallel using pytest-xdist (already installed):

```bash
mise run test:e2e:parallel                 # Auto-detect CPU count
uv run pytest tests/integration/ -n 4 -v   # Specific worker count
```

**Why parallelization works:**
- Each test creates its own sandbox instance (natural isolation)
- No shared state between tests
- Module-scoped fixtures (sandbox_defaults) are thread-safe

**Performance characteristics:**
- Sequential: ~3 minutes for 31 tests
- Parallel: Scales well since sandbox startup (30-60s) is the bottleneck
- Backend is the limiting factor, not local CPU

**When to use sequential:**
- Debugging test failures (`-n 0` or omit `-n`)
- Investigating flaky tests
- When backend rate limits are hit
