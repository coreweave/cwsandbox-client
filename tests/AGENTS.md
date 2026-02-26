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
└── integration/cwsandbox/ # Real backend operations (31 tests)
    ├── conftest.py     # Shared fixtures
    └── test_*.py       # 3 test files (auth, sandbox, session)
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
| `mock_wandb_api_key` | function | No | Sets `WANDB_API_KEY` to `"test-wandb-api-key"`, returns value |
| `mock_wandb_entity_name` | function | No | Sets `WANDB_ENTITY_NAME` to `"test-entity"`, returns value |

### Integration Test Fixtures (`tests/integration/cwsandbox/conftest.py`)

| Fixture | Scope | Autouse | Description |
|---------|-------|---------|-------------|
| `require_auth` | module | Yes | Skips tests if no auth configured (except test_auth.py) |
| `sandbox_defaults` | module | No | Returns `SandboxDefaults` with `python:3.11`, 300s lifetime, `("integration-test",)` tags |

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
2. `WANDB_API_KEY` + `WANDB_ENTITY_NAME` environment variables
3. `~/.netrc` (api.wandb.ai) + `WANDB_ENTITY_NAME`

A `.env` file in the project root is automatically loaded via python-dotenv.

Tests skip gracefully with clear messages when no auth is configured. The `require_auth` fixture in conftest.py validates auth upfront rather than letting tests fail with opaque RPC errors.

## Test File Reference

### Unit Test Files (`tests/unit/cwsandbox/`)

| File | Coverage |
|------|----------|
| `test_auth.py` | Auth resolution priority, env vars, netrc parsing |
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
| `test_auth.py` | W&B auth paths (env vars, netrc). Has own skip logic. |
| `test_sandbox.py` | Sandbox lifecycle, file ops, exec. Uses `require_auth`. |
| `test_session.py` | Session management, function execution. Uses `require_auth`. |

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
