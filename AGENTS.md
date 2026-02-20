# AGENTS.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Project Overview

Python client library for Aviato sandboxes - a remote code execution platform. The SDK provides a sync/async hybrid API for creating, managing, and executing code in containerized sandbox environments.

## Development Setup

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup, workflow, and all development tasks.

## Architecture

### Core Classes

**`Sandbox`** (`_sandbox.py`): Main entry point with sync/async hybrid API. All methods return immediately; operations execute in background via `_LoopManager`.

Construction patterns:
```python
# Factory method (recommended)
sb = Sandbox.run("echo", "hello")  # Returns immediately
result = sb.exec(["echo", "more"]).result()  # Block for result
sb.stop().result()  # Block for completion

# Context manager (recommended for most use cases)
with Sandbox.run("sleep", "infinity") as sb:
    result = sb.exec(["echo", "hello"]).result()
# Automatically stopped on exit

# Streaming output before getting result
with Sandbox.run("sleep", "infinity") as sb:
    process = sb.exec(["echo", "hello"])
    for line in process.stdout:  # Stream lines as they arrive
        print(line, end="")
    result = process.result()  # Get final ProcessResult

# Async context manager
async with Sandbox.run("sleep", "infinity") as sb:
    result = await sb.exec(["echo", "hello"])
```

Key methods:
- `run(*args, **kwargs)`: Create and start sandbox, return immediately. Accepts advanced configuration kwargs (see below).
- `start()`: Send start request, return `OperationRef[None]`. Call `.result()` to block until backend accepts.
- `wait()`: Block until RUNNING status, returns self for chaining
- `wait_until_complete(timeout=None, raise_on_termination=True)`: Wait until terminal state (COMPLETED, FAILED, TERMINATED), return `OperationRef[Sandbox]`. Call `.result()` to block or `await` in async contexts. Set `raise_on_termination=False` to handle externally-terminated sandboxes without raising `SandboxTerminatedError`.
- `exec(command, cwd=None, check=False, timeout_seconds=None, stdin=False)`: Execute command, return `Process`. Call `.result()` to block for `ProcessResult`. Iterate `process.stdout` before `.result()` for real-time streaming. Set `check=True` to raise `SandboxExecutionError` on non-zero returncode. Set `cwd` to an absolute path to run the command in a specific working directory (implemented via shell wrapping, requires /bin/sh in container). Set `stdin=True` to enable stdin streaming via `process.stdin`.
- `read_file(path)`: Return `OperationRef[bytes]`
- `write_file(path, content)`: Return `OperationRef[None]`
- `stop(snapshot_on_stop=False, graceful_shutdown_seconds=10.0, missing_ok=False)`: Stop sandbox and return `OperationRef[None]`. Raises `SandboxError` on failure. Set `snapshot_on_stop=True` to capture sandbox state before shutdown. Set `missing_ok=True` to suppress `SandboxNotFoundError`.
- `get_status()`: Fetch fresh status from API (sync)

Properties:
- `status`: Cached status from last API call (use `get_status()` for fresh)
- `status_updated_at`: When status was last fetched
- `sandbox_id`, `tower_id`, `runway_id`, `tower_group_id`, `returncode`, `started_at`

Advanced configuration kwargs (for `run()`, `Session.sandbox()`, and `@session.function()`):
- `resources` - Resource requests (CPU, memory, GPU)
- `mounted_files` - Files to mount into the sandbox
- `s3_mount` - S3 bucket mount configuration
- `ports` - Port mappings for the sandbox
- `network` - Network configuration via `NetworkOptions` or dict (ingress/egress modes, exposed ports)
- `max_timeout_seconds` - Maximum timeout for sandbox operations

Class methods:
- `Sandbox.session(defaults)`: Create a `Session` for managing multiple sandboxes (sync)
- `Sandbox.list(tags=None, status=None, runway_ids=None, tower_ids=None, include_stopped=False, ...)`: Query existing sandboxes, return `OperationRef[list[Sandbox]]`. Use `.result()` to block or `await` in async contexts. By default, terminal sandboxes (completed, failed, terminated) are excluded. Set `include_stopped=True` to include them from persistent storage.
- `Sandbox.from_id(sandbox_id)`: Attach to existing sandbox by ID, return `OperationRef[Sandbox]`. Works for both active and stopped sandboxes (the backend falls back to the DB for stopped sandboxes).
- `Sandbox.delete(sandbox_id, missing_ok=False)`: Delete sandbox by ID, return `OperationRef[None]`. Raises `SandboxError` on failure. Set `missing_ok=True` to suppress `SandboxNotFoundError` for already-deleted sandboxes.

**`Session`** (`_session.py`): Manages multiple sandboxes with shared defaults. Supports both sync and async context managers for the hybrid API.

Key methods:
- `session.sandbox(command, args, **kwargs)` - create an unstarted sandbox with session defaults. Auto-starts on first operation (exec, read_file, write_file, wait). Accepts advanced configuration kwargs.
- `session.function()` - decorator for remote function execution
- `session.adopt(sandbox)` - register an existing Sandbox (from `Sandbox.list()` or `Sandbox.from_id()`) for cleanup when session closes
- `session.close()` - return `OperationRef[None]` for cleanup
- `session.list(tags=None, status=None, runway_ids=None, tower_ids=None, include_stopped=False, adopt=False)` - find sandboxes matching session tags, return `OperationRef[list[Sandbox]]`. Use `.result()` to block or `await` in async contexts. Set `include_stopped=True` to include terminal sandboxes from persistent storage.
- `session.from_id(sandbox_id, adopt=True)` - attach to existing sandbox by ID, return `OperationRef[Sandbox]`

Properties:
- `sandbox_count`: Number of sandboxes currently tracked by this session

Usage pattern:
```python
with Session(defaults) as session:
    sb = session.sandbox(command="sleep", args=["infinity"])
    result = sb.exec(["echo", "hello"]).result()
# Automatically cleans up all sandboxes on exit
```

**`SandboxDefaults`** (`_defaults.py`): Immutable configuration dataclass. Tags propagate to backend for filtering.

Fields (all optional with sensible defaults):
- `container_image`, `command`, `args` - Container configuration
- `base_url` - API endpoint (default: `https://atc.cwaviato.com`)
- `request_timeout_seconds` - Client-side HTTP timeout (default: 300.0)
- `max_lifetime_seconds` - Server-side sandbox lifetime limit (default: None, backend controls)
- `temp_dir` - Sandbox temp directory (default: `/tmp`)
- `tags` - Tuple of tags for filtering
- `runway_ids`, `tower_ids` - Infrastructure filtering (optional tuple of IDs)
- `resources` - Resource requests (CPU, memory, GPU)
- `network` - Network configuration via `NetworkOptions`
- `environment_variables` - Environment variables to inject

Utility methods:
- `merge_tags(additional)` - Combine default tags with additional tags list
- `with_overrides(**kwargs)` - Create new defaults with some values overridden

Key constants (from `_defaults.py`):
- `DEFAULT_CONTAINER_IMAGE = "python:3.11"`
- `DEFAULT_COMMAND = "tail"`, `DEFAULT_ARGS = ("-f", "/dev/null")`
- `DEFAULT_BASE_URL = "https://atc.cwaviato.com"`
- `DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0` - Client-side HTTP timeout
- `DEFAULT_MAX_LIFETIME_SECONDS = None` - Server controls sandbox lifetime
- `DEFAULT_GRACEFUL_SHUTDOWN_SECONDS = 10.0`
- `DEFAULT_TEMP_DIR = "/tmp"`
- Polling: `DEFAULT_POLL_INTERVAL_SECONDS = 0.2`, `DEFAULT_POLL_BACKOFF_FACTOR = 1.5`, `DEFAULT_MAX_POLL_INTERVAL_SECONDS = 2.0`
- `DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS = 5.0` - Buffer added to exec timeout
- W&B auth: `WANDB_NETRC_HOST = "api.wandb.ai"`, `DEFAULT_PROJECT_NAME = "uncategorized"`

**`OperationRef[T]`** (`_types.py`): Generic wrapper for async operations with lazy result retrieval. Bridges `concurrent.futures.Future` to asyncio for the sync/async hybrid API.

Key methods:
- `result(timeout=None)` - Block until complete and return result
- `__await__` - Awaitable in async contexts

Usage pattern:
```python
ref = sandbox.read_file("/path")  # Returns immediately
data = ref.result()               # Block when result needed
# Or in async context:
data = await ref
```

**`SandboxStatus`** (`_sandbox.py`): StrEnum for sandbox lifecycle states: `PENDING`, `CREATING`, `RUNNING`, `PAUSED`, `COMPLETED`, `TERMINATED`, `FAILED`, `UNSPECIFIED`. Methods `from_proto()` and `to_proto()` for protobuf conversion.

**Exec Types** (`_types.py`): Types for command execution, returned by `Sandbox.exec()`:

- `Process`: Handle for running process with `stdout`/`stderr` StreamReaders and optional `stdin` StreamWriter. Properties: `returncode` (exit code or None), `command` (list executed), `stdin` (StreamWriter when `stdin=True`, or None). Methods: `poll()`, `wait(timeout)`, `result(timeout)`, `cancel()`. Awaitable in async contexts.
- `StreamReader`: Dual sync/async iterable wrapping asyncio.Queue. Supports both `for line in reader` and `async for line in reader`.
- `StreamWriter`: Writable stream for stdin. Methods: `write(data: bytes)`, `writeline(text: str)`, `close()`. All return `OperationRef[None]`. Property: `closed` (bool). Uses bounded queue (16 items, ~1MB with 64KB chunks) for backpressure.
- `ProcessResult`: Dataclass with `stdout`, `stderr`, `returncode`, `command`, plus raw byte variants (`stdout_bytes`, `stderr_bytes`).

**`NetworkOptions`** (`_types.py`): Frozen dataclass for typed network configuration. Controls sandbox ingress and egress modes. The `network` parameter accepts either a `NetworkOptions` instance or a plain dict (which is automatically converted).

Fields:
- `ingress_mode: str | None` - Inbound traffic mode: `"public"` (internet accessible), `"internal"` (cluster only), etc.
- `exposed_ports: tuple[int, ...] | None` - Ports to expose (required with `ingress_mode`). Lists are normalized to tuples for immutability.
- `egress_mode: str | None` - Outbound traffic mode: `"internet"` (full access), `"isolated"` (no external), `"org"` (org-internal only), etc.

Usage:
```python
from aviato import NetworkOptions

# Using NetworkOptions (recommended for type safety)
sandbox = Sandbox.run(
    network=NetworkOptions(
        ingress_mode="public",
        exposed_ports=(8080,),
        egress_mode="internet",
    ),
)

# Using dict (convenient for quick scripts)
sandbox = Sandbox.run(
    network={"ingress_mode": "public", "exposed_ports": [8080]},
)
```

### Authentication Flow

`_auth.py` implements a priority-based resolution:
1. `AVIATO_API_KEY` env var - Bearer token auth
2. `WANDB_API_KEY` + `WANDB_ENTITY_NAME` - W&B headers (x-api-key, x-entity-id, x-project-name)
3. `~/.netrc` (api.wandb.ai) + `WANDB_ENTITY_NAME`

### Function Execution (`_function.py`)

**`RemoteFunction[P, R]`**: Wrapper class returned by `@session.function()` decorator. Provides sync/async hybrid API for remote function execution.

Usage pattern:
```python
with Session(defaults) as session:
    @session.function()
    def compute(x: int, y: int) -> int:
        return x + y

    # Call .remote() to execute in sandbox
    ref = compute.remote(2, 3)  # Returns OperationRef immediately
    result = ref.result()       # Block for result: 5

    # Parallel execution across inputs
    refs = compute.map([(1, 2), (3, 4), (5, 6)])
    results = [r.result() for r in refs]  # [3, 7, 11]

    # Local testing without sandbox
    result = compute.local(2, 3)  # Runs in current process
```

Key methods:
- `__call__(*args, **kwargs)` - Execute in sandbox via `.remote()`, enabling natural `func(args)` syntax
- `remote(*args, **kwargs)` - Execute in sandbox, return `OperationRef[R]` immediately
- `map(items)` - Execute for each item tuple in parallel, return list of `OperationRef[R]`
- `local(*args, **kwargs)` - Execute locally without sandbox (for testing)

Configuration options (passed to decorator):
- `container_image` - Override image for this function
- `serialization` - `Serialization.JSON` (default) or `Serialization.PICKLE`
- Plus advanced configuration kwargs (see Sandbox section above)

Internals:
1. Extracts function source via AST, removes the `@session.function` decorator
2. Captures closure variables from `__closure__` and `co_freevars`
3. Walks bytecode (`LOAD_GLOBAL`, `STORE_GLOBAL`, `DELETE_GLOBAL`) to find referenced globals
4. Serializes payload (JSON or PICKLE), creates ephemeral sandbox, executes, reads result

Serialization modes via `Serialization` enum:
- `JSON` (default) - Safe, human-readable, limited to JSON-serializable types
- `PICKLE` - Supports complex Python objects (numpy arrays, custom classes) but requires trust

### Event Loop Management (`_loop_manager.py`)

**`_LoopManager`**: Singleton managing a background daemon thread with asyncio event loop. Enables sync code to execute async operations without user-managed event loops.

Key methods:
- `_LoopManager.get()` - Get singleton instance (thread-safe, double-checked locking)
- `run_sync(coro)` - Execute coroutine and block until complete
- `run_async(coro)` - Execute coroutine and return Future immediately
- `register_session(session)` - Track session in WeakSet for cleanup
- `cleanup_all()` - Stop all sandboxes in registered sessions

The daemon thread approach:
- Works in Jupyter notebooks without nest_asyncio
- Independent of user-managed event loops
- Allows cleanup via atexit and signal handlers

### Cleanup Handlers (`_cleanup.py`)

Auto-installed handlers for graceful sandbox shutdown on process exit. Installed automatically on module import.

- `_cleanup()`: Calls `_LoopManager.cleanup_all()` with re-entrancy guard
- `_signal_handler()`: Handles SIGINT/SIGTERM, chains to original handlers
- `_install_handlers()`: Registers atexit handler and signal handlers
- `_reset_for_testing()`: Resets module state for test isolation

On first signal, performs cleanup then chains to original handler. On second signal during cleanup, forces immediate exit.

### Module-Level Utilities

**`aviato.result()`**: Block for one or more OperationRefs and return results.

```python
# Single ref
data = aviato.result(sandbox.read_file("/path"))

# Multiple refs
results = aviato.result([sb.read_file(f) for f in files])
```

**`aviato.wait()`**: Wait for Sandbox, OperationRef, or Process objects to complete. Returns `(done, pending)` tuple.

```python
# Wait for all sandboxes to be running
sandboxes = [Sandbox.run(...) for _ in range(5)]
done, pending = aviato.wait(sandboxes)

# Wait for first N to complete
done, pending = aviato.wait(refs, num_returns=2)

# Wait with timeout
done, pending = aviato.wait(procs, timeout=30.0)
```

**`Waitable`**: Type alias for objects that can be waited on: `Sandbox | OperationRef[Any] | Process`.

### Backend Communication

Uses gRPC via `coreweave-aviato-grpc-python` and `grpcio` packages. Proto definitions generate `atc_pb2`, `atc_pb2_grpc`, `streaming_pb2`, and `streaming_pb2_grpc` modules.

**Channel management** (`_network.py`): Provides `parse_grpc_target()` for URL-to-target conversion and `create_channel()` for secure/insecure async channel creation. Auth headers are passed directly to streaming calls via metadata (interceptors don't work with request iterators).

**Streaming exec**: Uses native gRPC bidirectional streaming with request iterator pattern for proper half-close semantics via iterator completion.

### Related Repositories

- **Backend**: [github.com/coreweave/aviato](https://github.com/coreweave/aviato) - Server-side implementation (Go). Use `/repo-explore` to investigate backend behavior, API contracts, or debug client-server issues.

## Test Structure

- `tests/unit/` - Mock-based tests, no network calls (284 tests). Default pytest path.
- `tests/integration/` - Real sandbox operations, requires auth (31 tests). Run explicitly.

Unit test conftest clears all auth env vars before each test (`autouse=True` fixture).

### Integration Test Timing

Integration tests create real sandboxes and take significant time:
- **Individual test**: 5-15 seconds (sandbox startup + operation)
- **Full suite (31 tests)**: ~3 minutes total
- **Sandbox startup**: 30-60 seconds (mostly backend scheduling)

When running integration tests:
```bash
mise run test:e2e                         # Full suite (~2.5 minutes)
mise run test:e2e:parallel                # Parallel execution (faster)

# Individual test with timeout
timeout 120 uv run pytest tests/integration/aviato/test_sandbox.py::test_sandbox_lifecycle -v
```

**Important**: If integration tests hang beyond expected times, check:
1. API patterns match current sync/async hybrid design (use `.result()`, not `await`)
2. Sandbox reaches RUNNING status before file operations

### Integration Test Patterns

Tests should use the sync/async hybrid API:
```python
# Correct pattern
def test_sandbox_example(sandbox_defaults: SandboxDefaults) -> None:
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["echo", "hello"]).result()
        assert result.returncode == 0
```

## Exception Hierarchy

```
AviatoError
├── AviatoAuthenticationError
│   └── WandbAuthError
├── SandboxError
│   ├── SandboxNotRunningError
│   ├── SandboxTimeoutError
│   ├── SandboxTerminatedError
│   ├── SandboxFailedError
│   ├── SandboxNotFoundError         # .sandbox_id attribute
│   ├── SandboxExecutionError        # .exec_result, .exception_type, .exception_message attributes
│   └── SandboxFileError             # .filepath attribute
└── FunctionError
    ├── AsyncFunctionError
    └── FunctionSerializationError
```

## Examples

The `examples/` directory contains runnable scripts demonstrating common patterns:
- `quick_start.py`, `basic_execution.py`, `streaming_exec.py`, `stdin_streaming.py` - Sandbox creation and execution
- `function_decorator.py` - Remote function execution with `@session.function()`
- `multiple_sandboxes.py` - Session-based parallel execution
- `reconnect_to_sandbox.py`, `async_patterns.py` - Discovery and reconnection
- `delete_sandboxes.py` - Deletion patterns with `Sandbox.delete()`
- `error_handling.py` - Exception hierarchy and error recovery patterns
- `session_adopt_orphans.py`, `cleanup_by_tag.py`, `cleanup_old_sandboxes.py` - Orphan management and cleanup
- `parallel_batch_job.py` - Parallel batch processing with progress tracking

See `examples/README.md` and `examples/AGENTS.md` for full documentation. For detailed guides, see `docs/guides/`.

## Design Documentation

For comprehensive API design details, see `docs/` directory:
- `docs/guides/` - How-to guides for common tasks

When adding new documentation files to `docs/`, update `mkdocs.yml` nav section to include them.

### Key Design Decisions

**Thread Safety**: The sync API is designed for **single-threaded use**. Calling `.result()` from multiple threads simultaneously is not supported without external synchronization. Users wanting multi-threaded access should use one sandbox per thread or add their own locking. This is intentional to keep the implementation simple.

**Lazy-Start Model**: `Sandbox.run()` returns immediately once the backend accepts the request - it does NOT wait for RUNNING status. Blocking happens explicitly via `.result()` or `.wait()`.

**Single Internal Implementation**: There is one async implementation internally. The sync/async flexibility comes from how users consume results (`.result()` vs `await`), not from duplicate code paths.

## License Headers

All new files MUST include an SPDX license header. See [CONTRIBUTING.md](CONTRIBUTING.md) for full policy.

**License by directory:**
- Everything: `Apache-2.0`
- `examples/`: `BSD-3-Clause`

**Python files** (`.py`):
```python
# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client
```

**Markdown files** (`.md`):
```html
<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: aviato-client
-->
```

Use `BSD-3-Clause` instead for files under `examples/`. Validate with `reuse lint`.

## Temporary File Conventions

When creating temporary analysis or planning documents, use these filename suffixes to ensure they are gitignored:

| Suffix | Use Case |
|--------|----------|
| `-OLD.md` | Superseded or archived versions of documents |
| `-draft.md` | Work in progress, not ready for review |
| `-tmp.md` | Temporary files for single-session analysis |
| `-notes.md` | Personal analysis notes |

Example: `docs/api-redesign-draft.md` or `docs/spec-sync-api-OLD.md`

Files with these suffixes are excluded from git via `.gitignore`. For permanent documentation, use clear names without temporary markers.
