# AGENTS.md

This file provides guidance to AI coding assistants when working with code in this repository.

## Project Overview

Python client library for CoreWeave Sandbox - a remote code execution platform. The SDK provides a sync/async hybrid API for creating, managing, and executing code in containerized sandbox environments.

## Public API and Documentation

When adding, removing, or renaming public exports in `src/cwsandbox/__init__.py`, the API reference generator in the `coreweave/docs` repo needs its `MANIFEST_GROUPS` updated (in `scripts/cwsandbox-api-ref/generate.py`). Docstrings use Google style with `Examples:` and `Attributes:` sections for structured parsing by Griffe.

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
with Sandbox.run() as sb:  # Default command keeps sandbox alive
    result = sb.exec(["echo", "hello"]).result()
# Automatically stopped on exit

# Streaming output before getting result
with Sandbox.run() as sb:
    process = sb.exec(["echo", "hello"])
    for line in process.stdout:  # Stream lines as they arrive
        print(line, end="")
    result = process.result()  # Get final ProcessResult

# Async context manager
async with Sandbox.run() as sb:
    result = await sb.exec(["echo", "hello"])
```

Key methods:
- `run(*args, **kwargs)`: Create and start sandbox, return immediately. Accepts advanced configuration kwargs (see below).
- `start()`: Send start request, return `OperationRef[None]`. Call `.result()` to block until backend accepts.
- `wait()`: Block until RUNNING status, returns self for chaining
- `wait_until_complete(timeout=None, raise_on_termination=True)`: Wait until terminal state (COMPLETED, FAILED, TERMINATED), return `OperationRef[Sandbox]`. Call `.result()` to block or `await` in async contexts. Set `raise_on_termination=False` to handle externally-terminated sandboxes without raising `SandboxTerminatedError`.
- `exec(command, cwd=None, check=False, timeout_seconds=None, stdin=False)`: Execute command, return `Process`. Call `.result()` to block for `ProcessResult`. Iterate `process.stdout` before `.result()` for real-time streaming. Set `check=True` to raise `SandboxExecutionError` on non-zero returncode. Set `cwd` to an absolute path to run the command in a specific working directory (implemented via shell wrapping, requires /bin/sh in container). Set `stdin=True` to enable stdin streaming via `process.stdin`.
- `shell(command=None, *, width=None, height=None)`: Start an interactive TTY session, return `TerminalSession`. Always allocates a TTY and enables stdin. Output is raw bytes (merged stdout/stderr) with no buffering — safe for long-running interactive sessions. Defaults to `["/bin/bash"]`.
- `stream_logs(*, follow=False, tail_lines=None, since_time=None, timestamps=False)`: Stream logs from the sandbox's main process (PID 1), return `StreamReader[str]`. Only captures stdout/stderr from the command passed to `Sandbox.run()` — output from `exec()` commands is **not** included. Set `follow=True` for continuous streaming (like `tail -f`). Uses bounded queues for backpressure in follow mode.
- `read_file(path)`: Return `OperationRef[bytes]`
- `write_file(path, content)`: Return `OperationRef[None]`
- `stop(snapshot_on_stop=False, graceful_shutdown_seconds=10.0, missing_ok=False)`: Stop sandbox and return `OperationRef[None]`. Raises `SandboxError` on failure. Set `snapshot_on_stop=True` to capture sandbox state before shutdown. Set `missing_ok=True` to suppress `SandboxNotFoundError`.
- `get_status()`: Fetch fresh status from API (sync). Returns cached status for terminal sandboxes (COMPLETED, FAILED, TERMINATED) since terminal states are immutable.

Properties:
- `status`: Cached status from last API call (use `get_status()` for fresh)
- `status_updated_at`: When status was last fetched
- `sandbox_id`, `tower_id`, `runway_id`, `tower_group_id`, `returncode`, `started_at`
- `resource_requests`, `resource_limits` - Confirmed resources from start response (None for discovered sandboxes)

Advanced configuration kwargs (for `run()`, `Session.sandbox()`, and `@session.function()`):
- `resources` - Resource configuration via `ResourceOptions`, nested dict, or legacy flat dict (CPU, memory, GPU)
- `mounted_files` - Files to mount into the sandbox
- `s3_mount` - S3 bucket mount configuration
- `ports` - Port mappings for the sandbox
- `network` - Network configuration via `NetworkOptions` or dict (ingress/egress modes, exposed ports)
- `secrets` - Secrets to inject from secret stores as environment variables, via `Secret` or dict
- `max_timeout_seconds` - Maximum timeout for sandbox operations
- `environment_variables` - Environment variables to inject (merges with defaults)
- `annotations` - Kubernetes pod annotations (merges with defaults, explicit keys win)

Class methods:
- `Sandbox.session(defaults)`: Create a `Session` for managing multiple sandboxes (sync)
- `Sandbox.list(tags=None, status=None, runway_ids=None, tower_ids=None, include_stopped=False, ...)`: Query existing sandboxes, return `OperationRef[list[Sandbox]]`. Use `.result()` to block or `await` in async contexts. By default, terminal sandboxes (completed, failed, terminated) are excluded. Set `include_stopped=True` to include them.
- `Sandbox.from_id(sandbox_id)`: Attach to existing sandbox by ID, return `OperationRef[Sandbox]`. Works for both active and stopped sandboxes.
- `Sandbox.delete(sandbox_id, missing_ok=False)`: Delete sandbox by ID, return `OperationRef[None]`. Raises `SandboxError` on failure. Set `missing_ok=True` to suppress `SandboxNotFoundError` for already-deleted sandboxes.

**`Session`** (`_session.py`): Manages multiple sandboxes with shared defaults. Supports both sync and async context managers for the hybrid API.

Key methods:
- `session.sandbox(command, args, **kwargs)` - create an unstarted sandbox with session defaults. Auto-starts on first operation (exec, read_file, write_file, wait). Accepts advanced configuration kwargs.
- `session.function()` - decorator for remote function execution
- `session.adopt(sandbox)` - register an existing Sandbox (from `Sandbox.list()` or `Sandbox.from_id()`) for cleanup when session closes
- `session.close()` - return `OperationRef[None]` for cleanup
- `session.list(tags=None, status=None, runway_ids=None, tower_ids=None, include_stopped=False, adopt=False)` - find sandboxes matching session tags, return `OperationRef[list[Sandbox]]`. Use `.result()` to block or `await` in async contexts. Set `include_stopped=True` to include terminal sandboxes.
- `session.from_id(sandbox_id, adopt=True)` - attach to existing sandbox by ID, return `OperationRef[Sandbox]`

Properties:
- `sandbox_count`: Number of sandboxes currently tracked by this session

Usage pattern:
```python
with Session(defaults) as session:
    sb = session.sandbox()  # Default command keeps sandbox alive
    result = sb.exec(["echo", "hello"]).result()
# Automatically cleans up all sandboxes on exit
```

**`SandboxDefaults`** (`_defaults.py`): Immutable configuration dataclass. Tags propagate to backend for filtering.

Fields (all optional with sensible defaults):
- `container_image`, `command`, `args` - Container configuration
- `base_url` - API endpoint (default: `https://atc.cw-sandbox.com`)
- `request_timeout_seconds` - Client-side HTTP timeout (default: 300.0)
- `max_lifetime_seconds` - Server-side sandbox lifetime limit (default: None, backend controls)
- `temp_dir` - Sandbox temp directory (default: `/tmp`)
- `tags` - Tuple of tags for filtering
- `runway_ids`, `tower_ids` - Infrastructure filtering (optional tuple of IDs)
- `resources` - Resource configuration (`ResourceOptions | dict[str, Any] | None`)
- `network` - Network configuration via `NetworkOptions`
- `secrets` - Secrets to inject from secret stores (tuple of `Secret`)
- `environment_variables` - Environment variables to inject
- `annotations` - Kubernetes pod annotations (`dict[str, str]`, default: empty)

Utility methods:
- `merge_tags(additional)` - Combine default tags with additional tags list
- `merge_annotations(additional)` - Combine default annotations with additional dict (explicit keys win)
- `merge_environment_variables(additional)` - Combine default env vars with additional dict (explicit keys win)
- `with_overrides(**kwargs)` - Create new defaults with some values overridden

Key constants (from `_defaults.py`):
- `DEFAULT_CONTAINER_IMAGE = "python:3.11"`
- `DEFAULT_COMMAND = "tail"`, `DEFAULT_ARGS = ("-f", "/dev/null")`
- `DEFAULT_BASE_URL = "https://atc.cw-sandbox.com"`
- `DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0` - Client-side HTTP timeout
- `DEFAULT_MAX_LIFETIME_SECONDS = None` - Server controls sandbox lifetime
- `DEFAULT_GRACEFUL_SHUTDOWN_SECONDS = 10.0`
- `DEFAULT_TEMP_DIR = "/tmp"`
- Polling: `DEFAULT_POLL_INTERVAL_SECONDS = 0.2`, `DEFAULT_POLL_BACKOFF_FACTOR = 1.5`, `DEFAULT_MAX_POLL_INTERVAL_SECONDS = 2.0`
- `DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS = 5.0` - Buffer added to exec timeout

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
- `StreamReader`: Dual sync/async iterable wrapping asyncio.Queue. Supports both `for line in reader` and `async for line in reader`. Parameterized: `StreamReader[str]` for text (exec output, logs), `StreamReader[bytes]` for raw bytes (TTY output). Call `close()` to stop the underlying producer and end iteration.
- `StreamWriter`: Writable stream for stdin. Methods: `write(data: bytes)`, `writeline(text: str)`, `close()`. All return `OperationRef[None]`. Property: `closed` (bool). Uses bounded queue (16 items, ~1MB with 64KB chunks) for backpressure.
- `ProcessResult`: Dataclass with `stdout`, `stderr`, `returncode`, `command`, plus raw byte variants (`stdout_bytes`, `stderr_bytes`).

**Terminal Types** (`_types.py`): Types for interactive TTY sessions, returned by `Sandbox.shell()`:

- `TerminalSession`: Handle for an interactive TTY session. Extends `OperationRef[TerminalResult]`. Properties: `output` (StreamReader[bytes] — merged stdout/stderr as raw bytes), `stdin` (StreamWriter — always present), `command` (list executed). Methods: `resize(width, height)` (fire-and-forget), `wait(timeout)` (blocks until session ends, returns exit code), `result(timeout)` (returns TerminalResult). Awaitable in async contexts.
- `TerminalResult`: Frozen dataclass with `returncode` and `command`. Unlike `ProcessResult`, does not contain captured stdout/stderr because TTY sessions do not buffer output.

**`NetworkOptions`** (`_types.py`): Frozen dataclass for typed network configuration. Controls sandbox ingress and egress modes. The `network` parameter accepts either a `NetworkOptions` instance or a plain dict (which is automatically converted).

Fields:
- `ingress_mode: str | None` - Inbound traffic mode. Available modes depend on the runway configurations of towers you have access to.
- `exposed_ports: tuple[int, ...] | None` - Ports to expose (required with `ingress_mode`). Lists are normalized to tuples for immutability.
- `egress_mode: str | None` - Outbound traffic mode. Available modes depend on the runway configurations of towers you have access to.

Usage:
```python
from cwsandbox import NetworkOptions

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

**`Secret`** (`_types.py`): Frozen dataclass for injecting secrets from secret stores into sandbox environment variables. The `secrets` parameter accepts `Secret` instances or plain dicts (which are automatically converted via `Secret(**d)`).

Fields:
- `store: str` - Name of the secret store (e.g. `"wandb"`).
- `name: str` - Name of the secret in the store.
- `field: str` - Specific field within a structured secret (optional, defaults to `""`).
- `env_var: str | None` - Environment variable name the secret is injected as (defaults to `name`).

Duplicate `env_var` targets across secrets raise `ValueError` at merge time.

Usage:
```python
from cwsandbox import Secret

# Minimal: env_var defaults to name
sandbox = Sandbox.run(
    secrets=[Secret(store="wandb", name="HF_TOKEN")],
)

# Extracting a field from a structured secret
sandbox = Sandbox.run(
    secrets=[
        Secret(store="wandb", name="db-credentials", field="password", env_var="DB_PASS"),
    ],
)

# Using dicts (convenient for config files)
sandbox = Sandbox.run(
    secrets=[{"store": "wandb", "name": "HF_TOKEN"}],
)
```

**`ResourceOptions`** (`_types.py`): Frozen dataclass for typed resource configuration. Supports separate requests and limits for Burstable QoS pods. GPU is a separate top-level field because GPU overcommit is not supported by the backend. The `resources` parameter accepts a `ResourceOptions` instance, a nested dict, or a legacy flat dict (which is automatically coerced).

Fields:
- `requests: dict[str, str] | None` - CPU/memory resource requests (e.g. `{"cpu": "1", "memory": "256Mi"}`)
- `limits: dict[str, str] | None` - CPU/memory resource limits (e.g. `{"cpu": "8", "memory": "2Gi"}`)
- `gpu: dict[str, Any] | None` - GPU configuration (e.g. `{"count": 1, "type": "A100"}`)

Usage:
```python
from cwsandbox import ResourceOptions

# Using ResourceOptions (recommended for overcommit)
sandbox = Sandbox.run(
    resources=ResourceOptions(
        requests={"cpu": "1", "memory": "256Mi"},
        limits={"cpu": "8", "memory": "2Gi"},
    ),
)

# Using nested dict
sandbox = Sandbox.run(
    resources={"requests": {"cpu": "1"}, "limits": {"cpu": "8"}},
)

# Legacy flat dict (Guaranteed QoS - requests == limits)
sandbox = Sandbox.run(
    resources={"cpu": "8", "memory": "2Gi"},
)
```

### Authentication Flow

`_auth.py` implements a pluggable auth mode system with a single active mode:
1. `CWSANDBOX_API_KEY` env var - Bearer token auth (built-in default)
2. No auth (built-in fallback)

Provider integrations (e.g. `wandb.sandbox`) can replace the active mode for the current process via `set_auth_mode()`.

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

**`cwsandbox.results()`**: Block for one or more OperationRefs and return results.

```python
# Single ref
data = cwsandbox.results(sandbox.read_file("/path"))

# Multiple refs
all_data = cwsandbox.results([sb.read_file(f) for f in files])
```

**`cwsandbox.wait()`**: Wait for Sandbox, OperationRef, or Process objects to complete. Returns `(done, pending)` tuple.

```python
# Wait for all sandboxes to be running
sandboxes = [Sandbox.run(...) for _ in range(5)]
done, pending = cwsandbox.wait(sandboxes)

# Wait for first N to complete
done, pending = cwsandbox.wait(refs, num_returns=2)

# Wait with timeout
done, pending = cwsandbox.wait(procs, timeout=30.0)
```

**`Waitable`**: Type alias for objects that can be waited on: `Sandbox | OperationRef[Any] | Process | TerminalSession`.

### Discovery API

Module-level sync functions (`_discovery.py`) for querying available towers and runways. These are simple read-only queries that return results directly (no `OperationRef`/`await` needed).

**Functions:**
- `list_towers(*, tower_group_id=None, runway_name=None, gpu_type=None, architecture=None, include_resources=False)` -> `list[Tower]`: List available towers with optional filtering. Set `include_resources=True` for live resource availability. Auto-paginates.
- `get_tower(tower_id)` -> `Tower`: Get a single tower by ID. Always returns full details including resources. Raises `TowerNotFoundError` if not found.
- `list_runways(*, gpu_type=None, architecture=None, tower_id=None)` -> `list[Runway]`: List available runways with optional filtering. Auto-paginates.
- `get_runway(runway_name, *, tower_id=None)` -> `Runway`: Get a single runway by name. Raises `RunwayNotFoundError` if not found.

**Types:**
- `Tower`: Frozen dataclass with tower capabilities (CPU, memory, GPU), health status, `runway_names`, and optional `TowerResources`. Has human-readable `__repr__`.
- `TowerResources`: Live resource availability (`available_cpu_millicores`, `available_memory_bytes`, `available_gpu_count`, `running_sandboxes`).
- `Runway`: Frozen dataclass with `runway_name`, `tower_id`, `supported_gpu_types`, `supported_architectures`, and `ingress_modes`/`egress_modes`.
- `IngressMode`, `EgressMode`: Wrapper dataclasses (with `name` field) for forward compatibility.

**Utilities:**
- `format_bytes(value)`: Format bytes as human-readable string (e.g., `17179869184` -> `'16.0 GiB'`).
- `format_cpu(millicores)`: Format CPU millicores (e.g., `4000` -> `'4.0 cores'`).

Usage:
```python
import cwsandbox

# List all towers with resources
towers = cwsandbox.list_towers(include_resources=True)
for t in towers:
    print(t)  # Human-readable repr

# Get a specific tower
tower = cwsandbox.get_tower("tower-123")
print(f"CPU: {cwsandbox.format_cpu(tower.max_cpu_millicores)}")

# List runways, filter by GPU type
runways = cwsandbox.list_runways(gpu_type="A100")
```

Note: `runway_names` from discovery are the same strings used in `SandboxDefaults(runway_ids=[...])` for infrastructure targeting. The API reference generator in `coreweave/docs` repo needs `MANIFEST_GROUPS` updated in `scripts/cwsandbox-api-ref/generate.py` to include the new discovery types and functions.

### Backend Communication

Uses gRPC via `grpcio` with vendored proto stubs in `src/cwsandbox/_proto/`. The stubs (`atc_pb2`, `atc_pb2_grpc`, `streaming_pb2`, `streaming_pb2_grpc`) are updated via `scripts/update-protos.sh`.

**Channel management** (`_network.py`): Provides `parse_grpc_target()` for URL-to-target conversion and `create_channel()` for secure/insecure async channel creation. Auth headers are passed directly to streaming calls via metadata (interceptors don't work with request iterators).

**Streaming exec**: Uses native gRPC bidirectional streaming with request iterator pattern for proper half-close semantics via iterator completion.

### Related Repositories

- **Backend**: [github.com/coreweave/aviato](https://github.com/coreweave/aviato) - Server-side implementation (Go). Use `/repo-explore` to investigate backend behavior, API contracts, or debug client-server issues.

## Test Structure

- `tests/unit/` - Mock-based tests, no network calls. Default pytest path.
- `tests/integration/` - Real sandbox operations, requires auth. Run explicitly.

Unit test conftest clears all auth env vars before each test (`autouse=True` fixture).

### Integration Test Timing

Integration tests create real sandboxes and take significant time:
- **Individual test**: 5-15 seconds (sandbox startup + operation)
- **Full suite**: ~3 minutes total
- **Sandbox startup**: 30-60 seconds (mostly backend scheduling)

When running integration tests:
```bash
mise run test:e2e                         # Full suite (~2.5 minutes)
mise run test:e2e:parallel                # Parallel execution (faster)

# Individual test with timeout
timeout 120 uv run pytest tests/integration/cwsandbox/test_sandbox.py::test_sandbox_lifecycle -v
```

**Important**: If integration tests hang beyond expected times, check:
1. API patterns match current sync/async hybrid design (use `.result()`, not `await`)
2. Sandbox reaches RUNNING status before file operations

### Integration Test Patterns

Tests should use the sync/async hybrid API:
```python
# Correct pattern
def test_sandbox_example(sandbox_defaults: SandboxDefaults) -> None:
    with Sandbox.run(defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["echo", "hello"]).result()
        assert result.returncode == 0
```

## Exception Hierarchy

```
CWSandboxError
├── CWSandboxAuthenticationError
├── SandboxError
│   ├── SandboxNotRunningError
│   ├── SandboxTimeoutError
│   ├── SandboxTerminatedError
│   ├── SandboxFailedError
│   ├── SandboxNotFoundError         # .sandbox_id attribute
│   ├── SandboxExecutionError        # .exec_result, .exception_type, .exception_message attributes
│   └── SandboxFileError             # .filepath attribute
├── TowerNotFoundError               # .tower_id attribute
├── RunwayNotFoundError              # .runway_name, .tower_id attributes
└── FunctionError
    ├── AsyncFunctionError
    └── FunctionSerializationError
```

## Examples

The `examples/` directory contains runnable scripts demonstrating common patterns:
- `quick_start.py`, `basic_execution.py`, `streaming_exec.py`, `stdin_streaming.py` - Sandbox creation and execution
- `resource_configuration.py` - ResourceOptions, flat dict, nested dict, GPU, and response properties
- `function_decorator.py` - Remote function execution with `@session.function()`
- `multiple_sandboxes.py` - Session-based parallel execution
- `interactive_streaming_sandbox.py` - Log streaming with `stream_logs()` and CLI interaction (`exec`, `sh`, `logs`)
- `reconnect_to_sandbox.py`, `async_patterns.py` - Discovery and reconnection
- `delete_sandboxes.py` - Deletion patterns with `Sandbox.delete()`
- `error_handling.py` - Exception hierarchy and error recovery patterns
- `session_adopt_orphans.py`, `cleanup_by_tag.py`, `cleanup_old_sandboxes.py` - Orphan management and cleanup
- `parallel_batch_job.py` - Parallel batch processing with progress tracking

See `examples/README.md` and `examples/AGENTS.md` for full documentation. For detailed guides, see [docs.coreweave.com](https://docs.coreweave.com/products/coreweave-sandbox/client).

### Key Design Decisions

**Thread Safety**: The sync API is designed for **single-threaded use**. Calling `.result()` from multiple threads simultaneously is not supported without external synchronization. Users wanting multi-threaded access should use one sandbox per thread or add their own locking. This is intentional to keep the implementation simple.

**Lazy-Start Model**: `Sandbox.run()` returns immediately once the backend accepts the request - it does NOT wait for RUNNING status. Blocking happens explicitly via `.result()` or `.wait()`.

**Single Internal Implementation**: There is one async implementation internally. The sync/async flexibility comes from how users consume results (`.result()` vs `await`), not from duplicate code paths.

**ResourceOptions and Overcommit**: GPU is a separate field from requests/limits because the backend does not support GPU overcommit. Flat dict backward compatibility is maintained for CPU/memory (legacy form sets requests == limits for Guaranteed QoS). The flat dict form for GPU (`{"gpu_count": N}`) is a breaking change - use `ResourceOptions(gpu={"count": N})` instead. The `resource_requests` and `resource_limits` properties are populated only from start-response data, so they are None for sandboxes discovered via `Sandbox.from_id()` or `Sandbox.list()`.

## License Headers

All new files MUST include an SPDX license header. See [CONTRIBUTING.md](CONTRIBUTING.md) for full policy.

**License by directory:**
- Everything: `Apache-2.0`
- `examples/`: `BSD-3-Clause`

**Python files** (`.py`):
```python
# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client
```

**Markdown files** (`.md`):
```html
<!--
SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: cwsandbox-client
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
