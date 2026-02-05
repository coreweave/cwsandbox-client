# Aviato Examples

This directory contains example scripts demonstrating how to use the `aviato` package.

## Prerequisites

1. Install the package:
   ```bash
   pip install aviato
   ```

2. Set your API key:
   ```bash
   export AVIATO_API_KEY="your-api-key"
   ```

## Sync vs Async Patterns

The aviato SDK uses a sync/async hybrid API. **Most users should use sync patterns** - they require no asyncio boilerplate and work in Jupyter notebooks without `nest_asyncio`.

**Sync patterns** (recommended): All examples except `async_patterns.py`

**Async patterns** (alternative): `async_patterns.py` demonstrates using `await` with the hybrid API

See [Sync vs Async Guide](../docs/guides/sync-vs-async.md) for when to use each pattern.

---

## Sync Examples (Recommended)

### Quick Start (`quick_start.py`)

The simplest way to get started with sandboxes:

```bash
python examples/quick_start.py
```

Demonstrates:
- Using `Sandbox.run()` factory method for quick sandbox creation
- Auto-starting sandboxes with positional command arguments
- Using `wait()` to wait for RUNNING status

### Basic Execution (`basic_execution.py`)

More detailed sandbox usage with full control:

```bash
python examples/basic_execution.py
```

Demonstrates:
- Creating sandboxes with explicit constructor
- Using `SandboxDefaults` for configuration
- Executing commands with `exec()`
- Reading and writing files

### Streaming Exec (`streaming_exec.py`)

Real-time command output with streaming exec:

```bash
python examples/streaming_exec.py
```

Demonstrates:
- Real-time stdout iteration as lines arrive
- Separate stdout/stderr handling
- Process lifecycle (returncode, poll, wait, result)

### Stdin Streaming (`stdin_streaming.py`)

Send input to interactive commands:

```bash
python examples/stdin_streaming.py
```

Demonstrates:
- Enabling stdin with `exec(stdin=True)`
- Writing data with `process.stdin.write()`
- Writing lines with `process.stdin.writeline()`
- Closing stdin to signal EOF
- Combining stdin with stdout streaming
- Async stdin patterns

### Function Decorator (`function_decorator.py`)

Execute Python functions in sandboxes:

```bash
python examples/function_decorator.py
```

Demonstrates:
- Using `Session(defaults)` for session management
- The `@session.function()` decorator
- Pickle and JSON serialization modes
- Functions with closure variables

### Error Handling (`error_handling.py`)

Proper error handling with the exception hierarchy:

```bash
python examples/error_handling.py
```

Demonstrates:
- `SandboxExecutionError` with `check=True`
- `SandboxTimeoutError` from exec timeout
- `SandboxNotFoundError` with `missing_ok=True`
- `SandboxTerminatedError` from `wait_until_complete()`

### Multiple Sandboxes (`multiple_sandboxes.py`)

Managing multiple sandboxes concurrently:

```bash
python examples/multiple_sandboxes.py
```

Demonstrates:
- Creating multiple sandboxes via session
- Running parallel commands with fire-then-collect pattern
- Automatic cleanup on session exit

### Delete Sandboxes (`delete_sandboxes.py`)

Multiple ways to delete sandboxes:

```bash
python examples/delete_sandboxes.py
```

Demonstrates:
- Using `Sandbox.delete(sandbox_id)` class method for deletion by ID
- Using `sandbox.stop()` instance method
- Using `Sandbox.from_id()` to attach and then stop
- Handling `SandboxNotFoundError` with `missing_ok=True`

### Reconnect to Sandbox (`reconnect_to_sandbox.py`)

Attach to existing sandboxes by ID:

```bash
# Create a long-running sandbox
python examples/reconnect_to_sandbox.py --create

# Reconnect to it later
python examples/reconnect_to_sandbox.py --sandbox-id <id>

# Stop it after reconnecting
python examples/reconnect_to_sandbox.py --sandbox-id <id> --stop
```

Demonstrates:
- Using `Sandbox.from_id()` to attach to existing sandboxes
- Executing commands on reconnected sandboxes
- Managing long-running sandboxes across script invocations

### Session Adopt Orphans (`session_adopt_orphans.py`)

Use sessions to adopt and clean up orphaned sandboxes:

```bash
# Create orphaned sandboxes
python examples/session_adopt_orphans.py --create-orphans

# Adopt and clean them up
python examples/session_adopt_orphans.py --cleanup

# Demo session.adopt()
python examples/session_adopt_orphans.py --demo-adopt

# Demo session.from_id()
python examples/session_adopt_orphans.py --demo-from-id
```

Demonstrates:
- Using `session.list(adopt=True)` to adopt orphaned sandboxes
- Using `session.adopt()` to manually adopt sandboxes
- Using `session.from_id()` to attach and adopt by ID
- Automatic cleanup of adopted sandboxes on session exit

### Parallel Batch Job (`parallel_batch_job.py`)

Parallel batch processing with progress tracking:

```bash
python examples/parallel_batch_job.py
```

Demonstrates:
- Creating multiple sandboxes in parallel via Session
- Submitting long-running commands concurrently
- Using `aviato.wait()` to process results as they complete
- Progress tracking through a batch job

### Cleanup by Tag (`cleanup_by_tag.py`)

Clean up sandboxes identified by tags:

```bash
# Create tagged sandboxes
python examples/cleanup_by_tag.py --create

# Clean them up
python examples/cleanup_by_tag.py --cleanup
```

Demonstrates:
- Tagging sandboxes for easy identification
- Using `Sandbox.list(tags=...)` to find sandboxes
- Parallel cleanup with fire-then-collect pattern

### Cleanup Old Sandboxes (`cleanup_old_sandboxes.py`)

Clean up sandboxes older than a threshold:

```bash
# Dry run
python examples/cleanup_old_sandboxes.py --dry-run

# Actually clean up sandboxes older than 2 hours
python examples/cleanup_old_sandboxes.py --max-age-hours 2
```

Demonstrates:
- Filtering sandboxes by age using `started_at` timestamp
- Client-side filtering after `Sandbox.list()`
- Dry run mode for safe testing

### SWE-bench Evaluation (`swebench/run_evaluation.py`)

Demonstrates using Aviato as a backend for running agentic model evaluations at scale. The script runs SWE-bench evaluations in parallel using Aviato sandboxes, showing how to orchestrate many concurrent evaluation instances.

Demonstrates:
- Using `Session` with `ThreadPoolExecutor` for parallel sandbox execution
- Integration with evaluation frameworks (SWE-bench)
- Pre-built container images from external registries (Epoch AI)
- Per-instance cleanup patterns for batch workloads

See [SWE-bench Guide](../docs/guides/swebench.md) for full documentation.

---

## Async Example

### Async Patterns (`async_patterns.py`)

Using await with OperationRef and Process:

```bash
python examples/async_patterns.py
```

Demonstrates:
- Awaiting `OperationRef` from `Sandbox.list()`, `Sandbox.from_id()`, `Sandbox.delete()`
- Awaiting `Process` from `exec()`
- Awaiting file operations (`read_file()`, `write_file()`)
- Parallel operations with `asyncio.gather()`
- Async session context managers

---

## API Patterns

The aviato SDK uses a sync/async hybrid API. Operations return immediately and results can be retrieved with `.result()` (sync) or `await` (async).

The `exec()` method returns a `Process` object. Call `.result()` to block for the final result. Iterate over `process.stdout` before calling `.result()` if you need real-time streaming output.

### Quick Usage (Factory Method)

```python
# One-liner creation - returns immediately
sb = Sandbox.run("echo", "hello")
sb.stop().result()  # Block for completion

# Context manager for automatic cleanup
with Sandbox.run() as sb:
    result = sb.exec(["echo", "hello"]).result()
```

### With Defaults

```python
defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=60,
    tags=("my-app",),
)

with Sandbox.run(defaults=defaults) as sb:
    result = sb.exec(["echo", "hello"]).result()
```

### Session for Multiple Sandboxes

```python
with Session(defaults) as session:
    sb1 = session.sandbox()
    sb2 = session.sandbox()

    result1 = sb1.exec(["echo", "from sb1"]).result()
    result2 = sb2.exec(["echo", "from sb2"]).result()
# Automatically cleans up all sandboxes on exit
```

### Function Execution

```python
with Session(defaults) as session:
    @session.function()
    def compute(x: int, y: int) -> int:
        return x + y

    ref = compute.remote(2, 3)  # Returns OperationRef immediately
    result = ref.result()       # Block for result: 5
```
