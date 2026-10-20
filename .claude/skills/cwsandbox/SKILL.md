---
name: cwsandbox
description: "Use when working with cwsandbox-client library for CoreWeave Sandbox remote code execution. Covers Sandbox.run(), exec(), streaming, file operations, Session management, function decorators, cleanup patterns, and troubleshooting. Relevant for requests involving sandboxes, remote execution, containerized code running, RL training, model evaluation, or agent tool use."
disable-model-invocation: false
---

# CoreWeave Sandbox Skill

Python SDK for CoreWeave Sandbox — a compute platform for orchestrating isolated execution environments at scale.

## Quick Start

```python
from cwsandbox import Sandbox

with Sandbox.run(container_image="python:3.11") as sb:
    result = sb.exec(["python", "-c", "print('Hello!')"]).result()
    print(result.stdout)
```

## Core Patterns

### Sandbox Creation

```python
# Factory method (recommended) - blocks until backend accepts
sb = Sandbox.run("echo", "hello")
result = sb.exec(["echo", "more"]).result()
sb.stop().result()

# Context manager - auto-stops on exit
with Sandbox.run("sleep", "infinity") as sb:
    result = sb.exec(["echo", "hello"]).result()

# Session for multiple sandboxes with shared defaults
with Session(defaults) as session:
    sb = session.sandbox(command="sleep", args=["infinity"])
    result = sb.exec(["echo", "hello"]).result()
```

### Streaming Output

```python
with Sandbox.run("sleep", "infinity") as sb:
    process = sb.exec(["echo", "hello"])
    for line in process.stdout:  # Stream lines as they arrive
        print(line, end="")
    result = process.result()
```

### Async Pattern

```python
async with Sandbox.run("sleep", "infinity") as sb:
    result = await sb.exec(["echo", "hello"])
```

## Key Methods

| Method | Returns | Notes |
|--------|---------|-------|
| `Sandbox.run(*args, **kwargs)` | Sandbox | Factory - creates and starts sandbox |
| `sandbox.exec(command, cwd, check, timeout_seconds, stdin)` | Process | Execute command |
| `sandbox.read_file(path)` | OperationRef[bytes] | Read file |
| `sandbox.write_file(path, content)` | OperationRef[None] | Write file |
| `sandbox.stream_logs(follow, tail_lines, timestamps)` | StreamReader[str] | Stream PID 1 logs |
| `sandbox.shell(command, width, height)` | TerminalSession | Interactive TTY |
| `sandbox.stop(snapshot_on_stop, graceful_shutdown_seconds, missing_ok)` | OperationRef[None] | Stop sandbox |
| `sandbox.wait()` | self | Block until RUNNING |
| `sandbox.wait_until_complete(timeout, raise_on_termination)` | OperationRef[Sandbox] | Block until terminal |

## Lifecycle States

```
PENDING -> CREATING -> RUNNING -> COMPLETED/FAILED/TERMINATED
```

- `wait()` blocks until RUNNING or terminal
- `wait_until_complete()` blocks until terminal state
- `COMPLETED` = command exited 0
- `TERMINATED` = external kill / lifetime exceeded
- `FAILED` = startup or runtime error

## SandboxStatus States

`PENDING`, `CREATING`, `RUNNING`, `PAUSED`, `COMPLETED`, `TERMINATED`, `FAILED`, `UNSPECIFIED`

## Configuration Kwargs

```python
Sandbox.run(
    container_image="python:3.11",  # Container image
    resources={...},                 # CPU, memory, GPU requests
    mounted_files=[...],              # Files to mount
    s3_mount={...},                  # S3 bucket mount
    ports={...},                     # Port mappings
    network=NetworkOptions(           # Network config
        ingress_mode="public",
        exposed_ports=(8080,),
        egress_mode="internet",
    ),
    secrets=[Secret(...)],           # Secret injection
    max_timeout_seconds=3600,        # Max timeout
    environment_variables={...},     # Env vars
    annotations={...},               # Kubernetes annotations
)
```

## Session Management

```python
from cwsandbox import Session, SandboxDefaults

with Session(SandboxDefaults(tags=("my-tag",))) as session:
    sb = session.sandbox(command="sleep", args=["infinity"])
    result = sb.exec(["echo", "hello"]).result()

# Parallel execution
refs = compute.map([(1, 2), (3, 4), (5, 6)])
results = [r.result() for r in refs]
```

## Remote Functions

```python
with Session(defaults) as session:
    @session.function()
    def compute(x: int, y: int) -> int:
        return x + y

    ref = compute.remote(2, 3)
    result = ref.result()
    # or: result = compute.local(2, 3)  # local testing
```

## Exception Hierarchy

```
CWSandboxError
├── CWSandboxAuthenticationError
│   └── WandbAuthError
├── SandboxError
│   ├── SandboxNotRunningError
│   ├── SandboxTimeoutError
│   ├── SandboxTerminatedError
│   ├── SandboxFailedError
│   ├── SandboxNotFoundError
│   ├── SandboxExecutionError
│   └── SandboxFileError
└── FunctionError
    ├── AsyncFunctionError
    └── FunctionSerializationError
```

## Authentication

1. `CWSANDBOX_API_KEY` env var (Bearer token)
2. `WANDB_API_KEY` + `WANDB_ENTITY` (W&B headers)
3. `~/.netrc` (api.wandb.ai) + `WANDB_ENTITY`

## Important Design Points

- **Sync/async hybrid**: All methods return immediately; use `.result()` to block (sync) or `await` (async)
- **Single-threaded**: Not safe to call `.result()` from multiple threads simultaneously
- **Lazy-start**: `Sandbox.run()` returns once backend accepts — not when RUNNING
- **Auto-start**: `exec()`, `read_file()`, `write_file()`, `wait()` all auto-start if not started

## Common Issues

| Issue | Solution |
|-------|----------|
| "Sandbox not running" on exec | Sandbox hasn't reached RUNNING yet — `exec()` waits internally |
| Timeout during startup | Startup wait and operation timeout are separate phases |
| Integration tests hang | Use `.result()`, not `await` in sync tests |
| Stdin not working | Must pass `stdin=True` to `exec()` |

## Running Tests

```bash
# Unit tests (284 tests, no network)
mise run test

# Integration tests (31 tests, requires auth)
mise run test:e2e
mise run test:e2e:parallel

# Individual test
timeout 120 uv run pytest tests/integration/cwsandbox/test_sandbox.py::test_sandbox_lifecycle -v
```

## Examples

See `examples/` directory:
- `quick_start.py` — Context manager with exec
- `streaming_exec.py` — Real-time stdout iteration
- `function_decorator.py` — Remote function execution
- `error_handling.py` — Exception hierarchy
- `multiple_sandboxes.py` — Session-based multi-sandbox
- `parallel_batch_job.py` — Batch processing with `cwsandbox.wait()`
- `cleanup_by_tag.py` — Tag-based cleanup
- `interactive_streaming_sandbox.py` — Log streaming and CLI

## References

- Docs: https://docs.coreweave.com/products/coreweave-sandbox
- Backend: github.com/coreweave/aviato
