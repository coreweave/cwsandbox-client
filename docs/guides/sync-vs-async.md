# Sync vs Async Patterns

The aviato SDK provides a unified sync/async hybrid API. All operations work with both patterns.

## Quick Decision Guide

| Use Case | Pattern | Reason |
|----------|---------|--------|
| Most operations | Sync | Simpler code, no asyncio boilerplate |
| Parallel execution | Sync | Operations are non-blocking by design |
| Jupyter notebooks | Sync | No nest_asyncio needed |
| Async codebase | Async | Integrates with existing async code |

**Rule of thumb**: Use sync patterns (`.get()`) for simplicity. All methods work with both sync and async patterns.

## Sync Pattern (Recommended Default)

All SDK operations work synchronously. Operations return immediately with lazy result objects:

```python
from aviato import Sandbox

# Sandbox.run() returns immediately
sandbox = Sandbox.run()

# .result() blocks for result
result = sandbox.exec(["echo", "hello"]).result()
print(result.stdout)  # "hello\n"

# Context manager for automatic cleanup
with Sandbox.run() as sandbox:
    result = sandbox.exec(["ls"]).result()
    print(result.stdout)
```

### Key Sync Methods

Most methods return `OperationRef`, a lazy object that implements a `.get()` blocking function to get the value
while also be await-able for async codebases.

Sandbox `exec` calls return a `Process` object which uses `.result()` to block and return a `ProcessResult` instance
with information about the command that was run. `Process` has additional functionality like ouput streaming, and is
also is await-able for async codebases. 

- `Sandbox.run()` - Create and start sandbox (returns immediately)
- `Sandbox.list()` - Query existing sandboxes (returns OperationRef)
- `Sandbox.from_id()` - Attach to existing sandbox (returns OperationRef)
- `Sandbox.delete()` - Delete sandbox by ID (returns OperationRef)
- `sandbox.exec()` - Execute command (returns Process)
- `sandbox.read_file()` - Read file contents (returns OperationRef)
- `sandbox.write_file()` - Write file contents (returns OperationRef)
- `sandbox.stop()` - Stop sandbox (returns OperationRef)
- `sandbox.wait()` - Wait until RUNNING status
- `sandbox.wait_until_complete()` - Wait until terminal status
- `sandbox.get_status()` - Fetch fresh status
- `session.list()` - Query sandboxes matching session tags (returns OperationRef)
- `session.from_id()` - Attach and optionally adopt by ID (returns OperationRef)

## Async Codebases

`OperationRef` is awaitable, so the same methods work in async code without `.get()`:

```python
import asyncio
from aviato import Sandbox

async def main() -> None:
    # Sandbox.run() returns immediately
    sandbox = Sandbox.run()

    # await instead of .result()
    result = await sandbox.exec(["echo", "hello"])
    print(result.stdout)  # "hello\n"

    # Async context manager for automatic cleanup
    async with Sandbox.run() as sandbox:
        result = await sandbox.exec(["ls"])
        print(result.stdout)

asyncio.run(main())
```

## Parallel Execution

The sync API supports parallel execution because operations are **non-blocking by design**. Methods like `exec()`, `read_file()`, and `write_file()` return immediately - you only block when you call `.result()` or `.get()`.

```python
from aviato import Sandbox

# Start multiple sandboxes (each returns immediately)
sandboxes = [Sandbox.run() for _ in range(3)]

# Fire off commands on each sandbox (non-blocking)
processes = [sb.exec(["echo", f"sandbox-{i}"]) for i, sb in enumerate(sandboxes)]

# Now block for all results
results = [p.result() for p in processes]
for r in results:
    print(r.stdout)

# Clean up
for sb in sandboxes:
    sb.stop()
```

This pattern executes commands in parallel without any async code. The key insight: **non-blocking != async**. The sync API is non-blocking, you just use `.result()` or `.get()` to block when you need results.

## Jupyter Notebooks

The sync API works in Jupyter without `nest_asyncio`:

```python
# Cell 1 - Create sandbox
from aviato import Sandbox
sandbox = Sandbox.run()
sandbox.wait()  # Wait until RUNNING

# Cell 2 - Execute commands
result = sandbox.exec(["python", "-c", "print(1+1)"]).result()
print(result.stdout)

# Cell 3 - Discovery
sandboxes = Sandbox.list(tags=["notebook"]).get()
print(f"Found {len(sandboxes)} sandboxes")

# Cell 4 - Cleanup
sandbox.stop().get()
```

For async operations in Jupyter, use `await` directly (Jupyter has a built-in event loop):

```python
# Works in Jupyter without asyncio.run()
sandboxes = await Sandbox.list(tags=["notebook"])
```
