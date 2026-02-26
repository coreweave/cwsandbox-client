# Sync vs Async Patterns

The cwsandbox SDK has a single async implementation internally. The sync/async flexibility comes from
how you consume results: `.result()` for sync, `await` for async.

## Quick Decision Guide

| Use Case | Pattern | Reason |
|----------|---------|--------|
| Most operations | Sync | Simpler code, no asyncio boilerplate |
| Parallel execution | Sync | Operations are non-blocking by design |
| Jupyter notebooks | Sync | No nest_asyncio needed |
| Async codebase | Async | Integrates with existing async code |

**Rule of thumb**: Use sync patterns (`.result()`) for simplicity. Switch to `await` only when you are already
in an async codebase.

## Core Concept: OperationRef

Most SDK methods return `OperationRef[T]`, a wrapper that is both `.result()`-able and awaitable.
The same exceptions are raised in both paths.

```python
ref = sandbox.read_file("/output/data.txt")

# Sync: block for the value
data = ref.result()

# Async: await instead
data = await ref
```

`Sandbox.exec()` returns `Process`, which extends `OperationRef[ProcessResult]` with streaming
and stdin.

!!! warning "Never use `.result()` in async contexts"
    Calling `.result()` blocks the thread. In an async context this blocks the event loop and
    can deadlock your application. Use `await` instead.

## Auto-Start Behavior

`session.sandbox()` returns an unstarted sandbox. The sandbox auto-starts on the first operation
that needs it:

**Triggers auto-start**: `exec()`, `read_file()`, `write_file()`, `wait()`, `wait_until_complete()`

**Does not auto-start**: `get_status()` (raises `SandboxNotRunningError`), `stop()` (no-op if never started)

## Operations

### Creating Sandboxes

=== "Sync"

    ```python
    from cwsandbox import Sandbox

    # Context manager (recommended)
    with Sandbox.run("sleep", "infinity") as sb:
        result = sb.exec(["echo", "hello"]).result()
    # Automatically stopped on exit
    ```

=== "Async"

    ```python
    from cwsandbox import Sandbox

    # Construct + await to reach RUNNING, then async context manager for cleanup
    sb = Sandbox(command="sleep", args=["infinity"])
    async with sb as sb:
        result = await sb.exec(["echo", "hello"])
    # Automatically stopped on exit
    ```

!!! note "Why not `Sandbox.run()` in async?"
    `Sandbox.run()` calls `start().result()` internally, which blocks the event loop.
    In async code, construct with `Sandbox(...)` and use `async with` or `await sandbox`
    to reach RUNNING status without blocking.

### start()

=== "Sync"

    ```python
    sb = Sandbox(command="sleep", args=["infinity"])
    sb.start().result()  # Block until backend accepts
    print(sb.sandbox_id)
    ```

=== "Async"

    ```python
    sb = Sandbox(command="sleep", args=["infinity"])
    await sb.start()  # Await until backend accepts
    print(sb.sandbox_id)
    ```

### exec()

=== "Sync"

    ```python
    result = sb.exec(["echo", "hello"]).result()
    print(result.stdout)  # "hello\n"

    # Raise on non-zero exit code
    result = sb.exec(["python", "-c", "exit(1)"], check=True).result()
    ```

=== "Async"

    ```python
    result = await sb.exec(["echo", "hello"])
    print(result.stdout)  # "hello\n"

    # Raise on non-zero exit code
    result = await sb.exec(["python", "-c", "exit(1)"], check=True)
    ```

### read_file()

=== "Sync"

    ```python
    data = sb.read_file("/output/result.txt").result()
    print(data.decode())
    ```

=== "Async"

    ```python
    data = await sb.read_file("/output/result.txt")
    print(data.decode())
    ```

### write_file()

=== "Sync"

    ```python
    sb.write_file("/input/data.txt", b"content").result()
    ```

=== "Async"

    ```python
    await sb.write_file("/input/data.txt", b"content")
    ```

### stop()

=== "Sync"

    ```python
    sb.stop().result()

    # Ignore if already deleted
    sb.stop(missing_ok=True).result()
    ```

=== "Async"

    ```python
    await sb.stop()

    # Ignore if already deleted
    await sb.stop(missing_ok=True)
    ```

### wait()

`wait()` blocks until the sandbox reaches RUNNING status. It is sync-only because blocking is
the intent. In async code, `await sandbox` achieves the same thing.

```python
# Sync: block until RUNNING
sb = Sandbox.run("sleep", "infinity").wait()
result = sb.exec(["echo", "ready"]).result()
```

```python
# Async: await the sandbox directly
sb = Sandbox(command="sleep", args=["infinity"])
await sb  # Wait until RUNNING
result = await sb.exec(["echo", "ready"])
```

### wait_until_complete()

=== "Sync"

    ```python
    sb = Sandbox.run("python", "-c", "print('done')")
    sb.wait_until_complete().result()
    print(f"Exit code: {sb.returncode}")

    # Handle externally-terminated sandboxes without raising
    sb.wait_until_complete(raise_on_termination=False).result()
    ```

=== "Async"

    ```python
    sb = Sandbox(command="python", args=["-c", "print('done')"])
    await sb.wait_until_complete()
    print(f"Exit code: {sb.returncode}")

    # Handle externally-terminated sandboxes without raising
    await sb.wait_until_complete(raise_on_termination=False)
    ```

### get_status()

`get_status()` is sync-only. It fetches fresh status from the API.

```python
status = sb.get_status()
print(f"Sandbox is {status}")  # e.g. SandboxStatus.RUNNING
```

### Sandbox.list()

=== "Sync"

    ```python
    sandboxes = Sandbox.list(tags=["my-job"]).result()
    for sb in sandboxes:
        print(f"{sb.sandbox_id}: {sb.status}")
    ```

=== "Async"

    ```python
    sandboxes = await Sandbox.list(tags=["my-job"])
    for sb in sandboxes:
        print(f"{sb.sandbox_id}: {sb.status}")
    ```

### Sandbox.from_id()

=== "Sync"

    ```python
    sb = Sandbox.from_id("sandbox-abc123").result()
    result = sb.exec(["echo", "reconnected"]).result()
    ```

=== "Async"

    ```python
    sb = await Sandbox.from_id("sandbox-abc123")
    result = await sb.exec(["echo", "reconnected"])
    ```

### Sandbox.delete()

=== "Sync"

    ```python
    Sandbox.delete("sandbox-abc123").result()

    # Ignore if already deleted
    Sandbox.delete("sandbox-abc123", missing_ok=True).result()
    ```

=== "Async"

    ```python
    await Sandbox.delete("sandbox-abc123")

    # Ignore if already deleted
    await Sandbox.delete("sandbox-abc123", missing_ok=True)
    ```

### session.list()

=== "Sync"

    ```python
    sandboxes = session.list().result()
    for sb in sandboxes:
        print(f"{sb.sandbox_id}: {sb.status}")
    ```

=== "Async"

    ```python
    sandboxes = await session.list()
    for sb in sandboxes:
        print(f"{sb.sandbox_id}: {sb.status}")
    ```

### session.from_id()

=== "Sync"

    ```python
    sb = session.from_id("sandbox-abc123").result()
    result = sb.exec(["echo", "adopted"]).result()
    ```

=== "Async"

    ```python
    sb = await session.from_id("sandbox-abc123")
    result = await sb.exec(["echo", "adopted"])
    ```

### Streaming stdout

=== "Sync"

    ```python
    process = sb.exec(["python", "-c", "import time; [print(i) or time.sleep(0.1) for i in range(5)]"])
    for line in process.stdout:
        print(line, end="")
    result = process.result()
    ```

=== "Async"

    ```python
    process = sb.exec(["python", "-c", "import time; [print(i) or time.sleep(0.1) for i in range(5)]"])
    async for line in process.stdout:
        print(line, end="")
    result = await process
    ```

### Stdin streaming

Enable stdin with `stdin=True`. Use `write()` for raw bytes, `writeline()` for text lines,
and `close()` to signal EOF.

=== "Sync"

    ```python
    process = sb.exec(["cat"], stdin=True)
    process.stdin.write(b"hello ").result()
    process.stdin.writeline("world").result()
    process.stdin.close().result()  # Signals EOF to the process
    result = process.result()
    print(result.stdout)  # "hello world\n"
    ```

=== "Async"

    ```python
    process = sb.exec(["cat"], stdin=True)
    await process.stdin.write(b"hello ")
    await process.stdin.writeline("world")
    await process.stdin.close()  # Signals EOF to the process
    result = await process
    print(result.stdout)  # "hello world\n"
    ```

### cwsandbox.results()

`cwsandbox.results()` is a sync-only batch helper. It calls `.result()` on one or more OperationRefs.

```python
import cwsandbox

# Single ref
data = cwsandbox.results(sandbox.read_file("/path"))

# Multiple refs
all_data = cwsandbox.results([sb.read_file(f) for f in files])
```

### cwsandbox.wait()

`cwsandbox.wait()` is sync-only. It waits for a sequence of `Sandbox`, `OperationRef`, or `Process`
objects and returns `(done, pending)`. Sandboxes resolve when they reach RUNNING status, not when
they complete.

```python
import cwsandbox

# Wait for all sandboxes to reach RUNNING
sandboxes = [Sandbox.run() for _ in range(5)]
done, pending = cwsandbox.wait(sandboxes)

# Wait for first 2 operations to complete
refs = [sb.read_file(f) for f in files]
done, pending = cwsandbox.wait(refs, num_returns=2)

# Wait with timeout
done, pending = cwsandbox.wait(procs, timeout=30.0)
```

### @session.function()

=== "Sync"

    ```python
    with Session(defaults) as session:
        @session.function()
        def compute(x: int, y: int) -> int:
            return x + y

        result = compute.remote(2, 3).result()
        print(result)  # 5

        # Parallel map
        refs = compute.map([(1, 2), (3, 4), (5, 6)])
        results = [r.result() for r in refs]  # [3, 7, 11]
    ```

=== "Async"

    ```python
    async with Session(defaults) as session:
        @session.function()
        def compute(x: int, y: int) -> int:
            return x + y

        result = await compute.remote(2, 3)
        print(result)  # 5

        # Parallel map
        refs = compute.map([(1, 2), (3, 4), (5, 6)])
        results = [await r for r in refs]  # [3, 7, 11]
    ```

## Parallel Execution

The sync API supports parallel execution because operations are **non-blocking by design**. Methods
like `exec()`, `read_file()`, and `write_file()` return immediately. You only block when you call
`.result()`.

`Sandbox.run()` blocks until the backend accepts the start request (it calls `start().result()`
internally). For parallel startup, use `session.sandbox()` (auto-starts on first operation) or
collect `start()` refs:

```python
from cwsandbox import Sandbox, Session, SandboxDefaults

# Option 1: session.sandbox() - sandboxes auto-start on first exec()
with Session(SandboxDefaults()) as session:
    sandboxes = [session.sandbox(command="sleep", args=["infinity"]) for _ in range(3)]
    processes = [sb.exec(["echo", f"sb-{i}"]) for i, sb in enumerate(sandboxes)]
    results = [p.result() for p in processes]

# Option 2: Sandbox() + collect start() refs
sandboxes = [Sandbox(command="sleep", args=["infinity"]) for _ in range(3)]
start_refs = [sb.start() for sb in sandboxes]
for ref in start_refs:
    ref.result()  # All starts proceed in parallel

processes = [sb.exec(["echo", f"sb-{i}"]) for i, sb in enumerate(sandboxes)]
results = [p.result() for p in processes]
for sb in sandboxes:
    sb.stop().result()
```

## Jupyter Notebooks

The sync API works in Jupyter without `nest_asyncio` because the SDK runs its own background event
loop in a daemon thread:

```python
# Cell 1 - Create sandbox
from cwsandbox import Sandbox
sandbox = Sandbox.run()
sandbox.wait()  # Wait until RUNNING

# Cell 2 - Execute commands
result = sandbox.exec(["python", "-c", "print(1+1)"]).result()
print(result.stdout)

# Cell 3 - Cleanup
sandbox.stop().result()
```

For async in Jupyter, `await` works directly since Jupyter has a built-in event loop:

```python
# Works in Jupyter without asyncio.run()
sandboxes = await Sandbox.list(tags=["notebook"])
```

## Error Handling

The same exceptions are raised in both sync and async paths. `.result()` re-raises any exception
from the underlying operation; `await` does the same. See the [Troubleshooting](troubleshooting.md) guide
for the full exception hierarchy and recovery patterns.
