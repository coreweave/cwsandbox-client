---
name: sandbox-async
description: "Use when working with async patterns in CoreWeave Sandbox, in Jupyter notebooks, or when integrating with async codebases. Covers await vs .result(), OperationRef, parallel execution, and async context managers."
disable-model-invocation: false
---

# Sync vs Async Patterns

The SDK has a single async implementation. Sync/async flexibility comes from how you consume results.

## Quick Decision

| Use Case | Pattern | Reason |
|----------|---------|--------|
| Most operations | Sync | Simpler, no asyncio boilerplate |
| Parallel execution | Sync | Non-blocking by design |
| Jupyter notebooks | Sync | No nest_asyncio needed |
| Async codebase | Async | Integrates with existing async |

## Core Concept: OperationRef

Most methods return `OperationRef[T]` — both `.result()`-able and awaitable:

```python
ref = sandbox.read_file("/path")

# Sync: block
data = ref.result()

# Async: await
data = await ref
```

## Key Rule

> **Never use `.result()` in async contexts** — it blocks the event loop. Use `await` instead.

## Sandbox Creation

<Tabs>
<Tab title="Sync (Recommended)">

```python
from cwsandbox import Sandbox

with Sandbox.run() as sb:
    result = sb.exec(["echo", "hello"]).result()
```

</Tab>
<Tab title="Async">

```python
from cwsandbox import Sandbox

sb = Sandbox()
async with sb:
    result = await sb.exec(["echo", "hello"])
```

</Tab>
</Tabs>

## exec()

<Tabs>
<Tab title="Sync">

```python
result = sb.exec(["echo", "hello"]).result()
```

</Tab>
<Tab title="Async">

```python
result = await sb.exec(["echo", "hello"])
```

</Tab>
</Tabs>

## read_file()

<Tabs>
<Tab title="Sync">

```python
data = sb.read_file("/path").result()
```

</Tab>
<Tab title="Async">

```python
data = await sb.read_file("/path")
```

</Tab>
</Tabs>

## stop()

<Tabs>
<Tab title="Sync">

```python
sb.stop().result()
```

</Tab>
<Tab title="Async">

```python
await sb.stop()
```

</Tab>
</Tabs>

## start()

<Tabs>
<Tab title="Sync">

```python
sb = Sandbox()
sb.start().result()
```

</Tab>
<Tab title="Async">

```python
sb = Sandbox()
await sb.start()
```

</Tab>
</Tabs>

## Sandbox.list()

<Tabs>
<Tab title="Sync">

```python
sandboxes = Sandbox.list(tags=["my-tag"]).result()
```

</Tab>
<Tab title="Async">

```python
sandboxes = await Sandbox.list(tags=["my-tag"])
```

</Tab>
</Tabs>

## Streaming (async)

```python
process = sb.exec(["python", "-c", "import time; [print(i) for i in range(5)]"])

async for line in process.stdout:
    print(line, end="")

result = await process
```

## stdin Streaming (async)

```python
process = sb.exec(["cat"], stdin=True)
await process.stdin.write(b"hello ")
await process.stdin.writeline("world")
await process.stdin.close()
result = await process
```

## Parallel Execution

Operations are non-blocking by design — `exec()` returns immediately:

```python
# All start in parallel (sync API)
p1 = sb.exec(["sleep", "1"])
p2 = sb.exec(["sleep", "1"])
p3 = sb.exec(["sleep", "1"])

# Block ~1 second total, not 3
cwsandbox.result([p1, p2, p3])
```

## Jupyter Notebooks

Sync API works without nest_asyncio (SDK runs its own daemon thread with event loop):

```python
# Cell 1
from cwsandbox import Sandbox
sandbox = Sandbox.run()
sandbox.wait()

# Cell 2
result = sandbox.exec(["python", "-c", "print(1+1)"]).result()

# Cell 3
sandbox.stop().result()
```

For async in Jupyter, `await` works directly:

```python
sandboxes = await Sandbox.list(tags=["my-tag"])
```

## @session.function() (async)

```python
async with Session(defaults) as session:
    @session.function()
    def compute(x: int, y: int) -> int:
        return x + y

    result = await compute.remote(2, 3)

    refs = compute.map([(1, 2), (3, 4)])
    results = [await r for r in refs]
```

## Module-Level Helpers

```python
import cwsandbox

# cwsandbox.results() - batch retrieve refs
data = cwsandbox.results(sandbox.read_file("/path"))

# cwsandbox.wait() - wait for sandboxes/processes
done, pending = cwsandbox.wait(processes)
```

## References

- [Sync vs Async Guide](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/sync-vs-async)
