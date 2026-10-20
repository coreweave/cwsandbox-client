---
name: sandbox-session
description: "Use when managing multiple CoreWeave Sandboxes simultaneously, creating sandbox pools, or sharing configuration across sandboxes via Session. Covers Session, SandboxDefaults, adopt patterns, and parallel execution."
disable-model-invocation: false
---

# Multiple Sandboxes with Session

Manage multiple sandboxes with shared defaults and automatic cleanup.

## Basic Session Usage

```python
from cwsandbox import Sandbox, SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("my-app",),
)

with Sandbox.session(defaults) as session:
    sb1 = session.sandbox()
    sb2 = session.sandbox()

    p1 = sb1.exec(["echo", "one"])
    p2 = sb2.exec(["echo", "two"])

    print(p1.result().stdout, p2.result().stdout)
# All sandboxes cleaned up automatically
```

## Session Methods

| Method | Description |
|--------|-------------|
| `session.sandbox(command, args, **kwargs)` | Create sandbox with session defaults |
| `session.function(**kwargs)` | Decorator for remote functions |
| `session.adopt(sandbox)` | Register existing sandbox for cleanup |
| `session.close()` | Cleanup all sandboxes |
| `session.list(tags, status, adopt)` | Find sandboxes matching criteria |
| `session.from_id(id, adopt)` | Attach to sandbox by ID |

## Sandbox Pools

Pre-start all sandboxes for faster first-execution:

```python
with Session(defaults) as session:
    sandboxes = [session.sandbox() for _ in range(5)]
    
    # Pre-start all at once
    refs = [sb.start() for sb in sandboxes]
    [r.result() for r in refs]

    # Now exec on each runs immediately
    processes = [sb.exec(["echo", f"hello-{i}"]) for i, sb in enumerate(sandboxes)]
```

## Adopt Existing Sandboxes

Attach to sandboxes discovered via `Sandbox.list()` or `Sandbox.from_id()`:

```python
from cwsandbox import Sandbox, SandboxStatus

# Find existing sandboxes
sandboxes = Sandbox.list(
    tags=["training-run"],
    status=[SandboxStatus.RUNNING],
).result()

with Session(defaults) as session:
    for sb in sandboxes:
        session.adopt(sb)  # Will cleanup when session closes
```

## Parallel Execution with wait()

```python
import cwsandbox

with Session(defaults) as session:
    sandboxes = [session.sandbox() for _ in range(10)]
    
    processes = [sb.exec(["python", "-c", f"print({i**2})"]) for i, sb in enumerate(sandboxes)]
    
    # Wait for all to complete
    done, pending = cwsandbox.wait(processes)
```

## References

- [Sessions Guide](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/sessions)
