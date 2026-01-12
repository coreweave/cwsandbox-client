# Cleanup Patterns Guide

This guide covers resource management and cleanup strategies for sandboxes.

## Automatic Cleanup

### Context Managers (Recommended)

Sandboxes are stopped when exiting the context:

```python
from aviato import Sandbox

with Sandbox.run() as sandbox:
    result = sandbox.exec(["echo", "hello"]).result()
# sandbox.stop() called automatically
```

Sessions clean up all their sandboxes:

```python
import aviato

with aviato.Session(container_image="python:3.11") as session:
    sb1 = session.sandbox()
    sb2 = session.sandbox()
# Both sandboxes stopped automatically
```

### Global Cleanup Handlers

The SDK registers cleanup handlers for process termination:

| Scenario | Behavior |
|----------|----------|
| Normal script exit | `atexit` handler stops all sandboxes |
| Ctrl+C (SIGINT) | Signal handler stops all sandboxes |
| SIGTERM | Signal handler stops all sandboxes |
| Second Ctrl+C | Force exit (prevents hang) |

## Manual Cleanup

### stop()

```python
sandbox = Sandbox.run()
result = sandbox.exec(["echo", "hello"]).result()
sandbox.stop().result()

# With options
sandbox.stop(graceful_shutdown_seconds=30.0).result()
sandbox.stop(snapshot_on_stop=True).result()
```

### Session close()

```python
session = aviato.Session(container_image="python:3.11")
sandbox = session.sandbox()
# ...
session.close().result()  # Stops all sandboxes
```

### Batch Cleanup

```python
from aviato import get

sandboxes = [Sandbox.run() for _ in range(5)]
# ... use sandboxes ...
get([sb.stop() for sb in sandboxes])  # Stop all in parallel
```

## Orphan Management

### Tagging for Discovery

The SDK's automatic cleanup handlers prevent most orphans, but sandboxes can still be left running after forced shutdowns (kill -9), network failures, or when creating sandboxes outside of sessions and context managers. Use tags to make any orphans easily discoverable.

```python
from aviato import Sandbox, SandboxDefaults

# Tag at creation time
defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("my-project", "batch-job-123"),
)

with Sandbox.run(defaults=defaults) as sandbox:
    result = sandbox.exec(["echo", "hello"]).result()
```

Good tagging practices:
- Project or application name (`my-project`)
- Job or run identifier (`batch-job-123`, `run-2024-01-15`)
- Environment (`dev`, `staging`, `prod`)

### Finding Orphaned Sandboxes

Query by tags to find sandboxes from previous runs:

```python
from aviato import Sandbox

orphans = Sandbox.list(tags=["my-project"]).result()
for sandbox in orphans:
    sandbox.stop().result()
```

### Session Adoption

Bring orphans under session management for automatic cleanup:

```python
import aviato

with aviato.Session(container_image="python:3.11") as session:
    orphans = session.list(tags=["my-project"]).result()
    for sandbox in orphans:
        session.adopt(sandbox)
# All adopted sandboxes cleaned up with session
```

### Deleting by ID

```python
from aviato import Sandbox

Sandbox.delete("sandbox-abc123").result()
Sandbox.delete("sandbox-abc123", missing_ok=True)  # Ignore if gone
```
