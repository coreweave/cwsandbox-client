---
name: sandbox-cleanup
description: "Use when stopping CoreWeave Sandboxes, cleaning up resources, managing orphan sandboxes, batch deletion, or proper shutdown patterns. Covers stop(), delete(), session cleanup, and tagging strategies."
disable-model-invocation: false
---

# Sandbox Cleanup

Properly terminate sandboxes and manage resource cleanup.

## Context Manager (Recommended)

```python
with Sandbox.run() as sandbox:
    sandbox.exec(["echo", "hello"]).result()
# Stopped automatically
```

## Explicit stop()

```python
sandbox = Sandbox.run()
sandbox.exec(["echo", "hello"]).result()
sandbox.stop().result()
```

## stop() Options

```python
# Capture state before shutdown
sandbox.stop(snapshot_on_stop=True).result()

# Safe even if already stopped
sandbox.stop(missing_ok=True).result()

# Longer grace period
sandbox.stop(graceful_shutdown_seconds=30.0).result()
```

## Session Cleanup

Sessions auto-cleanup all sandboxes on close:

```python
with Session(defaults) as session:
    sandboxes = [session.sandbox() for _ in range(5)]
    # All cleaned up on exit
```

## Delete by ID

Remove sandboxes by ID without needing instance:

```python
from cwsandbox import Sandbox

# Delete specific sandbox
Sandbox.delete("sandbox-abc123", missing_ok=True).result()

# Batch delete old sandboxes
for sb in old_sandboxes:
    Sandbox.delete(sb.sandbox_id, missing_ok=True).result()
```

## Tag-Based Cleanup

Clean up all sandboxes with specific tags:

```python
from cwsandbox import Sandbox, SandboxStatus

# Find sandboxes by tag
sandboxes = Sandbox.list(
    tags=["training-run-2024-01-15"],
    include_stopped=True,
).result()

# Delete each
for sb in sandboxes:
    Sandbox.delete(sb.sandbox_id).result()
```

## Age-Based Cleanup

Delete sandboxes older than threshold:

```python
from datetime import datetime, timedelta

threshold = datetime.now() - timedelta(days=7)

for sb in Sandbox.list(include_stopped=True).result():
    if sb.started_at and sb.started_at < threshold:
        Sandbox.delete(sb.sandbox_id, missing_ok=True).result()
```

## Global Cleanup Handlers

SDK registers atexit and signal handlers automatically:
- On process exit: stops all registered sessions
- On Ctrl+C / SIGTERM: cleanup then exit
- Second interrupt during cleanup: force exit

## stop() vs delete()

| | stop() | delete() |
|--|--------|---------|
| Target | Live sandbox instance | Sandbox by ID |
| Purpose | Graceful shutdown | Permanent removal |
| Needs instance? | Yes | No |

## References

- [Cleanup Patterns Guide](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/cleanup-patterns)
