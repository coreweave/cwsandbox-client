# Sessions Guide

This guide covers using `Session` to manage multiple sandboxes with shared configuration.

## What is a Session?

A `Session` provides:
- Shared default configuration for all sandboxes
- Automatic cleanup when the session closes
- A scope for the `@session.function()` decorator

## Basic Usage

```python
import aviato
from aviato import SandboxDefaults

# Define shared configuration
defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("project-alpha",),
)

# Create a session with context manager
with aviato.Session(defaults=defaults) as session:
    # Create sandboxes through the session
    sb1 = session.sandbox()
    sb2 = session.sandbox()

    # Use the sandboxes
    result1 = sb1.exec(["echo", "sandbox 1"]).result()
    result2 = sb2.exec(["echo", "sandbox 2"]).result()

# All sandboxes automatically stopped when exiting context
```

## Creating Sandboxes

### session.sandbox()

Creates and starts a sandbox with session defaults:

```python
with aviato.Session(defaults=defaults) as session:
    # sandbox() creates AND starts the sandbox
    sandbox = session.sandbox()

    # Sandbox is already started, ready to use
    result = sandbox.exec(["echo", "hello"]).result()
```

### Overriding Defaults

Pass additional arguments to override session defaults:

```python
# Session with base configuration
defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("my-project",),
)

with aviato.Session(defaults=defaults) as session:
    # Override image and resources for this sandbox
    gpu_sandbox = session.sandbox(
        command="sleep",
        args=["infinity"],
        container_image="pytorch/pytorch:latest",
        resources={"cpu": "4000m", "memory": "16Gi", "gpu": "1"},
        tags=["gpu-workload"],  # Merged with session tags
    )
```

See the [Sandbox Configuration Guide](sandbox-configuration.md) for all available options.

## Multiple Sandbox Management

Sessions excel at managing sandbox pools:

```python
with aviato.Session(defaults=defaults) as session:
    # Create a pool of sandboxes
    sandboxes = [
        session.sandbox()
        for _ in range(5)
    ]

    # Distribute work across the pool
    processes = [
        sb.exec(["python", "-c", f"print({i})"])
        for i, sb in enumerate(sandboxes)
    ]

    # Collect results
    results = [p.result() for p in processes]

# All 5 sandboxes automatically cleaned up
```

## Session Lifecycle

### Manual Close

Close a session explicitly when not using context manager:

```python
session = aviato.Session(defaults=defaults)

sandbox = session.sandbox()
result = sandbox.exec(["echo", "hello"]).result()

# Close the session (stops all sandboxes)
session.close().result()
```

### What close() Does

1. Stops all sandboxes created through the session
2. Waits for cleanup to complete
3. Returns `OperationRef[None]`

### Error Handling

Sessions clean up even if exceptions occur:

```python
with aviato.Session(defaults=defaults) as session:
    sandbox = session.sandbox()
    raise RuntimeError("Something went wrong!")
# Sandbox is still cleaned up
```

## Adopting External Sandboxes

Bring sandboxes created outside the session under session management:

```python
import aviato
from aviato import Sandbox

with aviato.Session(defaults=defaults) as session:
    # Find existing sandboxes
    existing = Sandbox.list(tags=["orphaned-work"]).result()

    # Adopt them for cleanup
    for sandbox in existing:
        session.adopt(sandbox)

    # Now session.close() will clean them up too
```

## Session Properties

```python
session = aviato.Session(defaults=defaults)

# Number of sandboxes tracked
print(session.sandbox_count)  # 0

sandbox = session.sandbox()
print(session.sandbox_count)  # 1
```

## When to Use Sessions

| Use Case | Recommended Approach |
|----------|---------------------|
| Single sandbox, simple task | `Sandbox.run()` with context manager |
| Multiple sandboxes, shared config | Session |
| Function decorator API | Session required |
| Pool of workers | Session |
| One-off command | Direct `Sandbox.run()` |

## Without Sessions

For single sandboxes, sessions aren't required:

```python
from aviato import Sandbox

# Direct usage without session
with Sandbox.run() as sandbox:
    result = sandbox.exec(["echo", "hello"]).result()
# Cleanup handled by context manager
```
