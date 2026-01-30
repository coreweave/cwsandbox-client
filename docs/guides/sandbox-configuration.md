# Sandbox Configuration Guide

This guide covers sandbox configuration options: resources, mounted files, ports, and timeouts.

## Overview

Sandbox configuration can be set in three places:

1. **SandboxDefaults** - Shared defaults for all sandboxes in a session
2. **Sandbox.run() kwargs** - Per-sandbox overrides
3. **@session.function() kwargs** - Function-specific configuration

```python
from aviato import Sandbox, SandboxDefaults, Session

# Via SandboxDefaults
defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=3600,
)

# Via Sandbox.run() kwargs
sandbox = Sandbox.run(
    defaults=defaults,
    resources={"cpu": "500m", "memory": "1Gi"},
)

# Via @session.function() kwargs
with Session(defaults) as session:
    @session.function(resources={"cpu": "1000m"})
    def compute(x: int) -> int:
        return x * 2
```

## Resources

Request CPU, memory, and GPU resources for sandboxes:

```python
sandbox = Sandbox.run(
    resources={
        "cpu": "500m",       # 500 millicores (0.5 CPU)
        "memory": "512Mi",   # 512 MiB
    },
)
```

### CPU

CPU is specified in millicores or whole cores:

| Value | Meaning |
|-------|---------|
| `"100m"` | 100 millicores (0.1 CPU) |
| `"500m"` | 500 millicores (0.5 CPU) |
| `"1000m"` or `"1"` | 1 full CPU core |
| `"2000m"` or `"2"` | 2 CPU cores |

### Memory

Memory uses standard Kubernetes units:

| Value | Meaning |
|-------|---------|
| `"128Mi"` | 128 mebibytes |
| `"512Mi"` | 512 mebibytes |
| `"1Gi"` | 1 gibibyte |
| `"4Gi"` | 4 gibibytes |

### GPU

Request GPU resources:

```python
sandbox = Sandbox.run(
    resources={
        "cpu": "4000m",
        "memory": "16Gi",
        "gpu": "1",          # Request 1 GPU
    },
)
```

## Mounted Files

Mount files into the sandbox at startup:

```python
sandbox = Sandbox.run(
    mounted_files=[
        {
            "path": "/app/config.json",
            "content": '{"debug": true}',
        },
        {
            "path": "/app/script.py",
            "content": "print('hello')",
        },
    ],
)

# Files are available immediately
result = sandbox.exec(["python", "/app/script.py"]).result()
```

### Mount Options

| Field | Type | Description |
|-------|------|-------------|
| `path` | str | Absolute path in sandbox |
| `content` | str | File content (text) |

Mounted files are read-only. Use `write_file()` for files that need modification.

## Ports

Expose ports from the sandbox:

```python
sandbox = Sandbox.run(
    "python", "-m", "http.server", "8080",
    ports=[
        {"container_port": 8080},
    ],
)
```

### Port Configuration

| Field | Type | Description |
|-------|------|-------------|
| `container_port` | int | Port inside the sandbox |

## Network

Configure network options:

```python
sandbox = Sandbox.run(
    network={
        "ingress_mode": "public",
    },
)
```

## Timeouts

### timeout_seconds

Per-command timeout:

```python
# Per-exec timeout
result = sandbox.exec(
    ["python", "long_script.py"],
    timeout_seconds=300,  # 5 minute timeout
).result()
```

This controls how long the client waits for a response. If exceeded, raises `SandboxTimeoutError`.

### max_lifetime_seconds

Maximum sandbox lifetime (set in SandboxDefaults):

```python
defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=3600,  # 1 hour max lifetime
)

with Sandbox.run(defaults=defaults) as sandbox:
    # Sandbox automatically terminates after 1 hour
    pass
```

This is a server-side limit. The sandbox will be terminated when it reaches this age, regardless of activity.

## Complete Example

```python
from aviato import Sandbox, SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=1800,  # 30 minutes
    tags=("production", "ml-pipeline"),
)

with Sandbox.run(
    defaults=defaults,
    resources={
        "cpu": "2000m",
        "memory": "4Gi",
    },
    mounted_files=[
        {
            "path": "/app/config.yaml",
            "content": "model: gpt-4\nmax_tokens: 1000",
        },
    ],
    ports=[
        {"container_port": 8000},
    ],
) as sandbox:
    # Install dependencies
    sandbox.exec(["pip", "install", "fastapi", "uvicorn"]).result()

    # Run application
    result = sandbox.exec(
        ["python", "-c", "print('Server started')"],
        timeout_seconds=60,
    ).result()
    print(result.stdout)
```
