# Sandbox Configuration Guide

This guide covers sandbox configuration options: resources, mounted files, ports, and timeouts.

## Overview

Sandbox configuration can be set in three places:

1. **SandboxDefaults** - Shared defaults for all sandboxes in a session
2. **Sandbox.run() kwargs** - Per-sandbox overrides
3. **@session.function() kwargs** - Function-specific configuration

```python
from cwsandbox import Sandbox, SandboxDefaults, Session

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

Configure network options using the `NetworkOptions` dataclass or a plain dict:

```python
from cwsandbox import NetworkOptions, Sandbox

# Using NetworkOptions
sandbox = Sandbox.run(
    network=NetworkOptions(
        ingress_mode="public",
        exposed_ports=(8080,),
    ),
)

# Using dict
sandbox = Sandbox.run(
    network={"ingress_mode": "public", "exposed_ports": [8080]},
)
```

Both forms are equivalent: dicts are automatically converted to `NetworkOptions` internally.

### NetworkOptions Fields

| Field | Type | Description |
|-------|------|-------------|
| `ingress_mode` | `str \| None` | Controls inbound traffic: `"public"` (internet accessible), `"internal"` (cluster only), etc. |
| `exposed_ports` | `tuple[int, ...] \| None` | Ports to expose (required with `ingress_mode`). Pass as tuple `(8080,)` or list `[8080]`. |
| `egress_mode` | `str \| None` | Controls outbound traffic: `"internet"` (full access), `"isolated"` (no external), `"org"` (org-internal only), etc. |

All fields are optional and default to `None`, which uses backend defaults.

### Setting Network in SandboxDefaults

Set a default network configuration for all sandboxes:

```python
from cwsandbox import NetworkOptions, SandboxDefaults, Session

defaults = SandboxDefaults(
    network=NetworkOptions(egress_mode="internet"),
)

with Session(defaults) as session:
    # All sandboxes inherit the network config
    sb1 = session.sandbox()  # Uses egress_mode="internet"
    sb2 = session.sandbox()  # Uses egress_mode="internet"

    # Override for specific sandbox
    sb3 = session.sandbox(network=NetworkOptions(egress_mode="user"))
```

## Secrets

Inject secrets from secret stores as environment variables using the `Secret` type:

```python
from cwsandbox import Sandbox, Secret

with Sandbox.run(
    "pip", "install", "huggingface_hub",
    secrets=[
        Secret(store="wandb", name="HF_TOKEN"),
    ],
) as sandbox:
    sandbox.wait()
    result = sandbox.exec([
        "python", "-c",
        "from huggingface_hub import whoami; print(whoami()['name'])",
    ]).result()
    print(result.stdout.strip())  # Your Hugging Face username
```

Unlike `environment_variables`, secrets are never passed in plaintext - they are resolved
server-side from the named store and injected securely.

The `store` field refers to a secret store provider configured on the towers you are
using. Available stores depend on your tower configuration - contact your platform
administrator to find out which stores are available and what secrets they contain.
For example, towers with the W&B integration have a `"wandb"` store that provides
access to W&B team secrets.

### Secret Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `store` | `str` | required | Secret store provider configured on your tower (e.g. `"wandb"`) |
| `name` | `str` | required | Name of the secret in the store |
| `field` | `str` | `""` | Specific field within a structured secret |
| `env_var` | `str` | same as `name` | Environment variable the secret is injected as |

### Common Patterns

```python
from cwsandbox import Secret

# Minimal: env_var defaults to name
Secret(store="wandb", name="HF_TOKEN")
# -> injected as HF_TOKEN

# Custom env_var name
Secret(store="wandb", name="HF_TOKEN", env_var="HUGGINGFACE_TOKEN")
# -> injected as HUGGINGFACE_TOKEN

# Extracting a field from a structured secret
Secret(store="wandb", name="db-credentials", field="password", env_var="DB_PASS")
# -> injected as DB_PASS
```

### Setting Secrets in SandboxDefaults

Share secrets across all sandboxes in a session:

```python
from cwsandbox import SandboxDefaults, Secret, Session

defaults = SandboxDefaults(
    secrets=(
        Secret(store="wandb", name="HF_TOKEN"),
        Secret(store="wandb", name="OPENAI_API_KEY"),
    ),
)

with Session(defaults) as session:
    # All sandboxes get both secrets
    sb1 = session.sandbox()
    sb2 = session.sandbox()

    # Add more secrets for a specific sandbox
    sb3 = session.sandbox(
        secrets=[Secret(store="vault", name="DB_PASS")],
    )
    # sb3 gets all three secrets (defaults + explicit merged)
```

### Duplicate Detection

If two secrets target the same `env_var` but differ in store, name, or field, a `ValueError`
is raised at sandbox creation time. Identical duplicates (same store, name, field, env_var)
are allowed silently:

```python
# Raises ValueError: conflicting secrets for env_var 'HF_TOKEN'
Sandbox.run(
    defaults=SandboxDefaults(
        secrets=(Secret(store="wandb", name="HF_TOKEN"),),
    ),
    secrets=[Secret(store="vault", name="hf-token", env_var="HF_TOKEN")],
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
from cwsandbox import NetworkOptions, Sandbox, SandboxDefaults, Secret

defaults = SandboxDefaults(
    container_image="python:3.11",
    max_lifetime_seconds=1800,  # 30 minutes
    tags=("production", "ml-pipeline"),
    resources={"cpu": "2000m", "memory": "4Gi"},
    network=NetworkOptions(egress_mode="internet"),
    secrets=(Secret(store="wandb", name="HF_TOKEN"),),
)

with Sandbox.run(
    defaults=defaults,
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
