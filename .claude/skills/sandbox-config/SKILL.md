---
name: sandbox-config
description: "Use when configuring CoreWeave Sandbox resources, container images, environment variables, mounted files, ports, or network settings. Covers resources (CPU/memory/GPU), SandboxDefaults, tags, and configuration kwargs."
disable-model-invocation: false
---

# Sandbox Configuration

Configure sandbox resources, images, and environment settings.

## Basic Configuration

```python
from cwsandbox import Sandbox

with Sandbox.run(
    container_image="python:3.11",
    max_lifetime_seconds=300,
    tags=["my-app"],
) as sandbox:
    sandbox.exec(["python", "--version"]).result()
```

## Resources

Request CPU and memory:

```python
with Sandbox.run(resources={"cpu": "1", "memory": "1Gi"}) as sandbox:
    sandbox.exec(["python", "compute.py"]).result()
```

Kubernetes resource syntax:

| Resource | Format | Examples |
|----------|--------|---------|
| CPU | Cores or millicores | `"1"`, `"2"`, `"500m"` |
| Memory | Bytes with unit suffix | `"512Mi"`, `"1Gi"`, `"2Gi"` |

## Mounted Files

Pre-populate files at sandbox startup (read-only):

```python
with Sandbox.run(
    mounted_files=[
        {"path": "/app/config.json", "content": '{"debug": true}'},
        {"path": "/app/script.py", "content": "print('hello')"},
    ]
) as sandbox:
    sandbox.exec(["python", "/app/script.py"]).result()
```

## SandboxDefaults

For reusable configuration across multiple sandboxes:

```python
from cwsandbox import SandboxDefaults

defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("my-app", "production"),
    resources={"cpu": "2", "memory": "4Gi"},
)
```

## Configuration Kwargs

| Parameter | Description |
|-----------|-------------|
| `container_image` | Docker image to use |
| `resources` | CPU/memory/GPU requests |
| `mounted_files` | Files to mount at startup |
| `s3_mount` | S3 bucket mount |
| `ports` | Port mappings |
| `network` | NetworkOptions for ingress/egress |
| `secrets` | Secret injection |
| `max_timeout_seconds` | Max operation timeout |
| `environment_variables` | Env vars to inject |
| `annotations` | Kubernetes annotations |
| `tags` | Filtering tags |

## References

- [CoreWeave Sandbox Configuration Guide](https://docs.coreweave.com/products/coreweave-sandbox/client/guides/sandbox-configuration)
