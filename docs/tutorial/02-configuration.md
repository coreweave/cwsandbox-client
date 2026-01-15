# 2. Configuring Sandboxes

Pass configuration options directly to `Sandbox.run()`:

```python
from aviato import Sandbox

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

Uses [Kubernetes resource syntax](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/):

| Resource | Format | Examples |
|----------|--------|----------|
| CPU | Cores or millicores | `"1"`, `"2"`, `"500m"` (0.5 CPU) |
| Memory | Bytes with unit suffix | `"512Mi"`, `"1Gi"`, `"2Gi"` |

## Mounted Files

Pre-populate files at sandbox startup:

```python
with Sandbox.run(
    mounted_files=[
        {"path": "/app/config.json", "content": '{"debug": true}'},
        {"path": "/app/script.py", "content": "print('hello')"},
    ]
) as sandbox:
    sandbox.exec(["python", "/app/script.py"]).result()
```

Mounted files are read-only. Use `write_file()` for files that need modification.

---

For reusable configuration across multiple sandboxes, use `SandboxDefaults`. See the [Sandbox Configuration Guide](../guides/sandbox-configuration.md) for all available options including GPU resources, ports, and services.

