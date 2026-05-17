# cwsandbox-client

A Python client library for CWSandboxes.

## Documentation

See the [documentation site](https://docs.coreweave.com/products/coreweave-sandbox/client) for the full tutorial, guides, and API reference.

## Quick Start

```python
from cwsandbox import Sandbox

# Quick one-liner with factory method (sync/async hybrid API)
sb = Sandbox.run("echo", "Hello, World!")
sb.stop().result()  # Block for completion

# Context manager for automatic cleanup
with Sandbox.run("sleep", "infinity", container_image="python:3.11") as sb:
    result = sb.exec(["python", "-c", "print(2 + 2)"]).result()
    print(result.stdout)  # 4

# Also works in async contexts
async with Sandbox.run("sleep", "infinity") as sb:
    result = await sb.exec(["python", "-c", "print(2 + 2)"])
    print(result.stdout)  # 4
```

## Using with Harbor

CW Sandbox can serve as a [Harbor](https://github.com/harbor-framework/harbor)
execution environment. Install both packages together via the optional
`harbor` extra:

```bash
pip install "harbor>=0.6.6" "cwsandbox[harbor]"
```

Then point Harbor at the adapter with `--environment-import-path`:

```bash
harbor run -d "<org/dataset>" -m "<model>" -a "<agent>" \
  --environment-import-path cwsandbox.harbor:CWSandboxEnvironment \
  --ek docker_image=<your-pre-built-image>
```

Or in a job/trial YAML:

```yaml
environment:
  import_path: cwsandbox.harbor:CWSandboxEnvironment
  kwargs:
    docker_image: <your-pre-built-image>
```

CW Sandbox requires pre-built container images — it cannot build Dockerfiles.
Set `docker_image` either as a constructor kwarg (`--ek docker_image=...`) or
under `[environment]` in your `task.toml`.

## Development

See [DEVELOPMENT.md](https://github.com/coreweave/cwsandbox-client/blob/main/DEVELOPMENT.md) for setup and workflow.

For code standards and commit guidelines, see [CONTRIBUTING.md](https://github.com/coreweave/cwsandbox-client/blob/main/CONTRIBUTING.md).

## License
- The CWSandbox Client library is licensed under the Apache-2.0 license.
- The CWSandbox Client examples are licensed under the BSD-3-Clause license.
