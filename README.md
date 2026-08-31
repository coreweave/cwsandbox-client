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

## Sandbox data connections

Exec, log, and file operations prefer a sandbox-scoped direct mTLS connection.
Lifecycle and management operations continue to use the CWSandbox API. The SDK
generates the private key in memory, sends only a certificate signing request,
and never sends API bearer credentials to the sandbox data endpoint.

The default `auto` policy falls back to the API gateway when direct access is
not available. You can require either path for validation or rollback:

```python
from cwsandbox import DataPlaneMode, Sandbox, SandboxDefaults

# Fail instead of falling back, useful when validating direct connectivity.
with Sandbox.run(data_plane_mode=DataPlaneMode.DIRECT) as sb:
    print(sb.exec(["echo", "direct"]).result().stdout)

# Disable direct access for a group of sandboxes.
defaults = SandboxDefaults(data_plane_mode=DataPlaneMode.GATEWAY)
with Sandbox.run(defaults=defaults) as sb:
    print(sb.exec(["echo", "gateway"]).result().stdout)
```

Direct credentials and channels are created lazily. Active streams retain their
connection, while a process-wide bounded idle-channel cache prevents large
collections of inactive sandbox objects from retaining one socket per sandbox.

## Development

See [DEVELOPMENT.md](https://github.com/coreweave/cwsandbox-client/blob/main/DEVELOPMENT.md) for setup and workflow.

For code standards and commit guidelines, see [CONTRIBUTING.md](https://github.com/coreweave/cwsandbox-client/blob/main/CONTRIBUTING.md).

## License
- The CWSandbox Client library is licensed under the Apache-2.0 license.
- The CWSandbox Client examples are licensed under the BSD-3-Clause license.
