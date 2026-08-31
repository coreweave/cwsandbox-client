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

The default `auto` policy uses a short direct-connect budget, then falls back to
the API gateway. You can require either path for validation or rollback:

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

Direct credentials are scoped to the requested operation and created lazily.
Active streams retain their connection, while a process-wide bounded
idle-channel cache prevents large collections of inactive sandbox objects from
retaining one socket per sandbox.

## Authentication

Authentication defaults to `AuthStrategy.COREWEAVE_API_KEY`, which reads
`CWSANDBOX_API_KEY` and sends it as a Bearer token. The strategy argument is
optional, so existing callers do not need to change.

To use W&B credentials, install the optional integration and select it
explicitly. Credential resolution is delegated to the W&B SDK and supports an
active W&B session, `WANDB_API_KEY`, and the host-specific entry in `.netrc`:

```bash
pip install "cwsandbox[wandb]"
```

```python
from cwsandbox import AuthStrategy, Sandbox

with Sandbox.run(auth=AuthStrategy.WANDB) as sb:
    ...
```

W&B authentication is sent in the `x-wandb-api-key` header; it is not treated
as a CoreWeave Bearer token.

## Development

See [DEVELOPMENT.md](https://github.com/coreweave/cwsandbox-client/blob/main/DEVELOPMENT.md) for setup and workflow.

For code standards and commit guidelines, see [CONTRIBUTING.md](https://github.com/coreweave/cwsandbox-client/blob/main/CONTRIBUTING.md).

## License
- The CWSandbox Client library is licensed under the Apache-2.0 license.
- The CWSandbox Client examples are licensed under the BSD-3-Clause license.
