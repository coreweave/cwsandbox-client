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

## Command-line interface

Install the optional CLI to inspect and manage existing sandboxes from your terminal:

```bash
pip install "cwsandbox[cli]"
export CWSANDBOX_API_KEY="your-api-key"
export SANDBOX_ID="your-sandbox-id"
```

```bash
# Discover and inspect sandboxes
cwsandbox ls
cwsandbox get "$SANDBOX_ID"

# Run a command or open an interactive shell
cwsandbox exec "$SANDBOX_ID" python -c "print('hello')"
cwsandbox sh "$SANDBOX_ID"

# Inspect the main process logs, then stop the sandbox
cwsandbox logs "$SANDBOX_ID" --tail 100 --timestamps
cwsandbox stop "$SANDBOX_ID"
```

To authenticate with W&B instead, install `cwsandbox[cli,wandb]` and pass
`--auth wandb` (or set `CWSANDBOX_AUTH=wandb`). Credentials come from
`WANDB_API_KEY` or `wandb login`; choose the entity and project with
`WANDB_ENTITY` and `WANDB_PROJECT`:

```bash
WANDB_ENTITY=my-team cwsandbox --auth wandb ls
```

The CLI operates on sandboxes created with the Python SDK or another client. Run
`cwsandbox --help` or `cwsandbox <command> --help` for all commands and options.

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

Sandboxes are created in your default W&B entity unless you choose one. To pick
an entity (for example, a team in a different organization) or a project, pass
`WandbAuth`:

```python
from cwsandbox import Sandbox, WandbAuth

with Sandbox.run(auth=WandbAuth(entity="my-team")) as sb:
    ...
```

The entity is resolved in this order: `WandbAuth(entity=...)`, the active
`wandb.run`, then W&B settings such as `WANDB_ENTITY`. `AuthStrategy.WANDB`
is the same as `WandbAuth()`.

W&B authentication is sent in the `x-wandb-api-key` header; it is not treated
as a CoreWeave Bearer token.

## Embedding in a long-running host

Scripts and CLIs keep the default: the first owned sandbox registers `atexit`
and, on the main thread, SIGINT/SIGTERM handlers so a signal can stop Sessions
and exit.

Long-running hosts that already own process lifecycle should disable those
signal handlers before creating a sandbox. `atexit` still runs on normal
process exit. The host is responsible for graceful shutdown and for explicitly
stopping sandboxes it owns.

```python
import cwsandbox

cwsandbox.disable_signal_handlers()
```

```bash
CWSANDBOX_DISABLE_SIGNAL_HANDLERS=1
```

Accepted environment values: `1`, `true`, `yes`, `on` (case-insensitive).
Configure the API or environment variable before the first owned sandbox;
changing the variable afterward has no effect.

## Development

See [DEVELOPMENT.md](https://github.com/coreweave/cwsandbox-client/blob/main/DEVELOPMENT.md) for setup and workflow.

For code standards and commit guidelines, see [CONTRIBUTING.md](https://github.com/coreweave/cwsandbox-client/blob/main/CONTRIBUTING.md).

## License
- The CWSandbox Client library is licensed under the Apache-2.0 license.
- The CWSandbox Client examples are licensed under the BSD-3-Clause license.
