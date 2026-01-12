# aviato-client

A Python client library for Aviato sandboxes.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and workflow.

For code standards and commit guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start

```python
from aviato import Sandbox

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

## Usage

See the [documentation](docs/README.md) for comprehensive guides, or browse:

- **[Execution Guide](docs/guides/execution.md)** - Running commands with `exec()`
- **[Remote Functions](docs/guides/remote-functions.md)** - Execute Python functions with `@session.function()`
- **[Sessions Guide](docs/guides/sessions.md)** - Manage multiple sandboxes concurrently

For runnable scripts, see [examples/](examples/).

## Configuration

### Environment Variables

The SDK supports two authentication strategies. Aviato credentials take priority if both are configured.

#### Aviato Authentication

- `AVIATO_API_KEY` - API key for Aviato authentication
- `AVIATO_BASE_URL` - Aviato API URL (default: `https://atc.cwaviato.com`)

#### Weights & Biases Authentication

- `WANDB_API_KEY` - W&B API key (or use `~/.netrc` with `api.wandb.ai`)
- `WANDB_ENTITY_NAME` - W&B entity/team name (required for W&B auth)
- `WANDB_PROJECT_NAME` - W&B project name (default: `uncategorized`)
