# aviato-client

A Python client library for Aviato sandboxes.

## Installation

```bash
git clone https://github.com/coreweave/aviato-client.git
cd aviato-client
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```python
import asyncio
from aviato import Sandbox

async def main():
    # Quick one-liner with factory method
    sandbox = await Sandbox.create("echo", "Hello, World!")
    await sandbox.stop()

    # Or with context manager for automatic cleanup
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        result = await sandbox.exec(["python", "-c", "print(2 + 2)"])
        print(result.stdout)  # 4

asyncio.run(main())
```

## Usage

See the [examples/](examples/) directory for runnable scripts, or read the guides:

- **[Basic Execution](docs/examples/basic-execution.md)** - `SandboxDefaults`, `exec()`, and file operations
- **[Function Decorator](docs/examples/function-decorator.md)** - Execute Python functions with `@session.function()`
- **[Parallel Sandboxes](docs/examples/parallel-sandboxes.md)** - Manage multiple sandboxes concurrently

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
