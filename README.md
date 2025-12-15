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
        print(result.stdout.decode())  # 4

asyncio.run(main())
```

## Usage

See the [examples/](examples/) directory for runnable scripts, or read the guides:

- **[Basic Execution](docs/examples/basic-execution.md)** - `SandboxDefaults`, `exec()`, and file operations
- **[Function Decorator](docs/examples/function-decorator.md)** - Execute Python functions with `@session.function()`
- **[Parallel Sandboxes](docs/examples/parallel-sandboxes.md)** - Manage multiple sandboxes concurrently

## Configuration

### Environment Variables

- `AVIATO_API_KEY` - API key for authentication
- `AVIATO_BASE_URL` - Aviato API URL (default: `https://atc.cwaviato.com`)
