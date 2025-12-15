# Basic Execution

This example demonstrates basic sandbox operations: creating a sandbox, executing commands, and file operations.

## Quick Start with Factory Method

The simplest way to create a sandbox:

```python
import asyncio
from aviato import Sandbox

async def main():
    # Factory method parses positional args as command + args
    sandbox = await Sandbox.create("echo", "hello", "world")
    print(f"Sandbox ID: {sandbox.sandbox_id}")
    await sandbox.stop()

asyncio.run(main())
```

## Constructor with Context Manager

For more control, use the constructor with a context manager:

```python
import asyncio
from aviato import Sandbox, SandboxDefaults

async def main():
    # Define reusable defaults
    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=60.0,
        tags=("my-app", "production"),
    )

    async with Sandbox(
        command="sleep",
        args=["infinity"],
        defaults=defaults,
    ) as sandbox:
        # Execute commands
        result = await sandbox.exec(["echo", "Hello from sandbox"])
        print(result.stdout.decode())

        # Check Python version
        result = await sandbox.exec(["python", "--version"])
        print(result.stdout.decode())

asyncio.run(main())
```

## File Operations

Read and write files in the sandbox:

```python
import asyncio
from aviato import Sandbox

async def main():
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        # Write a file
        content = b"Hello, World!\n"
        success = await sandbox.write_file("/tmp/data.txt", content)
        print(f"Write success: {success}")

        # Read the file back
        read_back = await sandbox.read_file("/tmp/data.txt")
        print(f"Content: {read_back.decode()}")

        # Verify with cat
        result = await sandbox.exec(["cat", "/tmp/data.txt"])
        print(f"cat output: {result.stdout.decode()}")

asyncio.run(main())
```

## Sandbox Properties

Access sandbox metadata:

```python
async with Sandbox(
    command="sleep",
    args=["infinity"],
    container_image="python:3.11",
) as sandbox:
    print(f"Sandbox ID: {sandbox.sandbox_id}")
    print(f"Tower ID: {sandbox.tower_id}")
    print(f"Runway ID: {sandbox.runway_id}")
```
