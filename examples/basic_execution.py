"""Basic sandbox execution example using the new API.

This example demonstrates:
- Creating a sandbox using the context manager pattern
- Executing commands in the sandbox
- Reading and writing files
"""

import asyncio
import os

from aviato import Sandbox, SandboxDefaults


async def main() -> None:
    if not os.environ.get("AVIATO_API_KEY"):
        raise RuntimeError(
            "Missing AVIATO_API_KEY. Set it in your environment before running this example."
        )

    # Define reusable defaults
    defaults = SandboxDefaults(
        container_image="ubuntu:22.04",
        max_lifetime_seconds=60.0,
        tags=("example", "basic-execution"),
    )

    # Use the constructor with context manager for full control
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        defaults=defaults,
    ) as sandbox:
        print(f"Sandbox started: {sandbox.sandbox_id}")
        print(f"Running on tower: {sandbox.tower_id}")

        # Check sandbox status
        status = await sandbox.get_status()
        print(f"Sandbox status: {status}")

        # Execute a simple command
        result = await sandbox.exec(["echo", "Hello from Aviato sandbox"])
        print(result.stdout.rstrip())

        # Write a file
        content = b"Hello, World!\n"
        await sandbox.write_file("/tmp/data.txt", content)
        print("write_file: '/tmp/data.txt'")

        # Read the file back
        read_back = await sandbox.read_file("/tmp/data.txt")
        decoded = read_back.decode("utf-8", errors="replace").rstrip()
        print(f"read_file bytes={len(read_back)} content={decoded}")

        # Verify with cat
        result = await sandbox.exec(["cat", "/tmp/data.txt"])
        print(f"cat /tmp/data.txt -> {result.stdout.rstrip()}")


if __name__ == "__main__":
    asyncio.run(main())
