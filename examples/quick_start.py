"""Quick start example - job-style sandbox that runs to completion.

This example demonstrates:
- Using Sandbox.create() for one-shot job execution
- Running a command that finishes naturally (not sleep infinity)
- Waiting for the job to complete and checking exit status
"""

import asyncio

from aviato import Sandbox


async def main() -> None:
    # Create a sandbox that runs a job to completion
    # The command does some work and then exits
    sandbox = await Sandbox.create(
        "python",
        "-c",
        "print('Hello from sandbox!')",
        container_image="python:3.11",
    )

    print(f"Sandbox ID: {sandbox.sandbox_id}")
    print(f"Tower: {sandbox.tower_id}")

    # Wait for the job to complete
    await sandbox.wait()
    print(f"Job completed with exit code: {sandbox.returncode}")

    # Clean up the sandbox resources
    await sandbox.stop()


if __name__ == "__main__":
    asyncio.run(main())
