"""Streaming output example for real-time command execution.

This example demonstrates:
- Simple streaming with stream_output=True (output goes directly to console)
- Custom streaming callbacks for advanced use cases
"""

import asyncio

from aviato import (
    Sandbox,
    SandboxDefaults,
)


async def example_simple_streaming() -> None:
    """Stream output directly to console with stream_output=True.

    This is the simplest way to see real-time output - just add stream_output=True.
    Perfect for running training scripts where you want to watch progress.
    """
    print("=" * 60)
    print("Example 1: Simple streaming with stream_output=True")
    print("=" * 60)

    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=60.0,
    )

    async with Sandbox(
        command="sleep",
        args=["infinity"],
        defaults=defaults,
    ) as sandbox:
        print(f"Sandbox started: {sandbox.sandbox_id}\n")

        # Define a simple Python script that produces output
        script = """
import sys

for i in range(5):
    print(f"Processing step {i + 1}/5...")
    sys.stdout.flush()

print("Done!", file=sys.stderr)
sys.stderr.flush()
"""

        print("Running script with stream_output=True:\n", flush=True)

        # Simply add stream_output=True - output goes directly to console
        result = await sandbox.exec(["python", "-c", script], stream_output=True)

        print(f"\nCommand completed with exit code: {result.returncode}")


async def example_custom_callbacks() -> None:
    """Use custom callbacks for advanced output handling.

    Use on_stdout/on_stderr when you need to process output
    (e.g., logging, parsing, filtering).
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom streaming callbacks")
    print("=" * 60)

    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        print(f"Sandbox started: {sandbox.sandbox_id}\n")

        script = """
import sys
print("INFO: Starting process")
print("WARNING: Low memory", file=sys.stderr)
print("INFO: Process complete")
"""

        print("Running with custom callbacks:\n", flush=True)

        # Custom callbacks for processing output
        result = await sandbox.exec(
            ["python", "-c", script],
            on_stdout=lambda data: print(f"[LOG] {data.decode()}", end=""),
            on_stderr=lambda data: print(f"[ERR] {data.decode()}", end=""),
        )

        print(f"\nExit code: {result.returncode}")


async def example_progress_tracking() -> None:
    """Track progress of a long-running command.

    This shows a practical use case: simulated training with epoch progress.
    For simple "just show me the output" cases, use stream_output=True.
    """
    print("\n" + "=" * 60)
    print("Example 3: Progress tracking (simulated training)")
    print("=" * 60)

    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        print(f"Sandbox started: {sandbox.sandbox_id}\n")

        # Simulated training script - outputs epoch progress with ~1s per epoch
        training_script = """
import random
import time
import sys

for epoch in range(1, 6):
    time.sleep(1)  # Simulate training time
    loss = 2.5 / epoch + random.uniform(-0.1, 0.1)
    acc = min(0.95, 0.5 + epoch * 0.09 + random.uniform(-0.02, 0.02))
    print(f"Epoch {epoch}/5 | loss: {loss:.4f} | acc: {acc:.2%}", flush=True)
"""

        print("Running training with live output:\n", flush=True)

        result = await sandbox.exec(
            ["python", "-c", training_script],
            stream_output=True,
        )

        print(f"\nTraining finished with exit code: {result.returncode}")


async def main() -> None:
    """Run all streaming examples."""
    await example_simple_streaming()
    await example_custom_callbacks()
    await example_progress_tracking()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
