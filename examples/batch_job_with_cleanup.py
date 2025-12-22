#!/usr/bin/env python3
"""Example: Batch job with robust cleanup.

This example shows a pattern for running batch jobs where cleanup
happens even if the job is interrupted. It uses unique tags to
identify sandboxes from this run.

Usage:
    uv run examples/batch_job_with_cleanup.py
"""

import asyncio
import uuid

from aviato import Sandbox


async def run_batch_job(run_id: str, task_count: int) -> None:
    """Run a batch job with multiple sandboxes."""
    tag = f"batch-{run_id}"
    print(f"Starting batch job: {run_id}")
    print(f"Tag: {tag}\n")

    results = []

    try:
        # Create and run tasks
        for i in range(task_count):
            print(f"Running task {i + 1}/{task_count}...")

            async with Sandbox(
                command="python",
                args=["-c", f"print('Task {i} result: {i * 10}')"],
                tags=[tag, f"task-{i}"],
            ) as sandbox:
                await sandbox.exec(["cat", "/proc/1/cmdline"])
                results.append(f"Task {i}: completed")
                print(f"  Task {i} completed")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Cleaning up...")
        raise

    finally:
        # Cleanup any orphaned sandboxes from this run
        # This catches sandboxes that might have been left behind
        # if an error occurred between creation and context exit
        print(f"\nChecking for orphaned sandboxes with tag '{tag}'...")

        # list() returns Sandbox instances we can operate on directly
        orphans = await Sandbox.list(tags=[tag])

        if orphans:
            print(f"Found {len(orphans)} orphaned sandbox(es), cleaning up...")
            for sb in orphans:
                try:
                    # sb is a Sandbox instance, so we can call stop() directly
                    await sb.stop()
                    print(f"  Cleaned up: {sb.sandbox_id}")
                except Exception as e:
                    print(f"  Failed to clean up {sb.sandbox_id}: {e}")
        else:
            print("No orphaned sandboxes found.")

    print(f"\nBatch job complete. Results: {len(results)}/{task_count} tasks succeeded")


async def main() -> None:
    run_id = str(uuid.uuid4())[:8]
    await run_batch_job(run_id, task_count=3)


if __name__ == "__main__":
    asyncio.run(main())
