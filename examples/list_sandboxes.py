#!/usr/bin/env python3
"""Example: List sandboxes with various filters.

This example demonstrates how to query existing sandboxes using the
Sandbox.list() class method with different filter combinations.

Sandbox.list() returns actual Sandbox instances that you can use for
operations like exec(), stop(), get_status(), etc.

Usage:
    uv run examples/list_sandboxes.py
"""

import asyncio
from datetime import datetime, timedelta

from aviato import Sandbox


async def main() -> None:
    # List all sandboxes (no filters)
    # Returns Sandbox instances, not just metadata
    print("All sandboxes:")
    print("-" * 60)
    all_sandboxes = await Sandbox.list()
    for sb in all_sandboxes:
        print(f"  {sb.sandbox_id}: {sb.status} (started {sb.started_at})")
    print(f"Total: {len(all_sandboxes)}\n")

    # List only running sandboxes
    print("Running sandboxes:")
    print("-" * 60)
    running = await Sandbox.list(status="running")
    for sb in running:
        print(f"  {sb.sandbox_id}: tower={sb.tower_id}")
    print(f"Total: {len(running)}\n")

    # List sandboxes with specific tags
    print("Sandboxes tagged 'my-batch-job':")
    print("-" * 60)
    tagged = await Sandbox.list(tags=["my-batch-job"])
    for sb in tagged:
        print(f"  {sb.sandbox_id}: {sb.status}")
    print(f"Total: {len(tagged)}\n")

    # Combine multiple filters
    print("Running sandboxes on specific tower:")
    print("-" * 60)
    filtered = await Sandbox.list(
        status="running",
        tower_ids=["tower-us-east-1"],
    )
    for sb in filtered:
        print(f"  {sb.sandbox_id}")
    print(f"Total: {len(filtered)}\n")

    # Find old sandboxes (client-side filtering on started_at)
    print("Sandboxes older than 1 hour:")
    print("-" * 60)
    cutoff = datetime.now() - timedelta(hours=1)
    old_sandboxes = [sb for sb in all_sandboxes if sb.started_at and sb.started_at < cutoff]
    for sb in old_sandboxes:
        age = datetime.now() - sb.started_at
        print(f"  {sb.sandbox_id}: {sb.status} (age: {age})")
    print(f"Total: {len(old_sandboxes)}\n")

    # Demonstrate that listed sandboxes are operable
    if running:
        print("Executing command on first running sandbox:")
        print("-" * 60)
        sb = running[0]
        result = await sb.exec(["echo", "Hello from discovered sandbox!"])
        print(f"  Output: {result.stdout.strip()}")
        print(f"  Exit code: {result.returncode}\n")


if __name__ == "__main__":
    asyncio.run(main())
