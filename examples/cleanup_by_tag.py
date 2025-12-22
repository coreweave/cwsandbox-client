#!/usr/bin/env python3
"""Example: Clean up sandboxes by tag.

This is a common pattern for cleaning up orphaned sandboxes from a
batch job or test run. Tag your sandboxes with a unique identifier,
then use list() to find them and stop() to clean them up.

Since Sandbox.list() returns actual Sandbox instances, you can call
stop() directly on them without needing to use Sandbox.delete().

Usage:
    # First, run some sandboxes with a tag
    uv run examples/cleanup_by_tag.py --create

    # Later, clean them up
    uv run examples/cleanup_by_tag.py --cleanup
"""

import argparse
import asyncio

from aviato import Sandbox


async def create_tagged_sandboxes(tag: str, count: int) -> None:
    """Create some sandboxes with a specific tag."""
    print(f"Creating {count} sandboxes with tag '{tag}'...")

    create_tasks = [
        Sandbox.create(
            "sleep",
            "infinity",
            tags=[tag, f"instance-{i}"],
        )
        for i in range(count)
    ]
    sandboxes = await asyncio.gather(*create_tasks)

    for sb in sandboxes:
        print(f"  Created: {sb.sandbox_id} (tower: {sb.tower_id})")

    print(f"\nCreated {len(sandboxes)} sandboxes")
    print("Run with --cleanup to delete them.")


async def cleanup_tagged_sandboxes(tag: str) -> None:
    """Find and delete all sandboxes with a specific tag."""
    print(f"Finding sandboxes with tag '{tag}'...")

    # list() returns Sandbox instances we can operate on directly
    sandboxes = await Sandbox.list(tags=[tag])
    print(f"Found {len(sandboxes)} sandbox(es)")

    if not sandboxes:
        print("\nNothing to clean up.")
        return

    # Show which towers they're on
    for sb in sandboxes:
        print(f"  {sb.sandbox_id} (tower: {sb.tower_id}, status: {sb.status})")

    print("\nStopping sandboxes...")

    # Stop all sandboxes concurrently
    stop_tasks = [sb.stop() for sb in sandboxes]
    results = await asyncio.gather(*stop_tasks, return_exceptions=True)

    stopped = 0
    failed = 0
    for sb, result in zip(sandboxes, results, strict=False):
        if isinstance(result, Exception):
            print(f"  Failed to stop {sb.sandbox_id}: {result}")
            failed += 1
        else:
            print(f"  Stopped: {sb.sandbox_id}")
            stopped += 1

    print(f"\nResults: {stopped} stopped, {failed} failed")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Tag-based sandbox cleanup example")
    parser.add_argument("--create", action="store_true", help="Create test sandboxes")
    parser.add_argument("--cleanup", action="store_true", help="Clean up sandboxes")
    parser.add_argument("--tag", default="cleanup-example", help="Tag to use")
    parser.add_argument("--count", type=int, default=3, help="Number to create")
    args = parser.parse_args()

    if args.create:
        await create_tagged_sandboxes(args.tag, args.count)
    elif args.cleanup:
        await cleanup_tagged_sandboxes(args.tag)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
