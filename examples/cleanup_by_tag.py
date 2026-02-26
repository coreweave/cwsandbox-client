#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

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

from cwsandbox import Sandbox, SandboxError


def create_tagged_sandboxes(tag: str, count: int) -> None:
    """Create some sandboxes with a specific tag."""
    print(f"Creating {count} sandboxes with tag '{tag}'...")

    # Sandbox.run() returns immediately; sandboxes start in parallel
    sandboxes = [
        Sandbox.run(
            "sleep",
            "infinity",
            tags=[tag, f"instance-{i}"],
        )
        for i in range(count)
    ]

    for sb in sandboxes:
        print(f"  Created: {sb.sandbox_id} (tower: {sb.tower_id})")

    print(f"\nCreated {len(sandboxes)} sandboxes")
    print("Run with --cleanup to delete them.")


def cleanup_tagged_sandboxes(tag: str) -> None:
    """Find and delete all sandboxes with a specific tag."""
    print(f"Finding sandboxes with tag '{tag}'...")

    # list() returns OperationRef; use .result() to block for results
    sandboxes = Sandbox.list(tags=[tag]).result()
    print(f"Found {len(sandboxes)} sandbox(es)")

    if not sandboxes:
        print("\nNothing to clean up.")
        return

    # Show which towers they're on
    for sb in sandboxes:
        print(f"  {sb.sandbox_id} (tower: {sb.tower_id}, status: {sb.status})")

    print("\nStopping sandboxes...")

    # Stop all sandboxes concurrently
    stop_refs = [sb.stop() for sb in sandboxes]

    # Wait for each stop operation, handling failures individually
    stopped = 0
    failed = 0
    for sb, ref in zip(sandboxes, stop_refs, strict=False):
        try:
            ref.result()
            print(f"  Stopped: {sb.sandbox_id}")
            stopped += 1
        except SandboxError as e:
            print(f"  Failed to stop {sb.sandbox_id}: {e}")
            failed += 1

    print(f"\nResults: {stopped} stopped, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag-based sandbox cleanup example")
    parser.add_argument("--create", action="store_true", help="Create test sandboxes")
    parser.add_argument("--cleanup", action="store_true", help="Clean up sandboxes")
    parser.add_argument("--tag", default="cleanup-example", help="Tag to use")
    parser.add_argument("--count", type=int, default=3, help="Number to create")
    args = parser.parse_args()

    if args.create:
        create_tagged_sandboxes(args.tag, args.count)
    elif args.cleanup:
        cleanup_tagged_sandboxes(args.tag)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
