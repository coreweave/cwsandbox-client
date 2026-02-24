#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: aviato-client

"""Example: List sandboxes including stopped ones.

By default, Sandbox.list() only returns active sandboxes. Use
``include_stopped=True`` to also retrieve terminal sandboxes
(completed, failed, terminated) from persistent storage.

This is useful for:
- Auditing past sandbox runs
- Cleaning up or deleting old sandboxes
- Verifying that sandboxes completed successfully

Usage:
    # First, create some sandboxes that will complete immediately
    uv run examples/list_stopped_sandboxes.py --create

    # Then list them (active only by default)
    uv run examples/list_stopped_sandboxes.py --list

    # Include stopped sandboxes to see the completed ones
    uv run examples/list_stopped_sandboxes.py --list --include-stopped
"""

import argparse

from aviato import Sandbox, SandboxDefaults, Session

TAG = "list-stopped-example"


def create_sandboxes(count: int) -> None:
    """Create sandboxes that complete immediately."""
    print(f"Creating {count} sandboxes with tag '{TAG}'...")

    sandboxes = [
        Sandbox.run("echo", f"hello-{i}", tags=[TAG, f"instance-{i}"]) for i in range(count)
    ]

    for sb in sandboxes:
        print(f"  Created: {sb.sandbox_id}")

    print("\nWaiting for sandboxes to complete...")
    for sb in sandboxes:
        sb.wait_until_complete(timeout=60.0).result()
        print(f"  Completed: {sb.sandbox_id}")

    print(f"\nAll {count} sandboxes completed.")
    print("Run with --list --include-stopped to see them.")


def list_sandboxes(include_stopped: bool) -> None:
    """List sandboxes using both Sandbox and Session APIs."""
    label = "active + stopped" if include_stopped else "active only"
    print(f"Listing sandboxes ({label})\n")

    # Sandbox.list() - direct class method
    print("=== Sandbox.list() ===")
    sandboxes = Sandbox.list(tags=[TAG], include_stopped=include_stopped).result()

    if not sandboxes:
        print("  No sandboxes found.")
    else:
        for sb in sandboxes:
            print(f"  {sb.sandbox_id}  status={sb.status}  tower={sb.tower_id}")
        print(f"\n  Total: {len(sandboxes)} sandbox(es)")

    # Session.list() - auto-filters by session tags
    defaults = SandboxDefaults(tags=(TAG,))

    print("\n=== Session.list() ===")
    with Session(defaults) as session:
        sandboxes = session.list(include_stopped=include_stopped).result()

        if not sandboxes:
            print("  No sandboxes found for session tags.")
        else:
            for sb in sandboxes:
                print(f"  {sb.sandbox_id}  status={sb.status}  tower={sb.tower_id}")
            print(f"\n  Total: {len(sandboxes)} sandbox(es)")


def main() -> None:
    parser = argparse.ArgumentParser(description="List sandboxes including stopped ones")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create sandboxes that complete immediately",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_sandboxes",
        help="List sandboxes",
    )
    parser.add_argument(
        "--include-stopped",
        action="store_true",
        help="Include terminal sandboxes (completed, failed, terminated)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of sandboxes to create (default: 3)",
    )
    args = parser.parse_args()

    if args.create:
        create_sandboxes(args.count)
    elif args.list_sandboxes:
        list_sandboxes(args.include_stopped)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
