#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: aviato-client

"""Example: Use Session to adopt and clean up orphaned sandboxes.

This example demonstrates using Session's adopt functionality to
take ownership of sandboxes that were created elsewhere (e.g., from
a previous run that crashed) and have them cleaned up automatically
when the session closes.

Usage:
    # First, create some "orphaned" sandboxes
    uv run examples/session_adopt_orphans.py --create-orphans

    # Then adopt and clean them up via session
    uv run examples/session_adopt_orphans.py --cleanup
"""

import argparse

from aviato import Sandbox, Session


def create_orphans(tag: str, count: int) -> None:
    """Create sandboxes that simulate orphans from a crashed process."""
    print(f"Creating {count} 'orphaned' sandboxes with tag '{tag}'...")
    print("(In real life, these would be left behind by a crashed script)\n")

    for i in range(count):
        sb = Sandbox.run(tags=[tag, f"orphan-{i}"])
        print(f"  Created: {sb.sandbox_id}")

    print(f"\nCreated {count} orphans. Run with --cleanup --tag {tag} to adopt and clean them up.")


def cleanup_with_session(tag: str) -> None:
    """Use a Session to adopt orphans and clean them up."""
    print(f"Finding sandboxes with tag '{tag}'...\n")

    with Session() as session:
        # Find sandboxes by tag and adopt them into the session
        orphans = session.list(tags=[tag], adopt=True).result()

        print(f"Found and adopted {len(orphans)} sandbox(es)")
        for sb in orphans:
            print(f"  {sb.sandbox_id}: {sb.status}")

        if not orphans:
            print("\nNothing to clean up.")
            return

        print(f"\nSession tracking {session.sandbox_count} sandbox(es)")

    # Session has exited - all adopted sandboxes are now stopped
    print("Session closed - all adopted sandboxes have been stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Session adopt example")
    parser.add_argument("--create-orphans", action="store_true", help="Create orphan sandboxes")
    parser.add_argument("--cleanup", action="store_true", help="Adopt and cleanup orphans")
    parser.add_argument("--tag", default="adopt-orphans-example")
    parser.add_argument("--count", type=int, default=3, help="Number of orphans to create")
    args = parser.parse_args()

    if args.create_orphans:
        create_orphans(args.tag, args.count)
    elif args.cleanup:
        cleanup_with_session(args.tag)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
