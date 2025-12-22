#!/usr/bin/env python3
"""Example: Use Session to adopt and clean up orphaned sandboxes.

This example demonstrates using Session's adopt functionality to
take ownership of sandboxes that were created elsewhere (e.g., from
a previous run that crashed) and have them cleaned up automatically
when the session closes.

It also shows how to use SandboxDefaults with tags so that all sandboxes
created within a session can be easily discovered via session.list().

Usage:
    # First, create some "orphaned" sandboxes
    uv run examples/session_adopt_orphans.py --create-orphans

    # Then adopt and clean them up via session
    uv run examples/session_adopt_orphans.py --cleanup

    # Demo using SandboxDefaults with tags for easy discovery
    uv run examples/session_adopt_orphans.py --demo-defaults
"""

import argparse
import asyncio
import uuid

from aviato import Sandbox, SandboxDefaults, Session


async def create_orphans(tag: str, count: int) -> None:
    """Create sandboxes that simulate orphans from a crashed process.

    Uses SandboxDefaults with the tag so all sandboxes are automatically tagged.
    """
    print(f"Creating {count} 'orphaned' sandboxes with tag '{tag}'...")
    print("(In real life, these would be left behind by a crashed script)\n")

    # Use SandboxDefaults so all sandboxes get the tag automatically
    defaults = SandboxDefaults(tags=(tag,))

    for i in range(count):
        sb = await Sandbox.create(
            "sleep",
            "infinity",
            defaults=defaults,
            tags=[f"orphan-{i}"],  # Additional tag, merged with defaults
        )
        print(f"  Created: {sb.sandbox_id}")

    print(f"\nCreated {count} orphans. Run with --cleanup --tag {tag} to adopt and clean them up.")


async def cleanup_with_session(tag: str) -> None:
    """Use a Session to adopt orphans and clean them up.

    The session is created with SandboxDefaults that include the tag.
    When we call session.list(), it automatically filters by the session's
    default tags - no need to pass them explicitly.
    """
    print(f"Starting session with defaults containing tag '{tag}'...\n")

    # Create session with defaults that include our tag
    # session.list() will automatically filter by these tags
    defaults = SandboxDefaults(tags=(tag,))

    async with Session(defaults) as session:
        # session.list() uses the session's default tags automatically
        # No need to pass tags=[tag] - it's implied from defaults!
        print("Calling session.list(adopt=True)...")
        print("(Automatically filters by session's default tags)\n")
        orphans = await session.list(adopt=True)

        print(f"Found and adopted {len(orphans)} sandbox(es)")
        for sb in orphans:
            print(f"  {sb.sandbox_id}: {sb.status} (adopted)")
        print(f"\nSession now tracking {session.sandbox_count} sandbox(es)\n")

        # We could also do work with these sandboxes
        if orphans:
            print("Executing command on first adopted sandbox...")
            result = await orphans[0].exec(["echo", "Hello from adopted sandbox!"])
            print(f"  Output: {result.stdout.strip()}\n")

    # Session has exited - all adopted sandboxes are now stopped
    print("Session closed - all adopted sandboxes have been stopped.")

    # Verify they're gone
    remaining = await Sandbox.list(tags=[tag], status="running")
    print(f"Remaining running sandboxes with tag '{tag}': {len(remaining)}")


async def demo_adopt_method(tag: str) -> None:
    """Demonstrate the session.adopt() method."""
    print(f"Demonstrating session.adopt() with tag '{tag}'...\n")

    # First create some sandboxes
    print("Creating sandboxes outside session...")
    sb1 = await Sandbox.create("sleep", "infinity", tags=[tag])
    sb2 = await Sandbox.create("sleep", "infinity", tags=[tag])
    print(f"  Created: {sb1.sandbox_id}")
    print(f"  Created: {sb2.sandbox_id}\n")

    async with Session() as session:
        # Get sandboxes via Sandbox.list() (class method)
        sandboxes = await Sandbox.list(tags=[tag])
        print(f"Found {len(sandboxes)} sandbox(es) via Sandbox.list()")

        # Adopt them into the session
        print("Adopting into session...")
        for sb in sandboxes:
            session.adopt(sb)
            print(f"  Adopted: {sb.sandbox_id}")

        print(f"\nSession now tracking {session.sandbox_count} sandbox(es)")
        print("Exiting session (will stop all adopted sandboxes)...\n")

    print("Session closed.")
    remaining = await Sandbox.list(tags=[tag], status="running")
    print(f"Remaining running sandboxes: {len(remaining)}")


async def demo_from_id(tag: str) -> None:
    """Demonstrate session.from_id() method."""
    print("Demonstrating session.from_id()...\n")

    # Create a sandbox outside session
    print("Creating sandbox outside session...")
    standalone = await Sandbox.create("sleep", "infinity", tags=[tag])
    sandbox_id = standalone.sandbox_id
    print(f"  Created: {sandbox_id}\n")

    async with Session() as session:
        # Attach to it via session (adopts by default)
        print(f"Attaching to {sandbox_id} via session.from_id()...")
        attached = await session.from_id(sandbox_id)
        print(f"  Status: {attached.status}")
        print(f"  Tower: {attached.tower_id}")

        # Execute something
        result = await attached.exec(["hostname"])
        print(f"  Hostname: {result.stdout.strip()}")

        print(f"\nSession tracking {session.sandbox_count} sandbox(es)")
        print("Exiting session...\n")

    print("Session closed.")
    remaining = await Sandbox.list(tags=[tag], status="running")
    print(f"Remaining running sandboxes: {len(remaining)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Session adopt example")
    parser.add_argument("--create-orphans", action="store_true", help="Create orphan sandboxes")
    parser.add_argument("--cleanup", action="store_true", help="Adopt and cleanup orphans")
    parser.add_argument("--demo-adopt", action="store_true", help="Demo session.adopt()")
    parser.add_argument("--demo-from-id", action="store_true", help="Demo session.from_id()")
    parser.add_argument("--tag", default=f"session-example-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--count", type=int, default=3, help="Number of orphans to create")
    args = parser.parse_args()

    if args.create_orphans:
        await create_orphans(args.tag, args.count)
    elif args.cleanup:
        await cleanup_with_session(args.tag)
    elif args.demo_adopt:
        await demo_adopt_method(args.tag)
    elif args.demo_from_id:
        await demo_from_id(args.tag)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
