#!/usr/bin/env python3
"""Example: Reconnect to an existing sandbox by ID.

This example demonstrates using Sandbox.from_id() to attach to a
sandbox that was created elsewhere (e.g., in a previous script run,
or by another process).

This is useful for:
- Long-running sandboxes that outlive the creating process
- Debugging sandboxes interactively
- Handing off sandboxes between processes

Usage:
    # First, create a sandbox and note its ID
    uv run examples/reconnect_to_sandbox.py --create

    # Then reconnect to it
    uv run examples/reconnect_to_sandbox.py --sandbox-id <id>
"""

import argparse

from aviato import Sandbox
from aviato.exceptions import SandboxNotFoundError


def create_long_running_sandbox() -> str:
    """Create a sandbox that will keep running."""
    print("Creating a long-running sandbox...")

    sandbox = Sandbox.run(
        "sleep",
        "infinity",
        tags=["reconnect-example"],
    )

    print(f"Created sandbox: {sandbox.sandbox_id}")
    print(f"Tower: {sandbox.tower_id}")
    print("Status: running")
    print()
    print("This sandbox will keep running until you stop it.")
    print("To reconnect later, run:")
    print(f"  uv run examples/reconnect_to_sandbox.py --sandbox-id {sandbox.sandbox_id}")
    print()
    print("To stop it:")
    print(f"  uv run examples/reconnect_to_sandbox.py --sandbox-id {sandbox.sandbox_id} --stop")

    return sandbox.sandbox_id


def reconnect_to_sandbox(sandbox_id: str, stop: bool = False) -> None:
    """Reconnect to an existing sandbox and optionally stop it."""
    print(f"Reconnecting to sandbox: {sandbox_id}")

    try:
        # from_id() returns a Sandbox instance attached to the existing sandbox
        sandbox = Sandbox.from_id(sandbox_id).get()
    except SandboxNotFoundError:
        print(f"Error: Sandbox {sandbox_id} not found")
        print("It may have been stopped or never existed.")
        return

    print("Connected!")
    print(f"  Status: {sandbox.status}")
    print(f"  Tower: {sandbox.tower_id}")
    print(f"  Started at: {sandbox.started_at}")
    print()

    # Execute a command to prove we're connected
    print("Executing 'hostname' in the sandbox...")
    result = sandbox.exec(["hostname"]).result()
    print(f"  Hostname: {result.stdout.strip()}")
    print()

    # Execute another command
    print("Executing 'uptime' in the sandbox...")
    result = sandbox.exec(["uptime"]).result()
    print(f"  Uptime: {result.stdout.strip()}")
    print()

    if stop:
        print("Stopping sandbox...")
        sandbox.stop().get()
        print("Sandbox stopped.")
    else:
        print("Sandbox is still running. Use --stop to stop it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconnect to an existing sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new sandbox instead of reconnecting",
    )
    parser.add_argument(
        "--sandbox-id",
        help="ID of sandbox to reconnect to",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the sandbox after reconnecting",
    )
    args = parser.parse_args()

    if args.create:
        create_long_running_sandbox()
    elif args.sandbox_id:
        reconnect_to_sandbox(args.sandbox_id, stop=args.stop)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
