#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Example: Delete sandboxes by ID.

This example demonstrates multiple ways to delete sandboxes:
1. Sandbox.delete(sandbox_id) - class method for quick deletion by ID
2. sandbox.stop() - instance method on a Sandbox object

Usage:
    uv run examples/delete_sandboxes.py
"""

from cwsandbox import Sandbox
from cwsandbox.exceptions import SandboxNotFoundError


def main() -> None:
    # Create a sandbox to demonstrate deletion
    print("Creating a test sandbox...")
    sandbox = Sandbox.run(
        "sleep",
        "infinity",
        tags=["delete-example"],
    )
    sandbox_id = sandbox.sandbox_id
    print(f"Created sandbox: {sandbox_id}\n")

    # Method 1: Delete by ID using class method
    # Useful when you only have the ID, not a Sandbox instance
    print(f"Deleting sandbox {sandbox_id} using Sandbox.delete()...")
    Sandbox.delete(sandbox_id).result()
    print("Deletion completed\n")

    # Try to delete again - this will raise SandboxNotFoundError
    print("Attempting to delete the same sandbox again...")
    try:
        Sandbox.delete(sandbox_id).result()
    except SandboxNotFoundError as e:
        print(f"Expected error: {e}\n")

    # Use missing_ok=True to suppress the error
    print("Deleting with missing_ok=True...")
    Sandbox.delete(sandbox_id, missing_ok=True).result()
    print("Deletion completed (no error even though already deleted)\n")

    # Method 2: Delete using stop() on a discovered sandbox
    print("Creating another sandbox to demonstrate stop() on discovered sandbox...")
    sandbox2 = Sandbox.run(tags=["delete-example-2"])
    print(f"Created sandbox: {sandbox2.sandbox_id}")

    # Discover it via list() and stop it
    sandboxes = Sandbox.list(tags=["delete-example-2"]).result()
    if sandboxes:
        discovered = sandboxes[0]
        print(f"Found sandbox via list(): {discovered.sandbox_id}")
        discovered.stop().result()
        print("Stopped via discovered.stop()\n")

    # Method 3: Attach to sandbox by ID and stop
    print("Creating another sandbox to demonstrate from_id()...")
    sandbox3 = Sandbox.run(tags=["delete-example-3"])
    sandbox3_id = sandbox3.sandbox_id
    print(f"Created sandbox: {sandbox3_id}")

    # Attach to it by ID and stop
    attached = Sandbox.from_id(sandbox3_id).result()
    print(f"Attached to sandbox: {attached.sandbox_id}, status: {attached.status}")
    attached.stop().result()
    print("Stopped via attached.stop()\n")

    print("All examples completed!")


if __name__ == "__main__":
    main()
