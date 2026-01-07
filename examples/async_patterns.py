#!/usr/bin/env python3
"""Async patterns with aviato's sync/async hybrid API.

All aviato operations support both sync and async usage:
- Sync: call .get() or .result() to block for the result
- Async: use await on OperationRef or Process objects

This example shows how to use await in async contexts. The same operations
work identically with .get()/.result() in sync code.

Usage:
    uv run examples/async_patterns.py
"""

import asyncio

from aviato import Sandbox, SandboxDefaults, Session


async def main() -> None:
    defaults = SandboxDefaults(tags=("async-patterns-example",))

    # --- Awaiting OperationRef from discovery methods ---
    # Sandbox.list(), Sandbox.from_id(), Sandbox.delete() all return OperationRef
    # In sync code: sandboxes = Sandbox.list(...).get()
    # In async code: sandboxes = await Sandbox.list(...)
    print("Awaiting OperationRef from discovery methods")
    print("-" * 50)

    # Create a sandbox to work with
    sandbox = Sandbox.run(defaults=defaults)
    sandbox_id = sandbox.sandbox_id
    print(f"Created: {sandbox_id}")

    # await Sandbox.list() - returns list[Sandbox]
    sandboxes = await Sandbox.list(tags=["async-patterns-example"])
    print(f"Found {len(sandboxes)} sandbox(es)")

    # await Sandbox.from_id() - returns Sandbox
    attached = await Sandbox.from_id(sandbox_id)
    print(f"Attached to: {attached.sandbox_id}, status: {attached.status}")
    print()

    # --- Awaiting Process from exec() ---
    # exec() returns Process, which is awaitable
    # In sync code: result = sandbox.exec([...]).result()
    # In async code: result = await sandbox.exec([...])
    print("Awaiting Process from exec()")
    print("-" * 50)

    process = sandbox.exec(["echo", "hello from async"])
    result = await process  # Awaits until process completes
    print(f"Output: {result.stdout.strip()}")
    print()

    # --- Awaiting OperationRef from file operations ---
    # read_file() and write_file() return OperationRef
    print("Awaiting OperationRef from file operations")
    print("-" * 50)

    await sandbox.write_file("/tmp/test.txt", b"async content")
    content = await sandbox.read_file("/tmp/test.txt")
    print(f"File content: {content.decode()}")
    print()

    # --- Parallel operations with asyncio.gather() ---
    # Useful for concurrent execution of multiple operations
    print("Parallel operations with asyncio.gather()")
    print("-" * 50)

    # Run multiple commands in parallel
    results = await asyncio.gather(
        sandbox.exec(["echo", "one"]),
        sandbox.exec(["echo", "two"]),
        sandbox.exec(["echo", "three"]),
    )
    for r in results:
        print(f"  {r.stdout.strip()}")
    print()

    # --- Session with async context manager ---
    # Sessions support both sync and async context managers
    print("Session with async context manager")
    print("-" * 50)

    async with Session(defaults) as session:
        sb = session.sandbox()
        result = await sb.exec(["echo", "from session"])
        print(f"Output: {result.stdout.strip()}")

        # session.list() and session.from_id() also return OperationRef
        found = await session.list()
        print(f"Session has {len(found)} sandbox(es)")
    print("Session closed\n")

    # --- Cleanup ---
    print("Cleanup")
    print("-" * 50)

    # await Sandbox.delete() - returns None on success
    await Sandbox.delete(sandbox_id)
    print("Deleted sandbox")

    # missing_ok=True suppresses SandboxNotFoundError
    await Sandbox.delete(sandbox_id, missing_ok=True)
    print("Delete again (missing_ok=True): completed without error")
    print()

    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
