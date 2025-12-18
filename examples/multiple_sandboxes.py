"""Multiple sandboxes example using sessions.

This example demonstrates:
- Using Sandbox.session() to manage multiple sandboxes
- Creating sandboxes with session.create()
- Running parallel operations across sandboxes
"""

import asyncio

from aviato import Sandbox, SandboxDefaults


async def main() -> None:
    # Define session defaults
    defaults = SandboxDefaults(
        container_image="ubuntu:22.04",
        max_lifetime_seconds=60.0,
    )

    async with Sandbox.session(defaults) as session:
        # Create multiple sandboxes with session defaults
        sb1 = session.create(
            command="sleep",
            args=["infinity"],
            tags=["example", "multi", "sb1"],
        )
        sb2 = session.create(
            command="sleep",
            args=["infinity"],
            tags=["example", "multi", "sb2"],
        )

        # Use context managers to manage lifecycle
        async with sb1, sb2:
            print(f"Sandbox 1: {sb1.sandbox_id} on tower {sb1.tower_id}")
            print(f"Sandbox 2: {sb2.sandbox_id} on tower {sb2.tower_id}")

            # Run commands in parallel
            r1, r2 = await asyncio.gather(
                sb1.exec(["sh", "-c", "echo sandbox1 && sleep 0.2 && uname -s"]),
                sb2.exec(["sh", "-c", "echo sandbox2 && sleep 0.1 && uname -s"]),
            )

            sb1_text = r1.stdout.strip().replace("\n", " | ")
            sb2_text = r2.stdout.strip().replace("\n", " | ")

            print(f"sb1: {sb1_text}")
            print(f"sb2: {sb2_text}")

        # Sandboxes are automatically stopped when exiting context


if __name__ == "__main__":
    asyncio.run(main())
