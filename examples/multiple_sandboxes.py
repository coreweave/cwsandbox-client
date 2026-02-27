# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Multiple sandboxes example using sessions.

This example demonstrates:
- Using Sandbox.session() to manage multiple sandboxes
- Creating sandboxes with session.sandbox()
- Running parallel operations across sandboxes
- Automatic cleanup when session exits
"""

from cwsandbox import Sandbox, SandboxDefaults


def main() -> None:
    defaults = SandboxDefaults(
        container_image="ubuntu:22.04",
        max_lifetime_seconds=60.0,
    )

    with Sandbox.session(defaults) as session:
        # Create multiple sandboxes (unstarted until first operation)
        sb1 = session.sandbox(tags=["example", "multi", "sb1"])
        sb2 = session.sandbox(tags=["example", "multi", "sb2"])

        # Run commands in parallel using fire-then-collect pattern
        # Fire: start both executions (auto-starts sandboxes on first exec)
        p1 = sb1.exec(["sh", "-c", "echo sandbox1 && sleep 0.2 && uname -s"])
        p2 = sb2.exec(["sh", "-c", "echo sandbox2 && sleep 0.1 && uname -s"])

        # Collect: wait for both results
        r1 = p1.result()
        r2 = p2.result()

        sb1_text = r1.stdout.strip().replace("\n", " | ")
        sb2_text = r2.stdout.strip().replace("\n", " | ")

        print(f"sb1 ({sb1.sandbox_id}): {sb1_text}")
        print(f"sb2 ({sb2.sandbox_id}): {sb2_text}")

    # Sandboxes are automatically stopped when session exits


if __name__ == "__main__":
    main()
