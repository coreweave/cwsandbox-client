# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Streaming command execution with real-time output.

This example demonstrates iterating over process.stdout to receive
lines as they arrive, rather than waiting for the command to complete.
"""

from cwsandbox import Sandbox, SandboxDefaults


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "streaming-exec"),
    )

    with Sandbox.run(defaults=defaults) as sb:
        print("=== Real-time stdout iteration ===")
        # Sleep between prints to demonstrate real-time streaming
        cmd = (
            "import time\n"
            "for i in range(5):\n"
            "    print(f'Line {i}', flush=True)\n"
            "    time.sleep(0.3)"
        )
        process = sb.exec(["python", "-c", cmd])

        # Iterate over stdout as lines arrive
        for line in process.stdout:
            print(f"Received: {line.rstrip()}")

        # Get final result after iteration completes
        result = process.result()
        print(f"Exit code: {result.returncode}")


if __name__ == "__main__":
    main()
