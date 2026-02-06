# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: aviato-client

"""Quick start example - the most common sandbox usage pattern.

This example demonstrates:
- Using Sandbox.run() with a context manager
- Running commands with exec()
- Automatic cleanup when the context exits
"""

from aviato import Sandbox


def main() -> None:
    # Create a sandbox and run commands in it
    with Sandbox.run(container_image="python:3.11") as sandbox:
        print(f"Sandbox ID: {sandbox.sandbox_id}")

        # Execute a command and get the result
        result = sandbox.exec(["python", "-c", "print('Hello from sandbox!')"]).result()
        print(f"Output: {result.stdout.strip()}")
        print(f"Exit code: {result.returncode}")

    # Sandbox is automatically stopped when exiting the context


if __name__ == "__main__":
    main()
