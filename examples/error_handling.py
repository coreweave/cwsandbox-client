# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Error handling patterns for cwsandbox SDK.

Demonstrates:
- SandboxExecutionError with check=True
- SandboxTimeoutError from exec timeout
- SandboxNotFoundError with missing_ok

Usage:
    uv run examples/error_handling.py
"""

from cwsandbox import Sandbox, SandboxDefaults
from cwsandbox.exceptions import (
    SandboxExecutionError,
    SandboxNotFoundError,
    SandboxTimeoutError,
)


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "error-handling"),
    )

    # --- SandboxExecutionError with check=True ---
    print("1. SandboxExecutionError with check=True")
    print("-" * 50)

    with Sandbox.run(defaults=defaults) as sb:
        # Without check=True, non-zero exit codes don't raise
        result = sb.exec(["sh", "-c", "exit 42"]).result()
        print(f"   Without check: returncode={result.returncode} (no exception)")

        # With check=True, non-zero exit codes raise SandboxExecutionError
        try:
            sb.exec(["sh", "-c", "exit 1"], check=True).result()
        except SandboxExecutionError as e:
            print("   With check=True: caught SandboxExecutionError")
            print(f"   returncode={e.exec_result.returncode}")
    print()

    # --- SandboxTimeoutError ---
    print("2. SandboxTimeoutError from exec timeout")
    print("-" * 50)

    with Sandbox.run(defaults=defaults) as sb:
        try:
            # Command that takes longer than timeout
            sb.exec(["sleep", "10"], timeout_seconds=1).result()
        except SandboxTimeoutError:
            print("   Caught SandboxTimeoutError (command exceeded 1s timeout)")
    print()

    # --- SandboxNotFoundError with missing_ok ---
    print("3. SandboxNotFoundError with missing_ok")
    print("-" * 50)

    # Try to delete a non-existent sandbox
    fake_id = "non-existent-sandbox-id"

    # Without missing_ok, raises SandboxNotFoundError
    try:
        Sandbox.delete(fake_id).result()
    except SandboxNotFoundError as e:
        print("   Without missing_ok: caught SandboxNotFoundError")
        print(f"   sandbox_id={e.sandbox_id}")

    # With missing_ok=True, returns None instead of raising
    Sandbox.delete(fake_id, missing_ok=True).result()
    print("   With missing_ok=True: delete completed (no exception)")
    print()

    print("All error handling patterns demonstrated!")


if __name__ == "__main__":
    main()
