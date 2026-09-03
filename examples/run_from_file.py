# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Create a sandbox from a Compose file (CreateSandboxFromFile).

Demonstrates:
- Sandbox.run_from_file() reading a local Compose YAML path
- primary_service selecting the sandbox primary
- image_overrides supplying a pullable image (skip-build; leftover build:
  is still UNIMPLEMENTED)
- wait() for RUNNING (do not wait for PREPARING)

Compose ports stay in-pod; they are not published SandboxSpec.services.
Environment hostnames are left as written — the platform aliases them
in-pod. YAML is sent as raw bytes; reformatting it changes request identity.
"""

import tempfile
from pathlib import Path

from cwsandbox import ResourceOptions, Sandbox, SandboxDefaults

_COMPOSE = b"""\
services:
  main:
    image: python:3.11
    command: ["sleep", "infinity"]
"""


def main() -> None:
    defaults = SandboxDefaults(tags=("example", "run-from-file"))
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as handle:
        handle.write(_COMPOSE)
        compose_path = Path(handle.name)

    try:
        print("=== Create from Compose file ===")
        with Sandbox.run_from_file(
            compose_path,
            primary_service="main",
            image_overrides={"main": "python:3.11"},
            default_resources=ResourceOptions(
                requests={"cpu": "1", "memory": "256Mi"},
                limits={"cpu": "1", "memory": "256Mi"},
            ),
            defaults=defaults,
        ) as sb:
            print(f"Sandbox ID: {sb.sandbox_id}")
            sb.wait()
            print(f"Status: {sb.status}")
            result = sb.exec(["python", "-c", "print('from compose')"]).result()
            print(f"Output: {result.stdout.strip()}")
    finally:
        compose_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
