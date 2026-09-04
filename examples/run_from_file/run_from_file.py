# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Create a sandbox from a Compose file (CreateSandboxFromFile).

Demonstrates:
- Sandbox.run_from_file() reading a local Compose YAML path
- primary_service selecting the sandbox primary
- A three-service file (Redis, a Python API, Ubuntu primary) with
  healthchecks and depends_on
- wait() for RUNNING (do not wait for PREPARING)

Compose ports stay in-pod; they are not published SandboxSpec.services.
Environment hostnames are left as written — the platform aliases them
in-pod. YAML is sent as raw bytes; reformatting it changes request identity.
Images in this file are already pullable, so image_overrides is omitted.
A leftover build: stanza is still UNIMPLEMENTED.
"""

from pathlib import Path

from cwsandbox import ResourceOptions, Sandbox, SandboxDefaults

_COMPOSE = Path(__file__).with_name("docker-compose.yaml")
_HTTP_GET = """\
set -euo pipefail
exec 3<>/dev/tcp/api/8080
printf 'GET /health HTTP/1.0\\r\\nHost: api\\r\\n\\r\\n' >&3
cat <&3
"""


def main() -> None:
    defaults = SandboxDefaults(tags=("example", "run-from-file"))
    print(f"=== Create from Compose file ({_COMPOSE.name}) ===")
    with Sandbox.run_from_file(
        _COMPOSE,
        primary_service="main",
        default_resources=ResourceOptions(
            requests={"cpu": "500m", "memory": "256Mi"},
            limits={"cpu": "500m", "memory": "256Mi"},
        ),
        defaults=defaults,
    ) as sb:
        print(f"Sandbox ID: {sb.sandbox_id}")
        sb.wait()
        print(f"Status: {sb.status}")
        print(f"Containers: {[row.name for row in sb.containers]}")

        hosts = sb.exec(["getent", "hosts", "cache", "api"]).result()
        print(f"Hosts:\n{hosts.stdout.rstrip()}")

        ping = sb.exec(["redis-cli", "ping"], container="cache").result()
        print(f"Redis: {ping.stdout.strip()}")

        health = sb.exec(["bash", "-c", _HTTP_GET]).result()
        print(f"API /health: {health.stdout.strip()}")


if __name__ == "__main__":
    main()
