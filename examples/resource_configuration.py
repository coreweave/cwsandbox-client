# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Resource configuration examples showing different ways to specify CPU, memory, and GPU.

Demonstrates:
- ResourceOptions for separate requests and limits (Burstable QoS)
- Flat dict shorthand for equal requests and limits (Guaranteed QoS)
- Nested dict form as an alternative to ResourceOptions
- GPU configuration
- Inspecting confirmed resources from the sandbox response
"""

from cwsandbox import ResourceOptions, Sandbox, SandboxDefaults


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=60.0,
        tags=("example", "resource-configuration"),
    )

    # --- Burstable QoS: requests < limits ---
    # The sandbox requests fewer resources than its ceiling, allowing the
    # scheduler to bin-pack more pods while still permitting burst usage.
    print("=== Burstable QoS (ResourceOptions) ===")
    with Sandbox.run(
        defaults=defaults,
        resources=ResourceOptions(
            requests={"cpu": "500m", "memory": "512Mi"},
            limits={"cpu": "2", "memory": "2Gi"},
        ),
    ) as sb:
        print(f"Sandbox: {sb.sandbox_id}")
        print(f"Requests: {sb.resource_requests}")
        print(f"Limits:   {sb.resource_limits}")

        result = sb.exec(["python3", "-c", "import os; print(os.cpu_count(), 'CPUs')"]).result()
        print(f"Output:   {result.stdout.rstrip()}")

    # --- Guaranteed QoS: flat dict shorthand ---
    # When requests == limits, use a flat dict for brevity. The SDK
    # normalizes this to ResourceOptions with identical requests and limits.
    print("\n=== Guaranteed QoS (flat dict) ===")
    with Sandbox.run(
        defaults=defaults,
        resources={"cpu": "1", "memory": "1Gi"},
    ) as sb:
        print(f"Sandbox: {sb.sandbox_id}")
        print(f"Requests: {sb.resource_requests}")
        print(f"Limits:   {sb.resource_limits}")

    # --- Nested dict form ---
    # Equivalent to ResourceOptions but uses a plain dict.
    print("\n=== Nested dict form ===")
    with Sandbox.run(
        defaults=defaults,
        resources={
            "requests": {"cpu": "250m", "memory": "256Mi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        },
    ) as sb:
        print(f"Sandbox: {sb.sandbox_id}")
        print(f"Requests: {sb.resource_requests}")
        print(f"Limits:   {sb.resource_limits}")

    # --- GPU configuration ---
    # GPU is separate from requests/limits because GPU overcommit is not
    # supported by the backend. GPU count goes into both proto fields.
    print("\n=== GPU with resource limits ===")
    with Sandbox.run(
        defaults=defaults,
        resources=ResourceOptions(
            requests={"cpu": "1", "memory": "1Gi"},
            limits={"cpu": "4", "memory": "4Gi"},
            gpu={"count": 1, "type": "A100"},
        ),
    ) as sb:
        print(f"Sandbox: {sb.sandbox_id}")
        print(f"GPU:      {sb.resource_gpu}")
        print(f"Requests: {sb.resource_requests}")
        print(f"Limits:   {sb.resource_limits}")

    # --- Shared defaults ---
    # Set resources once in SandboxDefaults and reuse across sandboxes.
    print("\n=== Shared defaults ===")
    shared = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=60.0,
        tags=("example", "resource-configuration"),
        resources=ResourceOptions(
            requests={"cpu": "500m", "memory": "512Mi"},
            limits={"cpu": "2", "memory": "2Gi"},
        ),
    )
    with Sandbox.run(defaults=shared) as sb:
        print(f"Sandbox: {sb.sandbox_id}")
        print(f"Requests: {sb.resource_requests}")
        print(f"Limits:   {sb.resource_limits}")


if __name__ == "__main__":
    main()
