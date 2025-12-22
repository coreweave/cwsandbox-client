"""Example demonstrating kwargs validation in Aviato client.

This shows how the client validates sandbox parameters early, providing
helpful error messages for typos and invalid parameters.
"""

import asyncio

from aviato import Sandbox, Session


async def main() -> None:
    """Demonstrate kwargs validation."""
    print("=== Kwargs Validation Demo ===\n")

    # Example 1: Valid kwargs work as expected
    print("1. Creating sandbox with VALID kwargs:")
    try:
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            resources={"cpu": "100m", "memory": "128Mi"},
            ports=[{"container_port": 8080}],
            max_timeout_seconds=60,
        )
        print(f"   Sandbox created: {sandbox}\n")
    except ValueError as e:
        print(f"   Error: {e}\n")

    # Example 2: Invalid kwargs are caught early
    print("2. Creating sandbox with INVALID kwargs (typo in parameter name):")
    try:
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            resorces={"cpu": "100m"},  # Typo: 'resorces' instead of 'resources'
        )
        print(f"   Sandbox created: {sandbox}\n")
    except ValueError as e:
        print(f"   Error caught: {e}\n")

    # Example 3: Session.create validates kwargs
    print("3. Using Session.create with invalid kwargs:")
    try:
        session = Session()
        sandbox = session.create(
            command="echo",
            args=["hello"],
            invalid_param="value",
        )
        print(f"   Sandbox created: {sandbox}\n")
    except ValueError as e:
        print(f"   Error caught: {e}\n")

    # Example 4: Function decorator validates sandbox_kwargs
    print("4. Using @session.function() with invalid sandbox_kwargs:")
    try:
        session = Session()

        @session.function(
            wrong_kwarg="value",
        )
        def compute(x: int) -> int:
            return x * 2

        print(f"   Function decorated: {compute}\n")
    except ValueError as e:
        print(f"   Error caught: {e}\n")

    # Example 5: Show valid kwargs
    print("5. Valid kwargs that can be passed:")
    print("   - resources: ResourceRequest (cpu, memory, gpu)")
    print("   - mounted_files: List of MountedFile")
    print("   - s3_mount: S3Mount configuration")
    print("   - ports: List of Port configurations")
    print("   - service: ServiceOptions")
    print("   - max_timeout_seconds: Request timeout\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
