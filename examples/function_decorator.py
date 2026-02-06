# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: aviato-client

"""Function decorator example using sessions.

This example demonstrates:
- Using Session for managing sandboxes
- The @session.function() decorator for remote function execution
- Both JSON (default) and PICKLE serialization modes
- Closure and global variable capture
- .map() for parallel execution across multiple inputs
- .local() for testing without creating sandboxes
"""

from aviato import SandboxDefaults, Serialization, Session

# Module-level global variable (will be captured automatically)
GLOBAL_MULTIPLIER = 100


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "function-decorator"),
    )

    with Session(defaults) as session:
        # Basic function with JSON serialization (default, safe)
        @session.function()
        def add(x: int, y: int) -> int:
            return x + y

        result = add.remote(2, 3).result()
        print(f"add(2, 3) = {result}")
        print()

        # Function returning a dict (JSON-serializable)
        @session.function()
        def create_config(name: str, value: int) -> dict[str, object]:
            return {"name": name, "value": value, "computed": value * 2}

        result = create_config.remote("test", 42).result()
        print(f"create_config result: {result}")
        print()

        # Function with closure variable AND global variable
        local_offset = 10

        @session.function()
        def compute_with_context(x: int) -> int:
            # Uses both closure (local_offset) and global (GLOBAL_MULTIPLIER)
            return x * GLOBAL_MULTIPLIER + local_offset

        result = compute_with_context.remote(5).result()
        print(f"compute_with_context(5) = {result}")
        print(f"  (5 * {GLOBAL_MULTIPLIER} + {local_offset} = {result})")
        print()

        # Function with PICKLE serialization (for complex types)
        # Only use in trusted environments!
        @session.function(serialization=Serialization.PICKLE)
        def process_complex(data: list[dict]) -> dict:
            return {
                "count": len(data),
                "first": data[0] if data else None,
            }

        complex_data = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
        result = process_complex.remote(complex_data).result()
        print(f"process_complex result: {result}")
        print()

        # .map() for parallel execution across multiple inputs
        @session.function()
        def square(x: int) -> int:
            return x * x

        refs = square.map([(1,), (2,), (3,), (4,), (5,)])
        results = [ref.result() for ref in refs]
        print(f"square.map() results: {results}")
        print()

        # .local() for testing without creating a sandbox
        local_result = square.local(7)
        print(f"square.local(7) = {local_result}")
        print("  (Executed in current process, no sandbox created)")


if __name__ == "__main__":
    main()
