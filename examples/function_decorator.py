"""Function decorator example using sessions.

This example demonstrates:
- Using Sandbox.session() for managing sandboxes
- The @session.function() decorator for remote function execution
- Both JSON (default) and PICKLE serialization modes
- Closure and global variable capture
"""

import asyncio

from aviato import Sandbox, SandboxDefaults, Serialization

# Module-level global variable (will be captured automatically)
GLOBAL_MULTIPLIER = 100


async def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "function-decorator"),
    )

    async with Sandbox.session(defaults) as session:
        # Basic function with JSON serialization (default, safe)
        @session.function()
        def add(x: int, y: int) -> int:
            return x + y

        result = await add(2, 3)
        print(f"add(2, 3) = {result}")
        print()

        # Function returning a dict (JSON-serializable)
        @session.function()
        def create_config(name: str, value: int) -> dict[str, object]:
            return {"name": name, "value": value, "computed": value * 2}

        result = await create_config("test", 42)
        print(f"create_config result: {result}")
        print()

        # Function with closure variable AND global variable
        local_offset = 10

        @session.function()
        def compute_with_context(x: int) -> int:
            # Uses both closure (local_offset) and global (GLOBAL_MULTIPLIER)
            return x * GLOBAL_MULTIPLIER + local_offset

        result = await compute_with_context(5)
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
        result = await process_complex(complex_data)
        print(f"process_complex result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
