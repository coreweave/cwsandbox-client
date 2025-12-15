# Parallel Sandboxes

Use sessions to manage multiple sandboxes and run operations in parallel.

## Creating Multiple Sandboxes

```python
import asyncio
from aviato import Sandbox, SandboxDefaults

async def main():
    defaults = SandboxDefaults(
        container_image="python:3.11",
        max_lifetime_seconds=60.0,
    )

    async with Sandbox.session(defaults) as session:
        # Create multiple sandboxes
        sb1 = session.create(
            command="sleep",
            args=["infinity"],
            tags=["worker-1"],
        )
        sb2 = session.create(
            command="sleep",
            args=["infinity"],
            tags=["worker-2"],
        )

        async with sb1, sb2:
            print(f"Sandbox 1: {sb1.sandbox_id}")
            print(f"Sandbox 2: {sb2.sandbox_id}")

            # Run commands in parallel
            r1, r2 = await asyncio.gather(
                sb1.exec(["echo", "from sandbox 1"]),
                sb2.exec(["echo", "from sandbox 2"]),
            )

            print(f"sb1: {r1.stdout.decode().strip()}")
            print(f"sb2: {r2.stdout.decode().strip()}")

asyncio.run(main())
```

## Session Benefits

### Shared Configuration

All sandboxes created via `session.create()` inherit session defaults:

```python
defaults = SandboxDefaults(
    container_image="python:3.11",
    tags=("my-app",),
)

async with Sandbox.session(defaults) as session:
    # Both sandboxes use python:3.11 and have "my-app" tag
    sb1 = session.create(command="sleep", args=["infinity"])
    sb2 = session.create(command="sleep", args=["infinity"])
```

### Automatic Cleanup

The session tracks all sandboxes and stops any orphaned ones on exit:

```python
async with Sandbox.session(defaults) as session:
    sb1 = session.create(command="sleep", args=["infinity"])
    async with sb1:
        await sb1.exec(["echo", "working"])
    # sb1 stopped by context manager

    sb2 = session.create(command="sleep", args=["infinity"])
    await sb2.start()
    # Forgot to stop sb2!

# Session cleanup stops sb2 automatically
```

## Parallel Function Execution

Run multiple function calls in parallel:

```python
async with Sandbox.session(defaults) as session:
    @session.function()
    def compute(n: int) -> int:
        import time
        time.sleep(1)  # Simulate work
        return n * n

    # Run in parallel - total time ~1s instead of ~3s
    results = await asyncio.gather(
        compute(1),
        compute(2),
        compute(3),
    )

    values = [r.value for r in results]
    print(values)  # [1, 4, 9]
```
