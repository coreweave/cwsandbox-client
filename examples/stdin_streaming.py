"""Stdin streaming with exec(stdin=True).

This example demonstrates sending input to running commands via
process.stdin, including raw bytes, text lines, interactive Python,
combined stdin/stdout streaming, and async usage.
"""

import asyncio

from aviato import Sandbox, SandboxDefaults


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "stdin-streaming"),
    )

    with Sandbox.run(defaults=defaults) as sb:
        # --- 1. Basic cat roundtrip ---
        print("=== Basic cat roundtrip ===")
        process = sb.exec(["cat"], stdin=True)
        process.stdin.write(b"hello from stdin\n").result()
        process.stdin.close().result()
        result = process.result()
        print(f"Output: {result.stdout.rstrip()}")

        # --- 2. Using writeline() for convenience ---
        print("\n=== writeline() convenience ===")
        process = sb.exec(["cat"], stdin=True)
        process.stdin.writeline("writeline adds a newline").result()
        process.stdin.close().result()
        result = process.result()
        print(f"Output: {result.stdout.rstrip()}")

        # --- 3. Multiple writes before close ---
        print("\n=== Multiple writes ===")
        process = sb.exec(["cat"], stdin=True)
        for i in range(5):
            process.stdin.writeline(f"line {i}").result()
        process.stdin.close().result()
        result = process.result()
        print(f"Output:\n{result.stdout.rstrip()}")

        # --- 4. Interactive Python session via stdin ---
        print("\n=== Interactive Python via stdin ===")
        process = sb.exec(["python3"], stdin=True)
        process.stdin.writeline("x = 40 + 2").result()
        process.stdin.writeline("print(f'answer: {x}')").result()
        process.stdin.close().result()
        result = process.result()
        print(f"Output: {result.stdout.rstrip()}")

        # --- 5. Streaming output while sending input ---
        print("\n=== Combined stdin + stdout streaming ===")
        process = sb.exec(["cat"], stdin=True)
        process.stdin.writeline("streamed line 1").result()
        process.stdin.writeline("streamed line 2").result()
        process.stdin.close().result()
        for line in process.stdout:
            print(f"[stdout] {line}", end="")
        result = process.result()
        print(f"Exit code: {result.returncode}")

        # --- 6. Sort command (EOF-dependent) ---
        print("\n=== Sort command (needs EOF) ===")
        process = sb.exec(["sort"], stdin=True)
        process.stdin.writeline("banana").result()
        process.stdin.writeline("apple").result()
        process.stdin.writeline("cherry").result()
        process.stdin.close().result()
        result = process.result()
        print(f"Sorted:\n{result.stdout.rstrip()}")


async def async_example() -> None:
    """Async usage of stdin streaming."""
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "stdin-streaming-async"),
    )

    async with Sandbox.run(defaults=defaults) as sb:
        print("\n=== Async stdin streaming ===")
        process = sb.exec(["cat"], stdin=True)
        await process.stdin.write(b"async hello\n")
        await process.stdin.close()
        result = await process
        print(f"Output: {result.stdout.rstrip()}")


if __name__ == "__main__":
    main()
    asyncio.run(async_example())
