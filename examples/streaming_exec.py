"""Streaming command execution with real-time output.

This example demonstrates three patterns for handling output:
1. Silent (default): Access output via result.stdout
2. Streaming: Iterate over process.stdout for real-time processing
3. Auto-print (convenience): Use quiet=False for quick debugging
"""

from aviato import Sandbox, SandboxDefaults


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "streaming-exec"),
    )

    # Sleep between prints to demonstrate real-time streaming
    cmd = "import time\nfor i in range(5):\n    print(f'Line {i}', flush=True)\n    time.sleep(0.3)"

    with Sandbox.run(defaults=defaults) as sb:
        # Option 1: Silent (default) - access output via result
        print("=== Option 1: Silent (default) ===")
        result = sb.exec(["python", "-c", cmd]).result()
        print(f"Output:\n{result.stdout}")
        print(f"Exit code: {result.returncode}\n")

        # Option 2: Streaming - iterate for real-time processing
        print("=== Option 2: Streaming ===")
        process = sb.exec(["python", "-c", cmd])

        for line in process.stdout:
            print(f"[stdout] {line.rstrip()}")

        result = process.result()
        print(f"Exit code: {result.returncode}\n")

        # Option 3: Auto-print (convenience) - for quick debugging
        print("=== Option 3: Auto-print (convenience) ===")
        result = sb.exec(["python", "-c", cmd], quiet=False).result()
        print(f"Exit code: {result.returncode}")


if __name__ == "__main__":
    main()
