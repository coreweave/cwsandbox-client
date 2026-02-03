"""Basic sandbox execution example using the sync API.

This example demonstrates:
- Creating a sandbox using the context manager pattern
- Executing commands in the sandbox
- Reading and writing files
"""

from aviato import Sandbox, SandboxDefaults


def main() -> None:
    # Define reusable defaults
    defaults = SandboxDefaults(
        container_image="ubuntu:22.04",
        max_lifetime_seconds=60.0,
        tags=("example", "basic-execution"),
    )

    # Use Sandbox.run() with context manager for automatic cleanup
    with Sandbox.run(defaults=defaults) as sandbox:
        print(f"Sandbox started: {sandbox.sandbox_id}")

        # Wait for sandbox to be running before accessing tower_id
        # (tower assignment happens during scheduling, not at creation)
        sandbox.wait()
        print(f"Running on tower: {sandbox.tower_id}")
        print(f"Sandbox status: {sandbox.status}")

        # Execute a simple command
        result = sandbox.exec(["echo", "Hello from Aviato sandbox"]).result()
        print(result.stdout.rstrip())

        # Write a file
        content = b"Hello, World!\n"
        sandbox.write_file("/tmp/data.txt", content).result()
        print("write_file: '/tmp/data.txt'")

        # Read the file back
        read_back = sandbox.read_file("/tmp/data.txt").result()
        decoded = read_back.decode("utf-8", errors="replace").rstrip()
        print(f"read_file bytes={len(read_back)} content={decoded}")

        # Verify with cat
        result = sandbox.exec(["cat", "/tmp/data.txt"]).result()
        print(f"cat /tmp/data.txt -> {result.stdout.rstrip()}")


if __name__ == "__main__":
    main()
