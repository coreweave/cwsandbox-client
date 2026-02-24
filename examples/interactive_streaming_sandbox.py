# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: aviato-client

"""Start a sandbox that logs a configurable value, then exec to change it.

The entrypoint reads /tmp/message each iteration. Use `aviato exec` or
`aviato shell` to update the file and watch the log output change in real time.

Usage:
    python examples/interactive_streaming_sandbox.py

Then in another terminal:
    # One-off command (no PTY)
    aviato exec <sandbox-id> sh -c "echo 'changed by exec!' > /tmp/message"

    # Interactive shell
    aviato shell <sandbox-id>
    echo 'changed by shell!' > /tmp/message

    # Run a command with PTY (reads current value)
    aviato shell <sandbox-id> --cmd "cat /tmp/message"
"""

from aviato import Sandbox


def main() -> None:
    entrypoint = "echo 'hello' > /tmp/message; " "while true; do cat /tmp/message; sleep 3; done"

    with Sandbox.run("sh", "-c", entrypoint) as sb:
        print(f"Sandbox running: {sb.sandbox_id}")
        sid = sb.sandbox_id
        print("\nIn another terminal, try any of these:")
        print("\n  # One-off command (no PTY)")
        print(f"  aviato exec {sid} sh -c \"echo 'changed by exec!' > /tmp/message\"")
        print("\n  # Interactive shell")
        print(f"  aviato shell {sid}")
        print("  echo 'changed by shell!' > /tmp/message")
        print("\n  # Run a command with PTY (reads current value)")
        print(f'  aviato shell {sid} --cmd "cat /tmp/message"')
        print("\n  # Stream logs from another terminal")
        print(f"  aviato logs {sid} --follow --timestamps")
        print("\nStreaming logs (Ctrl-C to stop and clean up)...\n")
        try:
            for line in sb.stream_logs(follow=True, timestamps=True):
                print(line, end="")
        except KeyboardInterrupt:
            print("\nStopping sandbox...")
    print("Sandbox stopped.")


if __name__ == "__main__":
    main()
