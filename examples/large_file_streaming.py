# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Streaming large files without tripping backpressure.

When a command produces output faster than your code reads it, the output
stream is ended early with ``SandboxStreamBackpressureError`` rather than
silently dropping data. The fix is a *fast-drain* read loop: pull chunks off
the stream as fast as they arrive into a fast local sink (a file), and do any
slow per-chunk work (parsing, hashing, uploading) afterward — never inline in
the read loop.

This example shows:
  1. The fast-drain pattern for ``read_file_streaming`` (read → local file).
  2. The anti-pattern that causes backpressure, and how to fix it.
  3. Catching ``SandboxStreamBackpressureError`` and reacting usefully.
"""

from __future__ import annotations

import contextlib
import tempfile
import time
from pathlib import Path

from cwsandbox import (
    Sandbox,
    SandboxDefaults,
    SandboxStreamBackpressureError,
)

# 128 MiB — large enough to exercise the streaming path; adjust freely.
FILE_BYTES = 128 * 1024 * 1024
REMOTE_PATH = "/tmp/big.bin"


def fast_drain_read(sb: Sandbox, local_path: Path) -> int:
    """Stream a remote file to local disk with a tight read loop.

    The only work inside the loop is ``f.write(chunk)`` — a fast local sink.
    Anything slower (network upload, per-chunk parsing) is done after the
    stream is fully drained, so the read never falls behind the producer.
    """
    total = 0
    # contextlib.closing cancels the background reader if we exit early.
    with contextlib.closing(sb.read_file_streaming(REMOTE_PATH)) as reader:
        with open(local_path, "wb") as f:
            for chunk in reader:
                f.write(chunk)  # fast: just to the OS page cache / disk
                total += len(chunk)
    return total


def slow_inline_read_antipattern(sb: Sandbox) -> int:
    """Anti-pattern: slow work *inside* the read loop.

    Sleeping (or doing a network round-trip, or a synchronous DB insert) per
    chunk lets the producer outrun the consumer; the stream's buffer fills and
    the server ends it with SandboxStreamBackpressureError. Shown here only so
    you can recognize and avoid it — do NOT do this for large payloads.
    """
    total = 0
    with contextlib.closing(sb.read_file_streaming(REMOTE_PATH)) as reader:
        for chunk in reader:
            total += len(chunk)
            time.sleep(0.05)  # stand-in for slow per-chunk work — the problem
    return total


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "large-file-streaming"),
    )

    with Sandbox.run(defaults=defaults) as sb:
        # Produce a deterministic large file inside the sandbox (redirect to a
        # path with a shell so we don't stream the bytes back here just to make
        # the fixture).
        print(f"=== Creating a {FILE_BYTES // (1024 * 1024)} MiB file in the sandbox ===")
        sb.exec(
            ["sh", "-c", f"head -c {FILE_BYTES} /dev/zero > {REMOTE_PATH}"],
            check=True,
        ).result()

        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "big.bin"

            # 1) The right way: fast-drain to local disk.
            print("=== Fast-drain read (recommended) ===")
            received = fast_drain_read(sb, local)
            print(f"Delivered {received} bytes; local file is {local.stat().st_size} bytes")
            assert received == FILE_BYTES, "every byte should arrive — no truncation"

            # 2) Slow work belongs AFTER the drain, not inside the loop.
            print("=== Post-process the drained file (slow work off the read loop) ===")
            import hashlib

            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            print(f"sha256={digest}")

            # 3) Recognize and handle backpressure if a consumer can't keep up.
            print("=== Handling SandboxStreamBackpressureError ===")
            try:
                slow_inline_read_antipattern(sb)
                print("(consumer kept up this time — buffers absorbed the burst)")
            except SandboxStreamBackpressureError as e:
                # Not retryable as-is: retrying the same slow loop hits it again.
                # React by switching to the fast-drain pattern (or chunking).
                print(f"Backpressure (expected for the slow loop): {e}")
                print("Recovering with the fast-drain pattern instead...")
                received = fast_drain_read(sb, local)
                print(f"Recovered: delivered {received} bytes")


if __name__ == "__main__":
    main()
