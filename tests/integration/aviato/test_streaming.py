"""Integration tests for streaming functionality.

These tests require a running Aviato backend with ATCStreamingService deployed.
The streaming service uses HTTP/2 for bidirectional streaming.

Set AVIATO_BASE_URL and authentication environment variables before running.
"""

import pytest

from aviato import Sandbox


@pytest.mark.asyncio
async def test_exec_with_stdout_callback() -> None:
    """Test exec() with on_stdout callback streams output in real-time."""
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        stdout_chunks: list[bytes] = []

        result = await sandbox.exec(
            ["python", "-c", "print('hello'); print('world')"],
            on_stdout=lambda data: stdout_chunks.append(data),
        )

        assert result.returncode == 0
        assert result.stdout.strip() == "hello\nworld"
        # Callback should have been called with the output
        collected = b"".join(stdout_chunks)
        assert b"hello" in collected
        assert b"world" in collected


@pytest.mark.asyncio
async def test_exec_with_stderr_callback() -> None:
    """Test exec() with on_stderr callback streams error output."""
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        stderr_chunks: list[bytes] = []

        result = await sandbox.exec(
            ["python", "-c", "import sys; print('error', file=sys.stderr)"],
            on_stderr=lambda data: stderr_chunks.append(data),
        )

        assert result.returncode == 0
        assert "error" in result.stderr
        # Callback should have been called with the error output
        collected = b"".join(stderr_chunks)
        assert b"error" in collected


@pytest.mark.asyncio
async def test_exec_with_both_callbacks() -> None:
    """Test exec() with both stdout and stderr callbacks."""
    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        script = """
import sys
print('stdout line')
sys.stdout.flush()
print('stderr line', file=sys.stderr)
sys.stderr.flush()
"""

        result = await sandbox.exec(
            ["python", "-c", script],
            on_stdout=lambda data: stdout_chunks.append(data),
            on_stderr=lambda data: stderr_chunks.append(data),
        )

        assert result.returncode == 0
        assert "stdout line" in result.stdout
        assert "stderr line" in result.stderr

        stdout_collected = b"".join(stdout_chunks)
        stderr_collected = b"".join(stderr_chunks)
        assert b"stdout line" in stdout_collected
        assert b"stderr line" in stderr_collected


@pytest.mark.asyncio
async def test_exec_callback_with_check_raises() -> None:
    """Test exec() with callback and check=True raises on failure."""
    from aviato.exceptions import SandboxExecutionError

    async with Sandbox(
        command="sleep",
        args=["infinity"],
        container_image="python:3.11",
    ) as sandbox:
        stdout_chunks: list[bytes] = []

        with pytest.raises(SandboxExecutionError) as exc_info:
            await sandbox.exec(
                ["sh", "-c", "echo 'before fail'; exit 1"],
                check=True,
                on_stdout=lambda data: stdout_chunks.append(data),
            )

        assert exc_info.value.exec_result is not None
        assert exc_info.value.exec_result.returncode == 1
        # Callback should still have received output before failure
        collected = b"".join(stdout_chunks)
        assert b"before fail" in collected
