"""Integration tests for aviato._sandbox module.

These tests require a running Aviato backend.
Set AVIATO_BASE_URL and AVIATO_API_KEY environment variables before running.
"""

import uuid

import pytest

from aviato import Sandbox, SandboxDefaults
from tests.integration.aviato.conftest import _SESSION_TAG


def test_sandbox_lifecycle(sandbox_defaults: SandboxDefaults) -> None:
    """Test basic sandbox lifecycle: create, exec, stop."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        assert sandbox.sandbox_id is not None

        result = sandbox.exec(["echo", "hello"]).result()

        assert result.returncode == 0
        assert result.stdout.strip() == "hello"


def test_sandbox_run_factory(sandbox_defaults: SandboxDefaults) -> None:
    """Test Sandbox.run factory method returns sandbox with ID and reaches running state."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        assert sandbox.sandbox_id is not None
        sandbox.wait()
        assert sandbox.status == "running"


def test_sandbox_run_factory_no_defaults() -> None:
    """Test Sandbox.run factory method without defaults - exercises default config path."""
    sandbox = Sandbox.run("echo", "hello", "world", tags=[_SESSION_TAG])

    try:
        assert sandbox.sandbox_id is not None
    finally:
        sandbox.stop().result()


def test_sandbox_no_defaults_max_lifetime_only() -> None:
    """Test sandbox creation with only max_lifetime_seconds configured."""
    defaults = SandboxDefaults(max_lifetime_seconds=60)
    sandbox = Sandbox.run("echo", "hello", "world", defaults=defaults, tags=[_SESSION_TAG])

    try:
        assert sandbox.sandbox_id is not None
    finally:
        sandbox.stop().result()


def test_sandbox_no_defaults_tags_only() -> None:
    """Test sandbox creation with only tags configured."""
    defaults = SandboxDefaults(tags=("isolation-test", _SESSION_TAG))
    sandbox = Sandbox.run("echo", "hello", "world", defaults=defaults)

    try:
        assert sandbox.sandbox_id is not None
    finally:
        sandbox.stop().result()


def test_sandbox_no_defaults_sleep_command() -> None:
    """Test sandbox creation with sleep infinity command and no defaults."""
    sandbox = Sandbox.run("sleep", "infinity", tags=[_SESSION_TAG])

    try:
        assert sandbox.sandbox_id is not None
    finally:
        sandbox.stop().result()


def test_sandbox_file_operations(sandbox_defaults: SandboxDefaults) -> None:
    """Test sandbox file read/write operations."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        test_content = b"Hello, World!"
        filepath = f"/tmp/test_file_{uuid.uuid4().hex}.txt"

        sandbox.write_file(filepath, test_content).result()

        content = sandbox.read_file(filepath).result()
        assert content == test_content


def test_sandbox_with_defaults() -> None:
    """Test sandbox creation with SandboxDefaults."""
    defaults = SandboxDefaults(
        max_lifetime_seconds=60,
        tags=("test-integration",),
        resources={"cpu": "500m", "memory": "256Mi"},
    )

    with Sandbox.run("sleep", "infinity", defaults=defaults) as sandbox:
        assert sandbox.sandbox_id is not None

        result = sandbox.exec(["python", "--version"]).result()
        assert result.returncode == 0
        assert "Python 3.11" in result.stdout


def test_sandbox_python_exec(sandbox_defaults: SandboxDefaults) -> None:
    """Test executing Python code in sandbox."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["python", "-c", "print(2 + 2)"]).result()

        assert result.returncode == 0
        assert result.stdout.strip() == "4"


def test_sandbox_exec_check_raises_on_failure(sandbox_defaults: SandboxDefaults) -> None:
    """Test exec(check=True) raises SandboxExecutionError on non-zero exit."""
    from aviato.exceptions import SandboxExecutionError

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        with pytest.raises(SandboxExecutionError) as exc_info:
            sandbox.exec(["sh", "-c", "exit 42"], check=True).result()

        assert exc_info.value.exec_result is not None
        assert exc_info.value.exec_result.returncode == 42


def test_sandbox_exec_captures_stderr(sandbox_defaults: SandboxDefaults) -> None:
    """Test stderr is properly captured from commands."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["sh", "-c", "echo error_output >&2"]).result()

        assert "error_output" in result.stderr


def test_sandbox_exec_check_false_returns_result(sandbox_defaults: SandboxDefaults) -> None:
    """Test exec(check=False) returns result even on non-zero exit."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["sh", "-c", "exit 1"]).result()

        assert result.returncode == 1


def test_sandbox_read_nonexistent_file(sandbox_defaults: SandboxDefaults) -> None:
    """Test read_file raises SandboxFileError for missing files."""
    from aviato.exceptions import SandboxFileError

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        with pytest.raises(SandboxFileError) as exc_info:
            sandbox.read_file("/nonexistent/path/to/file.txt").result()

        assert exc_info.value.filepath == "/nonexistent/path/to/file.txt"


def test_sandbox_file_operations_binary(sandbox_defaults: SandboxDefaults) -> None:
    """Test reading/writing binary content (non-UTF8)."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        binary_content = bytes(range(256))
        filepath = f"/tmp/binary_test_{uuid.uuid4().hex}.bin"

        sandbox.write_file(filepath, binary_content).result()
        content = sandbox.read_file(filepath).result()

        assert content == binary_content


# Streaming exec tests (regression coverage for queue-decoupled pattern fix)


def test_sandbox_exec_streaming_basic(sandbox_defaults: SandboxDefaults) -> None:
    """Test basic streaming exec with sync iteration."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        process = sandbox.exec(["echo", "hello"])  # stream=True default
        lines = list(process.stdout)
        result = process.result()

        assert lines == ["hello\n"]
        assert result.returncode == 0
        assert result.stdout == "hello\n"


def test_sandbox_exec_streaming_stderr(sandbox_defaults: SandboxDefaults) -> None:
    """Test streaming exec captures both stdout and stderr."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        # Command producing both stdout and stderr
        process = sandbox.exec(["sh", "-c", "echo stdout_line && echo stderr_line >&2"])

        stdout_lines = list(process.stdout)
        stderr_lines = list(process.stderr)
        result = process.result()

        assert stdout_lines == ["stdout_line\n"]
        assert stderr_lines == ["stderr_line\n"]
        assert result.returncode == 0
        assert result.stdout == "stdout_line\n"
        assert result.stderr == "stderr_line\n"


def test_sandbox_exec_streaming_returncode_lifecycle(
    sandbox_defaults: SandboxDefaults,
) -> None:
    """Test returncode and poll() lifecycle during streaming."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        process = sandbox.exec(["sh", "-c", "echo line1; echo line2"])

        # Before iteration completes, returncode should be None
        assert process.returncode is None

        # Iterate through stdout (output may arrive in chunks, not line-by-line)
        chunks = list(process.stdout)
        combined = "".join(chunks)
        assert "line1" in combined
        assert "line2" in combined

        # After result(), returncode should be populated
        result = process.result()
        assert process.returncode == 0
        assert result.returncode == 0


def test_sandbox_exec_streaming_poll_wait(sandbox_defaults: SandboxDefaults) -> None:
    """Test poll() and wait() methods on streaming process."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        process = sandbox.exec(["echo", "test"])

        # Consume the stream first to let process complete
        _ = list(process.stdout)

        # After stream consumption, wait should return exit code
        exit_code = process.wait(timeout=5.0)
        assert exit_code == 0

        # poll() should also return exit code after completion
        assert process.poll() == 0


def test_sandbox_exec_streaming_large_output(sandbox_defaults: SandboxDefaults) -> None:
    """Test streaming exec with large output.

    Generates 2000 lines to validate streaming handles large outputs correctly.
    Critical regression test for commit 2ef1acb queue-decoupled pattern fix.

    Note: Streaming delivers data in chunks (not individual lines). The test
    validates all content arrives correctly by checking the combined output.
    """
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        process = sandbox.exec(
            ["python", "-c", "for i in range(2000): print(f'line_{i}')"],
            timeout_seconds=30.0,
        )

        chunks = list(process.stdout)
        result = process.result()

        # Combine all chunks and split by newlines to count actual lines
        combined = "".join(chunks)
        lines = [line for line in combined.split("\n") if line]

        assert len(lines) == 2000, f"Expected 2000 lines, got {len(lines)}"
        assert lines[0] == "line_0"
        assert lines[-1] == "line_1999"
        assert result.returncode == 0


def test_sandbox_exec_streaming_live_iteration(sandbox_defaults: SandboxDefaults) -> None:
    """Test lines arrive incrementally while command is running (proves true streaming).

    Uses slow-producing command and records timestamps for each line to verify
    that lines arrive spread out over time (streaming) vs all at once (buffered).
    """
    import time

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        # Command takes ~2 seconds total (10 lines x 0.2s sleep)
        process = sandbox.exec(
            ["bash", "-c", "for i in {1..10}; do echo line$i; sleep 0.2; done"],
            timeout_seconds=30.0,
        )

        line_times: list[float] = []
        lines: list[str] = []
        for line in process.stdout:
            line_times.append(time.time())
            lines.append(line)

        result = process.result()

        # Verify we got all lines
        assert len(lines) == 10
        assert lines[0] == "line1\n"
        assert lines[-1] == "line10\n"
        assert result.returncode == 0

        # Streaming proof: lines should arrive spread out over time, not all at once.
        # If buffered, all 10 lines would arrive within milliseconds of each other.
        # If streaming, lines arrive ~0.2s apart as the command produces them.
        # We check that the time spread from first to last line is significant.
        first_to_last_spread = line_times[-1] - line_times[0]

        # Command produces lines over ~2s (10 x 0.2s). With streaming, spread should
        # be at least 1s. With buffering, spread would be nearly 0.
        assert first_to_last_spread > 1.0, (
            f"Lines arrived over only {first_to_last_spread:.2f}s spread - "
            "expected incremental delivery proving true streaming"
        )


def test_sandbox_exec_streaming_check_raises(sandbox_defaults: SandboxDefaults) -> None:
    """Test check=True raises SandboxExecutionError after stream completes."""
    from aviato.exceptions import SandboxExecutionError

    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        process = sandbox.exec(
            ["sh", "-c", "echo 'output before exit' && exit 42"],
            check=True,
        )

        # Consume the stream - should complete without error
        _ = list(process.stdout)

        # Error should be raised when getting result with check=True
        with pytest.raises(SandboxExecutionError) as exc_info:
            process.result()

        # Verify exception contains captured output
        assert exc_info.value.exec_result is not None
        assert exc_info.value.exec_result.returncode == 42
        assert "output before exit" in exc_info.value.exec_result.stdout
        assert exc_info.value.exec_result.command == [
            "sh",
            "-c",
            "echo 'output before exit' && exit 42",
        ]


# Async context manager tests


@pytest.mark.asyncio
async def test_sandbox_async_context_manager(sandbox_defaults: SandboxDefaults) -> None:
    """Test async context manager for Sandbox.

    Verifies that 'async with Sandbox.run(...)' properly:
    1. Creates and starts the sandbox
    2. Allows await on exec() and file operations
    3. Properly cleans up (stops sandbox) on exit

    This is a regression test for event loop routing bugs where __aexit__
    directly awaited _stop_async() instead of routing through _LoopManager.
    """
    async with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        assert sandbox.sandbox_id is not None

        # Use await pattern (not .result())
        result = await sandbox.exec(["echo", "async context manager"])
        assert result.returncode == 0
        assert result.stdout.strip() == "async context manager"

        # Verify file operations also work with await
        await sandbox.write_file("/tmp/async_test.txt", b"async content")
        content = await sandbox.read_file("/tmp/async_test.txt")
        assert content == b"async content"

    # After exiting, sandbox should be stopped
    # (sandbox._stopped should be True, but we can't easily verify this externally)


# print_output tests


def test_sandbox_exec_default_is_silent(
    sandbox_defaults: SandboxDefaults, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test exec() is silent by default (print_output=False)."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        # Default: print_output=False, so no output is printed
        result = sandbox.exec(["echo", "silent by default"]).result()

        captured = capsys.readouterr()
        assert "silent by default" not in captured.out  # Not printed
        assert result.stdout.strip() == "silent by default"  # But available in result


def test_sandbox_exec_print_output_true_prints_output(
    sandbox_defaults: SandboxDefaults, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test exec() with print_output=True prints output."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["echo", "printed output"], print_output=True).result()

        captured = capsys.readouterr()
        assert "printed output" in captured.out
        assert result.stdout.strip() == "printed output"


def test_sandbox_exec_print_output_true_prints_stderr(
    sandbox_defaults: SandboxDefaults, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test exec() with print_output=True prints stderr to stdout."""
    with Sandbox.run("sleep", "infinity", defaults=sandbox_defaults) as sandbox:
        result = sandbox.exec(["sh", "-c", "echo stderr_msg >&2"], print_output=True).result()

        captured = capsys.readouterr()
        # stderr should be printed to stdout (no prefix)
        assert "stderr_msg" in captured.out
        assert result.stderr.strip() == "stderr_msg"
