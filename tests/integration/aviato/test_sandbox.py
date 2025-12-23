"""Integration tests for aviato._sandbox module.

These tests require a running Aviato backend.
Set AVIATO_BASE_URL and AVIATO_API_KEY environment variables before running.
"""

import uuid

import pytest

from aviato import Sandbox, SandboxDefaults


@pytest.mark.asyncio
async def test_sandbox_lifecycle(sandbox_defaults: SandboxDefaults) -> None:
    """Test basic sandbox lifecycle: create, exec, stop."""
    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        assert sandbox.sandbox_id is not None

        result = await sandbox.exec(["echo", "hello"])

        assert result.returncode == 0
        assert result.stdout.strip() == "hello"


@pytest.mark.asyncio
async def test_sandbox_create_factory() -> None:
    """Test Sandbox.create factory method."""
    sandbox = await Sandbox.create("echo", "hello", "world")

    try:
        assert sandbox.sandbox_id is not None
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_sandbox_file_operations(sandbox_defaults: SandboxDefaults) -> None:
    """Test sandbox file read/write operations."""
    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        test_content = b"Hello, World!"
        filepath = f"/tmp/test_file_{uuid.uuid4().hex}.txt"

        await sandbox.write_file(filepath, test_content)

        content = await sandbox.read_file(filepath)
        assert content == test_content


@pytest.mark.asyncio
async def test_sandbox_with_defaults() -> None:
    """Test sandbox creation with SandboxDefaults."""
    defaults = SandboxDefaults(
        max_lifetime_seconds=60,
        tags=("test-integration",),
    )

    async with Sandbox(
        defaults=defaults,
    ) as sandbox:
        assert sandbox.sandbox_id is not None

        result = await sandbox.exec(["python", "--version"])
        assert result.returncode == 0
        assert "Python 3.11" in result.stdout


@pytest.mark.asyncio
async def test_sandbox_python_exec(sandbox_defaults: SandboxDefaults) -> None:
    """Test executing Python code in sandbox."""
    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        result = await sandbox.exec(["python", "-c", "print(2 + 2)"])

        assert result.returncode == 0
        assert result.stdout.strip() == "4"


@pytest.mark.asyncio
async def test_sandbox_exec_check_raises_on_failure(sandbox_defaults: SandboxDefaults) -> None:
    """Test exec(check=True) raises SandboxExecutionError on non-zero exit."""
    from aviato.exceptions import SandboxExecutionError

    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        with pytest.raises(SandboxExecutionError) as exc_info:
            await sandbox.exec(["sh", "-c", "exit 42"], check=True)

        assert exc_info.value.exec_result is not None
        assert exc_info.value.exec_result.returncode == 42


@pytest.mark.asyncio
async def test_sandbox_exec_captures_stderr(sandbox_defaults: SandboxDefaults) -> None:
    """Test stderr is properly captured from commands."""
    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        result = await sandbox.exec(["sh", "-c", "echo error_output >&2"])

        assert "error_output" in result.stderr


@pytest.mark.asyncio
async def test_sandbox_exec_check_false_returns_result(sandbox_defaults: SandboxDefaults) -> None:
    """Test exec(check=False) returns result even on non-zero exit."""
    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        result = await sandbox.exec(["sh", "-c", "exit 1"])

        assert result.returncode == 1


@pytest.mark.asyncio
async def test_sandbox_read_nonexistent_file(sandbox_defaults: SandboxDefaults) -> None:
    """Test read_file raises SandboxFileError for missing files."""
    from aviato.exceptions import SandboxFileError

    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        with pytest.raises(SandboxFileError) as exc_info:
            await sandbox.read_file("/nonexistent/path/to/file.txt")

        assert exc_info.value.filepath == "/nonexistent/path/to/file.txt"


@pytest.mark.asyncio
async def test_sandbox_file_operations_binary(sandbox_defaults: SandboxDefaults) -> None:
    """Test reading/writing binary content (non-UTF8)."""
    async with Sandbox(defaults=sandbox_defaults) as sandbox:
        binary_content = bytes(range(256))
        filepath = f"/tmp/binary_test_{uuid.uuid4().hex}.bin"

        await sandbox.write_file(filepath, binary_content)
        content = await sandbox.read_file(filepath)

        assert content == binary_content
