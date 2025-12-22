"""Unit tests for aviato.exceptions module."""

import pytest

from aviato._types import ExecResult
from aviato.exceptions import (
    AsyncFunctionError,
    AviatoAuthenticationError,
    AviatoError,
    FunctionError,
    FunctionSerializationError,
    SandboxError,
    SandboxExecutionError,
    SandboxFailedError,
    SandboxFileError,
    SandboxNotRunningError,
    SandboxTerminatedError,
    SandboxTimeoutError,
    WandbAuthError,
)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_aviato_error_is_base_for_all_exceptions(self) -> None:
        """Test AviatoError is the base for all Aviato exceptions."""
        assert issubclass(SandboxError, AviatoError)
        assert issubclass(FunctionError, AviatoError)
        assert issubclass(AviatoAuthenticationError, AviatoError)

    def test_auth_error_is_base_for_auth_exceptions(self) -> None:
        """Test AviatoAuthenticationError is the base for auth-related exceptions."""
        assert issubclass(WandbAuthError, AviatoAuthenticationError)

    def test_sandbox_error_is_base_for_sandbox_exceptions(self) -> None:
        """Test SandboxError is the base for sandbox-related exceptions."""
        assert issubclass(SandboxNotRunningError, SandboxError)
        assert issubclass(SandboxTimeoutError, SandboxError)
        assert issubclass(SandboxTerminatedError, SandboxError)
        assert issubclass(SandboxFailedError, SandboxError)
        assert issubclass(SandboxExecutionError, SandboxError)
        assert issubclass(SandboxFileError, SandboxError)

    def test_function_error_is_base_for_function_exceptions(self) -> None:
        """Test FunctionError is the base for function-related exceptions."""
        assert issubclass(AsyncFunctionError, FunctionError)
        assert issubclass(FunctionSerializationError, FunctionError)

    def test_aviato_error_is_exception(self) -> None:
        """Test AviatoError inherits from Exception."""
        assert issubclass(AviatoError, Exception)


class TestExceptions:
    """Tests for individual exceptions."""

    def test_sandbox_not_running_error(self) -> None:
        """Test SandboxNotRunningError can be raised and caught."""
        with pytest.raises(SandboxNotRunningError, match="not running"):
            raise SandboxNotRunningError("Sandbox not running")

    def test_sandbox_timeout_error(self) -> None:
        """Test SandboxTimeoutError can be raised and caught."""
        with pytest.raises(SandboxTimeoutError, match="timed out"):
            raise SandboxTimeoutError("Operation timed out")

    def test_sandbox_terminated_error(self) -> None:
        """Test SandboxTerminatedError can be raised and caught."""
        with pytest.raises(SandboxTerminatedError, match="terminated"):
            raise SandboxTerminatedError("Sandbox was terminated")

    def test_sandbox_failed_error(self) -> None:
        """Test SandboxFailedError can be raised and caught."""
        with pytest.raises(SandboxFailedError, match="failed"):
            raise SandboxFailedError("Sandbox failed to start")


class TestSandboxExecutionError:
    """Tests for SandboxExecutionError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = SandboxExecutionError("Function failed")
        assert str(exc) == "Function failed"
        assert exc.exec_result is None
        assert exc.exception_type is None
        assert exc.exception_message is None

    def test_with_exec_result(self) -> None:
        """Test exception with exec result."""
        exec_result = ExecResult(
            stdout_bytes=b"output",
            stderr_bytes=b"error message",
            returncode=1,
            command=["python", "-c", "exit(1)"],
        )
        exc = SandboxExecutionError(
            "Function failed",
            exec_result=exec_result,
            exception_type="ValueError",
            exception_message="invalid value",
        )

        assert exc.exec_result is exec_result
        assert exc.exception_type == "ValueError"
        assert exc.exception_message == "invalid value"
        # Access via exec_result directly
        assert exc.exec_result.command == ["python", "-c", "exit(1)"]
        assert exc.exec_result.stdout == "output"
        assert exc.exec_result.stderr == "error message"
        assert exc.exec_result.stdout_bytes == b"output"
        assert exc.exec_result.stderr_bytes == b"error message"
        assert exc.exec_result.returncode == 1


class TestSandboxFileError:
    """Tests for SandboxFileError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = SandboxFileError("File not found")
        assert str(exc) == "File not found"
        assert exc.filepath is None

    def test_with_filepath(self) -> None:
        """Test exception with filepath."""
        exc = SandboxFileError(
            "Failed to read /tmp/data.txt",
            filepath="/tmp/data.txt",
        )
        assert exc.filepath == "/tmp/data.txt"


class TestAsyncFunctionError:
    """Tests for AsyncFunctionError."""

    def test_creation(self) -> None:
        """Test exception creation."""
        exc = AsyncFunctionError("Async functions not supported")
        assert "Async" in str(exc)
