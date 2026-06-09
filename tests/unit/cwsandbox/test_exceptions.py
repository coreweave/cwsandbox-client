# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox.exceptions module."""

from datetime import timedelta

from cwsandbox._types import ProcessResult
from cwsandbox.exceptions import (
    AsyncFunctionError,
    CWSandboxAuthenticationError,
    CWSandboxError,
    FunctionError,
    SandboxError,
    SandboxExecutionError,
    SandboxFailedError,
    SandboxFileError,
    SandboxNotFoundError,
    SandboxNotRunningError,
    SandboxStreamBackpressureError,
    SandboxStreamTruncatedError,
    SandboxTerminatedError,
    SandboxTimeoutError,
)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_cwsandbox_error_is_base_for_all_exceptions(self) -> None:
        """Test CWSandboxError is the base for all exceptions."""
        assert issubclass(SandboxError, CWSandboxError)
        assert issubclass(FunctionError, CWSandboxError)
        assert issubclass(CWSandboxAuthenticationError, CWSandboxError)

    def test_sandbox_error_is_base_for_sandbox_exceptions(self) -> None:
        """Test SandboxError is the base for sandbox-related exceptions."""
        assert issubclass(SandboxNotRunningError, SandboxError)
        assert issubclass(SandboxNotFoundError, SandboxError)
        assert issubclass(SandboxTimeoutError, SandboxError)
        assert issubclass(SandboxTerminatedError, SandboxError)
        assert issubclass(SandboxFailedError, SandboxError)
        assert issubclass(SandboxExecutionError, SandboxError)
        assert issubclass(SandboxFileError, SandboxError)

    def test_function_error_is_base_for_function_exceptions(self) -> None:
        """Test FunctionError is the base for function-related exceptions."""
        assert issubclass(AsyncFunctionError, FunctionError)

    def test_cwsandbox_error_is_exception(self) -> None:
        """Test CWSandboxError inherits from Exception."""
        assert issubclass(CWSandboxError, Exception)


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
        exec_result = ProcessResult(
            stdout="output",
            stderr="error message",
            returncode=1,
            stdout_bytes=b"output",
            stderr_bytes=b"error message",
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


class TestSandboxStreamBackpressureError:
    """Tests for SandboxStreamBackpressureError's stream_code attribute."""

    def test_is_execution_error_subclass(self) -> None:
        exc = SandboxStreamBackpressureError("too slow", stream_code="STREAM_BACKPRESSURE")
        # Existing `except SandboxExecutionError` handlers still catch it.
        assert isinstance(exc, SandboxExecutionError)

    def test_stream_code_kept_out_of_reason_namespace(self) -> None:
        """The streaming-channel code rides on .stream_code, NOT .reason (which
        is the AIP-193 ErrorInfo namespace)."""
        exc = SandboxStreamBackpressureError("too slow", stream_code="STREAM_BACKPRESSURE")
        assert exc.stream_code == "STREAM_BACKPRESSURE"
        # .reason stays None — the namespaces are deliberately distinct.
        assert exc.reason is None

    def test_does_not_accept_reason_kwarg(self) -> None:
        """STREAM_BACKPRESSURE must not be smuggled in via reason=."""
        import pytest

        with pytest.raises(TypeError):
            SandboxStreamBackpressureError("x", reason="STREAM_BACKPRESSURE")  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        exc = SandboxStreamBackpressureError("too slow")
        assert exc.stream_code is None
        assert exc.reason is None


class TestSandboxStreamTruncatedError:
    """Tests for SandboxStreamTruncatedError's stream_code attribute."""

    def test_is_execution_error_subclass(self) -> None:
        exc = SandboxStreamTruncatedError("truncated", stream_code="STREAM_TRUNCATED")
        # Existing `except SandboxExecutionError` handlers still catch it.
        assert isinstance(exc, SandboxExecutionError)

    def test_stream_code_kept_out_of_reason_namespace(self) -> None:
        """The streaming-channel code rides on .stream_code, NOT .reason (which
        is the AIP-193 ErrorInfo namespace)."""
        exc = SandboxStreamTruncatedError("truncated", stream_code="STREAM_TRUNCATED")
        assert exc.stream_code == "STREAM_TRUNCATED"
        # .reason stays None — the namespaces are deliberately distinct.
        assert exc.reason is None

    def test_does_not_accept_reason_kwarg(self) -> None:
        """STREAM_TRUNCATED must not be smuggled in via reason=."""
        import pytest

        with pytest.raises(TypeError):
            SandboxStreamTruncatedError("x", reason="STREAM_TRUNCATED")  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        exc = SandboxStreamTruncatedError("truncated")
        assert exc.stream_code is None
        assert exc.reason is None


class TestExecStreamErrorDispatch:
    """The in-band ExecStreamError code maps to the right typed exception."""

    def test_backpressure_code_maps_to_backpressure_error(self) -> None:
        from cwsandbox._error_info import STREAM_BACKPRESSURE
        from cwsandbox._sandbox import _exec_stream_error

        exc = _exec_stream_error("slow", STREAM_BACKPRESSURE)
        assert isinstance(exc, SandboxStreamBackpressureError)
        assert exc.stream_code == STREAM_BACKPRESSURE
        assert exc.reason is None

    def test_truncated_code_maps_to_truncated_error(self) -> None:
        from cwsandbox._error_info import STREAM_TRUNCATED
        from cwsandbox._sandbox import _exec_stream_error

        exc = _exec_stream_error("lost a tail", STREAM_TRUNCATED)
        assert isinstance(exc, SandboxStreamTruncatedError)
        # Caught by existing SandboxExecutionError handlers.
        assert isinstance(exc, SandboxExecutionError)
        assert exc.stream_code == STREAM_TRUNCATED
        # Streaming-channel code, not the AIP-193 reason namespace.
        assert exc.reason is None

    def test_unknown_code_stays_generic_execution_error(self) -> None:
        from cwsandbox._sandbox import _exec_stream_error

        exc = _exec_stream_error("boom", "SOMETHING_ELSE")
        # Exact type — not one of the typed streaming subclasses.
        assert type(exc) is SandboxExecutionError
        assert exc.reason == "SOMETHING_ELSE"


class TestStructuredErrorAttributes:
    """Tests for reason/metadata/retry_delay on CWSandboxError and subclasses."""

    def test_base_defaults(self) -> None:
        exc = CWSandboxError("boom")
        assert exc.reason is None
        assert exc.metadata == {}
        assert exc.retry_delay is None

    def test_base_accepts_structured_fields(self) -> None:
        exc = CWSandboxError(
            "boom",
            reason="CWSANDBOX_FILE_NOT_FOUND",
            metadata={"filepath": "/x"},
            retry_delay=timedelta(seconds=3),
        )
        assert exc.reason == "CWSANDBOX_FILE_NOT_FOUND"
        assert exc.metadata == {"filepath": "/x"}
        assert exc.retry_delay == timedelta(seconds=3)

    def test_metadata_is_copied(self) -> None:
        original = {"filepath": "/x"}
        exc = CWSandboxError("boom", metadata=original)
        original["filepath"] = "/y"
        assert exc.metadata == {"filepath": "/x"}

    def test_sandbox_not_found_carries_fields(self) -> None:
        exc = SandboxNotFoundError(
            "missing",
            sandbox_id="sb-1",
            reason="CWSANDBOX_SANDBOX_NOT_FOUND",
            metadata={"sandbox_id": "sb-1"},
        )
        assert exc.sandbox_id == "sb-1"
        assert exc.reason == "CWSANDBOX_SANDBOX_NOT_FOUND"
        assert exc.metadata == {"sandbox_id": "sb-1"}

    def test_sandbox_file_error_carries_fields(self) -> None:
        exc = SandboxFileError(
            "missing",
            filepath="/data/x.txt",
            reason="CWSANDBOX_FILE_NOT_FOUND",
            metadata={"filepath": "/data/x.txt"},
        )
        assert exc.filepath == "/data/x.txt"
        assert exc.reason == "CWSANDBOX_FILE_NOT_FOUND"
        assert exc.metadata == {"filepath": "/data/x.txt"}

    def test_sandbox_execution_error_carries_fields(self) -> None:
        exc = SandboxExecutionError(
            "exec failed",
            reason="CWSANDBOX_COMMAND_TIMEOUT",
            retry_delay=timedelta(seconds=1),
        )
        assert exc.reason == "CWSANDBOX_COMMAND_TIMEOUT"
        assert exc.retry_delay == timedelta(seconds=1)

    def test_backward_compat_no_kwargs(self) -> None:
        """Exceptions constructed without new kwargs still work."""
        assert SandboxNotRunningError("nope").reason is None
        assert SandboxTimeoutError("slow").reason is None
        assert SandboxTerminatedError("gone").reason is None
        assert SandboxFailedError("boom").reason is None
        assert CWSandboxAuthenticationError("denied").reason is None
