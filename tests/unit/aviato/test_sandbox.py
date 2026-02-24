# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Unit tests for aviato._sandbox module."""

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest

from aviato import NetworkOptions, Sandbox, SandboxDefaults
from aviato.exceptions import SandboxError, SandboxNotFoundError, SandboxNotRunningError


class MockRpcError(grpc.RpcError):
    """Mock gRPC error for testing."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class MockAioRpcError(grpc.aio.AioRpcError):
    """Mock gRPC async error for testing streaming calls."""

    def __init__(self, code: grpc.StatusCode, details: str = "") -> None:
        self._code = code
        self._details = details
        self._initial_metadata: grpc.aio.Metadata = grpc.aio.Metadata()
        self._trailing_metadata: grpc.aio.Metadata = grpc.aio.Metadata()

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def initial_metadata(self) -> grpc.aio.Metadata:
        return self._initial_metadata

    def trailing_metadata(self) -> grpc.aio.Metadata:
        return self._trailing_metadata


class MockStreamCall:
    """Mock gRPC bidirectional streaming call for testing.

    Simulates the gRPC StreamStreamCall interface with write(), done_writing(),
    and async iteration over responses. Supports both explicit write() pattern
    and request_iterator pattern.
    """

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        response_generator: Callable[[], AsyncIterator[Any]] | None = None,
        on_write: Callable[[Any], None] | None = None,
        error_on_read: Exception | None = None,
    ) -> None:
        """Initialize mock stream call.

        Args:
            responses: Fixed list of responses to yield.
            response_generator: Async generator that yields responses
                (takes priority over responses).
            on_write: Callback invoked on each request (from write or iterator).
            error_on_read: Exception to raise during iteration.
        """
        self._responses = list(responses) if responses else []
        self._response_generator = response_generator
        self._on_write = on_write
        self._error_on_read = error_on_read
        self._writes: list[Any] = []
        self._done_writing_called = False
        self._request_iterator: AsyncIterator[Any] | None = None

    def set_request_iterator(self, iterator: AsyncIterator[Any]) -> None:
        """Set the request iterator for consumption.

        In gRPC, when using request_iterator pattern, the call object
        internally consumes the iterator. This method allows tests to
        capture and process the iterator.
        """
        self._request_iterator = iterator

    async def consume_requests(self) -> None:
        """Consume all requests from the iterator and call on_write for each."""
        if self._request_iterator:
            async for request in self._request_iterator:
                self._writes.append(request)
                if self._on_write:
                    self._on_write(request)

    async def write(self, request: Any) -> None:
        """Record write and call optional callback."""
        self._writes.append(request)
        if self._on_write:
            self._on_write(request)

    async def done_writing(self) -> None:
        """Mark that writing is complete."""
        self._done_writing_called = True

    def __aiter__(self) -> "MockStreamCall":
        self._iter_index = 0
        return self

    async def __anext__(self) -> Any:
        # If we have a request iterator, consume it first to process requests
        if self._request_iterator and not hasattr(self, "_requests_consumed"):
            self._requests_consumed = True
            await self.consume_requests()

        if self._error_on_read:
            raise self._error_on_read
        if self._response_generator:
            # Use generator if provided
            if not hasattr(self, "_gen_instance"):
                self._gen_instance = self._response_generator()
            return await self._gen_instance.__anext__()
        # Use fixed responses list
        if self._iter_index >= len(self._responses):
            raise StopAsyncIteration
        response = self._responses[self._iter_index]
        self._iter_index += 1
        return response


class MockBidirectionalStreamCall:
    """Mock gRPC bidirectional streaming call that properly interleaves requests and responses.

    Unlike MockStreamCall, this class simulates true bidirectional streaming where
    responses are returned immediately while requests are consumed in the background.
    This is needed for testing stdin ready signal handling where the request generator
    waits for a ready response before sending stdin data.
    """

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        response_generator: Callable[[], AsyncIterator[Any]] | None = None,
        on_write: Callable[[Any], None] | None = None,
        error_on_read: Exception | None = None,
    ) -> None:
        """Initialize mock bidirectional stream call.

        Args:
            responses: List of responses to yield.
            response_generator: Async generator that yields responses (takes priority).
            on_write: Callback invoked on each request.
            error_on_read: Exception to raise during iteration.
        """
        import asyncio

        self._responses = list(responses) if responses else []
        self._response_generator = response_generator
        self._on_write = on_write
        self._error_on_read = error_on_read
        self._writes: list[Any] = []
        self._request_iterator: AsyncIterator[Any] | None = None
        self._request_task: asyncio.Task[None] | None = None
        self._iter_index = 0
        self._request_error: Exception | None = None

    def set_request_iterator(self, iterator: AsyncIterator[Any]) -> None:
        """Set the request iterator and start consuming it in background."""
        import asyncio

        self._request_iterator = iterator
        # Start consuming requests in background - don't block on it
        self._request_task = asyncio.create_task(self._consume_requests_background())

    async def _consume_requests_background(self) -> None:
        """Consume requests in background without blocking response iteration."""
        if self._request_iterator:
            try:
                async for request in self._request_iterator:
                    self._writes.append(request)
                    if self._on_write:
                        self._on_write(request)
            except Exception as e:
                # Request generator may raise on timeout/cancel - store for debugging
                self._request_error = e

    async def wait_for_requests(self) -> None:
        """Wait for all requests to be consumed. Use after process.result()."""
        import asyncio

        if self._request_task:
            # Wait for the background task to complete
            await asyncio.wait_for(self._request_task, timeout=5.0)

    def __aiter__(self) -> "MockBidirectionalStreamCall":
        return self

    async def __anext__(self) -> Any:
        if self._error_on_read:
            raise self._error_on_read
        if self._response_generator:
            # Use generator if provided
            if not hasattr(self, "_gen_instance"):
                self._gen_instance = self._response_generator()
            return await self._gen_instance.__anext__()
        if self._iter_index >= len(self._responses):
            raise StopAsyncIteration
        response = self._responses[self._iter_index]
        self._iter_index += 1
        return response


def create_mock_channel_and_stub(
    mock_call: MockStreamCall,
) -> tuple[MagicMock, MagicMock]:
    """Create mock channel and stub with the given mock call.

    Returns:
        Tuple of (mock_channel, mock_stub).
    """
    mock_channel = MagicMock()
    mock_channel.close = AsyncMock()
    mock_channel.channel_ready = AsyncMock()

    def stream_exec_side_effect(
        request_iterator: Any = None, timeout: float | None = None, metadata: Any = None
    ) -> MockStreamCall:
        # Capture the request iterator for processing
        if request_iterator is not None:
            mock_call.set_request_iterator(request_iterator)
        return mock_call

    mock_stub = MagicMock()
    mock_stub.StreamExec = MagicMock(side_effect=stream_exec_side_effect)

    return mock_channel, mock_stub


def create_mock_channel_and_stub_bidirectional(
    mock_call: MockBidirectionalStreamCall,
) -> tuple[MagicMock, MagicMock]:
    """Create mock channel and stub with bidirectional mock call.

    This version uses MockBidirectionalStreamCall which properly
    interleaves requests and responses for stdin ready signal testing.

    Returns:
        Tuple of (mock_channel, mock_stub).
    """
    mock_channel = MagicMock()
    mock_channel.close = AsyncMock()
    mock_channel.channel_ready = AsyncMock()

    def stream_exec_side_effect(
        request_iterator: Any = None, timeout: float | None = None, metadata: Any = None
    ) -> MockBidirectionalStreamCall:
        # Capture the request iterator for background processing
        if request_iterator is not None:
            mock_call.set_request_iterator(request_iterator)
        return mock_call

    mock_stub = MagicMock()
    mock_stub.StreamExec = MagicMock(side_effect=stream_exec_side_effect)

    return mock_channel, mock_stub


class TestSandboxRun:
    """Tests for Sandbox.run factory method."""

    def test_run_uses_defaults_without_args(self) -> None:
        """Test Sandbox.run uses default command when no args provided."""
        mock_ref = MagicMock()
        mock_ref.result = MagicMock(return_value=None)
        with patch.object(Sandbox, "start", return_value=mock_ref) as mock_start:
            sandbox = Sandbox.run()
            mock_start.assert_called_once()
            mock_ref.result.assert_called_once()
            # Default command is "tail" with args ["-f", "/dev/null"]
            assert sandbox._command == "tail"
            assert sandbox._args == ["-f", "/dev/null"]

    def test_run_calls_start(self) -> None:
        """Test Sandbox.run calls start().result() on the sandbox."""
        mock_ref = MagicMock()
        mock_ref.result = MagicMock(return_value=None)
        with patch.object(Sandbox, "start", return_value=mock_ref) as mock_start:
            Sandbox.run("echo", "hello", "world")
            mock_start.assert_called_once()
            mock_ref.result.assert_called_once()

    def test_run_has_sandbox_id_after_return(self) -> None:
        """Test Sandbox.run() has sandbox_id set after return."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "run-sandbox-id"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.sandbox_id == "run-sandbox-id"


class TestSandboxExec:
    """Tests for Sandbox.exec method."""

    def test_exec_without_start_auto_starts(self) -> None:
        """Test exec auto-starts sandbox via _ensure_started_async."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-id"

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        mock_call = MockStreamCall(responses=[exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            process = sandbox.exec(["echo", "test"])
            result = process.result()
            assert result.returncode == 0
            assert sandbox.sandbox_id == "auto-start-id"

    def test_exec_empty_command_raises_error(self) -> None:
        """Test exec with empty command raises ValueError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        with pytest.raises(ValueError, match="Command cannot be empty"):
            sandbox.exec([])

    def test_exec_check_raises_on_nonzero_returncode(self) -> None:
        """Test exec with check=True raises SandboxExecutionError on failure."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        from aviato.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Create mock responses: stderr output, then exit with code 127
        stderr_response = MagicMock()
        stderr_response.HasField = lambda field: field == "output"
        stderr_response.output.data = b"command not found"
        stderr_response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 127

        mock_call = MockStreamCall(responses=[stderr_response, exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["nonexistent"], check=True)
            with pytest.raises(SandboxExecutionError, match="exit code 127"):
                process.result()

    def test_exec_check_false_returns_result_on_failure(self) -> None:
        """Test exec with check=False returns result even on failure."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Create mock responses: stderr output, then exit with code 1
        stderr_response = MagicMock()
        stderr_response.HasField = lambda field: field == "output"
        stderr_response.output.data = b"error"
        stderr_response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 1

        mock_call = MockStreamCall(responses=[stderr_response, exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["failing-cmd"], check=False)
            result = process.result()

            assert result.returncode == 1

    def test_exec_streams_stdout_to_queue(self) -> None:
        """Test exec() streams stdout data to the queue as it arrives."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Create mock responses: multiple stdout chunks, then exit
        responses = []
        for chunk in [b"line1\n", b"line2\n", b"line3\n"]:
            response = MagicMock()
            response.HasField = lambda field, c=chunk: field == "output"
            response.output.data = chunk
            response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDOUT
            responses.append(response)

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0
        responses.append(exit_response)

        mock_call = MockStreamCall(responses=responses)
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        expected_metadata = (("authorization", "Bearer test-key"),)
        sandbox._auth_metadata = expected_metadata

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["echo", "test"])

            # Collect all stdout lines by iterating the stream
            lines = list(process.stdout)
            assert lines == ["line1\n", "line2\n", "line3\n"]

            # Result should have combined output
            result = process.result()
            assert result.stdout == "line1\nline2\nline3\n"
            assert result.returncode == 0

            call_kwargs = mock_stub.StreamExec.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    def test_exec_handles_stream_error(self) -> None:
        """Test exec() handles stream errors correctly."""

        from aviato.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Create mock response: error
        error_response = MagicMock()
        error_response.HasField = lambda field: field == "error"
        error_response.error.message = "Connection lost"

        mock_call = MockStreamCall(responses=[error_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["failing-cmd"])
            with pytest.raises(SandboxExecutionError, match="Connection lost"):
                process.result()

    def test_exec_cwd_empty_string_raises_error(self) -> None:
        """Test exec with empty cwd raises ValueError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"

        with pytest.raises(ValueError, match="cwd cannot be empty string"):
            sandbox.exec(["ls"], cwd="")

    def test_exec_cwd_relative_path_raises_error(self) -> None:
        """Test exec with relative cwd raises ValueError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"

        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            sandbox.exec(["ls"], cwd="relative/path")

    def test_exec_cwd_wraps_command_with_shell(self) -> None:
        """Test exec with cwd wraps command in shell wrapper."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        captured_command: list[str] = []

        def capture_write(request: Any) -> None:
            if hasattr(request, "init") and request.init.command:
                captured_command.extend(request.init.command)

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        mock_call = MockStreamCall(responses=[exit_response], on_write=capture_write)
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["ls", "-la"], cwd="/app")
            process.result()

        # Verify shell wrapping format
        assert captured_command == [
            "/bin/sh",
            "-c",
            "cd /app && exec ls -la",
        ]

    def test_exec_cwd_preserves_original_command_in_result(self) -> None:
        """Test exec with cwd preserves original command in ProcessResult."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        mock_call = MockStreamCall(responses=[exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["echo", "hello"], cwd="/app")
            result = process.result()

        # Original command should be preserved, not the wrapped command
        assert result.command == ["echo", "hello"]

    def test_exec_cwd_escapes_special_characters(self) -> None:
        """Test exec with cwd escapes special characters in path."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        captured_command: list[str] = []

        def capture_write(request: Any) -> None:
            if hasattr(request, "init") and request.init.command:
                captured_command.extend(request.init.command)

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        mock_call = MockStreamCall(responses=[exit_response], on_write=capture_write)
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["ls"], cwd="/path with spaces/and$special")
            process.result()

        # Verify special characters are escaped
        assert captured_command[0] == "/bin/sh"
        assert captured_command[1] == "-c"
        # The path should be properly quoted
        assert "'/path with spaces/and$special'" in captured_command[2]

    def test_exec_cwd_none_does_not_wrap_command(self) -> None:
        """Test exec without cwd does not wrap command."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        captured_command: list[str] = []

        def capture_write(request: Any) -> None:
            if hasattr(request, "init") and request.init.command:
                captured_command.extend(request.init.command)

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        mock_call = MockStreamCall(responses=[exit_response], on_write=capture_write)
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["ls", "-la"])
            process.result()

        # Without cwd, command should not be wrapped
        assert captured_command == ["ls", "-la"]


class TestExecCwdHelperFunctions:
    """Tests for cwd helper functions."""

    def test_validate_cwd_none_passes(self) -> None:
        """Test _validate_cwd allows None."""
        from aviato._sandbox import _validate_cwd

        _validate_cwd(None)  # Should not raise

    def test_validate_cwd_absolute_path_passes(self) -> None:
        """Test _validate_cwd allows absolute paths."""
        from aviato._sandbox import _validate_cwd

        _validate_cwd("/app")
        _validate_cwd("/var/log/app")
        _validate_cwd("/")

    def test_validate_cwd_empty_string_raises(self) -> None:
        """Test _validate_cwd raises on empty string."""
        from aviato._sandbox import _validate_cwd

        with pytest.raises(ValueError, match="cwd cannot be empty string"):
            _validate_cwd("")

    def test_validate_cwd_relative_path_raises(self) -> None:
        """Test _validate_cwd raises on relative paths."""
        from aviato._sandbox import _validate_cwd

        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            _validate_cwd("relative")
        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            _validate_cwd("./relative")
        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            _validate_cwd("../parent")

    def test_wrap_command_with_cwd_basic(self) -> None:
        """Test _wrap_command_with_cwd creates correct shell wrapper."""
        from aviato._sandbox import _wrap_command_with_cwd

        result = _wrap_command_with_cwd(["ls", "-la"], "/app")
        assert result == ["/bin/sh", "-c", "cd /app && exec ls -la"]

    def test_wrap_command_with_cwd_escapes_path(self) -> None:
        """Test _wrap_command_with_cwd escapes special characters in path."""
        from aviato._sandbox import _wrap_command_with_cwd

        result = _wrap_command_with_cwd(["ls"], "/path with spaces")
        assert result[0] == "/bin/sh"
        assert result[1] == "-c"
        # Path should be quoted
        assert "'/path with spaces'" in result[2]

    def test_wrap_command_with_cwd_escapes_command_args(self) -> None:
        """Test _wrap_command_with_cwd escapes command arguments."""
        from aviato._sandbox import _wrap_command_with_cwd

        result = _wrap_command_with_cwd(["echo", "hello world"], "/app")
        # Arguments with spaces should be quoted
        assert "'hello world'" in result[2]


class TestSandboxAuth:
    """Tests for Sandbox authentication."""

    @pytest.mark.asyncio
    async def test_resolves_auth_metadata(self, mock_aviato_api_key: str) -> None:
        """Test _ensure_client() resolves auth metadata and stores it."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with (
            patch("aviato._sandbox.create_channel") as mock_create_channel,
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub") as mock_stub_class,
            patch("aviato._sandbox.resolve_auth_metadata") as mock_resolve,
        ):
            mock_resolve.return_value = (("authorization", "Bearer test-api-key"),)
            await sandbox._ensure_client()

            mock_resolve.assert_called_once()
            mock_create_channel.assert_called_once()
            # Verify no interceptors arg passed to create_channel
            call_args = mock_create_channel.call_args
            assert len(call_args[0]) == 2  # only target, is_secure
            assert sandbox._auth_metadata == (("authorization", "Bearer test-api-key"),)
            mock_stub_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_metadata_passed_to_start_rpc(self, mock_aviato_api_key: str) -> None:
        """Test auth metadata is passed to the Start RPC call."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_id = "test-id"
        mock_response.service_address = ""
        mock_response.exposed_ports = []
        mock_response.applied_ingress_mode = ""
        mock_response.applied_egress_mode = ""
        mock_stub.Start = AsyncMock(return_value=mock_response)

        sandbox._channel = MagicMock()
        sandbox._stub = mock_stub
        sandbox._auth_metadata = (("authorization", "Bearer test-api-key"),)

        await sandbox._start_async()

        mock_stub.Start.assert_called_once()
        call_kwargs = mock_stub.Start.call_args[1]
        assert call_kwargs["metadata"] == (("authorization", "Bearer test-api-key"),)


class TestSandboxCleanup:
    """Tests for Sandbox cleanup and resource warnings."""

    def test_del_warns_if_sandbox_not_stopped(self) -> None:
        """Test __del__ emits ResourceWarning if sandbox was started but not stopped."""
        import warnings

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-sandbox-id"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del sandbox

            assert len(w) == 1
            assert issubclass(w[0].category, ResourceWarning)
            assert "was not stopped" in str(w[0].message)

    def test_del_no_warning_if_sandbox_never_started(self) -> None:
        """Test __del__ does not warn if sandbox was never started."""
        import warnings

        sandbox = Sandbox(command="sleep", args=["infinity"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del sandbox

            resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(resource_warnings) == 0

    def test_sync_context_manager_calls_stop_on_exit(self) -> None:
        """Test sync context manager calls stop() when exiting if sandbox was started."""
        import concurrent.futures

        from aviato import OperationRef

        sandbox = Sandbox(command="sleep", args=["infinity"])
        stop_called = False

        def mock_start() -> OperationRef[None]:
            sandbox._sandbox_id = "test-sandbox-id"
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return OperationRef(future)

        def mock_stop(**kwargs: object) -> OperationRef[bool]:
            nonlocal stop_called
            stop_called = True
            future: concurrent.futures.Future[bool] = concurrent.futures.Future()
            future.set_result(True)
            return OperationRef(future)

        sandbox.start = mock_start  # type: ignore[method-assign]
        sandbox.stop = mock_stop  # type: ignore[method-assign]

        with sandbox:
            pass

        assert stop_called

    def test_enter_starts_if_not_started(self) -> None:
        """Test __enter__ starts sandbox if not already started."""
        import concurrent.futures

        from aviato import OperationRef

        sandbox = Sandbox(command="sleep", args=["infinity"])
        start_called = False

        def mock_start() -> OperationRef[None]:
            nonlocal start_called
            start_called = True
            sandbox._sandbox_id = "enter-sandbox-id"
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return OperationRef(future)

        def mock_stop(**kwargs: object) -> OperationRef[None]:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return OperationRef(future)

        sandbox.start = mock_start  # type: ignore[method-assign]
        sandbox.stop = mock_stop  # type: ignore[method-assign]

        with sandbox:
            assert start_called
            assert sandbox.sandbox_id == "enter-sandbox-id"


class TestSandboxGetStatus:
    """Tests for Sandbox.get_status method."""

    def test_get_status_raises_without_start(self) -> None:
        """Test get_status raises SandboxNotRunningError if not started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(SandboxNotRunningError, match="has not been started"):
            sandbox.get_status()


class TestSandboxWait:
    """Tests for Sandbox.wait method (wait until RUNNING)."""

    def test_wait_raises_on_failed(self) -> None:
        """Test wait raises SandboxFailedError when sandbox fails."""
        from coreweave.aviato.v1beta1 import atc_pb2

        from aviato.exceptions import SandboxFailedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_FAILED
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        with pytest.raises(SandboxFailedError, match="failed"):
            sandbox.wait()

    def test_wait_raises_on_terminated_by_default(self) -> None:
        """Test wait raises SandboxTerminatedError when terminated."""
        from coreweave.aviato.v1beta1 import atc_pb2

        from aviato.exceptions import SandboxTerminatedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_TERMINATED
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        with pytest.raises(SandboxTerminatedError, match="terminated"):
            sandbox.wait()

    def test_wait_auto_starts_unstarted_sandbox(self) -> None:
        """Test wait() triggers auto-start on unstarted sandbox."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="echo", args=["hello"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-wait-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.tower_id = "tower-1"
        mock_get_response.runway_id = "runway-1"
        mock_get_response.tower_group_id = None
        mock_get_response.started_at_time = None

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)
            sandbox._stub.Get = AsyncMock(return_value=mock_get_response)

            assert sandbox.sandbox_id is None
            sandbox.wait()

            assert sandbox.sandbox_id == "auto-start-wait-id"
            sandbox._stub.Start.assert_called_once()


class TestSandboxAutoStartFileOps:
    """Tests for auto-start behavior on read_file and write_file."""

    def test_read_file_auto_starts_unstarted_sandbox(self) -> None:
        """Test read_file() triggers auto-start on unstarted sandbox."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-read-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.tower_id = "tower-1"
        mock_get_response.runway_id = "runway-1"
        mock_get_response.tower_group_id = None
        mock_get_response.started_at_time = None

        mock_read_response = MagicMock()
        mock_read_response.success = True
        mock_read_response.file_contents = b"file data"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)
            sandbox._stub.Get = AsyncMock(return_value=mock_get_response)
            sandbox._stub.RetrieveFile = AsyncMock(return_value=mock_read_response)

            assert sandbox.sandbox_id is None
            data = sandbox.read_file("/test.txt").result()

            assert sandbox.sandbox_id == "auto-start-read-id"
            assert data == b"file data"
            sandbox._stub.Start.assert_called_once()
            call_kwargs = sandbox._stub.RetrieveFile.call_args[1]
            assert "metadata" in call_kwargs

    def test_write_file_auto_starts_unstarted_sandbox(self) -> None:
        """Test write_file() triggers auto-start on unstarted sandbox."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-write-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.tower_id = "tower-1"
        mock_get_response.runway_id = "runway-1"
        mock_get_response.tower_group_id = None
        mock_get_response.started_at_time = None

        mock_write_response = MagicMock()
        mock_write_response.success = True

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)
            sandbox._stub.Get = AsyncMock(return_value=mock_get_response)
            sandbox._stub.AddFile = AsyncMock(return_value=mock_write_response)

            assert sandbox.sandbox_id is None
            sandbox.write_file("/test.txt", b"hello").result()

            assert sandbox.sandbox_id == "auto-start-write-id"
            sandbox._stub.Start.assert_called_once()
            call_kwargs = sandbox._stub.AddFile.call_args[1]
            assert "metadata" in call_kwargs


class TestSandboxWaitUntilComplete:
    """Tests for Sandbox.wait_until_complete method."""

    def test_wait_until_complete_auto_starts(self) -> None:
        """Test wait_until_complete auto-starts via _ensure_started_async."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="echo", args=["hello"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-complete-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_COMPLETED

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)
            sandbox._stub.Get = AsyncMock(return_value=mock_get_response)

            sandbox.wait_until_complete().result()

            assert sandbox.sandbox_id == "auto-start-complete-id"
            assert sandbox.returncode == 0

    def test_wait_until_complete_no_raise_on_terminated_when_disabled(self) -> None:
        """Test wait_until_complete returns normally when raise_on_termination=False."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_TERMINATED
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        sandbox.wait_until_complete(raise_on_termination=False).result()

        assert sandbox.returncode is None


class TestSandboxStart:
    """Tests for Sandbox.start method."""

    def test_start_returns_operation_ref(self) -> None:
        """Test start returns OperationRef[None]."""
        from aviato import OperationRef

        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "new-sandbox-id"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            ref = sandbox.start()
            assert isinstance(ref, OperationRef)
            result = ref.result()
            assert result is None

    def test_start_sets_sandbox_id(self) -> None:
        """Test start sets the sandbox ID (does NOT wait for RUNNING)."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "new-sandbox-id"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.sandbox_id == "new-sandbox-id"

    def test_start_sends_correct_request(self) -> None:
        """Test start sends request with correct parameters."""
        sandbox = Sandbox(
            command="python",
            args=["-c", "print('hello')"],
            container_image="python:3.12",
            tags=["test-tag"],
            max_lifetime_seconds=3600,
        )

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            start_call = sandbox._stub.Start.call_args[0][0]
            assert start_call.command == "python"
            assert start_call.args == ["-c", "print('hello')"]
            assert start_call.container_image == "python:3.12"
            assert start_call.tags == ["test-tag"]
            assert start_call.max_lifetime_seconds == 3600

    def test_start_raises_not_running_on_canceled(self) -> None:
        """Test start raises SandboxNotRunningError when request is cancelled."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        # Create a mock gRPC error
        mock_error = MockRpcError(grpc.StatusCode.CANCELLED, "request cancelled")

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(side_effect=mock_error)

            with pytest.raises(SandboxNotRunningError, match="was cancelled"):
                sandbox.start().result()


class TestSandboxWaitForRunning:
    """Tests for wait() which waits until RUNNING status."""

    def test_wait_raises_on_failed_status(self) -> None:
        """Test wait raises SandboxFailedError when sandbox fails to start."""
        from coreweave.aviato.v1beta1 import atc_pb2

        from aviato.exceptions import SandboxFailedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "failing-sandbox-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_FAILED

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Get = AsyncMock(return_value=mock_get_response)

            with pytest.raises(SandboxFailedError, match="failed to start"):
                sandbox.wait()

    def test_wait_handles_fast_completion(self) -> None:
        """Test wait handles sandbox that completes during startup."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="echo", args=["hello"])
        sandbox._sandbox_id = "fast-sandbox-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.tower_id = "tower-1"
        mock_get_response.runway_id = "runway-1"
        mock_get_response.tower_group_id = None
        mock_get_response.started_at_time = None

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Get = AsyncMock(return_value=mock_get_response)

            sandbox.wait()

            assert sandbox.returncode == 0


class TestSandboxEnvironmentVariables:
    """Tests for environment variables merging behavior."""

    def test_env_vars_inherit_from_defaults(self) -> None:
        """Test sandbox inherits environment variables from defaults."""
        defaults = SandboxDefaults(
            environment_variables={
                "LOG_LEVEL": "info",
                "REGION": "us-west",
            }
        )

        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            defaults=defaults,
        )

        assert sandbox._environment_variables == {
            "LOG_LEVEL": "info",
            "REGION": "us-west",
        }

    def test_env_vars_merge_with_defaults(self) -> None:
        """Test sandbox-specific env vars merge with defaults."""
        defaults = SandboxDefaults(
            environment_variables={
                "LOG_LEVEL": "info",
                "REGION": "us-west",
            }
        )

        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            defaults=defaults,
            environment_variables={
                "LOG_LEVEL": "debug",  # Override default
                "USER_ID": "123",  # Add new
            },
        )

        assert sandbox._environment_variables == {
            "LOG_LEVEL": "debug",  # Overridden
            "REGION": "us-west",  # From defaults
            "USER_ID": "123",  # Added
        }

    def test_env_vars_empty_when_no_defaults_or_params(self) -> None:
        """Test env vars is empty dict when nothing specified."""
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
        )

        assert sandbox._environment_variables == {}

    def test_env_vars_only_from_params_when_no_defaults(self) -> None:
        """Test env vars uses only params when no defaults."""
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            environment_variables={"API_KEY": "secret"},
        )

        assert sandbox._environment_variables == {"API_KEY": "secret"}


class TestSandboxStop:
    """Tests for Sandbox.stop method."""

    def test_stop_raises_on_backend_failure(self) -> None:
        """Test stop raises SandboxError when backend reports failure."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error_message = "backend error"
        sandbox._stub.Stop = AsyncMock(return_value=mock_response)
        sandbox._channel.close = AsyncMock()

        with pytest.raises(SandboxError, match="Failed to stop sandbox"):
            sandbox.stop().result()

    def test_stop_is_idempotent(self) -> None:
        """Test stop() is idempotent - safe to call multiple times."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        # Calling stop on never-started sandbox returns None (no-op)
        result = sandbox.stop().result()
        assert result is None

        # Calling stop again is also safe
        result = sandbox.stop().result()
        assert result is None

    def test_stop_missing_ok_true_suppresses_not_found(self) -> None:
        """Test stop(missing_ok=True) suppresses SandboxNotFoundError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        sandbox._stub.Stop = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")
        )
        sandbox._channel.close = AsyncMock()

        # Should not raise, returns None
        result = sandbox.stop(missing_ok=True).result()
        assert result is None

    def test_stop_missing_ok_false_raises_not_found(self) -> None:
        """Test stop(missing_ok=False) raises SandboxNotFoundError."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        sandbox._stub.Stop = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")
        )
        sandbox._channel.close = AsyncMock()

        with pytest.raises(SandboxNotFoundError):
            sandbox.stop().result()

    def test_stop_on_never_started_is_noop(self) -> None:
        """Test stop() on a never-started sandbox is a no-op."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.sandbox_id is None

        result = sandbox.stop().result()
        assert result is None

    def test_stop_waits_for_inflight_start(self) -> None:
        """Test stop() acquires _start_lock and waits for in-flight start()."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        expected_metadata = (("authorization", "Bearer test-key"),)
        sandbox._auth_metadata = expected_metadata

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "race-sandbox-id"

        mock_stop_response = MagicMock()
        mock_stop_response.success = True

        mock_stub = MagicMock()
        mock_stub.Start = AsyncMock(return_value=mock_start_response)
        mock_stub.Stop = AsyncMock(return_value=mock_stop_response)

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._channel.close = AsyncMock()
            sandbox._stub = mock_stub

            # Start the sandbox first so stop has something to stop
            sandbox.start().result()
            assert sandbox.sandbox_id == "race-sandbox-id"

            # Now stop should proceed normally
            sandbox.stop().result()

        # Assert after the with block since stop() clears _stub
        mock_stub.Stop.assert_called_once()
        call_kwargs = mock_stub.Stop.call_args[1]
        assert call_kwargs["metadata"] == expected_metadata


class TestSandboxTimeouts:
    """Tests for Sandbox timeout behavior."""

    def test_exec_respects_timeout_seconds(self) -> None:
        """Test exec() raises SandboxTimeoutError when timeout_seconds is exceeded."""
        from aviato.exceptions import SandboxTimeoutError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Mock call that raises DEADLINE_EXCEEDED when iterating
        mock_call = MockStreamCall(
            error_on_read=MockAioRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded")
        )
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["sleep", "10"], timeout_seconds=0.1)
            with pytest.raises(SandboxTimeoutError):
                process.result()


class TestSandboxKwargsValidation:
    """Tests for kwargs validation in Sandbox methods."""

    def test_init_with_valid_kwargs(self) -> None:
        """Test Sandbox.__init__ accepts valid kwargs."""
        net_opts = NetworkOptions(ingress_mode="public")
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            resources={"cpu": "100m", "memory": "128Mi"},
            ports=[{"container_port": 8080}],
            network=net_opts,
            max_timeout_seconds=60,
            environment_variables={"TEST_ENV_VAR": "test-value"},
        )
        assert sandbox._start_kwargs["resources"] == {"cpu": "100m", "memory": "128Mi"}
        assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]
        assert sandbox._start_kwargs["network"] == net_opts
        assert sandbox._start_kwargs["max_timeout_seconds"] == 60
        assert sandbox._environment_variables == {"TEST_ENV_VAR": "test-value"}

    def test_init_with_invalid_kwargs(self) -> None:
        """Test Sandbox.__init__ rejects invalid kwargs."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Sandbox(
                command="echo",
                args=["hello"],
                invalid_param="value",
                another_bad_param=42,
            )

    def test_init_with_mixed_valid_invalid_kwargs(self) -> None:
        """Test Sandbox.__init__ rejects if any kwargs are invalid."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Sandbox(
                command="echo",
                args=["hello"],
                resources={"cpu": "100m"},
                invalid_param="value",
            )

    def test_run_with_valid_kwargs(self) -> None:
        """Test Sandbox.run accepts valid kwargs."""
        with patch.object(Sandbox, "start") as mock_start:
            sandbox = Sandbox.run(
                "echo",
                "hello",
                resources={"cpu": "100m"},
                ports=[{"container_port": 8080}],
            )
            mock_start.assert_called_once()
            assert sandbox._start_kwargs["resources"] == {"cpu": "100m"}
            assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]

    def test_run_with_invalid_kwargs(self) -> None:
        """Test Sandbox.run rejects invalid kwargs."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Sandbox.run(
                "echo",
                "hello",
                invalid_param="value",
            )


class TestSandboxList:
    """Tests for Sandbox.list class method."""

    @pytest.mark.asyncio
    async def test_list_returns_sandbox_instances(self, mock_aviato_api_key: str) -> None:
        """Test list() returns list of Sandbox instances."""
        from coreweave.aviato.v1beta1 import atc_pb2
        from google.protobuf import timestamp_pb2

        mock_sandbox_info = atc_pb2.SandboxInfo(
            sandbox_id="test-123",
            sandbox_status=atc_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            tower_id="tower-1",
            tower_group_id="group-1",
            runway_id="runway-1",
        )

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(
            return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
        )

        with (
            patch("aviato._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandboxes = await Sandbox.list(tags=["test-tag"])

            assert len(sandboxes) == 1
            assert isinstance(sandboxes[0], Sandbox)
            assert sandboxes[0].sandbox_id == "test-123"
            assert sandboxes[0].status == "running"
            call_kwargs = mock_stub.List.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, mock_aviato_api_key: str) -> None:
        """Test list() passes status filter to request."""
        from coreweave.aviato.v1beta1 import atc_pb2

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(return_value=atc_pb2.ListSandboxesResponse(sandboxes=[]))

        with (
            patch("aviato._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            await Sandbox.list(status="running")

            call_args = mock_stub.List.call_args[0][0]
            assert call_args.status == atc_pb2.SANDBOX_STATUS_RUNNING
            call_kwargs = mock_stub.List.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_list_with_invalid_status_raises(self, mock_aviato_api_key: str) -> None:
        """Test list() raises ValueError for invalid status."""
        with pytest.raises(ValueError, match="not a valid SandboxStatus"):
            await Sandbox.list(status="invalid_status")

    @pytest.mark.asyncio
    async def test_list_empty_result(self, mock_aviato_api_key: str) -> None:
        """Test list() returns empty list when no sandboxes match."""
        from coreweave.aviato.v1beta1 import atc_pb2

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(return_value=atc_pb2.ListSandboxesResponse(sandboxes=[]))

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandboxes = await Sandbox.list(tags=["nonexistent"])
            assert sandboxes == []


class TestSandboxFromId:
    """Tests for Sandbox.from_id class method."""

    @pytest.mark.asyncio
    async def test_from_id_returns_sandbox_instance(self, mock_aviato_api_key: str) -> None:
        """Test from_id() returns a Sandbox instance."""
        from coreweave.aviato.v1beta1 import atc_pb2
        from google.protobuf import timestamp_pb2

        mock_response = atc_pb2.GetSandboxResponse(
            sandbox_id="test-123",
            sandbox_status=atc_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            tower_id="tower-1",
            tower_group_id="group-1",
            runway_id="runway-1",
        )

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandbox = await Sandbox.from_id("test-123")

            assert isinstance(sandbox, Sandbox)
            assert sandbox.sandbox_id == "test-123"
            assert sandbox.status == "running"
            assert sandbox.tower_id == "tower-1"
            call_kwargs = mock_stub.Get.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_from_id_raises_not_found(self, mock_aviato_api_key: str) -> None:
        """Test from_id() raises SandboxNotFoundError for non-existent sandbox."""

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found"))

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            with pytest.raises(SandboxNotFoundError, match="not found"):
                await Sandbox.from_id("nonexistent-id")


class TestSandboxDeleteClassMethod:
    """Tests for Sandbox.delete class method."""

    @pytest.mark.asyncio
    async def test_delete_returns_none_on_success(self, mock_aviato_api_key: str) -> None:
        """Test delete() returns None when deletion succeeds."""
        from coreweave.aviato.v1beta1 import atc_pb2

        mock_response = atc_pb2.DeleteSandboxResponse(success=True, error_message="")

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Delete = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            result = await Sandbox.delete("test-123")

            assert result is None
            call_kwargs = mock_stub.Delete.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_delete_raises_not_found_by_default(self, mock_aviato_api_key: str) -> None:
        """Test delete() raises SandboxNotFoundError when sandbox doesn't exist."""

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Delete = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")
        )

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            with pytest.raises(SandboxNotFoundError):
                await Sandbox.delete("nonexistent-id")

    @pytest.mark.asyncio
    async def test_delete_missing_ok_suppresses_not_found(self, mock_aviato_api_key: str) -> None:
        """Test delete(missing_ok=True) returns None instead of raising."""
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Delete = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")
        )

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            result = await Sandbox.delete("nonexistent-id", missing_ok=True)
            assert result is None


class TestSandboxServiceAddressAndExposedPorts:
    """Tests for service_address and exposed_ports properties."""

    def test_service_address_none_before_start(self) -> None:
        """Test service_address is None before sandbox is started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.service_address is None

    def test_exposed_ports_none_before_start(self) -> None:
        """Test exposed_ports is None before sandbox is started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.exposed_ports is None

    def test_start_captures_service_address(self) -> None:
        """Test start() captures service_address from response."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = "166.19.9.70:8080"
        mock_start_response.exposed_ports = []

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.service_address == "166.19.9.70:8080"

    def test_start_treats_empty_service_address_as_none(self) -> None:
        """Test start() treats empty string service_address as None."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.service_address is None

    def test_start_captures_exposed_ports(self) -> None:
        """Test start() captures exposed_ports from response."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_port1 = MagicMock()
        mock_port1.container_port = 8080
        mock_port1.name = "http"

        mock_port2 = MagicMock()
        mock_port2.container_port = 22
        mock_port2.name = "ssh"

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = "166.19.9.70:8080"
        mock_start_response.exposed_ports = [mock_port1, mock_port2]

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.exposed_ports == ((8080, "http"), (22, "ssh"))

    def test_start_empty_exposed_ports_is_none(self) -> None:
        """Test start() returns None for empty exposed_ports list."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.exposed_ports is None

    @pytest.mark.asyncio
    async def test_from_id_has_none_for_service_address(self, mock_aviato_api_key: str) -> None:
        """Test from_id() returns sandbox with None service_address."""
        from coreweave.aviato.v1beta1 import atc_pb2
        from google.protobuf import timestamp_pb2

        mock_response = atc_pb2.GetSandboxResponse(
            sandbox_id="test-123",
            sandbox_status=atc_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            tower_id="tower-1",
            tower_group_id="group-1",
            runway_id="runway-1",
        )

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandbox = await Sandbox.from_id("test-123")

            assert sandbox.service_address is None
            assert sandbox.exposed_ports is None


class TestSandboxRunwayAndTowerIds:
    """Tests for runway_ids and tower_ids parameters."""

    def test_runway_ids_stored_on_sandbox(self) -> None:
        """Test runway_ids are stored on sandbox instance."""
        sandbox = Sandbox(runway_ids=["runway-1", "runway-2"])
        assert sandbox._runway_ids == ["runway-1", "runway-2"]

    def test_tower_ids_stored_on_sandbox(self) -> None:
        """Test tower_ids are stored on sandbox instance."""
        sandbox = Sandbox(tower_ids=["tower-1", "tower-2"])
        assert sandbox._tower_ids == ["tower-1", "tower-2"]

    def test_empty_runway_ids_overrides_defaults(self) -> None:
        """Test empty runway_ids list overrides defaults."""
        from aviato._defaults import SandboxDefaults

        defaults = SandboxDefaults(runway_ids=("default-runway",))
        sandbox = Sandbox(runway_ids=[], defaults=defaults)
        assert sandbox._runway_ids == []

    def test_empty_tower_ids_overrides_defaults(self) -> None:
        """Test empty tower_ids list overrides defaults."""
        from aviato._defaults import SandboxDefaults

        defaults = SandboxDefaults(tower_ids=("default-tower",))
        sandbox = Sandbox(tower_ids=[], defaults=defaults)
        assert sandbox._tower_ids == []

    def test_none_runway_ids_uses_defaults(self) -> None:
        """Test None runway_ids falls back to defaults."""
        from aviato._defaults import SandboxDefaults

        defaults = SandboxDefaults(runway_ids=("default-runway",))
        sandbox = Sandbox(defaults=defaults)
        assert sandbox._runway_ids == ["default-runway"]

    def test_none_tower_ids_uses_defaults(self) -> None:
        """Test None tower_ids falls back to defaults."""
        from aviato._defaults import SandboxDefaults

        defaults = SandboxDefaults(tower_ids=("default-tower",))
        sandbox = Sandbox(defaults=defaults)
        assert sandbox._tower_ids == ["default-tower"]

    def test_run_passes_runway_ids(self) -> None:
        """Test Sandbox.run passes runway_ids to sandbox."""
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run(runway_ids=["runway-1"])
            assert sandbox._runway_ids == ["runway-1"]

    def test_run_passes_tower_ids(self) -> None:
        """Test Sandbox.run passes tower_ids to sandbox."""
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run(tower_ids=["tower-1"])
            assert sandbox._tower_ids == ["tower-1"]


class TestNetworkOptionsTypeGuard:
    """Tests for network parameter type guard validation."""

    def test_init_network_none_accepted(self) -> None:
        """Test Sandbox.__init__ accepts network=None."""
        sandbox = Sandbox(command="echo", args=["hello"], network=None)
        assert "network" not in sandbox._start_kwargs

    def test_init_network_options_accepted(self) -> None:
        """Test Sandbox.__init__ accepts NetworkOptions instance."""
        net_opts = NetworkOptions(ingress_mode="public", exposed_ports=(8080,))
        sandbox = Sandbox(command="echo", args=["hello"], network=net_opts)
        assert sandbox._start_kwargs["network"] == net_opts

    def test_init_network_dict_accepted(self) -> None:
        """Test Sandbox.__init__ accepts dict and converts to NetworkOptions."""
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            network={"ingress_mode": "public", "exposed_ports": [8080]},
        )
        # Dict is converted to NetworkOptions
        net_opts = sandbox._start_kwargs["network"]
        assert isinstance(net_opts, NetworkOptions)
        assert net_opts.ingress_mode == "public"
        assert net_opts.exposed_ports == (8080,)

    def test_init_network_string_raises_type_error(self) -> None:
        """Test Sandbox.__init__ raises TypeError for string network."""
        with pytest.raises(TypeError, match="network must be.*got str"):
            Sandbox(
                command="echo",
                args=["hello"],
                network="public",  # type: ignore[arg-type]
            )

    def test_run_network_none_accepted(self) -> None:
        """Test Sandbox.run accepts network=None."""
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run("echo", "hello", network=None)
            assert "network" not in sandbox._start_kwargs

    def test_run_network_options_accepted(self) -> None:
        """Test Sandbox.run accepts NetworkOptions instance."""
        net_opts = NetworkOptions(egress_mode="internet")
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run("echo", "hello", network=net_opts)
            assert sandbox._start_kwargs["network"] == net_opts

    def test_run_network_dict_accepted(self) -> None:
        """Test Sandbox.run accepts dict and converts to NetworkOptions."""
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run("echo", "hello", network={"egress_mode": "internet"})
            net_opts = sandbox._start_kwargs["network"]
            assert isinstance(net_opts, NetworkOptions)
            assert net_opts.egress_mode == "internet"

    def test_init_uses_defaults_network_when_not_specified(self) -> None:
        """Test Sandbox.__init__ uses defaults.network when network is None."""
        defaults_network = NetworkOptions(ingress_mode="internal", exposed_ports=(9000,))
        defaults = SandboxDefaults(network=defaults_network)
        sandbox = Sandbox(command="echo", args=["hello"], defaults=defaults)

        assert sandbox._start_kwargs["network"] is defaults_network

    def test_init_explicit_network_overrides_defaults(self) -> None:
        """Test explicit network parameter overrides defaults.network."""
        defaults_network = NetworkOptions(ingress_mode="internal")
        explicit_network = NetworkOptions(ingress_mode="public", exposed_ports=(8080,))
        defaults = SandboxDefaults(network=defaults_network)

        sandbox = Sandbox(
            command="echo", args=["hello"], defaults=defaults, network=explicit_network
        )

        assert sandbox._start_kwargs["network"] is explicit_network

    def test_run_uses_defaults_network_when_not_specified(self) -> None:
        """Test Sandbox.run uses defaults.network when network is None."""
        defaults_network = NetworkOptions(egress_mode="isolated")
        defaults = SandboxDefaults(network=defaults_network)
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run("echo", "hello", defaults=defaults)
            assert sandbox._start_kwargs["network"] is defaults_network


class TestAppliedNetworkModes:
    """Tests for applied_ingress_mode and applied_egress_mode attributes."""

    def test_applied_ingress_mode_starts_as_none(self) -> None:
        """Test applied_ingress_mode is None before start."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.applied_ingress_mode is None

    def test_applied_egress_mode_starts_as_none(self) -> None:
        """Test applied_egress_mode is None before start."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.applied_egress_mode is None

    def test_applied_modes_captured_from_start_response(self) -> None:
        """Test applied_* modes are captured from StartSandboxResponse."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []
        mock_start_response.applied_ingress_mode = "public"
        mock_start_response.applied_egress_mode = "internet"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.applied_ingress_mode == "public"
            assert sandbox.applied_egress_mode == "internet"

    def test_applied_modes_empty_string_mapped_to_none(self) -> None:
        """Test empty string applied_* modes are mapped to None."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []
        mock_start_response.applied_ingress_mode = ""
        mock_start_response.applied_egress_mode = ""

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox.applied_ingress_mode is None
            assert sandbox.applied_egress_mode is None

    def test_applied_modes_preserved_after_get_status(self) -> None:
        """Test applied_* values are preserved after get_status() refresh."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])

        # Set up sandbox with applied modes from start
        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []
        mock_start_response.applied_ingress_mode = "public"
        mock_start_response.applied_egress_mode = "org"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)
            sandbox.start().result()

        # Mock get_status call
        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING

        sandbox._stub.Get = AsyncMock(return_value=mock_get_response)

        sandbox.get_status()

        # Applied modes should be preserved (not overwritten by get_status)
        assert sandbox.applied_ingress_mode == "public"
        assert sandbox.applied_egress_mode == "org"


class TestNetworkOptionsRequestPayload:
    """Tests for NetworkOptions conversion to request payload."""

    def test_network_options_converted_to_dict(self) -> None:
        """Test NetworkOptions is converted to dict in start request."""
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            network=NetworkOptions(ingress_mode="public", egress_mode="internet"),
        )

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []
        mock_start_response.applied_ingress_mode = ""
        mock_start_response.applied_egress_mode = ""

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            start_call = sandbox._stub.Start.call_args[0][0]
            # Network should be converted to dict in the request
            assert start_call.network.ingress_mode == "public"
            assert start_call.network.egress_mode == "internet"

    def test_exposed_ports_tuple_converted_to_list(self) -> None:
        """Test exposed_ports tuple is converted to list in request."""
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            network=NetworkOptions(exposed_ports=(8080, 443)),
        )

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []
        mock_start_response.applied_ingress_mode = ""
        mock_start_response.applied_egress_mode = ""

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            start_call = sandbox._stub.Start.call_args[0][0]
            # exposed_ports should be a list in the request
            assert list(start_call.network.exposed_ports) == [8080, 443]

    def test_none_network_fields_omitted_from_request(self) -> None:
        """Test None fields in NetworkOptions are omitted from request."""
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            network=NetworkOptions(ingress_mode="public"),  # Only ingress_mode set
        )

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []
        mock_start_response.applied_ingress_mode = ""
        mock_start_response.applied_egress_mode = ""

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            start_call = sandbox._stub.Start.call_args[0][0]
            # ingress_mode should be set
            assert start_call.network.ingress_mode == "public"
            # egress_mode should be empty (default protobuf value)
            assert start_call.network.egress_mode == ""

    def test_no_network_option_omits_network_from_request(self) -> None:
        """Test no network option means network is not in _start_kwargs."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        # network should not be in _start_kwargs
        assert "network" not in sandbox._start_kwargs


class TestSandboxExecutionStats:
    """Tests for sandbox execution statistics tracking."""

    def test_initial_stats_are_zero(self) -> None:
        """Test sandbox starts with zeroed exec stats."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        assert sandbox.exec_stats == {
            "exec_count": 0,
            "exec_completed_ok": 0,
            "exec_completed_nonzero": 0,
            "exec_failures": 0,
        }

    def test_exec_increments_exec_count(self) -> None:
        """Test exec() increments exec_count before execution."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"

        # Mock _exec_streaming_async with MagicMock to avoid creating a real coroutine
        # that would trigger "coroutine was never awaited" warnings when discarded
        with patch.object(sandbox, "_exec_streaming_async", new=MagicMock()):
            with patch.object(sandbox._loop_manager, "run_async", return_value=MagicMock()):
                sandbox.exec(["echo", "hello"])

                # exec_count should be incremented
                assert sandbox._exec_count == 1

    def test_on_exec_complete_tracks_success(self) -> None:
        """Test _on_exec_complete tracks successful completion."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        result = MagicMock()
        result.returncode = 0

        sandbox._on_exec_complete(result, None)

        assert sandbox._exec_completed_ok == 1
        assert sandbox._exec_completed_nonzero == 0
        assert sandbox._exec_failures == 0

    def test_on_exec_complete_tracks_nonzero(self) -> None:
        """Test _on_exec_complete tracks non-zero exit codes."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        result = MagicMock()
        result.returncode = 1

        sandbox._on_exec_complete(result, None)

        assert sandbox._exec_completed_ok == 0
        assert sandbox._exec_completed_nonzero == 1
        assert sandbox._exec_failures == 0

    def test_on_exec_complete_tracks_failure(self) -> None:
        """Test _on_exec_complete tracks failures."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        sandbox._on_exec_complete(None, RuntimeError("test error"))

        assert sandbox._exec_completed_ok == 0
        assert sandbox._exec_completed_nonzero == 0
        assert sandbox._exec_failures == 1

    def test_exec_stats_lock_provides_atomic_read(self) -> None:
        """exec_stats blocks while the lock is held by another thread."""
        import threading

        sandbox = Sandbox(command="sleep", args=["infinity"])

        acquired = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with sandbox._exec_stats_lock:
                acquired.set()
                release.wait(timeout=5.0)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        assert acquired.wait(timeout=5.0)

        done = threading.Event()
        stats_result: dict[str, int] = {}

        def read_stats() -> None:
            stats_result.update(sandbox.exec_stats)
            done.set()

        reader = threading.Thread(target=read_stats, daemon=True)
        reader.start()

        # Reader should not finish while lock is held
        assert not done.wait(timeout=0.2)

        release.set()
        assert done.wait(timeout=5.0)
        t.join(timeout=5.0)
        reader.join(timeout=5.0)

    def test_on_exec_complete_reports_to_session(self) -> None:
        """Test _on_exec_complete reports to session reporter."""
        from aviato._wandb import ExecOutcome

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"

        mock_session = MagicMock()
        sandbox._session = mock_session

        result = MagicMock()
        result.returncode = 0

        sandbox._on_exec_complete(result, None)

        mock_session._record_exec_outcome.assert_called_once_with(
            ExecOutcome.COMPLETED_OK, "test-id"
        )


class TestSandboxStartupTimeTracking:
    """Tests for sandbox startup time tracking."""

    def test_initial_startup_tracking_state(self) -> None:
        """Test sandbox starts with correct initial startup tracking state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        assert sandbox._start_accepted_at is None
        assert sandbox._startup_recorded is False

    def test_start_sets_start_accepted_at(self) -> None:
        """Test start() sets _start_accepted_at timestamp."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []
        mock_start_response.applied_ingress_mode = ""
        mock_start_response.applied_egress_mode = ""

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            sandbox.start().result()

            assert sandbox._start_accepted_at is not None
            assert sandbox._start_accepted_at > 0

    def test_from_sandbox_info_marks_startup_recorded(self) -> None:
        """Test _from_sandbox_info marks startup as already recorded."""
        from coreweave.aviato.v1beta1 import atc_pb2
        from google.protobuf import timestamp_pb2

        sandbox = Sandbox._from_sandbox_info(
            sandbox_id="test-123",
            sandbox_status=atc_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            tower_id="tower-1",
            tower_group_id="group-1",
            runway_id="runway-1",
            base_url="https://api.example.com",
            timeout_seconds=300.0,
        )

        assert sandbox._startup_recorded is True
        assert sandbox._start_accepted_at is None

    def test_wait_records_startup_time_to_session(self) -> None:
        """Test wait() records startup time to session reporter."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._start_accepted_at = 100.0

        mock_session = MagicMock()
        sandbox._session = mock_session
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        mock_response.tower_id = "tower-1"
        mock_response.tower_group_id = None
        mock_response.runway_id = "runway-1"
        mock_response.started_at_time = None
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        with patch("time.monotonic", return_value=102.5):
            sandbox.wait()

        mock_session._record_startup_time.assert_called_once()
        call_args = mock_session._record_startup_time.call_args[0]
        assert call_args[0] == pytest.approx(2.5, rel=0.1)

    def test_startup_time_only_recorded_once(self) -> None:
        """Test startup time is only recorded once."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._start_accepted_at = 100.0
        sandbox._startup_recorded = True

        mock_session = MagicMock()
        sandbox._session = mock_session
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        mock_response.tower_id = "tower-1"
        mock_response.tower_group_id = None
        mock_response.runway_id = "runway-1"
        mock_response.started_at_time = None
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        sandbox.wait()

        mock_session._record_startup_time.assert_not_called()


class TestTranslateRpcError:
    """Tests for _translate_rpc_error function."""

    def test_not_found_returns_sandbox_not_found_error(self) -> None:
        """Test NOT_FOUND status code returns SandboxNotFoundError."""
        from aviato._sandbox import _translate_rpc_error

        error = MockRpcError(grpc.StatusCode.NOT_FOUND, "sandbox not found")
        result = _translate_rpc_error(error, sandbox_id="test-123")

        assert isinstance(result, SandboxNotFoundError)
        assert result.sandbox_id == "test-123"
        assert "test-123" in str(result)

    def test_not_found_without_sandbox_id_uses_details(self) -> None:
        """Test NOT_FOUND without sandbox_id uses error details."""
        from aviato._sandbox import _translate_rpc_error

        error = MockRpcError(grpc.StatusCode.NOT_FOUND, "resource not found")
        result = _translate_rpc_error(error)

        assert isinstance(result, SandboxNotFoundError)
        assert "resource not found" in str(result)

    def test_cancelled_returns_sandbox_not_running_error(self) -> None:
        """Test CANCELLED status code returns SandboxNotRunningError."""
        from aviato._sandbox import _translate_rpc_error
        from aviato.exceptions import SandboxNotRunningError

        error = MockRpcError(grpc.StatusCode.CANCELLED, "request cancelled")
        result = _translate_rpc_error(error, operation="Start sandbox")

        assert isinstance(result, SandboxNotRunningError)
        assert "cancelled" in str(result)

    def test_cancelled_with_sandbox_id_includes_id(self) -> None:
        """Test CANCELLED with sandbox_id includes ID in message."""
        from aviato._sandbox import _translate_rpc_error
        from aviato.exceptions import SandboxNotRunningError

        error = MockRpcError(grpc.StatusCode.CANCELLED, "cancelled")
        result = _translate_rpc_error(error, sandbox_id="test-456", operation="Execute command")

        assert isinstance(result, SandboxNotRunningError)
        assert "test-456" in str(result)

    def test_deadline_exceeded_returns_sandbox_timeout_error(self) -> None:
        """Test DEADLINE_EXCEEDED status code returns SandboxTimeoutError."""
        from aviato._sandbox import _translate_rpc_error
        from aviato.exceptions import SandboxTimeoutError

        error = MockRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "timeout after 30s")
        result = _translate_rpc_error(error, operation="Execute command")

        assert isinstance(result, SandboxTimeoutError)
        assert "timed out" in str(result)

    def test_unavailable_returns_sandbox_not_running_error(self) -> None:
        """Test UNAVAILABLE status code returns SandboxNotRunningError."""
        from aviato._sandbox import _translate_rpc_error
        from aviato.exceptions import SandboxNotRunningError

        error = MockRpcError(grpc.StatusCode.UNAVAILABLE, "connection refused")
        result = _translate_rpc_error(error)

        assert isinstance(result, SandboxNotRunningError)
        assert "unavailable" in str(result).lower()

    def test_permission_denied_returns_auth_error(self) -> None:
        """Test PERMISSION_DENIED status code returns AviatoAuthenticationError."""
        from aviato._sandbox import _translate_rpc_error
        from aviato.exceptions import AviatoAuthenticationError

        error = MockRpcError(grpc.StatusCode.PERMISSION_DENIED, "access denied")
        result = _translate_rpc_error(error)

        assert isinstance(result, AviatoAuthenticationError)
        assert "denied" in str(result).lower()

    def test_unauthenticated_returns_auth_error(self) -> None:
        """Test UNAUTHENTICATED status code returns AviatoAuthenticationError."""
        from aviato._sandbox import _translate_rpc_error
        from aviato.exceptions import AviatoAuthenticationError

        error = MockRpcError(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
        result = _translate_rpc_error(error)

        assert isinstance(result, AviatoAuthenticationError)
        assert "authentication" in str(result).lower()

    def test_other_status_returns_sandbox_error(self) -> None:
        """Test other status codes return generic SandboxError."""
        from aviato._sandbox import _translate_rpc_error

        error = MockRpcError(grpc.StatusCode.INTERNAL, "internal error")
        result = _translate_rpc_error(error, operation="Test operation")

        assert isinstance(result, SandboxError)
        assert "failed" in str(result)

    def test_empty_details_uses_string_repr(self) -> None:
        """Test that empty details falls back to str(e)."""
        from aviato._sandbox import _translate_rpc_error

        error = MockRpcError(grpc.StatusCode.INTERNAL, "")
        result = _translate_rpc_error(error, operation="Test")

        assert isinstance(result, SandboxError)


class TestExecStdinReadySignal:
    """Tests for stdin ready signal handling in exec streaming."""

    def test_exec_stdin_waits_for_ready(self) -> None:
        """Test stdin data is not sent until ready signal is received.

        This test verifies the timing behavior: stdin data should only be
        sent after the ready signal is received from the server.
        """
        import asyncio
        import time

        from google.protobuf import timestamp_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Track timing: when was stdin data received relative to ready being sent?
        ready_sent_time: float | None = None
        stdin_received_time: float | None = None
        close_received_time: float | None = None

        # Use event to synchronize: wait for close before sending exit
        close_event = asyncio.Event()

        def on_write_with_event(request: Any) -> None:
            nonlocal stdin_received_time, close_received_time
            if hasattr(request, "stdin") and request.HasField("stdin"):
                stdin_received_time = time.monotonic()
            elif hasattr(request, "close") and request.HasField("close"):
                close_received_time = time.monotonic()
                close_event.set()

        async def response_generator() -> AsyncIterator[Any]:
            """Generate responses with proper sequencing like a real server."""
            nonlocal ready_sent_time
            # Send ready signal first
            ready_response = MagicMock()
            ready_response.HasField = lambda field: field == "ready"
            ready_response.ready.ready_at = timestamp_pb2.Timestamp(seconds=1234567890)
            ready_sent_time = time.monotonic()
            yield ready_response

            # Wait for stdin close before sending exit (like a real server)
            try:
                await asyncio.wait_for(close_event.wait(), timeout=5.0)
            except TimeoutError:
                pass

            # Send exit after stdin is closed
            exit_response = MagicMock()
            exit_response.HasField = lambda field: field == "exit"
            exit_response.exit.exit_code = 0
            yield exit_response

        # Use bidirectional mock with response generator - properly interleaves
        mock_call = MockBidirectionalStreamCall(
            response_generator=response_generator, on_write=on_write_with_event
        )
        mock_channel, mock_stub = create_mock_channel_and_stub_bidirectional(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["cat"], stdin=True)
            assert process.stdin is not None  # stdin=True provides StreamWriter

            # Write data to stdin and close
            process.stdin.write(b"hello").result()
            process.stdin.close().result()

            result = process.result()
            assert result.returncode == 0

        # Verify timing: stdin was received after ready was sent
        assert ready_sent_time is not None, "Ready signal was never sent"
        assert stdin_received_time is not None, "Stdin data was never received"
        assert close_received_time is not None, "Stdin close was never received"

        # Stdin should be sent after ready (with some tolerance for timing)
        assert (
            stdin_received_time >= ready_sent_time
        ), f"Stdin received at {stdin_received_time} but ready sent at {ready_sent_time}"

    def test_exec_stdin_ready_timeout(self) -> None:
        """Test SandboxTimeoutError raised when ready signal not received."""
        from aviato.exceptions import SandboxTimeoutError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Never send ready signal - just hang (simulated by no responses)
        # The test will timeout waiting for ready
        mock_call = MockStreamCall(responses=[])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["cat"], stdin=True, timeout_seconds=0.1)

            with pytest.raises(SandboxTimeoutError, match="ready signal"):
                process.result()

    def test_exec_stdin_error_before_ready(self) -> None:
        """Test error before ready signal unblocks and propagates error."""
        from aviato.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Error response before ready signal
        error_response = MagicMock()
        error_response.HasField = lambda field: field == "error"
        error_response.error.message = "Process crashed"

        # Use bidirectional mock to allow responses while requests are still being sent
        mock_call = MockBidirectionalStreamCall(responses=[error_response])
        mock_channel, mock_stub = create_mock_channel_and_stub_bidirectional(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["cat"], stdin=True, timeout_seconds=5.0)

            # Should not deadlock - error sets ready_event
            with pytest.raises(SandboxExecutionError, match="Process crashed"):
                process.result()

    def test_exec_stdin_exit_before_ready(self) -> None:
        """Test exit before ready signal unblocks and returns result."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Exit response before ready signal (fast-completing process)
        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        # Use bidirectional mock to allow responses while requests are still being sent
        mock_call = MockBidirectionalStreamCall(responses=[exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub_bidirectional(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["true"], stdin=True, timeout_seconds=5.0)

            # Should not deadlock - exit sets ready_event
            result = process.result()
            assert result.returncode == 0

    def test_exec_stdin_cancel_unblocks(self) -> None:
        """Test Process.cancel() unblocks stdin waiting for ready."""
        import asyncio
        import concurrent.futures

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Never send ready - simulates hung connection
        mock_call = MockStreamCall(responses=[])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            # Use longer timeout so cancel is what terminates it
            process = sandbox.exec(["cat"], stdin=True, timeout_seconds=30.0)

            # Cancel the process - this should unblock the stdin waiter
            process.cancel()

            # Result should raise CancelledError or return quickly
            with pytest.raises((concurrent.futures.CancelledError, asyncio.CancelledError)):
                process.result(timeout=1.0)

    def test_exec_no_stdin_no_ready_wait(self) -> None:
        """Test stdin=False does not wait for ready signal."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # No ready signal in responses - just stdout and exit
        stdout_response = MagicMock()
        stdout_response.HasField = lambda field: field == "output"
        stdout_response.output.data = b"hello\n"
        stdout_response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDOUT

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        mock_call = MockStreamCall(responses=[stdout_response, exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth_metadata", return_value=()),
            patch("aviato._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch(
                "aviato._sandbox.streaming_pb2_grpc.ATCStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            # stdin=False (default) - should not wait for ready
            process = sandbox.exec(["echo", "hello"])

            # Should complete without ready signal
            result = process.result()
            assert result.returncode == 0
            assert result.stdout == "hello\n"

            # Verify stdin is None when stdin=False
            assert process.stdin is None


class TestCrossLoopRegression:
    """Regression tests for cross-event-loop bug.

    The sync/async hybrid API uses a background daemon thread (Loop B via _LoopManager)
    for all gRPC operations. Results bridge back to the caller's loop (Loop A) via
    asyncio.wrap_future(). If any entrypoint submits work directly to Loop A instead
    of routing through _LoopManager, it causes RuntimeError when nest_asyncio is not
    installed (or a subtle deadlock when it is).

    Key invariant: all gRPC operations go through _loop_manager.run_async(), results
    bridge back via asyncio.wrap_future().
    """

    def test_await_sandbox_routes_through_loop_manager(self) -> None:
        """Verify __await__ routes through _loop_manager, not the caller's loop."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"

        async def do_await() -> Sandbox:
            return await sandbox

        with patch.object(
            sandbox._loop_manager, "run_async", wraps=sandbox._loop_manager.run_async
        ) as mock_run_async:
            with patch.object(sandbox, "_ensure_started_async", new_callable=AsyncMock):
                with patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock):
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(do_await())
                        assert result is sandbox
                        mock_run_async.assert_called_once()
                    finally:
                        loop.close()

    def test_aenter_routes_through_loop_manager(self) -> None:
        """Verify __aenter__ routes through _loop_manager for unstarted sandbox."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox._sandbox_id is None

        with patch.object(
            sandbox._loop_manager, "run_async", wraps=sandbox._loop_manager.run_async
        ) as mock_run_async:
            with patch.object(sandbox, "_start_async", new_callable=AsyncMock):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(sandbox.__aenter__())
                    assert result is sandbox
                    mock_run_async.assert_called_once()
                finally:
                    loop.close()

    def test_aenter_skips_start_when_already_started(self) -> None:
        """Verify __aenter__ skips _loop_manager when sandbox already has an ID."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "already-started"

        with patch.object(
            sandbox._loop_manager, "run_async", wraps=sandbox._loop_manager.run_async
        ) as mock_run_async:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(sandbox.__aenter__())
                assert result is sandbox
                mock_run_async.assert_not_called()
            finally:
                loop.close()

    def test_start_routes_through_loop_manager(self) -> None:
        """Verify start() routes through _loop_manager.run_async."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "cross-loop-id"

        with patch.object(
            sandbox._loop_manager, "run_async", wraps=sandbox._loop_manager.run_async
        ) as mock_run_async:
            with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
                sandbox._channel = MagicMock()
                sandbox._stub = MagicMock()
                sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

                ref = sandbox.start()
                ref.result()

                mock_run_async.assert_called_once()

    def test_operation_ref_await_uses_wrap_future(self) -> None:
        """Verify OperationRef.__await__ uses asyncio.wrap_future for cross-loop safety."""
        from aviato import OperationRef

        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        future.set_result("test-value")
        ref = OperationRef(future)

        async def do_await() -> str:
            return await ref

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(do_await())
            assert result == "test-value"
        finally:
            loop.close()

    def test_dual_loop_sandbox_await(self) -> None:
        """Deterministic dual-loop test: create on one loop, await on another.

        This is the core regression test. The bug manifested when:
        1. Sandbox created on Loop A (or no loop)
        2. Awaited on Loop B (different from _LoopManager's background loop)
        3. If __await__ submitted work to Loop B directly, it would fail

        The fix routes through _LoopManager.run_async() which always submits
        to the background daemon loop, making the caller's loop irrelevant.
        """
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "dual-loop-id"

        async def do_await() -> Sandbox:
            return await sandbox

        with patch.object(sandbox, "_ensure_started_async", new_callable=AsyncMock):
            with patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock):
                # Await on a fresh event loop (simulates different caller context)
                loop_a = asyncio.new_event_loop()
                try:
                    result = loop_a.run_until_complete(do_await())
                    assert result is sandbox
                finally:
                    loop_a.close()

                # Await on yet another fresh loop (simulates Jupyter cell re-execution)
                loop_b = asyncio.new_event_loop()
                try:
                    result = loop_b.run_until_complete(do_await())
                    assert result is sandbox
                finally:
                    loop_b.close()

    def test_dual_loop_aenter_aexit(self) -> None:
        """Create sandbox on one loop, use async context manager on another."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "dual-loop-ctx-id"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            # Start on Loop A (sync)
            sandbox.start().result()
            assert sandbox.sandbox_id == "dual-loop-ctx-id"

            # Use async context manager on Loop B
            async def use_async_ctx() -> Sandbox:
                async with sandbox as sb:
                    return sb

            with patch.object(sandbox, "_stop_async", new_callable=AsyncMock):
                loop_b = asyncio.new_event_loop()
                try:
                    result = loop_b.run_until_complete(use_async_ctx())
                    assert result is sandbox
                finally:
                    loop_b.close()

    def test_start_then_await_on_different_loop(self) -> None:
        """Call start() synchronously, then await the sandbox on a different loop."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "start-then-await-id"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_start_response)

            # Start synchronously (uses _loop_manager internally)
            sandbox.start().result()
            assert sandbox.sandbox_id == "start-then-await-id"

            # Now await on a completely different loop
            async def do_await() -> Sandbox:
                return await sandbox

            with patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock):
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(do_await())
                    assert result is sandbox
                finally:
                    loop.close()
