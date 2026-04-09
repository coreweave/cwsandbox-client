# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._sandbox module."""

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest

from cwsandbox import NetworkOptions, Sandbox, SandboxDefaults, Secret
from cwsandbox._sandbox import SandboxStatus, _Running, _Starting, _Stopping, _Terminal
from cwsandbox.exceptions import (
    SandboxError,
    SandboxNotFoundError,
    SandboxNotRunningError,
)


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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        with pytest.raises(ValueError, match="Command cannot be empty"):
            sandbox.exec([])

    def test_exec_check_raises_on_nonzero_returncode(self) -> None:
        """Test exec with check=True raises SandboxExecutionError on failure."""
        from cwsandbox._proto import streaming_pb2
        from cwsandbox.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["nonexistent"], check=True)
            with pytest.raises(SandboxExecutionError, match="exit code 127"):
                process.result()

    def test_exec_check_false_returns_result_on_failure(self) -> None:
        """Test exec with check=False returns result even on failure."""
        from cwsandbox._proto import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["failing-cmd"], check=False)
            result = process.result()

            assert result.returncode == 1

    def test_exec_streams_stdout_to_queue(self) -> None:
        """Test exec() streams stdout data to the queue as it arrives."""
        from cwsandbox._proto import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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

        from cwsandbox.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        sandbox._state = _Running(sandbox_id="test-id")

        with pytest.raises(ValueError, match="cwd cannot be empty string"):
            sandbox.exec(["ls"], cwd="")

    def test_exec_cwd_relative_path_raises_error(self) -> None:
        """Test exec with relative cwd raises ValueError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")

        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            sandbox.exec(["ls"], cwd="relative/path")

    def test_exec_cwd_wraps_command_with_shell(self) -> None:
        """Test exec with cwd wraps command in shell wrapper."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        exit_response = MagicMock()
        exit_response.HasField = lambda field: field == "exit"
        exit_response.exit.exit_code = 0

        mock_call = MockStreamCall(responses=[exit_response])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["ls", "-la"])
            process.result()

        # Without cwd, command should not be wrapped
        assert captured_command == ["ls", "-la"]

    def test_exec_raises_on_terminal_sandbox(self) -> None:
        """exec() on a terminal sandbox raises SandboxNotRunningError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Terminal(sandbox_id="test-id", status=SandboxStatus.COMPLETED)
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            sandbox.exec(["echo", "hello"]).result()

    def test_read_file_raises_on_terminal_sandbox(self) -> None:
        """read_file() on a terminal sandbox raises SandboxNotRunningError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Terminal(sandbox_id="test-id", status=SandboxStatus.COMPLETED)
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            sandbox.read_file("/tmp/test.txt").result()

    def test_write_file_raises_on_terminal_sandbox(self) -> None:
        """write_file() on a terminal sandbox raises SandboxNotRunningError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Terminal(sandbox_id="test-id", status=SandboxStatus.COMPLETED)
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            sandbox.write_file("/tmp/test.txt", b"data").result()


class TestExecCwdHelperFunctions:
    """Tests for cwd helper functions."""

    def test_validate_cwd_none_passes(self) -> None:
        """Test _validate_cwd allows None."""
        from cwsandbox._sandbox import _validate_cwd

        _validate_cwd(None)  # Should not raise

    def test_validate_cwd_absolute_path_passes(self) -> None:
        """Test _validate_cwd allows absolute paths."""
        from cwsandbox._sandbox import _validate_cwd

        _validate_cwd("/app")
        _validate_cwd("/var/log/app")
        _validate_cwd("/")

    def test_validate_cwd_empty_string_raises(self) -> None:
        """Test _validate_cwd raises on empty string."""
        from cwsandbox._sandbox import _validate_cwd

        with pytest.raises(ValueError, match="cwd cannot be empty string"):
            _validate_cwd("")

    def test_validate_cwd_relative_path_raises(self) -> None:
        """Test _validate_cwd raises on relative paths."""
        from cwsandbox._sandbox import _validate_cwd

        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            _validate_cwd("relative")
        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            _validate_cwd("./relative")
        with pytest.raises(ValueError, match="cwd must be an absolute path"):
            _validate_cwd("../parent")

    def test_wrap_command_with_cwd_basic(self) -> None:
        """Test _wrap_command_with_cwd creates correct shell wrapper."""
        from cwsandbox._sandbox import _wrap_command_with_cwd

        result = _wrap_command_with_cwd(["ls", "-la"], "/app")
        assert result == ["/bin/sh", "-c", "cd /app && exec ls -la"]

    def test_wrap_command_with_cwd_escapes_path(self) -> None:
        """Test _wrap_command_with_cwd escapes special characters in path."""
        from cwsandbox._sandbox import _wrap_command_with_cwd

        result = _wrap_command_with_cwd(["ls"], "/path with spaces")
        assert result[0] == "/bin/sh"
        assert result[1] == "-c"
        # Path should be quoted
        assert "'/path with spaces'" in result[2]

    def test_wrap_command_with_cwd_escapes_command_args(self) -> None:
        """Test _wrap_command_with_cwd escapes command arguments."""
        from cwsandbox._sandbox import _wrap_command_with_cwd

        result = _wrap_command_with_cwd(["echo", "hello world"], "/app")
        # Arguments with spaces should be quoted
        assert "'hello world'" in result[2]


class TestSandboxAuth:
    """Tests for Sandbox authentication."""

    @pytest.mark.asyncio
    async def test_resolves_auth_metadata(self, mock_api_key: str) -> None:
        """Test _ensure_client() resolves auth metadata and stores it."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with (
            patch("cwsandbox._sandbox.create_channel") as mock_create_channel,
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub") as mock_stub_class,
            patch("cwsandbox._sandbox.resolve_auth_metadata") as mock_resolve,
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
    async def test_auth_metadata_passed_to_start_rpc(self, mock_api_key: str) -> None:
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

    @pytest.mark.asyncio
    async def test_start_async_includes_secrets_in_request(self, mock_api_key: str) -> None:
        """Test _start_async passes secrets into the Start RPC request."""
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            secrets=[
                Secret(store="wandb", name="HF_TOKEN", field="api_key", env_var="HF_TOKEN"),
            ],
        )

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
        request = mock_stub.Start.call_args[0][0]
        assert len(request.secret_stores) == 1
        assert request.secret_stores[0].store_name == "wandb"
        assert len(request.secret_stores[0].secrets) == 1
        assert request.secret_stores[0].secrets[0].path == "HF_TOKEN"
        assert request.secret_stores[0].secrets[0].field == "api_key"
        assert request.secret_stores[0].secrets[0].env_var == "HF_TOKEN"


class TestSandboxCleanup:
    """Tests for Sandbox cleanup and resource warnings."""

    def test_del_warns_if_sandbox_not_stopped(self) -> None:
        """Test __del__ emits ResourceWarning if sandbox was started but not stopped."""
        import warnings

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-sandbox-id"
        sandbox._state = _Starting(sandbox_id="test-sandbox-id")

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

        from cwsandbox import OperationRef

        sandbox = Sandbox(command="sleep", args=["infinity"])
        stop_called = False

        def mock_start() -> OperationRef[None]:
            sandbox._sandbox_id = "test-sandbox-id"
            sandbox._state = _Starting(sandbox_id="test-sandbox-id")
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

    def test_sync_context_manager_skips_stop_when_done(self) -> None:
        """Test __exit__ cleans up channels without calling stop() when sandbox is terminal."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Terminal(sandbox_id="test-id", status=SandboxStatus.COMPLETED)

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_streaming_channel = MagicMock()
        mock_streaming_channel.close = AsyncMock()
        sandbox._channel = mock_channel
        sandbox._streaming_channel = mock_streaming_channel

        stop_called = False

        def mock_stop(**kwargs: object) -> None:
            nonlocal stop_called
            stop_called = True

        sandbox.stop = mock_stop  # type: ignore[method-assign]

        with sandbox:
            pass

        assert not stop_called
        mock_channel.close.assert_called_once_with(grace=None)
        mock_streaming_channel.close.assert_called_once_with(grace=None)
        assert sandbox._channel is None
        assert sandbox._streaming_channel is None

    def test_enter_starts_if_not_started(self) -> None:
        """Test __enter__ starts sandbox if not already started."""
        import concurrent.futures

        from cwsandbox import OperationRef

        sandbox = Sandbox(command="sleep", args=["infinity"])
        start_called = False

        def mock_start() -> OperationRef[None]:
            nonlocal start_called
            start_called = True
            sandbox._sandbox_id = "enter-sandbox-id"
            sandbox._state = _Starting(sandbox_id="enter-sandbox-id")
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

    def test_get_status_raises_for_cancelled(self) -> None:
        """get_status raises SandboxNotRunningError for a cancelled sandbox."""
        from cwsandbox._sandbox import _NotStarted

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _NotStarted(cancelled=True)

        with pytest.raises(SandboxNotRunningError, match="cancelled before starting"):
            sandbox.get_status()

    @pytest.mark.parametrize(
        "terminal_status",
        [SandboxStatus.COMPLETED, SandboxStatus.FAILED, SandboxStatus.TERMINATED],
    )
    def test_get_status_returns_cached_for_terminal(self, terminal_status: SandboxStatus) -> None:
        """get_status returns cached status for terminal sandboxes without API call."""
        from cwsandbox._sandbox import _Terminal

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Terminal(sandbox_id="test-id", status=terminal_status)

        result = sandbox.get_status()

        assert result == terminal_status
        assert sandbox._status_updated_at is not None


class TestSandboxWait:
    """Tests for Sandbox.wait method (wait until RUNNING)."""

    def test_wait_raises_on_failed(self) -> None:
        """Test wait raises SandboxFailedError when sandbox fails."""
        from cwsandbox._proto import gateway_pb2
        from cwsandbox.exceptions import SandboxFailedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Starting(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_FAILED
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        with pytest.raises(SandboxFailedError, match="failed"):
            sandbox.wait()

    def test_wait_raises_on_terminated_by_default(self) -> None:
        """Test wait raises SandboxTerminatedError when terminated."""
        from cwsandbox._proto import gateway_pb2
        from cwsandbox.exceptions import SandboxTerminatedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Starting(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_TERMINATED
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        with pytest.raises(SandboxTerminatedError, match="terminated"):
            sandbox.wait()

    def test_wait_auto_starts_unstarted_sandbox(self) -> None:
        """Test wait() triggers auto-start on unstarted sandbox."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="echo", args=["hello"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-wait-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.runner_id = "tower-1"
        mock_get_response.profile_id = "runway-1"
        mock_get_response.runner_group_id = None
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
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-read-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.runner_id = "tower-1"
        mock_get_response.profile_id = "runway-1"
        mock_get_response.runner_group_id = None
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
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-write-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.runner_id = "tower-1"
        mock_get_response.profile_id = "runway-1"
        mock_get_response.runner_group_id = None
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
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="echo", args=["hello"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "auto-start-complete-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.returncode = 0

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
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Starting(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_TERMINATED
        mock_response.returncode = 0
        mock_response.started_at_time = None
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        sandbox.wait_until_complete(raise_on_termination=False).result()

        assert sandbox.returncode is None


class TestSandboxStart:
    """Tests for Sandbox.start method."""

    def test_start_returns_operation_ref(self) -> None:
        """Test start returns OperationRef[None]."""
        from cwsandbox import OperationRef

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
        from cwsandbox._proto import gateway_pb2
        from cwsandbox.exceptions import SandboxFailedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "failing-sandbox-id"
        sandbox._state = _Starting(sandbox_id="failing-sandbox-id")

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_FAILED

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Get = AsyncMock(return_value=mock_get_response)

            with pytest.raises(SandboxFailedError, match="failed to start"):
                sandbox.wait()

    def test_wait_handles_fast_completion(self) -> None:
        """Test wait handles sandbox that completes during startup."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="echo", args=["hello"])
        sandbox._sandbox_id = "fast-sandbox-id"
        sandbox._state = _Starting(sandbox_id="fast-sandbox-id")

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.returncode = 0
        mock_get_response.runner_id = "tower-1"
        mock_get_response.profile_id = "runway-1"
        mock_get_response.runner_group_id = None
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


class TestSandboxAnnotations:
    """Tests for annotations merging behavior."""

    def test_init_with_annotations(self) -> None:
        """Test sandbox stores annotations when provided."""
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            annotations={"prometheus.io/scrape": "true"},
        )

        assert sandbox._annotations == {"prometheus.io/scrape": "true"}

    def test_init_annotations_defaults_fallback(self) -> None:
        """Test sandbox inherits annotations from defaults when not specified."""
        defaults = SandboxDefaults(
            annotations={"team": "platform", "env": "staging"},
        )

        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            defaults=defaults,
        )

        assert sandbox._annotations == {"team": "platform", "env": "staging"}

    def test_init_annotations_merge(self) -> None:
        """Test explicit annotations merge with defaults."""
        defaults = SandboxDefaults(
            annotations={"team": "platform", "env": "staging"},
        )

        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            defaults=defaults,
            annotations={"tier": "frontend"},
        )

        assert sandbox._annotations == {
            "team": "platform",
            "env": "staging",
            "tier": "frontend",
        }

    def test_init_annotations_explicit_overrides_default_keys(self) -> None:
        """Test explicit annotations override defaults on key collision."""
        defaults = SandboxDefaults(
            annotations={"team": "platform", "env": "staging"},
        )

        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            defaults=defaults,
            annotations={"env": "production", "tier": "backend"},
        )

        assert sandbox._annotations == {
            "team": "platform",
            "env": "production",
            "tier": "backend",
        }

    def test_init_annotations_empty_when_no_defaults_or_params(self) -> None:
        """Test annotations is empty dict when nothing specified."""
        sandbox = Sandbox(command="echo", args=["hello"])

        assert sandbox._annotations == {}

    def test_run_with_annotations(self) -> None:
        """Test Sandbox.run() passes annotations through to __init__."""
        with patch.object(Sandbox, "start") as mock_start:
            mock_start.return_value = MagicMock(result=MagicMock(return_value=None))
            sandbox = Sandbox.run(
                "echo",
                "hello",
                annotations={"team": "platform"},
            )

        assert sandbox._annotations == {"team": "platform"}

    @pytest.mark.asyncio
    async def test_start_async_maps_to_pod_annotations(self) -> None:
        """Test _start_async maps annotations to pod_annotations in request."""
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            annotations={"team": "platform"},
        )

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
        sandbox._auth_metadata = ()

        await sandbox._start_async()

        mock_stub.Start.assert_called_once()
        call_args = mock_stub.Start.call_args
        request = call_args[0][0]
        assert request.pod_annotations == {"team": "platform"}

    @pytest.mark.asyncio
    async def test_start_async_empty_annotations_not_sent(self) -> None:
        """Test _start_async omits pod_annotations when annotations is empty."""
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
        sandbox._auth_metadata = ()

        await sandbox._start_async()

        mock_stub.Start.assert_called_once()
        call_args = mock_stub.Start.call_args
        request = call_args[0][0]
        assert len(request.pod_annotations) == 0

    def test_from_sandbox_info_initializes_annotations(self) -> None:
        """Test _from_sandbox_info sets _annotations to empty dict."""
        mock_info = MagicMock()
        mock_info.sandbox_id = "test-id"
        mock_info.sandbox_status = SandboxStatus.RUNNING.to_proto()
        mock_info.started_at_time = None
        mock_info.runner_id = None
        mock_info.profile_id = None
        mock_info.runner_group_id = None

        sandbox = Sandbox._from_sandbox_info(
            mock_info,
            base_url="https://test.example.com",
            timeout_seconds=30.0,
        )

        assert sandbox._annotations == {}


class TestSandboxStop:
    """Tests for Sandbox.stop method."""

    def test_stop_raises_on_backend_failure(self) -> None:
        """Test stop raises SandboxError when backend reports failure."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Starting(sandbox_id="test-id")
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
        sandbox._state = _Starting(sandbox_id="test-id")
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
        sandbox._state = _Starting(sandbox_id="test-id")
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
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        expected_metadata = (("authorization", "Bearer test-key"),)
        sandbox._auth_metadata = expected_metadata

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "race-sandbox-id"

        mock_stop_response = MagicMock()
        mock_stop_response.success = True

        # Mock Get to return terminal so _do_poll_complete resolves
        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.sandbox_id = "race-sandbox-id"
        mock_get_response.runner_id = ""
        mock_get_response.profile_id = ""
        mock_get_response.runner_group_id = ""
        mock_get_response.started_at_time = None
        mock_get_response.returncode = 0

        mock_stub = MagicMock()
        mock_stub.Start = AsyncMock(return_value=mock_start_response)
        mock_stub.Stop = AsyncMock(return_value=mock_stop_response)
        mock_stub.Get = AsyncMock(return_value=mock_get_response)

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
        from cwsandbox.exceptions import SandboxTimeoutError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Mock call that raises DEADLINE_EXCEEDED when iterating
        mock_call = MockStreamCall(
            error_on_read=MockAioRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded")
        )
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        secrets = [
            Secret(store="wandb", name="HF_TOKEN", field="api_key", env_var="HF_TOKEN"),
        ]
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            resources={"cpu": "100m", "memory": "128Mi"},
            ports=[{"container_port": 8080}],
            network=net_opts,
            max_timeout_seconds=60,
            environment_variables={"TEST_ENV_VAR": "test-value"},
            secrets=secrets,
        )
        from cwsandbox._types import ResourceOptions

        stored_res = sandbox._start_kwargs["resources"]
        assert isinstance(stored_res, ResourceOptions)
        assert stored_res.requests == {"cpu": "100m", "memory": "128Mi"}
        assert stored_res.limits == {"cpu": "100m", "memory": "128Mi"}
        assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]
        assert sandbox._start_kwargs["network"] == net_opts
        assert sandbox._start_kwargs["max_timeout_seconds"] == 60
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 1
        assert stored[0].store == "wandb"
        assert stored[0].name == "HF_TOKEN"
        assert stored[0].field == "api_key"
        assert stored[0].env_var == "HF_TOKEN"
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
        secrets = [
            Secret(store="wandb", name="OPENAI_API_KEY", field="api_key", env_var="OPENAI_KEY"),
        ]
        with patch.object(Sandbox, "start") as mock_start:
            sandbox = Sandbox.run(
                "echo",
                "hello",
                resources={"cpu": "100m"},
                ports=[{"container_port": 8080}],
                secrets=secrets,
            )
            from cwsandbox._types import ResourceOptions

            mock_start.assert_called_once()
            stored_res = sandbox._start_kwargs["resources"]
            assert isinstance(stored_res, ResourceOptions)
            assert stored_res.requests == {"cpu": "100m"}
            assert stored_res.limits == {"cpu": "100m"}
            assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]
            stored = sandbox._start_kwargs["secrets"]
            assert len(stored) == 1 and stored[0].store == "wandb"
            assert stored[0].name == "OPENAI_API_KEY"
            assert stored[0].field == "api_key"
            assert stored[0].env_var == "OPENAI_KEY"

    def test_run_with_invalid_kwargs(self) -> None:
        """Test Sandbox.run rejects invalid kwargs."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Sandbox.run(
                "echo",
                "hello",
                invalid_param="value",
            )


class TestResourceOptionsWiring:
    """Tests for ResourceOptions normalization and proto conversion."""

    def test_init_normalizes_flat_dict_to_resource_options(self) -> None:
        """Flat resource dict is normalized to ResourceOptions in __init__."""
        from cwsandbox._types import ResourceOptions

        sandbox = Sandbox(command="echo", resources={"cpu": "2", "memory": "4Gi"})
        stored = sandbox._start_kwargs["resources"]
        assert isinstance(stored, ResourceOptions)
        assert stored.requests == {"cpu": "2", "memory": "4Gi"}
        assert stored.limits == {"cpu": "2", "memory": "4Gi"}

    def test_init_accepts_resource_options_directly(self) -> None:
        """ResourceOptions instance is accepted directly in __init__."""
        from cwsandbox._types import ResourceOptions

        opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )
        sandbox = Sandbox(command="echo", resources=opts)
        stored = sandbox._start_kwargs["resources"]
        assert isinstance(stored, ResourceOptions)
        assert stored.requests == {"cpu": "1", "memory": "256Mi"}
        assert stored.limits == {"cpu": "8", "memory": "2Gi"}

    def test_init_falls_back_to_defaults_resources(self) -> None:
        """When no resources kwarg, uses SandboxDefaults.resources."""
        from cwsandbox._types import ResourceOptions

        defaults = SandboxDefaults(resources={"cpu": "500m", "memory": "1Gi"})
        sandbox = Sandbox(command="echo", defaults=defaults)
        stored = sandbox._start_kwargs["resources"]
        assert isinstance(stored, ResourceOptions)
        assert stored.requests == {"cpu": "500m", "memory": "1Gi"}

    def test_init_none_resources_omitted(self) -> None:
        """When resources is None everywhere, _start_kwargs has no resources key."""
        sandbox = Sandbox(command="echo")
        assert "resources" not in sandbox._start_kwargs

    @pytest.mark.asyncio
    async def test_start_async_maps_resource_options_to_proto_fields(
        self, mock_api_key: str
    ) -> None:
        """ResourceOptions maps to resource_limits and resource_requests proto fields."""
        from cwsandbox._types import ResourceOptions

        opts = ResourceOptions(
            requests={"cpu": "1", "memory": "256Mi"},
            limits={"cpu": "8", "memory": "2Gi"},
        )
        sandbox = Sandbox(command="echo", resources=opts)

        mock_response = MagicMock()
        mock_response.sandbox_id = "sb-123"
        mock_response.service_address = ""
        mock_response.exposed_ports = []
        mock_response.applied_ingress_mode = ""
        mock_response.applied_egress_mode = ""
        mock_response.HasField = MagicMock(return_value=False)

        mock_stub = AsyncMock()
        mock_stub.Start = AsyncMock(return_value=mock_response)

        sandbox._stub = mock_stub
        sandbox._channel = MagicMock()
        sandbox._auth_metadata = (("authorization", "Bearer test"),)

        await sandbox._start_async()

        call_kwargs = mock_stub.Start.call_args
        request = call_kwargs[0][0]
        assert hasattr(request, "resource_limits")
        assert hasattr(request, "resource_requests")
        # resource_limits should have cpu="8", memory="2Gi"
        assert request.resource_limits.cpu == "8"
        assert request.resource_limits.memory == "2Gi"
        # resource_requests should have cpu="1", memory="256Mi"
        assert request.resource_requests.cpu == "1"
        assert request.resource_requests.memory == "256Mi"
        # Legacy resources field should NOT be set
        assert not request.HasField("resources")

    @pytest.mark.asyncio
    async def test_start_async_maps_gpu_to_both_proto_fields(self, mock_api_key: str) -> None:
        """GPU config maps to both resource_limits and resource_requests."""
        from cwsandbox._types import ResourceOptions

        opts = ResourceOptions(
            requests={"cpu": "1"},
            limits={"cpu": "4"},
            gpu={"count": 2, "type": "A100"},
        )
        sandbox = Sandbox(command="echo", resources=opts)

        mock_response = MagicMock()
        mock_response.sandbox_id = "sb-456"
        mock_response.service_address = ""
        mock_response.exposed_ports = []
        mock_response.applied_ingress_mode = ""
        mock_response.applied_egress_mode = ""
        mock_response.HasField = MagicMock(return_value=False)

        mock_stub = AsyncMock()
        mock_stub.Start = AsyncMock(return_value=mock_response)

        sandbox._stub = mock_stub
        sandbox._channel = MagicMock()
        sandbox._auth_metadata = (("authorization", "Bearer test"),)

        await sandbox._start_async()

        request = mock_stub.Start.call_args[0][0]
        assert request.resource_limits.gpu.gpu_count == 2
        assert request.resource_limits.gpu.gpu_type == "A100"
        assert request.resource_requests.gpu.gpu_count == 2
        assert request.resource_requests.gpu.gpu_type == "A100"

    @pytest.mark.asyncio
    async def test_start_async_extracts_response_resource_fields(self, mock_api_key: str) -> None:
        """Resource limits/requests are extracted from StartSandboxResponse."""
        sandbox = Sandbox(command="echo")

        mock_response = MagicMock()
        mock_response.sandbox_id = "sb-789"
        mock_response.service_address = ""
        mock_response.exposed_ports = []
        mock_response.applied_ingress_mode = ""
        mock_response.applied_egress_mode = ""

        # Simulate response with resource fields present (no GPU)
        mock_limits = MagicMock()
        mock_limits.cpu = "8"
        mock_limits.memory = "2Gi"
        mock_limits.HasField = MagicMock(return_value=False)
        mock_requests = MagicMock()
        mock_requests.cpu = "1"
        mock_requests.memory = "256Mi"
        mock_response.requested_resource_limits = mock_limits
        mock_response.requested_resource_requests = mock_requests
        mock_response.HasField = MagicMock(
            side_effect=lambda f: f in ("requested_resource_limits", "requested_resource_requests")
        )

        mock_stub = AsyncMock()
        mock_stub.Start = AsyncMock(return_value=mock_response)

        sandbox._stub = mock_stub
        sandbox._channel = MagicMock()
        sandbox._auth_metadata = (("authorization", "Bearer test"),)

        await sandbox._start_async()

        assert sandbox.resource_limits == {"cpu": "8", "memory": "2Gi"}
        assert sandbox.resource_requests == {"cpu": "1", "memory": "256Mi"}
        assert sandbox.resource_gpu is None

    @pytest.mark.asyncio
    async def test_start_async_extracts_gpu_from_response(self, mock_api_key: str) -> None:
        """GPU fields are extracted from requested_resource_limits in StartSandboxResponse."""
        sandbox = Sandbox(command="echo")

        mock_gpu = MagicMock()
        mock_gpu.gpu_count = 2
        mock_gpu.gpu_type = "A100"
        mock_gpu.gpu_memory_gb = 80

        mock_limits = MagicMock()
        mock_limits.cpu = "8"
        mock_limits.memory = "2Gi"
        mock_limits.gpu = mock_gpu
        mock_limits.HasField = MagicMock(side_effect=lambda f: f == "gpu")

        mock_response = MagicMock()
        mock_response.sandbox_id = "sb-gpu"
        mock_response.service_address = ""
        mock_response.exposed_ports = []
        mock_response.applied_ingress_mode = ""
        mock_response.applied_egress_mode = ""
        mock_response.requested_resource_limits = mock_limits
        mock_response.HasField = MagicMock(side_effect=lambda f: f == "requested_resource_limits")

        mock_stub = AsyncMock()
        mock_stub.Start = AsyncMock(return_value=mock_response)

        sandbox._stub = mock_stub
        sandbox._channel = MagicMock()
        sandbox._auth_metadata = (("authorization", "Bearer test"),)

        await sandbox._start_async()

        assert sandbox.resource_gpu == {"count": 2, "type": "A100", "memory_gb": 80}

    def test_resource_properties_default_none(self) -> None:
        """resource_limits, resource_requests, and resource_gpu default to None."""
        sandbox = Sandbox(command="echo")
        assert sandbox.resource_limits is None
        assert sandbox.resource_requests is None
        assert sandbox.resource_gpu is None

    def test_from_sandbox_info_has_none_resources(self, mock_api_key: str) -> None:
        """Discovered sandboxes have None resource properties."""
        from google.protobuf import timestamp_pb2

        from cwsandbox._proto import gateway_pb2

        info = MagicMock()
        info.sandbox_id = "sb-disc"
        info.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        info.started_at_time = timestamp_pb2.Timestamp()
        info.tower_id = "tower-1"
        info.runway_id = "runway-1"
        info.tower_group_id = "tg-1"
        info.returncode = 0

        sandbox = Sandbox._from_sandbox_info(info, base_url="http://test", timeout_seconds=30.0)
        assert sandbox.resource_limits is None
        assert sandbox.resource_requests is None
        assert sandbox.resource_gpu is None


class TestSandboxList:
    """Tests for Sandbox.list class method."""

    @pytest.mark.asyncio
    async def test_list_returns_sandbox_instances(self, mock_api_key: str) -> None:
        """Test list() returns list of Sandbox instances."""
        from google.protobuf import timestamp_pb2

        from cwsandbox._proto import gateway_pb2

        mock_sandbox_info = gateway_pb2.SandboxInfo(
            sandbox_id="test-123",
            sandbox_status=gateway_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            runner_id="tower-1",
            runner_group_id="group-1",
            profile_id="runway-1",
        )

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(
            return_value=gateway_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
        )

        with (
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            sandboxes = await Sandbox.list(tags=["test-tag"])

            assert len(sandboxes) == 1
            assert isinstance(sandboxes[0], Sandbox)
            assert sandboxes[0].sandbox_id == "test-123"
            assert sandboxes[0].status == "running"
            call_kwargs = mock_stub.List.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, mock_api_key: str) -> None:
        """Test list() passes status filter to request."""
        from cwsandbox._proto import gateway_pb2

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(return_value=gateway_pb2.ListSandboxesResponse(sandboxes=[]))

        with (
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            await Sandbox.list(status="running")

            call_args = mock_stub.List.call_args[0][0]
            assert call_args.status == gateway_pb2.SANDBOX_STATUS_RUNNING
            call_kwargs = mock_stub.List.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_list_with_invalid_status_raises(self, mock_api_key: str) -> None:
        """Test list() raises ValueError for invalid status."""
        with pytest.raises(ValueError, match="not a valid SandboxStatus"):
            await Sandbox.list(status="invalid_status")

    @pytest.mark.asyncio
    async def test_list_empty_result(self, mock_api_key: str) -> None:
        """Test list() returns empty list when no sandboxes match."""
        from cwsandbox._proto import gateway_pb2

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(return_value=gateway_pb2.ListSandboxesResponse(sandboxes=[]))

        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            sandboxes = await Sandbox.list(tags=["nonexistent"])
            assert sandboxes == []

    @pytest.mark.asyncio
    async def test_list_include_stopped_passes_field(self, mock_api_key: str) -> None:
        """Test list(include_stopped=True) sets the field on the request."""
        from cwsandbox._proto import gateway_pb2

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(return_value=gateway_pb2.ListSandboxesResponse(sandboxes=[]))

        with (
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            await Sandbox.list(include_stopped=True)

            call_args = mock_stub.List.call_args[0][0]
            assert call_args.include_stopped is True

    @pytest.mark.asyncio
    async def test_list_include_stopped_default_false(self, mock_api_key: str) -> None:
        """Test list() does not set include_stopped by default."""
        from cwsandbox._proto import gateway_pb2

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(return_value=gateway_pb2.ListSandboxesResponse(sandboxes=[]))

        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            await Sandbox.list()

            call_args = mock_stub.List.call_args[0][0]
            assert call_args.include_stopped is False


class TestSandboxFromId:
    """Tests for Sandbox.from_id class method."""

    @pytest.mark.asyncio
    async def test_from_id_returns_sandbox_instance(self, mock_api_key: str) -> None:
        """Test from_id() returns a Sandbox instance."""
        from google.protobuf import timestamp_pb2

        from cwsandbox._proto import gateway_pb2

        mock_response = gateway_pb2.GetSandboxResponse(
            sandbox_id="test-123",
            sandbox_status=gateway_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            runner_id="tower-1",
            runner_group_id="group-1",
            profile_id="runway-1",
        )

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(return_value=mock_response)

        with (
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            sandbox = await Sandbox.from_id("test-123")

            assert isinstance(sandbox, Sandbox)
            assert sandbox.sandbox_id == "test-123"
            assert sandbox.status == "running"
            assert sandbox.runner_id == "tower-1"
            call_kwargs = mock_stub.Get.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_from_id_raises_not_found(self, mock_api_key: str) -> None:
        """Test from_id() raises SandboxNotFoundError for non-existent sandbox."""
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found"))

        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            with pytest.raises(SandboxNotFoundError, match="not found"):
                await Sandbox.from_id("nonexistent-id")


class TestSandboxDeleteClassMethod:
    """Tests for Sandbox.delete class method."""

    @pytest.mark.asyncio
    async def test_delete_returns_none_on_success(self, mock_api_key: str) -> None:
        """Test delete() returns None when deletion succeeds."""
        from cwsandbox._proto import gateway_pb2

        mock_response = gateway_pb2.DeleteSandboxResponse(success=True, error_message="")

        expected_metadata = (("authorization", "Bearer test-api-key"),)
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Delete = AsyncMock(return_value=mock_response)

        with (
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=expected_metadata),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            result = await Sandbox.delete("test-123")

            assert result is None
            call_kwargs = mock_stub.Delete.call_args[1]
            assert call_kwargs["metadata"] == expected_metadata

    @pytest.mark.asyncio
    async def test_delete_raises_not_found_by_default(self, mock_api_key: str) -> None:
        """Test delete() raises SandboxNotFoundError when sandbox doesn't exist."""
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Delete = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")
        )

        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            with pytest.raises(SandboxNotFoundError):
                await Sandbox.delete("nonexistent-id")

    @pytest.mark.asyncio
    async def test_delete_missing_ok_suppresses_not_found(self, mock_api_key: str) -> None:
        """Test delete(missing_ok=True) returns None instead of raising."""
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Delete = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")
        )

        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
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
    async def test_from_id_has_none_for_service_address(self, mock_api_key: str) -> None:
        """Test from_id() returns sandbox with None service_address."""
        from google.protobuf import timestamp_pb2

        from cwsandbox._proto import gateway_pb2

        mock_response = gateway_pb2.GetSandboxResponse(
            sandbox_id="test-123",
            sandbox_status=gateway_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            runner_id="tower-1",
            runner_group_id="group-1",
            profile_id="runway-1",
        )

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(return_value=mock_response)

        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch("cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub", return_value=mock_stub),
        ):
            sandbox = await Sandbox.from_id("test-123")

            assert sandbox.service_address is None
            assert sandbox.exposed_ports is None


class TestSandboxRunwayAndTowerIds:
    """Tests for profile_ids and runner_ids parameters."""

    def test_profile_ids_stored_on_sandbox(self) -> None:
        """Test profile_ids are stored on sandbox instance."""
        sandbox = Sandbox(profile_ids=["runway-1", "runway-2"])
        assert sandbox._profile_ids == ["runway-1", "runway-2"]

    def test_runner_ids_stored_on_sandbox(self) -> None:
        """Test runner_ids are stored on sandbox instance."""
        sandbox = Sandbox(runner_ids=["tower-1", "tower-2"])
        assert sandbox._runner_ids == ["tower-1", "tower-2"]

    def test_empty_profile_ids_overrides_defaults(self) -> None:
        """Test empty profile_ids list overrides defaults."""
        from cwsandbox._defaults import SandboxDefaults

        defaults = SandboxDefaults(profile_ids=("default-runway",))
        sandbox = Sandbox(profile_ids=[], defaults=defaults)
        assert sandbox._profile_ids == []

    def test_empty_runner_ids_overrides_defaults(self) -> None:
        """Test empty runner_ids list overrides defaults."""
        from cwsandbox._defaults import SandboxDefaults

        defaults = SandboxDefaults(runner_ids=("default-tower",))
        sandbox = Sandbox(runner_ids=[], defaults=defaults)
        assert sandbox._runner_ids == []

    def test_none_profile_ids_uses_defaults(self) -> None:
        """Test None profile_ids falls back to defaults."""
        from cwsandbox._defaults import SandboxDefaults

        defaults = SandboxDefaults(profile_ids=("default-runway",))
        sandbox = Sandbox(defaults=defaults)
        assert sandbox._profile_ids == ["default-runway"]

    def test_none_runner_ids_uses_defaults(self) -> None:
        """Test None runner_ids falls back to defaults."""
        from cwsandbox._defaults import SandboxDefaults

        defaults = SandboxDefaults(runner_ids=("default-tower",))
        sandbox = Sandbox(defaults=defaults)
        assert sandbox._runner_ids == ["default-tower"]

    def test_run_passes_profile_ids(self) -> None:
        """Test Sandbox.run passes profile_ids to sandbox."""
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run(profile_ids=["runway-1"])
            assert sandbox._profile_ids == ["runway-1"]

    def test_run_passes_runner_ids(self) -> None:
        """Test Sandbox.run passes runner_ids to sandbox."""
        with patch.object(Sandbox, "start"):
            sandbox = Sandbox.run(runner_ids=["tower-1"])
            assert sandbox._runner_ids == ["tower-1"]


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
        from cwsandbox._proto import gateway_pb2

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
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING

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


class TestSecretsParameter:
    """Tests for secrets parameter."""

    def test_init_secrets_none_omits_from_kwargs(self) -> None:
        """Test Sandbox.__init__ omits secrets when None and no defaults."""
        sandbox = Sandbox(command="echo", args=["hello"], secrets=None)
        assert "secrets" not in sandbox._start_kwargs

    def test_init_secrets_accepted(self) -> None:
        """Test Sandbox.__init__ accepts list of Secret."""
        secret = Secret(store="wandb", name="HF_TOKEN", env_var="HF_TOKEN", field="api_key")
        sandbox = Sandbox(command="echo", args=["hello"], secrets=[secret])
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 1 and stored[0] is secret

    def test_init_secret_env_var_defaults_to_name(self) -> None:
        """Test Secret.env_var defaults to name when not specified."""
        secret = Secret(store="wandb", name="WANDB_API_KEY")
        sandbox = Sandbox(command="echo", args=["hello"], secrets=[secret])
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 1
        assert stored[0].store == "wandb"
        assert stored[0].name == "WANDB_API_KEY"
        assert stored[0].env_var == "WANDB_API_KEY"
        assert stored[0].field == ""

    def test_init_uses_defaults_secrets_when_not_specified(self) -> None:
        """Test Sandbox.__init__ uses defaults.secrets when secrets is None."""
        secret = Secret(store="wandb", name="HF_TOKEN")
        defaults = SandboxDefaults(secrets=(secret,))
        sandbox = Sandbox(command="echo", args=["hello"], defaults=defaults)
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 1 and stored[0].store == "wandb"

    def test_init_merges_defaults_and_explicit_secrets(self) -> None:
        """Test secrets from defaults and call-site are merged (defaults first)."""
        default_secret = Secret(store="wandb", name="HF_TOKEN")
        defaults = SandboxDefaults(secrets=(default_secret,))
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            defaults=defaults,
            secrets=[
                Secret(store="vault", name="OPENAI_API_KEY", env_var="OPENAI_KEY"),
            ],
        )
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 2
        assert stored[0].store == "wandb"
        assert stored[1].store == "vault"

    def test_init_raises_on_conflicting_env_var(self) -> None:
        """Test that conflicting secrets targeting the same env_var raise ValueError."""
        defaults = SandboxDefaults(
            secrets=(Secret(store="wandb", name="HF_TOKEN", env_var="HF_TOKEN"),)
        )
        with pytest.raises(ValueError, match="Conflicting secrets for env_var 'HF_TOKEN'"):
            Sandbox(
                command="echo",
                args=["hello"],
                defaults=defaults,
                secrets=[Secret(store="vault", name="hf-token", env_var="HF_TOKEN")],
            )

    def test_init_allows_identical_duplicate_secrets(self) -> None:
        """Test that identical secrets targeting the same env_var are allowed."""
        secret = Secret(store="wandb", name="HF_TOKEN", env_var="HF_TOKEN")
        defaults = SandboxDefaults(secrets=(secret,))
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            defaults=defaults,
            secrets=[Secret(store="wandb", name="HF_TOKEN", env_var="HF_TOKEN")],
        )
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 1
        assert stored[0].store == "wandb" and stored[0].env_var == "HF_TOKEN"

    @pytest.mark.asyncio
    async def test_identical_duplicate_secrets_deduplicated_in_proto(
        self, mock_api_key: str
    ) -> None:
        """Identical secrets in defaults + explicit are deduplicated before proto serialization.

        When the same Secret appears in both SandboxDefaults and the per-sandbox
        secrets list, the merge logic deduplicates them so only one copy is
        serialized into the proto request, avoiding backend rejection of duplicate
        env_var targets.
        """
        # Same secret in defaults AND explicit — a natural "belt and suspenders" pattern
        shared_secret = Secret(store="wandb", name="HF_TOKEN", env_var="HF_TOKEN")
        defaults = SandboxDefaults(secrets=(shared_secret,))
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            defaults=defaults,
            secrets=[Secret(store="wandb", name="HF_TOKEN", env_var="HF_TOKEN")],
        )

        # Step 1: Merge logic deduplicates identical secrets
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 1, "Identical secrets should be deduplicated during merge"

        # Step 2: Wire up mock to reach proto conversion in _start_async
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

        # Step 3: Inspect the proto request — only one entry, no duplicate
        request = mock_stub.Start.call_args[0][0]
        assert len(request.secret_stores) == 1, "Both secrets share store 'wandb'"
        wandb_store = request.secret_stores[0]
        assert wandb_store.store_name == "wandb"
        assert len(wandb_store.secrets) == 1, "Deduplicated: only one SecretMapping entry"
        assert wandb_store.secrets[0].path == "HF_TOKEN"
        assert wandb_store.secrets[0].env_var == "HF_TOKEN"

    def test_init_raises_on_conflicting_field(self) -> None:
        """Test that secrets differing only in field raise ValueError."""
        with pytest.raises(ValueError, match="Conflicting secrets for env_var"):
            Sandbox(
                command="echo",
                args=["hello"],
                secrets=[
                    Secret(store="vault", name="creds", field="password", env_var="DB_PASS"),
                    Secret(store="vault", name="creds", field="token", env_var="DB_PASS"),
                ],
            )

    def test_init_coerces_dicts_to_secret(self) -> None:
        """Test Sandbox.__init__ converts dicts to Secret objects."""
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            secrets=[{"store": "wandb", "name": "HF_TOKEN"}],
        )
        stored = sandbox._start_kwargs["secrets"]
        assert len(stored) == 1
        assert isinstance(stored[0], Secret)
        assert stored[0].store == "wandb"
        assert stored[0].name == "HF_TOKEN"
        assert stored[0].env_var == "HF_TOKEN"

    def test_init_coerces_dicts_with_all_fields(self) -> None:
        """Test dict coercion works with all Secret fields specified."""
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            secrets=[
                {"store": "wandb", "name": "db-creds", "field": "password", "env_var": "DB_PASS"},
            ],
        )
        stored = sandbox._start_kwargs["secrets"]
        assert stored[0].field == "password"
        assert stored[0].env_var == "DB_PASS"

    def test_init_rejects_dict_with_bad_keys(self) -> None:
        """Test dict coercion raises TypeError on invalid keys."""
        with pytest.raises(TypeError):
            Sandbox(
                command="echo",
                args=["hello"],
                secrets=[{"store_name": "wandb", "secrets": []}],
            )


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
        sandbox._state = _Running(sandbox_id="test-id")

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
        from cwsandbox._wandb import ExecOutcome

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")

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
        from unittest.mock import MagicMock

        from google.protobuf import timestamp_pb2

        from cwsandbox._proto import gateway_pb2

        info = MagicMock()
        info.sandbox_id = "test-123"
        info.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        info.started_at_time = timestamp_pb2.Timestamp(seconds=1234567890)
        info.runner_id = "tower-1"
        info.runner_group_id = "group-1"
        info.profile_id = "runway-1"
        sandbox = Sandbox._from_sandbox_info(
            info,
            base_url="https://api.example.com",
            timeout_seconds=300.0,
        )

        assert sandbox._startup_recorded is True
        assert sandbox._start_accepted_at is None

    def test_wait_records_startup_time_to_session(self) -> None:
        """Test wait() records startup time to session reporter."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Starting(sandbox_id="test-id")
        sandbox._start_accepted_at = 100.0

        mock_session = MagicMock()
        sandbox._session = mock_session
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        mock_response.runner_id = "tower-1"
        mock_response.runner_group_id = None
        mock_response.profile_id = "runway-1"
        mock_response.started_at_time = None
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        with patch("time.monotonic", return_value=102.5):
            sandbox.wait()

        mock_session._record_startup_time.assert_called_once()
        call_args = mock_session._record_startup_time.call_args[0]
        assert call_args[0] == pytest.approx(2.5, rel=0.1)

    def test_startup_time_only_recorded_once(self) -> None:
        """Test startup time is only recorded once."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Starting(sandbox_id="test-id")
        sandbox._start_accepted_at = 100.0
        sandbox._startup_recorded = True

        mock_session = MagicMock()
        sandbox._session = mock_session
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        mock_response.runner_id = "tower-1"
        mock_response.runner_group_id = None
        mock_response.profile_id = "runway-1"
        mock_response.started_at_time = None
        sandbox._stub.Get = AsyncMock(return_value=mock_response)

        sandbox.wait()

        mock_session._record_startup_time.assert_not_called()


class TestTranslateRpcError:
    """Tests for _translate_rpc_error function."""

    def test_not_found_returns_sandbox_not_found_error(self) -> None:
        """Test NOT_FOUND status code returns SandboxNotFoundError."""
        from cwsandbox._sandbox import _translate_rpc_error

        error = MockRpcError(grpc.StatusCode.NOT_FOUND, "sandbox not found")
        result = _translate_rpc_error(error, sandbox_id="test-123")

        assert isinstance(result, SandboxNotFoundError)
        assert result.sandbox_id == "test-123"
        assert "test-123" in str(result)

    def test_not_found_without_sandbox_id_uses_details(self) -> None:
        """Test NOT_FOUND without sandbox_id uses error details."""
        from cwsandbox._sandbox import _translate_rpc_error

        error = MockRpcError(grpc.StatusCode.NOT_FOUND, "resource not found")
        result = _translate_rpc_error(error)

        assert isinstance(result, SandboxNotFoundError)
        assert "resource not found" in str(result)

    def test_cancelled_returns_sandbox_not_running_error(self) -> None:
        """Test CANCELLED status code returns SandboxNotRunningError."""
        from cwsandbox._sandbox import _translate_rpc_error
        from cwsandbox.exceptions import SandboxNotRunningError

        error = MockRpcError(grpc.StatusCode.CANCELLED, "request cancelled")
        result = _translate_rpc_error(error, operation="Start sandbox")

        assert isinstance(result, SandboxNotRunningError)
        assert "cancelled" in str(result)

    def test_cancelled_with_sandbox_id_includes_id(self) -> None:
        """Test CANCELLED with sandbox_id includes ID in message."""
        from cwsandbox._sandbox import _translate_rpc_error
        from cwsandbox.exceptions import SandboxNotRunningError

        error = MockRpcError(grpc.StatusCode.CANCELLED, "cancelled")
        result = _translate_rpc_error(error, sandbox_id="test-456", operation="Execute command")

        assert isinstance(result, SandboxNotRunningError)
        assert "test-456" in str(result)

    def test_deadline_exceeded_returns_sandbox_timeout_error(self) -> None:
        """Test DEADLINE_EXCEEDED status code returns SandboxTimeoutError."""
        from cwsandbox._sandbox import _translate_rpc_error
        from cwsandbox.exceptions import SandboxTimeoutError

        error = MockRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "timeout after 30s")
        result = _translate_rpc_error(error, operation="Execute command")

        assert isinstance(result, SandboxTimeoutError)
        assert "timed out" in str(result)

    def test_unavailable_returns_sandbox_not_running_error(self) -> None:
        """Test UNAVAILABLE status code returns SandboxNotRunningError."""
        from cwsandbox._sandbox import _translate_rpc_error
        from cwsandbox.exceptions import SandboxNotRunningError

        error = MockRpcError(grpc.StatusCode.UNAVAILABLE, "connection refused")
        result = _translate_rpc_error(error)

        assert isinstance(result, SandboxNotRunningError)
        assert "unavailable" in str(result).lower()

    def test_permission_denied_returns_auth_error(self) -> None:
        """Test PERMISSION_DENIED status code returns CWSandboxAuthenticationError."""
        from cwsandbox._sandbox import _translate_rpc_error
        from cwsandbox.exceptions import CWSandboxAuthenticationError

        error = MockRpcError(grpc.StatusCode.PERMISSION_DENIED, "access denied")
        result = _translate_rpc_error(error)

        assert isinstance(result, CWSandboxAuthenticationError)
        assert "denied" in str(result).lower()

    def test_unauthenticated_returns_auth_error(self) -> None:
        """Test UNAUTHENTICATED status code returns CWSandboxAuthenticationError."""
        from cwsandbox._sandbox import _translate_rpc_error
        from cwsandbox.exceptions import CWSandboxAuthenticationError

        error = MockRpcError(grpc.StatusCode.UNAUTHENTICATED, "invalid token")
        result = _translate_rpc_error(error)

        assert isinstance(result, CWSandboxAuthenticationError)
        assert "authentication" in str(result).lower()

    def test_other_status_delegates_to_shared_translator(self) -> None:
        """Test other status codes delegate to shared transport translator."""
        from cwsandbox._sandbox import _translate_rpc_error

        error = MockRpcError(grpc.StatusCode.INTERNAL, "internal error")
        result = _translate_rpc_error(error, operation="Test operation")

        assert isinstance(result, SandboxError)
        assert "failed" in str(result)

    def test_empty_details_uses_string_repr(self) -> None:
        """Test that empty details falls back to str(e)."""
        from cwsandbox._sandbox import _translate_rpc_error

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
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        assert stdin_received_time >= ready_sent_time, (
            f"Stdin received at {stdin_received_time} but ready sent at {ready_sent_time}"
        )

    def test_exec_stdin_ready_timeout(self) -> None:
        """Test SandboxTimeoutError raised when ready signal not received."""
        from cwsandbox.exceptions import SandboxTimeoutError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Never send ready signal - just hang (simulated by no responses)
        # The test will timeout waiting for ready
        mock_call = MockStreamCall(responses=[])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
                return_value=mock_stub,
            ),
        ):
            process = sandbox.exec(["cat"], stdin=True, timeout_seconds=0.1)

            with pytest.raises(SandboxTimeoutError, match="ready signal"):
                process.result()

    def test_exec_stdin_error_before_ready(self) -> None:
        """Test error before ready signal unblocks and propagates error."""
        from cwsandbox.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        # Never send ready - simulates hung connection
        mock_call = MockStreamCall(responses=[])
        mock_channel, mock_stub = create_mock_channel_and_stub(mock_call)

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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
        from cwsandbox._proto import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
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
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
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


class TestShellStreamingTTY:
    """Tests for _exec_streaming_tty_async via the Sandbox.shell() public API."""

    @staticmethod
    def _make_sandbox() -> Sandbox:
        """Create a sandbox pre-configured in RUNNING state for shell tests."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        return sandbox

    @staticmethod
    def _streaming_patches(sandbox: Sandbox, mock_call: MockBidirectionalStreamCall):  # type: ignore[no-untyped-def]
        """Return a context manager that patches streaming infra for shell tests."""
        from contextlib import ExitStack

        mock_channel, mock_stub = create_mock_channel_and_stub_bidirectional(mock_call)

        stack = ExitStack()
        stack.enter_context(
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock)
        )
        stack.enter_context(patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()))
        stack.enter_context(
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("localhost:443", True))
        )
        stack.enter_context(patch("cwsandbox._sandbox.create_channel", return_value=mock_channel))
        stack.enter_context(
            patch(
                "cwsandbox._sandbox.streaming_pb2_grpc.GatewayStreamingServiceStub",
                return_value=mock_stub,
            )
        )
        return stack

    def test_shell_happy_path_returns_terminal_result(self) -> None:
        """shell() returns TerminalResult with exit code on normal completion."""
        from google.protobuf import timestamp_pb2

        from cwsandbox._types import TerminalResult

        sandbox = self._make_sandbox()

        async def responses() -> AsyncIterator[Any]:
            ready = MagicMock()
            ready.HasField = lambda f: f == "ready"
            ready.ready.ready_at = timestamp_pb2.Timestamp(seconds=1)
            yield ready

            out = MagicMock()
            out.HasField = lambda f: f == "output"
            out.output.data = b"hello\r\n"
            yield out

            exit_resp = MagicMock()
            exit_resp.HasField = lambda f: f == "exit"
            exit_resp.exit.exit_code = 0
            yield exit_resp

        mock_call = MockBidirectionalStreamCall(response_generator=responses)
        with self._streaming_patches(sandbox, mock_call):
            session = sandbox.shell(["/bin/bash"], width=80, height=24)

            chunks: list[bytes] = []
            for chunk in session.output:
                chunks.append(chunk)

            result = session.result()

        assert isinstance(result, TerminalResult)
        assert result.returncode == 0
        assert result.command == ["/bin/bash"]
        assert b"hello\r\n" in chunks

    def test_shell_server_error_raises(self) -> None:
        """Server error response raises SandboxExecutionError."""
        from google.protobuf import timestamp_pb2

        from cwsandbox.exceptions import SandboxExecutionError

        sandbox = self._make_sandbox()

        async def responses() -> AsyncIterator[Any]:
            ready = MagicMock()
            ready.HasField = lambda f: f == "ready"
            ready.ready.ready_at = timestamp_pb2.Timestamp(seconds=1)
            yield ready

            err = MagicMock()
            err.HasField = lambda f: f == "error"
            err.error.message = "command not found"
            yield err

        mock_call = MockBidirectionalStreamCall(response_generator=responses)
        with self._streaming_patches(sandbox, mock_call):
            session = sandbox.shell(["/bin/bash"])
            # Drain output — error propagates as exception in StreamReader
            chunks: list[bytes] = []
            try:
                for chunk in session.output:
                    chunks.append(chunk)
            except SandboxExecutionError:
                pass

            with pytest.raises(SandboxExecutionError, match="command not found"):
                session.result()

    def test_shell_grpc_error_propagates(self) -> None:
        """gRPC error during streaming is translated and raised."""
        sandbox = self._make_sandbox()

        mock_call = MockBidirectionalStreamCall(
            error_on_read=MockAioRpcError(grpc.StatusCode.UNAVAILABLE, "connection lost"),
        )
        with self._streaming_patches(sandbox, mock_call):
            session = sandbox.shell(["/bin/bash"])

            # The error surfaces through the output stream or result
            try:
                list(session.output)
            except SandboxError:
                pass

            with pytest.raises(SandboxError):
                session.result()

    def test_shell_stdin_forwarding(self) -> None:
        """stdin.write() sends data chunks to the server."""
        from google.protobuf import timestamp_pb2

        sandbox = self._make_sandbox()

        stdin_received: list[bytes] = []

        def on_write(request: Any) -> None:
            if hasattr(request, "stdin") and request.HasField("stdin"):
                stdin_received.append(request.stdin.data)

        close_event = asyncio.Event()

        def on_write_with_close(request: Any) -> None:
            on_write(request)
            if hasattr(request, "close") and request.HasField("close"):
                close_event.set()

        async def responses() -> AsyncIterator[Any]:
            ready = MagicMock()
            ready.HasField = lambda f: f == "ready"
            ready.ready.ready_at = timestamp_pb2.Timestamp(seconds=1)
            yield ready

            try:
                await asyncio.wait_for(close_event.wait(), timeout=5.0)
            except TimeoutError:
                pass

            exit_resp = MagicMock()
            exit_resp.HasField = lambda f: f == "exit"
            exit_resp.exit.exit_code = 0
            yield exit_resp

        mock_call = MockBidirectionalStreamCall(
            response_generator=responses, on_write=on_write_with_close
        )
        with self._streaming_patches(sandbox, mock_call):
            session = sandbox.shell(["/bin/bash"])
            session.stdin.write(b"echo hello\n").result()
            session.stdin.close().result()
            list(session.output)
            result = session.result()

        assert result.returncode == 0
        assert b"echo hello\n" in stdin_received

    def test_shell_resize_sends_message(self) -> None:
        """session.resize() sends a resize request to the server."""
        from google.protobuf import timestamp_pb2

        sandbox = self._make_sandbox()

        resize_received: list[tuple[int, int]] = []

        def on_write(request: Any) -> None:
            if hasattr(request, "resize") and request.HasField("resize"):
                resize_received.append((request.resize.width, request.resize.height))

        resize_done = asyncio.Event()

        def on_write_with_resize(request: Any) -> None:
            on_write(request)
            if hasattr(request, "resize") and request.HasField("resize"):
                resize_done.set()

        async def responses() -> AsyncIterator[Any]:
            ready = MagicMock()
            ready.HasField = lambda f: f == "ready"
            ready.ready.ready_at = timestamp_pb2.Timestamp(seconds=1)
            yield ready

            # Wait for resize to be received before sending exit
            try:
                await asyncio.wait_for(resize_done.wait(), timeout=5.0)
            except TimeoutError:
                pass

            exit_resp = MagicMock()
            exit_resp.HasField = lambda f: f == "exit"
            exit_resp.exit.exit_code = 0
            yield exit_resp

        mock_call = MockBidirectionalStreamCall(
            response_generator=responses, on_write=on_write_with_resize
        )
        with self._streaming_patches(sandbox, mock_call):
            session = sandbox.shell(["/bin/bash"], width=80, height=24)
            session.resize(120, 40)
            list(session.output)
            session.result()

        assert (120, 40) in resize_received

    def test_shell_early_failure_unblocks_output_stream(self) -> None:
        """Failure before gRPC call propagates to output queue so readers don't hang."""
        sandbox = self._make_sandbox()

        # Patch _ensure_started_async to raise — simulates an auth or network
        # failure before the gRPC stream is established.
        async def fail_start() -> None:
            raise SandboxNotRunningError("Sandbox has been stopped")

        with patch.object(sandbox, "_ensure_started_async", side_effect=fail_start):
            session = sandbox.shell(["/bin/bash"])

            # The output stream must terminate (with an exception), not hang
            with pytest.raises(SandboxNotRunningError):
                list(session.output)

    def test_shell_cancel_terminates_output_stream(self) -> None:
        """Cancelled shell session must deliver a sentinel to the output stream.

        When a TerminalSession future is cancelled, _exec_streaming_tty_async
        receives CancelledError. The finally block must always enqueue the
        sentinel so StreamReader consumers don't block
        forever waiting for a termination signal that never arrives.
        """
        import threading
        import time

        from google.protobuf import timestamp_pb2

        sandbox = self._make_sandbox()

        # Simulate a long-running shell: ready → output → hang forever
        hang_event = asyncio.Event()

        async def response_generator() -> AsyncIterator[Any]:
            ready = MagicMock()
            ready.HasField = lambda field: field == "ready"
            ready.ready.ready_at = timestamp_pb2.Timestamp(seconds=1234567890)
            yield ready

            output = MagicMock()
            output.HasField = lambda field: field == "output"
            output.output.data = b"bash-5.2$ "
            yield output

            # Block indefinitely — simulates ongoing interactive session
            await hang_event.wait()

        mock_call = MockBidirectionalStreamCall(response_generator=response_generator)
        with self._streaming_patches(sandbox, mock_call):
            session = sandbox.shell(["/bin/bash"])

            # Drain output in a background thread (like a real consumer would)
            output_chunks: list[bytes] = []
            output_done = threading.Event()

            def drain_output() -> None:
                try:
                    for chunk in session.output:
                        output_chunks.append(chunk)
                except Exception:
                    pass
                output_done.set()

            reader = threading.Thread(target=drain_output, daemon=True)
            reader.start()

            # Wait for the first chunk so we know the session is active
            time.sleep(1.0)
            assert len(output_chunks) >= 1, "Should have received the prompt chunk"

            # Cancel the session future — sends CancelledError to the coroutine
            session._future.cancel()

            # The output consumer MUST terminate after cancellation.
            # Bug: it hangs forever because the sentinel is skipped.
            finished = output_done.wait(timeout=5.0)
            if not finished:
                # Force cleanup so the test doesn't hang the suite
                session.output.close()
                output_done.wait(timeout=2.0)
            assert finished, (
                "Output stream did not terminate after cancellation — "
                "CancelledError path skipped the output_queue sentinel"
            )


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
        from cwsandbox import OperationRef

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


class TestTerminalStateProperties:
    """Verify property accessors return correct values from _Terminal state."""

    def test_returncode_from_terminal_state(self) -> None:
        """Properties read returncode from _Terminal state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED, returncode=42)
        assert sandbox.returncode == 42

    def test_status_from_terminal_state(self) -> None:
        """Properties read status from _Terminal state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.FAILED)
        assert sandbox.status == SandboxStatus.FAILED

    def test_sandbox_id_from_terminal_state(self) -> None:
        """Properties read sandbox_id from _Terminal state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Terminal(sandbox_id="sb-terminal", status=SandboxStatus.COMPLETED)
        assert sandbox.sandbox_id == "sb-terminal"

    def test_runner_id_from_terminal_state(self) -> None:
        """Properties read runner_id from _Terminal state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Terminal(
            sandbox_id="sb-1", status=SandboxStatus.COMPLETED, runner_id="tower-99"
        )
        assert sandbox.runner_id == "tower-99"

    def test_profile_id_from_terminal_state(self) -> None:
        """Properties read profile_id from _Terminal state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Terminal(
            sandbox_id="sb-1", status=SandboxStatus.COMPLETED, profile_id="runway-99"
        )
        assert sandbox.profile_id == "runway-99"


class TestTerminatingStatus:
    """Tests for TERMINATING status and _Stopping lifecycle state."""

    def test_from_proto_terminating(self) -> None:
        """from_proto(9) returns TERMINATING."""
        from cwsandbox._proto import gateway_pb2

        status = SandboxStatus.from_proto(gateway_pb2.SANDBOX_STATUS_TERMINATING)
        assert status == SandboxStatus.TERMINATING

    def test_to_proto_terminating(self) -> None:
        """TERMINATING round-trips through to_proto."""
        from cwsandbox._proto import gateway_pb2

        assert SandboxStatus.TERMINATING.to_proto() == gateway_pb2.SANDBOX_STATUS_TERMINATING

    def test_terminating_not_in_terminal_statuses(self) -> None:
        """TERMINATING is not a terminal status."""
        from cwsandbox._sandbox import _TERMINAL_STATUSES

        assert SandboxStatus.TERMINATING not in _TERMINAL_STATUSES

    def test_lifecycle_state_from_info_terminating(self) -> None:
        """_lifecycle_state_from_info maps TERMINATING to _Stopping with metadata."""
        from cwsandbox._sandbox import _lifecycle_state_from_info

        state = _lifecycle_state_from_info(
            sandbox_id="sb-1",
            status=SandboxStatus.TERMINATING,
            runner_id="tower-1",
            profile_id="runway-1",
            runner_group_id="group-1",
        )
        assert isinstance(state, _Stopping)
        assert state.sandbox_id == "sb-1"
        assert state.status == SandboxStatus.TERMINATING
        assert state.runner_id == "tower-1"
        assert state.profile_id == "runway-1"
        assert state.runner_group_id == "group-1"

    def test_stopping_is_frozen(self) -> None:
        """_Stopping dataclass is frozen (immutable)."""
        state = _Stopping(sandbox_id="sb-1")
        with pytest.raises(AttributeError):
            state.sandbox_id = "sb-2"  # type: ignore[misc]


class TestStoppingStateTransitions:
    """Tests for state transition guards involving _Stopping."""

    def test_stopping_to_terminal_completed_allowed(self) -> None:
        """_Stopping -> _Terminal(COMPLETED) is allowed."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1", runner_id="tower-1")

        mock_info = MagicMock()
        mock_info.sandbox_id = "sb-1"
        mock_info.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_info.runner_id = "tower-1"
        mock_info.profile_id = ""
        mock_info.runner_group_id = ""
        mock_info.started_at_time = None
        mock_info.returncode = 0

        new_state = sandbox._apply_sandbox_info(mock_info, source="poll")
        assert isinstance(new_state, _Terminal)
        assert new_state.status == SandboxStatus.COMPLETED

    def test_stopping_to_running_rejected(self) -> None:
        """_Stopping -> _Running is rejected (stale poll response)."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")

        mock_info = MagicMock()
        mock_info.sandbox_id = "sb-1"
        mock_info.sandbox_status = gateway_pb2.SANDBOX_STATUS_RUNNING
        mock_info.runner_id = ""
        mock_info.profile_id = ""
        mock_info.runner_group_id = ""
        mock_info.started_at_time = None

        new_state = sandbox._apply_sandbox_info(mock_info, source="poll")
        assert isinstance(new_state, _Stopping)

    def test_stopping_to_starting_rejected(self) -> None:
        """_Stopping -> _Starting is rejected (stale poll response)."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")

        mock_info = MagicMock()
        mock_info.sandbox_id = "sb-1"
        mock_info.sandbox_status = gateway_pb2.SANDBOX_STATUS_PENDING
        mock_info.runner_id = ""
        mock_info.profile_id = ""
        mock_info.runner_group_id = ""
        mock_info.started_at_time = None

        new_state = sandbox._apply_sandbox_info(mock_info, source="poll")
        assert isinstance(new_state, _Stopping)

    def test_stopping_to_terminal_failed_allowed(self) -> None:
        """_Stopping -> _Terminal(FAILED) is allowed."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")

        mock_info = MagicMock()
        mock_info.sandbox_id = "sb-1"
        mock_info.sandbox_status = gateway_pb2.SANDBOX_STATUS_FAILED
        mock_info.runner_id = ""
        mock_info.profile_id = ""
        mock_info.runner_group_id = ""
        mock_info.started_at_time = None

        new_state = sandbox._apply_sandbox_info(mock_info, source="poll")
        assert isinstance(new_state, _Terminal)
        assert new_state.status == SandboxStatus.FAILED


class TestStoppingOperationGuards:
    """Tests for operation guards during _Stopping state."""

    def test_exec_blocked_in_stopping(self) -> None:
        """exec() raises SandboxNotRunningError in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            sandbox.exec(["echo", "hello"]).result()

    def test_read_file_blocked_in_stopping(self) -> None:
        """read_file() raises SandboxNotRunningError in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            sandbox.read_file("/tmp/test").result()

    def test_write_file_blocked_in_stopping(self) -> None:
        """write_file() raises SandboxNotRunningError in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            sandbox.write_file("/tmp/test", b"data").result()

    def test_shell_blocked_in_stopping(self) -> None:
        """shell() raises SandboxNotRunningError in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            session = sandbox.shell()
            session.result()

    def test_stream_logs_follow_blocked_in_stopping(self) -> None:
        """stream_logs(follow=True) raises SandboxNotRunningError in _Stopping."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()

        with pytest.raises(SandboxNotRunningError, match="terminating"):
            reader = sandbox.stream_logs(follow=True)
            # Iterate to trigger the async method
            for _ in reader:
                pass

    def test_stream_logs_no_follow_allowed_in_stopping(self) -> None:
        """stream_logs(follow=False) is allowed in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")

        # The guard check should not block non-follow log streaming.
        # _is_stopping is True but _is_done is False, so follow=False passes.
        assert sandbox._is_stopping is True
        assert not sandbox._is_done


class TestStoppingStopFlow:
    """Tests for stop() behavior with _Stopping lifecycle."""

    def test_stop_sends_rpc_then_sets_stopping(self) -> None:
        """stop() sends Stop RPC then transitions to _Stopping."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id", runner_id="tower-1")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()

        mock_stub = MagicMock()
        mock_stop_response = MagicMock()
        mock_stop_response.success = True
        mock_stub.Stop = AsyncMock(return_value=mock_stop_response)

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.sandbox_id = "test-id"
        mock_get_response.runner_id = "tower-1"
        mock_get_response.profile_id = ""
        mock_get_response.runner_group_id = ""
        mock_get_response.started_at_time = None
        mock_get_response.returncode = 0
        mock_stub.Get = AsyncMock(return_value=mock_get_response)

        sandbox._stub = mock_stub
        sandbox.stop().result()
        mock_stub.Stop.assert_called_once()

    def test_stop_rpc_failure_no_state_change(self) -> None:
        """Stop RPC failure does not change state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = MagicMock()

        sandbox._stub.Stop = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.INTERNAL, "server error")
        )

        with pytest.raises(SandboxError):
            sandbox.stop().result()

    def test_stop_missing_ok_not_found_sets_terminal(self) -> None:
        """stop(missing_ok=True) + NOT_FOUND -> _Terminal(TERMINATED)."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id", runner_id="tower-1")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = MagicMock()

        sandbox._stub.Stop = AsyncMock(
            side_effect=MockRpcError(grpc.StatusCode.NOT_FOUND, "Not found")
        )

        result = sandbox.stop(missing_ok=True).result()
        assert result is None
        assert isinstance(sandbox._state, _Terminal)
        assert sandbox._state.status == SandboxStatus.TERMINATED

    def test_repeated_stop_joins_shared_task(self) -> None:
        """Repeated stop() calls join the same task."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()

        mock_stub = MagicMock()
        mock_stop_response = MagicMock()
        mock_stop_response.success = True
        mock_stub.Stop = AsyncMock(return_value=mock_stop_response)

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.sandbox_id = "test-id"
        mock_get_response.runner_id = ""
        mock_get_response.profile_id = ""
        mock_get_response.runner_group_id = ""
        mock_get_response.started_at_time = None
        mock_get_response.returncode = 0
        mock_stub.Get = AsyncMock(return_value=mock_get_response)

        sandbox._stub = mock_stub
        sandbox.stop().result()
        sandbox.stop().result()  # Second call should be idempotent
        # Stop RPC should only be called once
        mock_stub.Stop.assert_called_once()


class TestStoppingProperties:
    """Tests for property accessors in _Stopping state."""

    def test_status_returns_terminating(self) -> None:
        """status property returns TERMINATING in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1")
        assert sandbox.status == SandboxStatus.TERMINATING

    def test_returncode_none_in_stopping(self) -> None:
        """returncode is None in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1")
        assert sandbox.returncode is None

    def test_runner_id_accessible_in_stopping(self) -> None:
        """runner_id is accessible in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1", runner_id="tower-1")
        assert sandbox.runner_id == "tower-1"

    def test_profile_id_accessible_in_stopping(self) -> None:
        """profile_id is accessible in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1", profile_id="runway-1")
        assert sandbox.profile_id == "runway-1"

    def test_runner_group_id_accessible_in_stopping(self) -> None:
        """runner_group_id is accessible in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1", runner_group_id="group-1")
        assert sandbox.runner_group_id == "group-1"

    def test_started_at_accessible_in_stopping(self) -> None:
        """started_at is accessible in _Stopping state."""
        from datetime import UTC, datetime

        ts = datetime.now(UTC)
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1", started_at=ts)
        assert sandbox.started_at == ts

    def test_sandbox_id_accessible_in_stopping(self) -> None:
        """sandbox_id is accessible in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1")
        assert sandbox.sandbox_id == "sb-1"

    def test_is_stopping_true(self) -> None:
        """_is_stopping is True in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1")
        assert sandbox._is_stopping is True
        assert sandbox._is_done is False

    def test_is_stopping_false_in_running(self) -> None:
        """_is_stopping is False in _Running state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Running(sandbox_id="sb-1")
        assert sandbox._is_stopping is False


class TestStopOwnedTermination:
    """Tests for _stop_owned-based termination detection.

    Verifies that raise_on_termination triggers on local stop() provenance
    (_stop_owned), not on mere observation of TERMINATING status.
    """

    def test_stop_owned_raises_on_completed(self) -> None:
        """stop() + COMPLETED + raise_on_termination=True raises SandboxTerminatedError."""
        from cwsandbox.exceptions import SandboxTerminatedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._stop_owned = True

        terminal = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED, returncode=0)
        with pytest.raises(SandboxTerminatedError):
            sandbox._raise_or_return_for_terminal(terminal, raise_on_termination=True)

    def test_stop_owned_false_does_not_raise_on_completed(self) -> None:
        """Normal exit (no stop()) + COMPLETED does NOT raise with raise_on_termination=True."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._stop_owned = False

        terminal = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED, returncode=0)
        sandbox._raise_or_return_for_terminal(terminal, raise_on_termination=True)

    def test_stop_owned_respects_raise_on_termination_false(self) -> None:
        """stop() + COMPLETED + raise_on_termination=False does NOT raise."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._stop_owned = True

        terminal = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED, returncode=0)
        sandbox._raise_or_return_for_terminal(terminal, raise_on_termination=False)

    def test_normal_exit_through_terminating_no_raise(self) -> None:
        """Sandbox polling through TERMINATING to COMPLETED does NOT raise.

        This is the false-positive case that _termination_observed triggered:
        a sandbox naturally exits, polls see TERMINATING during drain, and
        then COMPLETED. Without _stop_owned, no termination error is raised.
        """
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Running(sandbox_id="sb-1")

        # Observe TERMINATING (normal exit draining)
        mock_info = MagicMock()
        mock_info.sandbox_id = "sb-1"
        mock_info.sandbox_status = gateway_pb2.SANDBOX_STATUS_TERMINATING
        mock_info.runner_id = ""
        mock_info.profile_id = ""
        mock_info.runner_group_id = ""
        mock_info.started_at_time = None

        new_state = sandbox._apply_sandbox_info(mock_info, source="poll")
        assert isinstance(new_state, _Stopping)
        assert sandbox._stop_owned is False

        # Then observe COMPLETED
        mock_info.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_info.returncode = 0
        sandbox._state = new_state
        terminal_state = sandbox._apply_sandbox_info(mock_info, source="poll")
        assert isinstance(terminal_state, _Terminal)

        # No raise with raise_on_termination=True because _stop_owned is False
        sandbox._raise_or_return_for_terminal(terminal_state, raise_on_termination=True)

    def test_discovered_sandbox_stop_owned_false(self) -> None:
        """Sandboxes discovered via from_id/list always have _stop_owned=False."""
        from cwsandbox._proto import gateway_pb2

        info = gateway_pb2.SandboxInfo(
            sandbox_id="sb-discovered",
            sandbox_status=gateway_pb2.SANDBOX_STATUS_TERMINATING,
        )
        sandbox = Sandbox._from_sandbox_info(
            info,
            base_url="https://api.example.com",
            timeout_seconds=300.0,
        )
        assert sandbox._stop_owned is False


class TestStoppingDiscovery:
    """Tests for discovering sandboxes in TERMINATING state."""

    def test_from_sandbox_info_with_terminating_status(self) -> None:
        """_from_sandbox_info creates _Stopping state for TERMINATING sandbox."""
        from cwsandbox._proto import gateway_pb2

        info = gateway_pb2.SandboxInfo(
            sandbox_id="sb-terminating",
            sandbox_status=gateway_pb2.SANDBOX_STATUS_TERMINATING,
        )
        sandbox = Sandbox._from_sandbox_info(
            info,
            base_url="https://api.example.com",
            timeout_seconds=300.0,
        )
        assert isinstance(sandbox._state, _Stopping)
        assert sandbox.status == SandboxStatus.TERMINATING
        assert sandbox.sandbox_id == "sb-terminating"


class TestStoppingSessionClose:
    """Tests for Session.close() interaction with _Stopping sandboxes."""

    @pytest.mark.asyncio
    async def test_close_joins_stopping_sandbox(self) -> None:
        """Session.close() on a _Stopping sandbox joins the stop task, not double-stops."""
        from cwsandbox._proto import gateway_pb2
        from cwsandbox._session import Session

        session = Session()
        sandbox = session.sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Running(sandbox_id="test-id")

        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stop_response = MagicMock()
        mock_stop_response.success = True
        mock_stub.Stop = AsyncMock(return_value=mock_stop_response)

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.sandbox_id = "test-id"
        mock_get_response.runner_id = ""
        mock_get_response.profile_id = ""
        mock_get_response.runner_group_id = ""
        mock_get_response.started_at_time = None
        mock_get_response.returncode = 0
        mock_stub.Get = AsyncMock(return_value=mock_get_response)
        sandbox._stub = mock_stub

        # First stop() puts sandbox into _Stopping then polls to terminal
        await sandbox._stop_async()
        assert isinstance(sandbox._state, _Terminal)

        # Session.close() should not call Stop RPC again
        mock_stub.Stop.reset_mock()
        await session._close_async()
        mock_stub.Stop.assert_not_called()


class TestStoppingCancelledError:
    """Tests for CancelledError handling with _stop_owned."""

    @pytest.mark.asyncio
    async def test_cancelled_error_during_wait_running_with_stop_owned(self) -> None:
        """CancelledError with _stop_owned raises SandboxNotRunningError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Stopping(sandbox_id="test-id")
        sandbox._stop_owned = True

        cancelled_task = asyncio.Future()
        cancelled_task.cancel()
        sandbox._running_task = cancelled_task

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            await sandbox._wait_until_running_async()

    @pytest.mark.asyncio
    async def test_cancelled_error_during_wait_complete_with_stop_owned(self) -> None:
        """CancelledError with _stop_owned raises SandboxNotRunningError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Stopping(sandbox_id="test-id")
        sandbox._stop_owned = True

        cancelled_task = asyncio.Future()
        cancelled_task.cancel()
        sandbox._complete_task = cancelled_task

        with pytest.raises(SandboxNotRunningError, match="has been stopped"):
            await sandbox._wait_until_complete_async()

    @pytest.mark.asyncio
    async def test_cancelled_error_without_stop_owned_propagates(self) -> None:
        """CancelledError without _stop_owned propagates as CancelledError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._state = _Stopping(sandbox_id="test-id")
        sandbox._stop_owned = False

        cancelled_task = asyncio.Future()
        cancelled_task.cancel()
        sandbox._running_task = cancelled_task

        with pytest.raises(asyncio.CancelledError):
            await sandbox._wait_until_running_async()


class TestDoPolRunningStoppingBranch:
    """Tests for _do_poll_running behavior when sandbox enters _Stopping."""

    @pytest.mark.asyncio
    async def test_do_poll_running_stopping_returns_normally(self) -> None:
        """_do_poll_running returns without raising when sandbox enters _Stopping.

        The sandbox is draining through its grace period and will reach a
        terminal state via _do_poll_complete. Raising here was a false positive.
        """
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Starting(sandbox_id="sb-1")

        mock_response = MagicMock()
        mock_response.sandbox_id = "sb-1"
        mock_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_TERMINATING
        mock_response.runner_id = ""
        mock_response.profile_id = ""
        mock_response.runner_group_id = ""
        mock_response.started_at_time = None

        with patch.object(sandbox, "_poll_until_stable", return_value=mock_response):
            await sandbox._do_poll_running()

        assert isinstance(sandbox._state, _Stopping)

    @pytest.mark.asyncio
    async def test_wait_regression_starting_to_terminating_to_completed(self) -> None:
        """wait() does not raise when sandbox transitions STARTING -> TERMINATING -> COMPLETED.

        Regression test: the old _termination_observed flag would have caused
        a SandboxTerminatedError on the COMPLETED transition because TERMINATING
        was observed during the poll.
        """
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="echo", args=["done"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Starting(sandbox_id="sb-1")

        # First poll returns TERMINATING (sandbox exiting naturally)
        terminating_response = MagicMock()
        terminating_response.sandbox_id = "sb-1"
        terminating_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_TERMINATING
        terminating_response.runner_id = ""
        terminating_response.profile_id = ""
        terminating_response.runner_group_id = ""
        terminating_response.started_at_time = None

        with patch.object(sandbox, "_poll_until_stable", return_value=terminating_response):
            await sandbox._do_poll_running()

        assert isinstance(sandbox._state, _Stopping)
        assert sandbox._stop_owned is False

        # Second poll (via _do_poll_complete) returns COMPLETED
        completed_response = MagicMock()
        completed_response.sandbox_id = "sb-1"
        completed_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        completed_response.runner_id = ""
        completed_response.profile_id = ""
        completed_response.runner_group_id = ""
        completed_response.started_at_time = None
        completed_response.returncode = 0

        with patch.object(sandbox, "_poll_until_stable", return_value=completed_response):
            await sandbox._do_poll_complete()

        assert isinstance(sandbox._state, _Terminal)
        assert sandbox._state.status == SandboxStatus.COMPLETED

        # No raise with raise_on_termination=True because _stop_owned is False
        sandbox._raise_or_return_for_terminal(sandbox._state, raise_on_termination=True)


class TestStoppingWaitUntilComplete:
    """Tests for wait_until_complete with _Stopping sandboxes."""

    @pytest.mark.asyncio
    async def test_stop_then_wait_until_complete_raises(self) -> None:
        """stop() + wait_until_complete(raise_on_termination=True) raises."""
        from cwsandbox._proto import gateway_pb2
        from cwsandbox.exceptions import SandboxTerminatedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Running(sandbox_id="sb-1")

        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stop_response = MagicMock()
        mock_stop_response.success = True
        mock_stub.Stop = AsyncMock(return_value=mock_stop_response)

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.sandbox_id = "sb-1"
        mock_get_response.runner_id = ""
        mock_get_response.profile_id = ""
        mock_get_response.runner_group_id = ""
        mock_get_response.started_at_time = None
        mock_get_response.returncode = 0
        mock_stub.Get = AsyncMock(return_value=mock_get_response)
        sandbox._stub = mock_stub

        await sandbox._stop_async()
        assert sandbox._stop_owned is True
        assert isinstance(sandbox._state, _Terminal)

        with pytest.raises(SandboxTerminatedError):
            sandbox._raise_or_return_for_terminal(sandbox._state, raise_on_termination=True)

    @pytest.mark.asyncio
    async def test_stop_then_wait_until_complete_no_raise_when_false(self) -> None:
        """stop() + wait_until_complete(raise_on_termination=False) does NOT raise."""
        from cwsandbox._proto import gateway_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Running(sandbox_id="sb-1")

        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stop_response = MagicMock()
        mock_stop_response.success = True
        mock_stub.Stop = AsyncMock(return_value=mock_stop_response)

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = gateway_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.sandbox_id = "sb-1"
        mock_get_response.runner_id = ""
        mock_get_response.profile_id = ""
        mock_get_response.runner_group_id = ""
        mock_get_response.started_at_time = None
        mock_get_response.returncode = 0
        mock_stub.Get = AsyncMock(return_value=mock_get_response)
        sandbox._stub = mock_stub

        await sandbox._stop_async()
        assert sandbox._stop_owned is True
        assert isinstance(sandbox._state, _Terminal)

        # No raise with raise_on_termination=False
        sandbox._raise_or_return_for_terminal(sandbox._state, raise_on_termination=False)

    def test_discovered_stopping_sandbox_wait_until_complete(self) -> None:
        """Sandbox.from_id() returning _Stopping then wait_until_complete does not raise.

        Discovered sandboxes have _stop_owned=False, so even if they are in
        TERMINATING and eventually reach COMPLETED, no SandboxTerminatedError
        is raised with raise_on_termination=True.
        """
        from cwsandbox._proto import gateway_pb2

        info = gateway_pb2.SandboxInfo(
            sandbox_id="sb-discovered",
            sandbox_status=gateway_pb2.SANDBOX_STATUS_TERMINATING,
        )
        sandbox = Sandbox._from_sandbox_info(
            info,
            base_url="https://api.example.com",
            timeout_seconds=300.0,
        )
        assert isinstance(sandbox._state, _Stopping)
        assert sandbox._stop_owned is False

        terminal = _Terminal(
            sandbox_id="sb-discovered", status=SandboxStatus.COMPLETED, returncode=0
        )
        # No raise because _stop_owned is False
        sandbox._raise_or_return_for_terminal(terminal, raise_on_termination=True)


class TestStoppingDel:
    """Tests for __del__ warning with _Stopping state."""

    def test_del_warns_for_stopping(self) -> None:
        """__del__ warns about unstopped sandbox in _Stopping state."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _Stopping(sandbox_id="sb-1")

        with pytest.warns(ResourceWarning, match="was not stopped"):
            sandbox.__del__()
