"""Unit tests for aviato._sandbox module."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aviato import Sandbox
from aviato.exceptions import SandboxNotRunningError


class TestSandboxRun:
    """Tests for Sandbox.run factory method."""

    def test_run_uses_defaults_without_args(self) -> None:
        """Test Sandbox.run uses default command when no args provided."""
        with patch.object(Sandbox, "start") as mock_start:
            sandbox = Sandbox.run()
            mock_start.assert_called_once()
            # Default command is "tail" with args ["-f", "/dev/null"]
            assert sandbox._command == "tail"
            assert sandbox._args == ["-f", "/dev/null"]

    def test_run_calls_start(self) -> None:
        """Test Sandbox.run calls start() on the sandbox."""
        with patch.object(Sandbox, "start") as mock_start:
            Sandbox.run("echo", "hello", "world")
            mock_start.assert_called_once()


class TestSandboxExec:
    """Tests for Sandbox.exec method."""

    def test_exec_without_start_raises_error(self) -> None:
        """Test exec raises SandboxNotRunningError before start."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        # exec() returns Process immediately, error occurs when result() is called
        process = sandbox.exec(["echo", "test"])
        with pytest.raises(SandboxNotRunningError, match="No sandbox is running"):
            process.result()

    def test_exec_empty_command_raises_error(self) -> None:
        """Test exec with empty command raises ValueError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        with pytest.raises(ValueError, match="Command cannot be empty"):
            sandbox.exec([])

    def test_exec_check_raises_on_nonzero_returncode(self) -> None:
        """Test exec with check=True raises SandboxExecutionError on failure."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        from aviato.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        # Mock streaming responses: stderr output, then exit with code 127
        async def mock_stream(*args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            # stderr output
            response = MagicMock()
            response.HasField = lambda field: field == "output"
            response.output.data = b"command not found"
            response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR
            yield response

            # exit with non-zero code
            response = MagicMock()
            response.HasField = lambda field: field == "exit"
            response.exit.exit_code = 127
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
            process = sandbox.exec(["nonexistent"], check=True)
            with pytest.raises(SandboxExecutionError, match="exit code 127"):
                process.result()

    def test_exec_check_false_returns_result_on_failure(self) -> None:
        """Test exec with check=False returns result even on failure."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        # Mock streaming responses: stderr output, then exit with code 1
        async def mock_stream(*args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            # stderr output
            response = MagicMock()
            response.HasField = lambda field: field == "output"
            response.output.data = b"error"
            response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR
            yield response

            # exit with non-zero code
            response = MagicMock()
            response.HasField = lambda field: field == "exit"
            response.exit.exit_code = 1
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
            process = sandbox.exec(["failing-cmd"], check=False)
            result = process.result()

            assert result.returncode == 1

    def test_exec_streams_stdout_to_queue(self) -> None:
        """Test exec() streams stdout data to the queue as it arrives."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        # Mock streaming responses: multiple stdout chunks, then exit
        async def mock_stream(*args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            for chunk in [b"line1\n", b"line2\n", b"line3\n"]:
                response = MagicMock()
                response.HasField = lambda field: field == "output"
                response.output.data = chunk
                response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDOUT
                yield response

            # exit with code 0
            response = MagicMock()
            response.HasField = lambda field: field == "exit"
            response.exit.exit_code = 0
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
            process = sandbox.exec(["echo", "test"])

            # Collect all stdout lines by iterating the stream
            lines = list(process.stdout)
            assert lines == ["line1\n", "line2\n", "line3\n"]

            # Result should have combined output
            result = process.result()
            assert result.stdout == "line1\nline2\nline3\n"
            assert result.returncode == 0

    def test_exec_handles_stream_error(self) -> None:
        """Test exec() handles stream errors correctly."""

        from aviato.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        # Mock streaming response: error
        async def mock_stream(*args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            response = MagicMock()
            response.HasField = lambda field: field == "error"
            response.error.message = "Connection lost"
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
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
        sandbox._client = MagicMock()

        captured_command: list[str] = []

        async def mock_stream(
            request_iter: AsyncIterator[MagicMock],
            **kwargs: object,
        ) -> AsyncIterator[MagicMock]:
            # Capture the command from the request
            async for req in request_iter:
                if hasattr(req, "init"):
                    captured_command.extend(req.init.command)
                break
            # Return exit response
            response = MagicMock()
            response.HasField = lambda field: field == "exit"
            response.exit.exit_code = 0
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
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
        sandbox._client = MagicMock()

        async def mock_stream(*args: object, **kwargs: object) -> AsyncIterator[MagicMock]:
            response = MagicMock()
            response.HasField = lambda field: field == "exit"
            response.exit.exit_code = 0
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
            process = sandbox.exec(["echo", "hello"], cwd="/app")
            result = process.result()

        # Original command should be preserved, not the wrapped command
        assert result.command == ["echo", "hello"]

    def test_exec_cwd_escapes_special_characters(self) -> None:
        """Test exec with cwd escapes special characters in path."""

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        captured_command: list[str] = []

        async def mock_stream(
            request_iter: AsyncIterator[MagicMock],
            **kwargs: object,
        ) -> AsyncIterator[MagicMock]:
            async for req in request_iter:
                if hasattr(req, "init"):
                    captured_command.extend(req.init.command)
                break
            response = MagicMock()
            response.HasField = lambda field: field == "exit"
            response.exit.exit_code = 0
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
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
        sandbox._client = MagicMock()

        captured_command: list[str] = []

        async def mock_stream(
            request_iter: AsyncIterator[MagicMock],
            **kwargs: object,
        ) -> AsyncIterator[MagicMock]:
            async for req in request_iter:
                if hasattr(req, "init"):
                    captured_command.extend(req.init.command)
                break
            response = MagicMock()
            response.HasField = lambda field: field == "exit"
            response.exit.exit_code = 0
            yield response

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = mock_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
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
    async def test_sets_auth_header(self, mock_aviato_api_key: str) -> None:
        """Test Sandbox sets Authorization header from AVIATO_API_KEY."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with patch("httpx.AsyncClient") as mock_client:
            await sandbox._ensure_client()

            mock_client.assert_called_once()
            headers = mock_client.call_args.kwargs["headers"]
            assert headers["Authorization"] == f"Bearer {mock_aviato_api_key}"


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
        from aviato import OperationRef

        sandbox = Sandbox(command="sleep", args=["infinity"])
        stop_called = False

        def mock_start() -> None:
            sandbox._sandbox_id = "test-sandbox-id"

        def mock_stop(**kwargs: object) -> OperationRef[bool]:
            nonlocal stop_called
            stop_called = True
            import concurrent.futures

            future: concurrent.futures.Future[bool] = concurrent.futures.Future()
            future.set_result(True)
            return OperationRef(future)

        sandbox.start = mock_start  # type: ignore[method-assign]
        sandbox.stop = mock_stop  # type: ignore[method-assign]

        with sandbox:
            pass

        assert stop_called


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
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_FAILED
        sandbox._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(SandboxFailedError, match="failed"):
            sandbox.wait()

    def test_wait_raises_on_terminated_by_default(self) -> None:
        """Test wait raises SandboxTerminatedError when terminated."""
        from coreweave.aviato.v1beta1 import atc_pb2

        from aviato.exceptions import SandboxTerminatedError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_TERMINATED
        sandbox._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(SandboxTerminatedError, match="terminated"):
            sandbox.wait()


class TestSandboxWaitUntilComplete:
    """Tests for Sandbox.wait_until_complete method."""

    def test_wait_until_complete_raises_without_start(self) -> None:
        """Test wait_until_complete raises SandboxNotRunningError if not started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(SandboxNotRunningError, match="No sandbox is running"):
            sandbox.wait_until_complete()

    def test_wait_until_complete_no_raise_on_terminated_when_disabled(self) -> None:
        """Test wait_until_complete returns normally when raise_on_termination=False."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_TERMINATED
        sandbox._client.get = AsyncMock(return_value=mock_response)

        sandbox.wait_until_complete(raise_on_termination=False)

        assert sandbox.returncode is None


class TestSandboxStart:
    """Tests for Sandbox.start method."""

    def test_start_sets_sandbox_id(self) -> None:
        """Test start sets the sandbox ID (does NOT wait for RUNNING)."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "new-sandbox-id"

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)

            sandbox.start()

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
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)

            sandbox.start()

            start_call = sandbox._client.start.call_args[0][0]
            assert start_call.command == "python"
            assert start_call.args == ["-c", "print('hello')"]
            assert start_call.container_image == "python:3.12"
            assert start_call.tags == ["test-tag"]
            assert start_call.max_lifetime_seconds == 3600

    def test_start_raises_not_running_on_canceled(self) -> None:
        """Test start raises SandboxNotRunningError when request is cancelled."""
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        sandbox = Sandbox(command="sleep", args=["infinity"])

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(
                side_effect=ConnectError(message="request cancelled", code=Code.CANCELED)
            )

            with pytest.raises(SandboxNotRunningError, match="Sandbox start was cancelled"):
                sandbox.start()


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
            sandbox._client = MagicMock()
            sandbox._client.get = AsyncMock(return_value=mock_get_response)

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
            sandbox._client = MagicMock()
            sandbox._client.get = AsyncMock(return_value=mock_get_response)

            sandbox.wait()

            assert sandbox.returncode == 0


class TestSandboxStop:
    """Tests for Sandbox.stop method."""

    def test_stop_raises_on_backend_failure(self) -> None:
        """Test stop raises SandboxError when backend reports failure."""
        from aviato.exceptions import SandboxError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error_message = "backend error"
        sandbox._client.stop = AsyncMock(return_value=mock_response)
        sandbox._client.close = AsyncMock()

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
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()
        sandbox._client.stop = AsyncMock(
            side_effect=ConnectError(message="Not found", code=Code.NOT_FOUND)
        )
        sandbox._client.close = AsyncMock()

        # Should not raise, returns None
        result = sandbox.stop(missing_ok=True).result()
        assert result is None

    def test_stop_missing_ok_false_raises_not_found(self) -> None:
        """Test stop(missing_ok=False) raises SandboxNotFoundError."""
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        from aviato.exceptions import SandboxNotFoundError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()
        sandbox._client.stop = AsyncMock(
            side_effect=ConnectError(message="Not found", code=Code.NOT_FOUND)
        )
        sandbox._client.close = AsyncMock()

        with pytest.raises(SandboxNotFoundError):
            sandbox.stop().result()


class TestSandboxTimeouts:
    """Tests for Sandbox timeout behavior."""

    def test_exec_respects_timeout_seconds(self) -> None:
        """Test exec() raises SandboxTimeoutError when timeout_seconds is exceeded."""
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        from aviato.exceptions import SandboxTimeoutError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        async def slow_stream(
            *args: object, timeout_ms: int | None = None, **kwargs: object
        ) -> AsyncIterator[MagicMock]:
            # Simulate server-side timeout behavior: wait for timeout then raise
            if timeout_ms is not None:
                await asyncio.sleep(timeout_ms / 1000 + 0.05)  # Slightly exceed timeout
            raise ConnectError(Code.DEADLINE_EXCEEDED, "deadline exceeded")
            yield MagicMock()  # Never reached, but needed for generator type

        mock_streaming_client = MagicMock()
        mock_streaming_client.stream_exec = slow_stream

        with (
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("aviato._sandbox.resolve_auth") as mock_auth,
            patch(
                "aviato._sandbox.streaming_connect.ATCStreamingServiceClient",
                return_value=mock_streaming_client,
            ),
        ):
            mock_auth.return_value.headers = {}
            process = sandbox.exec(["sleep", "10"], timeout_seconds=0.1)
            with pytest.raises(SandboxTimeoutError):
                process.result()


class TestSandboxKwargsValidation:
    """Tests for kwargs validation in Sandbox methods."""

    def test_init_with_valid_kwargs(self) -> None:
        """Test Sandbox.__init__ accepts valid kwargs."""
        sandbox = Sandbox(
            command="echo",
            args=["hello"],
            resources={"cpu": "100m", "memory": "128Mi"},
            ports=[{"container_port": 8080}],
            service={"public": True},
            max_timeout_seconds=60,
        )
        assert sandbox._start_kwargs["resources"] == {"cpu": "100m", "memory": "128Mi"}
        assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]
        assert sandbox._start_kwargs["service"] == {"public": True}
        assert sandbox._start_kwargs["max_timeout_seconds"] == 60

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.list = AsyncMock(
                    return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
                )

                sandboxes = await Sandbox.list(tags=["test-tag"])

                assert len(sandboxes) == 1
                assert isinstance(sandboxes[0], Sandbox)
                assert sandboxes[0].sandbox_id == "test-123"
                assert sandboxes[0].status == "running"

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, mock_aviato_api_key: str) -> None:
        """Test list() passes status filter to request."""
        from coreweave.aviato.v1beta1 import atc_pb2

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.list = AsyncMock(
                    return_value=atc_pb2.ListSandboxesResponse(sandboxes=[])
                )

                await Sandbox.list(status="running")

                call_args = mock_atc_instance.list.call_args[0][0]
                assert call_args.status == atc_pb2.SANDBOX_STATUS_RUNNING

    @pytest.mark.asyncio
    async def test_list_with_invalid_status_raises(self, mock_aviato_api_key: str) -> None:
        """Test list() raises ValueError for invalid status."""
        with pytest.raises(ValueError, match="not a valid SandboxStatus"):
            await Sandbox.list(status="invalid_status")

    @pytest.mark.asyncio
    async def test_list_empty_result(self, mock_aviato_api_key: str) -> None:
        """Test list() returns empty list when no sandboxes match."""
        from coreweave.aviato.v1beta1 import atc_pb2

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.list = AsyncMock(
                    return_value=atc_pb2.ListSandboxesResponse(sandboxes=[])
                )

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.get = AsyncMock(return_value=mock_response)

                sandbox = await Sandbox.from_id("test-123")

                assert isinstance(sandbox, Sandbox)
                assert sandbox.sandbox_id == "test-123"
                assert sandbox.status == "running"
                assert sandbox.tower_id == "tower-1"

    @pytest.mark.asyncio
    async def test_from_id_raises_not_found(self, mock_aviato_api_key: str) -> None:
        """Test from_id() raises SandboxNotFoundError for non-existent sandbox."""
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        from aviato.exceptions import SandboxNotFoundError

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.get = AsyncMock(
                    side_effect=ConnectError(message="Not found", code=Code.NOT_FOUND)
                )

                with pytest.raises(SandboxNotFoundError, match="not found"):
                    await Sandbox.from_id("nonexistent-id")


class TestSandboxDeleteClassMethod:
    """Tests for Sandbox.delete class method."""

    @pytest.mark.asyncio
    async def test_delete_returns_none_on_success(self, mock_aviato_api_key: str) -> None:
        """Test delete() returns None when deletion succeeds."""
        from coreweave.aviato.v1beta1 import atc_pb2

        mock_response = atc_pb2.DeleteSandboxResponse(success=True, error_message="")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.delete = AsyncMock(return_value=mock_response)

                result = await Sandbox.delete("test-123")

                assert result is None

    @pytest.mark.asyncio
    async def test_delete_raises_not_found_by_default(self, mock_aviato_api_key: str) -> None:
        """Test delete() raises SandboxNotFoundError when sandbox doesn't exist."""
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        from aviato.exceptions import SandboxNotFoundError

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.delete = AsyncMock(
                    side_effect=ConnectError(message="Not found", code=Code.NOT_FOUND)
                )

                with pytest.raises(SandboxNotFoundError):
                    await Sandbox.delete("nonexistent-id")

    @pytest.mark.asyncio
    async def test_delete_missing_ok_suppresses_not_found(self, mock_aviato_api_key: str) -> None:
        """Test delete(missing_ok=True) returns None instead of raising."""
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.delete = AsyncMock(
                    side_effect=ConnectError(message="Not found", code=Code.NOT_FOUND)
                )

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
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)

            sandbox.start()

            assert sandbox.service_address == "166.19.9.70:8080"

    def test_start_treats_empty_service_address_as_none(self) -> None:
        """Test start() treats empty string service_address as None."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)

            sandbox.start()

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
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)

            sandbox.start()

            assert sandbox.exposed_ports == ((8080, "http"), (22, "ssh"))

    def test_start_empty_exposed_ports_is_none(self) -> None:
        """Test start() returns None for empty exposed_ports list."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"
        mock_start_response.service_address = ""
        mock_start_response.exposed_ports = []

        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)

            sandbox.start()

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.get = AsyncMock(return_value=mock_response)

                sandbox = await Sandbox.from_id("test-123")

                assert sandbox.service_address is None
                assert sandbox.exposed_ports is None
