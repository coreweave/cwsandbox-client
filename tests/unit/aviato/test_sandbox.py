"""Unit tests for aviato._sandbox module."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aviato import Sandbox
from aviato.exceptions import SandboxNotRunningError


class TestSandboxCreate:
    """Tests for Sandbox.create factory method."""

    @pytest.mark.asyncio
    async def test_create_requires_positional_args(self) -> None:
        """Test Sandbox.create raises ValueError without positional args."""
        with pytest.raises(ValueError, match="At least one positional argument"):
            await Sandbox.create()

    @pytest.mark.asyncio
    async def test_create_calls_start(self) -> None:
        """Test Sandbox.create calls start() on the sandbox."""
        with patch.object(Sandbox, "start", return_value="test-id") as mock_start:
            await Sandbox.create("echo", "hello", "world")
            mock_start.assert_called_once()


class TestSandboxExec:
    """Tests for Sandbox.exec method."""

    @pytest.mark.asyncio
    async def test_exec_without_start_raises_error(self) -> None:
        """Test exec raises SandboxNotRunningError before start."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(SandboxNotRunningError, match="No sandbox is running"):
            await sandbox.exec(["echo", "test"])

    @pytest.mark.asyncio
    async def test_exec_empty_command_raises_error(self) -> None:
        """Test exec with empty command raises ValueError."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        with pytest.raises(ValueError, match="Command cannot be empty"):
            await sandbox.exec([])

    @pytest.mark.asyncio
    async def test_exec_check_raises_on_nonzero_returncode(self) -> None:
        """Test exec with check=True raises SandboxExecutionError on failure."""
        from unittest.mock import AsyncMock, MagicMock

        from aviato.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.result.stdout = ""
        mock_response.result.stderr = "command not found"
        mock_response.result.exit_code = 127
        sandbox._client.exec = AsyncMock(return_value=mock_response)

        with pytest.raises(SandboxExecutionError, match="exit code 127"):
            await sandbox.exec(["nonexistent"], check=True)

    @pytest.mark.asyncio
    async def test_exec_check_false_returns_result_on_failure(self) -> None:
        """Test exec with check=False returns result even on failure."""
        from unittest.mock import AsyncMock, MagicMock

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.result.stdout = ""
        mock_response.result.stderr = "error"
        mock_response.result.exit_code = 1
        sandbox._client.exec = AsyncMock(return_value=mock_response)

        result = await sandbox.exec(["failing-cmd"], check=False)

        assert result.returncode == 1

    @pytest.mark.asyncio
    async def test_exec_with_stream_output_uses_streaming(self) -> None:
        """Test exec with stream_output=True uses streaming API."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._streaming_client = MagicMock()

        output_response = MagicMock()
        output_response.HasField = lambda f: f == "output"
        output_response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDOUT
        output_response.output.data = b"streamed output"
        output_response.output.HasField = lambda f: False

        exit_response = MagicMock()
        exit_response.HasField = lambda f: f == "exit"
        exit_response.exit.exit_code = 0
        exit_response.exit.HasField = lambda f: False

        async def mock_stream_exec(
            request_iter: MagicMock, timeout_ms: int | None = None
        ) -> AsyncIterator[MagicMock]:
            async for _ in request_iter:
                pass
            yield output_response
            yield exit_response

        sandbox._streaming_client.stream_exec = mock_stream_exec

        # stream_output=True should use streaming API and return result
        result = await sandbox.exec(["echo", "test"], stream_output=True)

        assert result.stdout_bytes == b"streamed output"
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_exec_with_callbacks_streams_and_returns_result(self) -> None:
        """Test exec with callbacks streams output and returns complete result."""
        from coreweave.aviato.v1beta1 import streaming_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._streaming_client = MagicMock()

        stdout_response = MagicMock()
        stdout_response.HasField = lambda f: f == "output"
        stdout_response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDOUT
        stdout_response.output.data = b"stdout data"
        stdout_response.output.HasField = lambda f: False

        stderr_response = MagicMock()
        stderr_response.HasField = lambda f: f == "output"
        stderr_response.output.stream_type = streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR
        stderr_response.output.data = b"stderr data"
        stderr_response.output.HasField = lambda f: False

        exit_response = MagicMock()
        exit_response.HasField = lambda f: f == "exit"
        exit_response.exit.exit_code = 0
        exit_response.exit.HasField = lambda f: False

        async def mock_stream_exec(
            request_iter: MagicMock, timeout_ms: int | None = None
        ) -> AsyncIterator[MagicMock]:
            async for _ in request_iter:
                pass
            yield stdout_response
            yield stderr_response
            yield exit_response

        sandbox._streaming_client.stream_exec = mock_stream_exec

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        result = await sandbox.exec(
            ["cmd"],
            on_stdout=lambda data: stdout_chunks.append(data),
            on_stderr=lambda data: stderr_chunks.append(data),
        )

        # Callbacks should have been invoked
        assert stdout_chunks == [b"stdout data"]
        assert stderr_chunks == [b"stderr data"]
        # Result should contain all output
        assert result.stdout_bytes == b"stdout data"
        assert result.stderr_bytes == b"stderr data"
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_exec_streaming_with_check_raises_on_failure(self) -> None:
        """Test exec with streaming and check=True raises on non-zero exit."""
        from aviato.exceptions import SandboxExecutionError

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._streaming_client = MagicMock()

        exit_response = MagicMock()
        exit_response.HasField = lambda f: f == "exit"
        exit_response.exit.exit_code = 1
        exit_response.exit.HasField = lambda f: False

        async def mock_stream_exec(
            request_iter: MagicMock, timeout_ms: int | None = None
        ) -> AsyncIterator[MagicMock]:
            async for _ in request_iter:
                pass
            yield exit_response

        sandbox._streaming_client.stream_exec = mock_stream_exec

        with pytest.raises(SandboxExecutionError, match="failed with exit code 1"):
            await sandbox.exec(["cmd"], stream_output=True, check=True)


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

    @pytest.mark.asyncio
    async def test_context_manager_calls_stop_on_exit(self) -> None:
        """Test context manager calls stop() when exiting if sandbox was started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        stop_mock = AsyncMock()

        async def mock_start() -> str:
            sandbox._sandbox_id = "test-sandbox-id"
            return "test-sandbox-id"

        sandbox.start = mock_start
        sandbox.stop = stop_mock

        async with sandbox:
            pass

        stop_mock.assert_called_once()


class TestSandboxGetStatus:
    """Tests for Sandbox.get_status method."""

    @pytest.mark.asyncio
    async def test_get_status_raises_without_start(self) -> None:
        """Test get_status raises SandboxNotRunningError if not started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(SandboxNotRunningError, match="has not been started"):
            await sandbox.get_status()


class TestSandboxWait:
    """Tests for Sandbox.wait method."""

    @pytest.mark.asyncio
    async def test_wait_raises_without_start(self) -> None:
        """Test wait raises SandboxNotRunningError if not started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(SandboxNotRunningError, match="No sandbox is running"):
            await sandbox.wait()

    @pytest.mark.asyncio
    async def test_wait_raises_on_failed(self) -> None:
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
            await sandbox.wait()

    @pytest.mark.asyncio
    async def test_wait_raises_on_terminated_by_default(self) -> None:
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
            await sandbox.wait()

    @pytest.mark.asyncio
    async def test_wait_no_raise_on_terminated_when_disabled(self) -> None:
        """Test wait returns normally when raise_on_termination=False."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_TERMINATED
        sandbox._client.get = AsyncMock(return_value=mock_response)

        await sandbox.wait(raise_on_termination=False)

        assert sandbox.returncode is None


class TestSandboxStart:
    """Tests for Sandbox.start method."""

    @pytest.mark.asyncio
    async def test_start_returns_sandbox_id(self) -> None:
        """Test start returns the sandbox ID."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "new-sandbox-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.tower_id = "tower-1"
        mock_get_response.runway_id = "runway-1"

        with patch.object(sandbox, "_ensure_client"):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)
            sandbox._client.get = AsyncMock(return_value=mock_get_response)

            result = await sandbox.start()

            assert result == "new-sandbox-id"
            assert sandbox.sandbox_id == "new-sandbox-id"
            assert sandbox.tower_id == "tower-1"
            assert sandbox.runway_id == "runway-1"

    @pytest.mark.asyncio
    async def test_start_raises_on_failed_status(self) -> None:
        """Test start raises SandboxFailedError when sandbox fails to start."""
        from coreweave.aviato.v1beta1 import atc_pb2

        from aviato.exceptions import SandboxFailedError

        sandbox = Sandbox(command="sleep", args=["infinity"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "failing-sandbox-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_FAILED

        with patch.object(sandbox, "_ensure_client"):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)
            sandbox._client.get = AsyncMock(return_value=mock_get_response)

            with pytest.raises(SandboxFailedError, match="failed to start"):
                await sandbox.start()

    @pytest.mark.asyncio
    async def test_start_handles_fast_completion(self) -> None:
        """Test start handles sandbox that completes during startup."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="echo", args=["hello"])

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "fast-sandbox-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_COMPLETED
        mock_get_response.tower_id = "tower-1"
        mock_get_response.runway_id = "runway-1"

        with patch.object(sandbox, "_ensure_client"):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)
            sandbox._client.get = AsyncMock(return_value=mock_get_response)

            result = await sandbox.start()

            assert result == "fast-sandbox-id"
            assert sandbox.returncode == 0

    @pytest.mark.asyncio
    async def test_start_sends_correct_request(self) -> None:
        """Test start sends request with correct parameters."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(
            command="python",
            args=["-c", "print('hello')"],
            container_image="python:3.12",
            tags=["test-tag"],
            max_lifetime_seconds=3600,
        )

        mock_start_response = MagicMock()
        mock_start_response.sandbox_id = "test-sandbox-id"

        mock_get_response = MagicMock()
        mock_get_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        mock_get_response.tower_id = "tower-1"
        mock_get_response.runway_id = None

        with patch.object(sandbox, "_ensure_client"):
            sandbox._client = MagicMock()
            sandbox._client.start = AsyncMock(return_value=mock_start_response)
            sandbox._client.get = AsyncMock(return_value=mock_get_response)

            await sandbox.start()

            start_call = sandbox._client.start.call_args[0][0]
            assert start_call.command == "python"
            assert start_call.args == ["-c", "print('hello')"]
            assert start_call.container_image == "python:3.12"
            assert start_call.tags == ["test-tag"]
            assert start_call.max_lifetime_seconds == 3600


class TestSandboxStop:
    """Tests for Sandbox.stop method."""

    @pytest.mark.asyncio
    async def test_stop_returns_false_on_failure(self) -> None:
        """Test stop returns False when backend reports failure."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error_message = "sandbox not found"
        sandbox._client.stop = AsyncMock(return_value=mock_response)
        sandbox._client.close = AsyncMock()

        result = await sandbox.stop()

        assert result is False

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        """Test stop() is idempotent - safe to call multiple times."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        # Calling stop on never-started sandbox returns True (no-op)
        result = await sandbox.stop()
        assert result is True

        # Calling stop again is also safe
        result = await sandbox.stop()
        assert result is True


class TestSandboxTimeouts:
    """Tests for Sandbox timeout behavior."""

    @pytest.mark.asyncio
    async def test_exec_respects_timeout_seconds(self) -> None:
        """Test exec() raises TimeoutError when timeout_seconds is exceeded."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        async def slow_exec(request: MagicMock) -> MagicMock:
            # Simulate a hanging operation that will be cancelled by timeout
            await asyncio.Event().wait()
            return MagicMock()

        sandbox._client.exec = slow_exec

        with patch("aviato._sandbox.DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS", 0):
            with pytest.raises(asyncio.TimeoutError):
                await sandbox.exec(["sleep", "10"], timeout_seconds=0.1)


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

    @pytest.mark.asyncio
    async def test_create_with_valid_kwargs(self) -> None:
        """Test Sandbox.create accepts valid kwargs."""
        with patch.object(Sandbox, "start", return_value="test-id") as mock_start:
            sandbox = await Sandbox.create(
                "echo",
                "hello",
                resources={"cpu": "100m"},
                ports=[{"container_port": 8080}],
            )
            mock_start.assert_called_once()
            assert sandbox._start_kwargs["resources"] == {"cpu": "100m"}
            assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]

    @pytest.mark.asyncio
    async def test_create_with_invalid_kwargs(self) -> None:
        """Test Sandbox.create rejects invalid kwargs."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            await Sandbox.create(
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
    async def test_delete_returns_true_on_success(self, mock_aviato_api_key: str) -> None:
        """Test delete() returns True when deletion succeeds."""
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

                assert result is True

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
        """Test delete(missing_ok=True) returns False instead of raising."""
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
                assert result is False
