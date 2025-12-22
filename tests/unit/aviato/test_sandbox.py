"""Unit tests for aviato._sandbox module."""

import asyncio
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
        with pytest.raises(ValueError, match="Invalid sandbox parameters"):
            Sandbox(
                command="echo",
                args=["hello"],
                invalid_param="value",
                another_bad_param=42,
            )

    def test_init_with_mixed_valid_invalid_kwargs(self) -> None:
        """Test Sandbox.__init__ rejects if any kwargs are invalid."""
        with pytest.raises(ValueError, match="Invalid sandbox parameters"):
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
        with pytest.raises(ValueError, match="Invalid sandbox parameters"):
            await Sandbox.create(
                "echo",
                "hello",
                invalid_param="value",
            )
