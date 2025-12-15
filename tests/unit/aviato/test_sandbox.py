"""Unit tests for aviato._sandbox module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aviato import Sandbox, SandboxDefaults
from aviato.exceptions import SandboxNotRunningError


class TestSandboxInit:
    """Tests for Sandbox initialization."""

    def test_sandbox_init_with_required_args(self) -> None:
        """Test Sandbox can be initialized with required arguments."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        assert sandbox._command == "sleep"
        assert sandbox._args == ["infinity"]
        assert sandbox._container_image == "python:3.11"  # default

    def test_sandbox_init_with_all_args(self) -> None:
        """Test Sandbox can be initialized with all arguments."""
        sandbox = Sandbox(
            command="python",
            args=["-c", "print('hello')"],
            container_image="python:3.12",
            tags=["test-tag"],
            base_url="http://custom-url.com",
            request_timeout_seconds=60.0,
            max_lifetime_seconds=3600,
        )

        assert sandbox._command == "python"
        assert sandbox._args == ["-c", "print('hello')"]
        assert sandbox._container_image == "python:3.12"
        assert sandbox._tags == ["test-tag"]
        assert sandbox._base_url == "http://custom-url.com"
        assert sandbox._request_timeout_seconds == 60.0
        assert sandbox._max_lifetime_seconds == 3600

    def test_sandbox_init_applies_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Sandbox applies SandboxDefaults."""
        # Remove env var so it doesn't override defaults
        monkeypatch.delenv("AVIATO_BASE_URL", raising=False)

        defaults = SandboxDefaults(
            container_image="python:3.10",
            base_url="http://defaults-url.com",
            request_timeout_seconds=120.0,
            tags=("default-tag",),
        )

        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            defaults=defaults,
        )

        assert sandbox._container_image == "python:3.10"
        assert sandbox._base_url == "http://defaults-url.com"
        assert sandbox._request_timeout_seconds == 120.0
        assert sandbox._tags == ["default-tag"]

    def test_sandbox_init_explicit_overrides_defaults(self) -> None:
        """Test explicit arguments override defaults."""
        defaults = SandboxDefaults(
            container_image="python:3.10",
            tags=("default-tag",),
        )

        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            defaults=defaults,
            container_image="python:3.12",
            tags=["explicit-tag"],
        )

        assert sandbox._container_image == "python:3.12"
        assert sandbox._tags == ["default-tag", "explicit-tag"]

    def test_sandbox_init_uses_env_base_url(self, mock_aviato_base_url: str) -> None:
        """Test Sandbox uses AVIATO_BASE_URL from environment."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        assert sandbox._base_url == mock_aviato_base_url


class TestSandboxCreate:
    """Tests for Sandbox.create factory method."""

    @pytest.mark.asyncio
    async def test_create_requires_positional_args(self) -> None:
        """Test Sandbox.create raises ValueError without positional args."""
        with pytest.raises(ValueError, match="At least one positional argument"):
            await Sandbox.create()

    @pytest.mark.asyncio
    async def test_create_parses_command_and_args(self) -> None:
        """Test Sandbox.create correctly parses positional arguments."""
        with patch.object(Sandbox, "start", return_value="test-id"):
            sandbox = await Sandbox.create("echo", "hello", "world")

            assert sandbox._command == "echo"
            assert sandbox._args == ["hello", "world"]


class TestSandboxProperties:
    """Tests for Sandbox properties."""

    def test_sandbox_id_none_before_start(self) -> None:
        """Test sandbox_id is None before start."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.sandbox_id is None

    def test_returncode_none_before_completion(self) -> None:
        """Test returncode is None before completion."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.returncode is None

    def test_tower_id_none_before_start(self) -> None:
        """Test tower_id is None before start."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.tower_id is None

    def test_runway_id_none_before_start(self) -> None:
        """Test runway_id is None before start."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        assert sandbox.runway_id is None


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


class TestSession:
    """Tests for Sandbox.session factory."""

    def test_session_returns_sandbox_session(self) -> None:
        """Test Sandbox.session returns a Session."""
        from aviato import Session

        session = Sandbox.session()
        assert isinstance(session, Session)

    def test_session_with_defaults(self) -> None:
        """Test Sandbox.session accepts defaults."""
        defaults = SandboxDefaults(container_image="python:3.10")
        session = Sandbox.session(defaults)
        assert session._defaults == defaults


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
            assert "test-sandbox-id" in str(w[0].message)
            assert "was not stopped" in str(w[0].message)

    def test_del_no_warning_if_sandbox_stopped(self) -> None:
        """Test __del__ does not warn if sandbox was properly stopped."""
        import warnings

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-sandbox-id"
        sandbox._returncode = 0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            del sandbox

            resource_warnings = [x for x in w if issubclass(x.category, ResourceWarning)]
            assert len(resource_warnings) == 0

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
        from unittest.mock import AsyncMock

        sandbox = Sandbox(command="sleep", args=["infinity"])
        stop_mock = AsyncMock()

        async def mock_start() -> str:
            sandbox._sandbox_id = "test-sandbox-id"
            return "test-sandbox-id"

        sandbox.start = mock_start
        sandbox.stop = stop_mock

        async with sandbox:
            assert sandbox._sandbox_id == "test-sandbox-id"

        stop_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_no_stop_if_never_started(self) -> None:
        """Test context manager does not call stop() if sandbox was never started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox.start = AsyncMock()
        sandbox.stop = AsyncMock()

        async with sandbox:
            pass

        sandbox.stop.assert_not_called()


class TestSandboxGetStatus:
    """Tests for Sandbox.get_status method."""

    @pytest.mark.asyncio
    async def test_get_status_returns_running(self) -> None:
        """Test get_status returns 'running' for running sandbox."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
        sandbox._client.get = AsyncMock(return_value=mock_response)

        status = await sandbox.get_status()

        assert status == "running"

    @pytest.mark.asyncio
    async def test_get_status_returns_completed(self) -> None:
        """Test get_status returns 'completed' for completed sandbox."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_COMPLETED
        sandbox._client.get = AsyncMock(return_value=mock_response)

        status = await sandbox.get_status()

        assert status == "completed"

    @pytest.mark.asyncio
    async def test_get_status_raises_without_start(self) -> None:
        """Test get_status raises SandboxNotRunningError if not started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(SandboxNotRunningError, match="has not been started"):
            await sandbox.get_status()


class TestSandboxWait:
    """Tests for Sandbox.wait method."""

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_if_already_completed(self) -> None:
        """Test wait returns immediately if returncode already set."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._returncode = 0

        await sandbox.wait()

        assert sandbox.returncode == 0

    @pytest.mark.asyncio
    async def test_wait_raises_without_start(self) -> None:
        """Test wait raises SandboxNotRunningError if not started."""
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(SandboxNotRunningError, match="No sandbox is running"):
            await sandbox.wait()

    @pytest.mark.asyncio
    async def test_wait_sets_returncode_on_completed(self) -> None:
        """Test wait sets returncode when sandbox completes."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_COMPLETED
        sandbox._client.get = AsyncMock(return_value=mock_response)

        await sandbox.wait()

        assert sandbox.returncode == 0

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

    @pytest.mark.asyncio
    async def test_wait_polls_until_stable(self) -> None:
        """Test wait polls through transient states until stable."""
        from coreweave.aviato.v1beta1 import atc_pb2

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        call_count = 0

        async def mock_get(request: MagicMock) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            if call_count < 3:
                mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_RUNNING
            else:
                mock_response.sandbox_status = atc_pb2.SANDBOX_STATUS_COMPLETED
            return mock_response

        sandbox._client.get = mock_get

        await sandbox.wait()

        assert call_count >= 3
        assert sandbox.returncode == 0


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
    async def test_stop_returns_success(self) -> None:
        """Test stop returns True on success."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.success = True
        sandbox._client.stop = AsyncMock(return_value=mock_response)
        sandbox._client.close = AsyncMock()

        result = await sandbox.stop()

        assert result is True

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
    async def test_stop_sends_correct_parameters(self) -> None:
        """Test stop sends correct request parameters."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        mock_client = MagicMock()
        sandbox._client = mock_client

        mock_response = MagicMock()
        mock_response.success = True
        mock_client.stop = AsyncMock(return_value=mock_response)
        mock_client.close = AsyncMock()

        await sandbox.stop(snapshot_on_stop=True, graceful_shutdown_seconds=30.0)

        stop_call = mock_client.stop.call_args[0][0]
        assert stop_call.sandbox_id == "test-id"
        assert stop_call.snapshot_on_stop is True
        assert stop_call.graceful_shutdown_seconds == 30

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        """Test stop() is idempotent - safe to call on unstarted or stopped sandbox."""
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
            await asyncio.sleep(10)
            return MagicMock()

        sandbox._client.exec = slow_exec

        with pytest.raises(asyncio.TimeoutError):
            await sandbox.exec(["sleep", "10"], timeout_seconds=0.1)

    @pytest.mark.asyncio
    async def test_read_file_respects_timeout_seconds(self) -> None:
        """Test read_file() raises TimeoutError when timeout_seconds is exceeded."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        async def slow_retrieve(request: MagicMock) -> MagicMock:
            await asyncio.sleep(10)
            return MagicMock()

        sandbox._client.retrieve_file = slow_retrieve

        with pytest.raises(asyncio.TimeoutError):
            await sandbox.read_file("/tmp/test.txt", timeout_seconds=0.1)

    @pytest.mark.asyncio
    async def test_write_file_respects_timeout_seconds(self) -> None:
        """Test write_file() raises TimeoutError when timeout_seconds is exceeded."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        async def slow_add_file(request: MagicMock) -> MagicMock:
            await asyncio.sleep(10)
            return MagicMock()

        sandbox._client.add_file = slow_add_file

        with pytest.raises(asyncio.TimeoutError):
            await sandbox.write_file("/tmp/test.txt", b"content", timeout_seconds=0.1)

    @pytest.mark.asyncio
    async def test_exec_timeout_uses_default_when_not_specified(self) -> None:
        """Test exec() uses request_timeout_seconds when timeout_seconds not given."""
        sandbox = Sandbox(command="sleep", args=["infinity"], request_timeout_seconds=60.0)
        sandbox._sandbox_id = "test-id"
        sandbox._client = MagicMock()

        mock_response = MagicMock()
        mock_response.result.stdout = ""
        mock_response.result.stderr = ""
        mock_response.result.exit_code = 0
        sandbox._client.exec = AsyncMock(return_value=mock_response)

        await sandbox.exec(["echo", "test"])

        call_args = sandbox._client.exec.call_args[0][0]
        assert call_args.max_timeout_seconds == 60
