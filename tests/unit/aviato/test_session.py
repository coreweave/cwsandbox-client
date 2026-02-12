# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Unit tests for aviato._session module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aviato import Sandbox, Session
from tests.unit.aviato.conftest import make_operation_ref, make_process


class TestSessionSandbox:
    """Tests for Session.sandbox method."""

    def test_sandbox_returns_sandbox(self) -> None:
        """Test session.sandbox returns a Sandbox instance."""
        session = Session()
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        assert isinstance(sandbox, Sandbox)

    def test_sandbox_inherits_environment_variables_from_session_defaults(self) -> None:
        """Test session.sandbox inherits environment variables from session defaults."""
        from aviato import SandboxDefaults

        defaults = SandboxDefaults(
            environment_variables={
                "PROJECT_ID": "test-project",
                "LOG_LEVEL": "info",
            }
        )
        session = Session(defaults)
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        assert sandbox._environment_variables == {
            "PROJECT_ID": "test-project",
            "LOG_LEVEL": "info",
        }

    def test_sandbox_overrides_session_environment_variables(self) -> None:
        """Test session.sandbox can override session environment variables."""
        from aviato import SandboxDefaults

        defaults = SandboxDefaults(
            environment_variables={
                "PROJECT_ID": "test-project",
                "LOG_LEVEL": "info",
            }
        )
        session = Session(defaults)
        sandbox = session.sandbox(
            command="sleep",
            args=["infinity"],
            environment_variables={
                "LOG_LEVEL": "debug",  # Override
                "MODEL_NAME": "gpt2",  # Add new
            },
        )

        assert sandbox._environment_variables == {
            "PROJECT_ID": "test-project",  # Inherited
            "LOG_LEVEL": "debug",  # Overridden
            "MODEL_NAME": "gpt2",  # Added
        }


class TestSessionContextManager:
    """Tests for Session context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_works(self) -> None:
        """Test Session can be used as async context manager."""
        async with Session() as session:
            assert isinstance(session, Session)


class TestSessionCleanup:
    """Tests for Session cleanup behavior."""

    @pytest.mark.asyncio
    async def test_close_stops_orphaned_sandboxes(self) -> None:
        """Test session.close() stops sandboxes that weren't manually stopped."""
        from unittest.mock import AsyncMock

        session = Session()
        sandbox1 = session.sandbox(command="sleep", args=["infinity"])
        sandbox2 = session.sandbox(command="sleep", args=["infinity"])

        # Mock the internal async stop method (close uses _stop_async directly)
        sandbox1._stop_async = AsyncMock()
        sandbox2._stop_async = AsyncMock()

        await session._close_async()

        sandbox1._stop_async.assert_called_once()
        sandbox2._stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandbox_deregisters_on_stop(self) -> None:
        """Test sandbox is deregistered from session when stopped."""
        from unittest.mock import AsyncMock, MagicMock

        session = Session()
        sandbox = session.sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-sandbox-id"

        sandbox._channel = MagicMock()
        sandbox._stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.exit_code = 0
        sandbox._stub.Stop = AsyncMock(return_value=mock_response)
        sandbox._channel.close = AsyncMock()

        assert id(sandbox) in session._sandboxes

        await sandbox.stop()

        assert id(sandbox) not in session._sandboxes

    @pytest.mark.asyncio
    async def test_close_attempts_all_sandboxes_on_partial_failure(self) -> None:
        """Test session.close() attempts to stop all sandboxes even if some fail."""
        from unittest.mock import AsyncMock

        from aviato.exceptions import SandboxError

        session = Session()
        sandbox1 = session.sandbox(command="sleep", args=["infinity"])
        sandbox2 = session.sandbox(command="sleep", args=["infinity"])
        sandbox3 = session.sandbox(command="sleep", args=["infinity"])

        # Mock the internal async stop method (close uses _stop_async directly)
        sandbox1._stop_async = AsyncMock()
        sandbox2._stop_async = AsyncMock(side_effect=Exception("Network error"))
        sandbox3._stop_async = AsyncMock()

        with pytest.raises(SandboxError, match="Failed to stop 1 sandbox"):
            await session._close_async()

        sandbox1._stop_async.assert_called_once()
        sandbox2._stop_async.assert_called_once()
        sandbox3._stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Test calling close() multiple times is safe."""
        session = Session()

        # Should not raise on repeated calls (use _close_async for async test)
        await session._close_async()
        await session._close_async()

    def test_close_returns_operation_ref(self) -> None:
        """Test close() returns OperationRef that can be awaited."""
        from aviato import OperationRef

        session = Session()
        ref = session.close()

        assert isinstance(ref, OperationRef)
        # Get the result (should complete without error)
        ref.result()


class TestSessionSandboxMethod:
    """Tests for Session.sandbox() method."""

    def test_sandbox_returns_sandbox(self) -> None:
        """Test session.sandbox() returns a Sandbox instance."""
        session = Session()
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        assert isinstance(sandbox, Sandbox)
        assert sandbox._session is session

    def test_sandbox_returns_unstarted(self) -> None:
        """Test session.sandbox() returns sandbox with no sandbox_id (unstarted)."""
        session = Session()
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        assert sandbox.sandbox_id is None

    def test_sandbox_does_not_make_network_call(self) -> None:
        """Test session.sandbox() does not call start() or _start_async()."""
        session = Session()

        with patch.object(Sandbox, "start") as mock_start:
            session.sandbox(command="sleep", args=["infinity"])
            mock_start.assert_not_called()

    def test_sandbox_raises_if_session_closed(self) -> None:
        """Test session.sandbox() raises SandboxError if session is closed."""
        from aviato.exceptions import SandboxError

        session = Session()
        session._closed = True

        with pytest.raises(SandboxError, match="session is closed"):
            session.sandbox(command="sleep", args=["infinity"])


class TestSessionSyncContextManager:
    """Tests for Session sync context manager."""

    def test_sync_context_manager_works(self) -> None:
        """Test Session can be used as sync context manager."""
        from unittest.mock import MagicMock, patch

        with patch.object(Session, "close") as mock_close:
            mock_ref = MagicMock()
            mock_close.return_value = mock_ref

            with Session() as session:
                assert isinstance(session, Session)

            mock_close.assert_called_once()
            mock_ref.result.assert_called_once()

    def test_sync_context_manager_closes_on_exception(self) -> None:
        """Test sync context manager closes session even on exception."""
        from unittest.mock import MagicMock, patch

        with patch.object(Session, "close") as mock_close:
            mock_ref = MagicMock()
            mock_close.return_value = mock_ref

            with pytest.raises(ValueError):
                with Session():
                    raise ValueError("test error")

            mock_close.assert_called_once()
            mock_ref.result.assert_called_once()


class TestSessionFromSandbox:
    """Tests for creating sessions via Sandbox.session()."""

    def test_sandbox_session_returns_session(self) -> None:
        """Test Sandbox.session returns a Session."""
        session = Sandbox.session()
        assert isinstance(session, Session)


class TestSessionFunctionDecorator:
    """Tests for Session.function() decorator."""

    def test_function_decorator_returns_remote_function(self) -> None:
        """Test @session.function() returns a RemoteFunction that preserves name."""
        from aviato._function import RemoteFunction

        session = Session()

        @session.function()
        def my_function(x: int, y: int) -> int:
            return x + y

        assert isinstance(my_function, RemoteFunction)
        assert my_function.__name__ == "my_function"

    @pytest.mark.asyncio
    async def test_function_decorator_executes_in_sandbox(self) -> None:
        """Test decorated function executes in sandbox."""
        session = Session()

        @session.function()
        def add(x: int, y: int) -> int:
            return x + y

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox._start_async = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = MagicMock(return_value=make_operation_ref(None))
        mock_sandbox.exec = MagicMock(return_value=make_process(returncode=0))

        result_json = json.dumps(5).encode()
        mock_sandbox.read_file = MagicMock(return_value=make_operation_ref(result_json))

        with patch("aviato._sandbox.Sandbox", return_value=mock_sandbox):
            result = await add.remote(2, 3)

            assert result == 5

    @pytest.mark.asyncio
    async def test_function_decorator_with_closure_variables(self) -> None:
        """Test decorated function captures closure variables."""
        session = Session()
        multiplier = 10

        @session.function()
        def compute_with_closure(x: int) -> int:
            return x * multiplier

        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = AsyncMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = AsyncMock(return_value=None)
        mock_sandbox._start_async = AsyncMock(return_value=None)
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = MagicMock(return_value=make_operation_ref(None))
        mock_sandbox.exec = MagicMock(return_value=make_process(returncode=0))

        result_json = json.dumps(50).encode()
        mock_sandbox.read_file = MagicMock(return_value=make_operation_ref(result_json))

        with patch("aviato._sandbox.Sandbox", return_value=mock_sandbox):
            result = await compute_with_closure.remote(5)

            assert result == 50

            write_call = mock_sandbox.write_file.call_args
            payload_bytes = write_call[0][1]
            payload = json.loads(payload_bytes)
            assert "multiplier" in payload["closure_vars"]
            assert payload["closure_vars"]["multiplier"] == 10

    @pytest.mark.asyncio
    async def test_function_decorator_rejects_async(self) -> None:
        """Test decorator rejects async functions."""
        from aviato.exceptions import AsyncFunctionError

        session = Session()

        with pytest.raises(AsyncFunctionError, match="async"):

            @session.function()
            async def async_func(x: int) -> int:
                return x


class TestSessionKwargsValidation:
    """Tests for kwargs validation in Session methods."""

    def test_sandbox_with_valid_kwargs(self) -> None:
        """Test Session.sandbox accepts valid kwargs."""
        session = Session()
        sandbox = session.sandbox(
            command="echo",
            args=["hello"],
            resources={"cpu": "100m"},
            ports=[{"container_port": 8080}],
        )
        assert sandbox._start_kwargs["resources"] == {"cpu": "100m"}
        assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]

    def test_sandbox_with_invalid_kwargs(self) -> None:
        """Test Session.sandbox rejects invalid kwargs."""
        session = Session()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            session.sandbox(
                command="echo",
                args=["hello"],
                invalid_param="value",
            )

    def test_function_with_valid_sandbox_kwargs(self) -> None:
        """Test session.function() accepts valid sandbox_kwargs."""
        session = Session()

        @session.function(
            resources={"cpu": "100m"},
            ports=[{"container_port": 8080}],
        )
        def add(x: int, y: int) -> int:
            return x + y

        # Decorator should work without raising
        assert callable(add.remote)

    def test_function_with_invalid_sandbox_kwargs(self) -> None:
        """Test session.function() rejects invalid sandbox_kwargs."""
        session = Session()

        with pytest.raises(TypeError, match="unexpected keyword argument"):

            @session.function(
                invalid_param="value",
            )
            def add(x: int, y: int) -> int:
                return x + y


class TestSessionList:
    """Tests for Session.list method."""

    @pytest.mark.asyncio
    async def test_list_returns_sandbox_instances(self, mock_aviato_api_key: str) -> None:
        """Test session.list() returns Sandbox instances."""
        from coreweave.aviato.v1beta1 import atc_pb2
        from google.protobuf import timestamp_pb2

        from aviato import SandboxDefaults

        mock_sandbox_info = atc_pb2.SandboxInfo(
            sandbox_id="test-123",
            sandbox_status=atc_pb2.SANDBOX_STATUS_RUNNING,
            started_at_time=timestamp_pb2.Timestamp(seconds=1234567890),
            tower_id="tower-1",
            tower_group_id="group-1",
            runway_id="runway-1",
        )

        defaults = SandboxDefaults(tags=("session-tag",))
        session = Session(defaults)

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(
            return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
        )

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandboxes = await session.list()

            assert len(sandboxes) == 1
            assert isinstance(sandboxes[0], Sandbox)

    @pytest.mark.asyncio
    async def test_list_uses_default_tags(self, mock_aviato_api_key: str) -> None:
        """Test session.list() automatically filters by session's default tags."""
        from coreweave.aviato.v1beta1 import atc_pb2

        from aviato import SandboxDefaults

        defaults = SandboxDefaults(tags=("session-tag",))
        session = Session(defaults)

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(return_value=atc_pb2.ListSandboxesResponse(sandboxes=[]))

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            await session.list()

            call_args = mock_stub.List.call_args[0][0]
            assert "session-tag" in call_args.tags

    @pytest.mark.asyncio
    async def test_list_with_adopt_registers_sandboxes(self, mock_aviato_api_key: str) -> None:
        """Test session.list(adopt=True) registers sandboxes with session."""
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

        session = Session()

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(
            return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
        )

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandboxes = await session.list(adopt=True)

            assert session.sandbox_count == 1
            assert sandboxes[0]._session is session

    @pytest.mark.asyncio
    async def test_list_without_adopt_does_not_register(self, mock_aviato_api_key: str) -> None:
        """Test session.list(adopt=False) does not register sandboxes."""
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

        session = Session()

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.List = AsyncMock(
            return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
        )

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            await session.list(adopt=False)

            assert session.sandbox_count == 0


class TestSessionFromId:
    """Tests for Session.from_id method."""

    @pytest.mark.asyncio
    async def test_from_id_returns_sandbox_instance(self, mock_aviato_api_key: str) -> None:
        """Test session.from_id() returns a Sandbox instance."""
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

        session = Session()

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandbox = await session.from_id("test-123")

            assert isinstance(sandbox, Sandbox)
            assert sandbox.sandbox_id == "test-123"

    @pytest.mark.asyncio
    async def test_from_id_adopts_by_default(self, mock_aviato_api_key: str) -> None:
        """Test session.from_id() adopts sandbox by default."""
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

        session = Session()

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            sandbox = await session.from_id("test-123")

            assert session.sandbox_count == 1
            assert sandbox._session is session

    @pytest.mark.asyncio
    async def test_from_id_adopt_false_does_not_register(self, mock_aviato_api_key: str) -> None:
        """Test session.from_id(adopt=False) does not register sandbox."""
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

        session = Session()

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.Get = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            await session.from_id("test-123", adopt=False)

            assert session.sandbox_count == 0


class TestSessionAdopt:
    """Tests for Session.adopt method."""

    def test_adopt_registers_sandbox(self) -> None:
        """Test session.adopt() registers sandbox for cleanup."""
        session = Session()
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-123"

        session.adopt(sandbox)

        assert session.sandbox_count == 1
        assert sandbox._session is session

    def test_adopt_raises_on_closed_session(self) -> None:
        """Test session.adopt() raises SandboxError if session is closed."""
        from aviato.exceptions import SandboxError

        session = Session()
        session._closed = True

        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-123"

        with pytest.raises(SandboxError, match="session is closed"):
            session.adopt(sandbox)

    def test_adopt_raises_on_sandbox_without_id(self) -> None:
        """Test session.adopt() raises ValueError if sandbox has no ID."""
        session = Session()
        sandbox = Sandbox(command="sleep", args=["infinity"])

        with pytest.raises(ValueError, match="without sandbox_id"):
            session.adopt(sandbox)


class TestSessionReportTo:
    """Tests for Session report_to parameter."""

    def test_report_to_none_always_creates_reporter(self) -> None:
        """Test report_to=None always creates reporter for lazy detection."""
        from aviato._wandb import WandbReporter

        session = Session(report_to=None)
        assert isinstance(session._reporter, WandbReporter)

    def test_report_to_empty_list_disables_reporting(self) -> None:
        """Test report_to=[] disables reporting."""
        session = Session(report_to=[])
        assert session._reporter is None

    def test_report_to_wandb_creates_reporter(self) -> None:
        """Test report_to=['wandb'] creates reporter."""
        from aviato._wandb import WandbReporter

        session = Session(report_to=["wandb"])
        assert isinstance(session._reporter, WandbReporter)

    def test_report_to_non_wandb_disables_reporting(self) -> None:
        """Test report_to with unrecognized values disables reporting."""
        session = Session(report_to=["other"])
        assert session._reporter is None

    def test_lazy_detection_finds_run_after_session_creation(self) -> None:
        """Test lazy detection: create Session, then wandb.init, then log_metrics."""
        session = Session(report_to=None)
        session._reporter.record_sandbox_created()

        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.run = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = session.log_metrics()

        assert result is True
        mock_run.log.assert_called_once()


class TestSessionSandboxMetrics:
    """Tests for Session.sandbox() metrics reporting."""

    def test_sandbox_construction_records_creation(self) -> None:
        """Test session.sandbox() records creation metric immediately."""
        session = Session(report_to=["wandb"])
        session.sandbox(command="sleep", args=["infinity"])

        assert session._reporter._sandboxes_created == 1

    @pytest.mark.asyncio
    async def test_start_async_does_not_double_count(self, mock_aviato_api_key: str) -> None:
        """Test _start_async() does not re-record creation (already counted by sandbox())."""
        session = Session(report_to=["wandb"])
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_id = "test-123"
        mock_response.service_address = ""
        mock_response.exposed_ports = []
        mock_response.applied_ingress_mode = ""
        mock_response.applied_egress_mode = ""
        mock_stub.Start = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            await sandbox._start_async()

        assert session._reporter._sandboxes_created == 1

    @pytest.mark.asyncio
    async def test_start_async_repeated_calls_no_extra_count(
        self, mock_aviato_api_key: str
    ) -> None:
        """Test repeated _start_async() calls do not inflate creation count."""
        session = Session(report_to=["wandb"])
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.sandbox_id = "test-123"
        mock_response.service_address = ""
        mock_response.exposed_ports = []
        mock_response.applied_ingress_mode = ""
        mock_response.applied_egress_mode = ""
        mock_stub.Start = AsyncMock(return_value=mock_response)

        with (
            patch("aviato._sandbox.parse_grpc_target", return_value=("test:443", True)),
            patch("aviato._sandbox.create_channel", return_value=mock_channel),
            patch("aviato._sandbox.atc_pb2_grpc.ATCServiceStub", return_value=mock_stub),
        ):
            await sandbox._start_async()
            await sandbox._start_async()

        assert session._reporter._sandboxes_created == 1

    def test_sandbox_noop_without_reporter(self) -> None:
        """Test session.sandbox() works without reporter (report_to=[])."""
        session = Session(report_to=[])
        sb = session.sandbox(command="sleep", args=["infinity"])

        assert sb is not None
        assert session._reporter is None


class TestSessionLogMetrics:
    """Tests for Session.log_metrics method."""

    def test_log_metrics_returns_false_without_reporter(self) -> None:
        """Test log_metrics returns False when no reporter."""
        session = Session(report_to=[])
        result = session.log_metrics()
        assert result is False

    def test_log_metrics_calls_reporter_log(self) -> None:
        """Test log_metrics delegates to reporter.log."""
        session = Session(report_to=["wandb"])
        session._reporter.record_sandbox_created()
        mock_run = MagicMock()
        session._reporter._run = mock_run

        result = session.log_metrics(step=42)

        assert result is True
        mock_run.log.assert_called_once()

    def test_log_metrics_resets_by_default(self) -> None:
        """Test log_metrics resets metrics by default."""
        session = Session(report_to=["wandb"])
        session._reporter.record_sandbox_created()
        mock_run = MagicMock()
        session._reporter._run = mock_run

        session.log_metrics(step=100)

        assert session._reporter._sandboxes_created == 0

    def test_log_metrics_preserves_when_reset_false(self) -> None:
        """Test log_metrics preserves metrics when reset=False."""
        session = Session(report_to=["wandb"])
        session._reporter.record_sandbox_created()
        mock_run = MagicMock()
        session._reporter._run = mock_run

        session.log_metrics(step=100, reset=False)

        assert session._reporter._sandboxes_created == 1

    def test_log_metrics_preserves_on_failed_log(self) -> None:
        """Test log_metrics preserves metrics when log() returns False (no run)."""
        session = Session(report_to=["wandb"])
        session._reporter.record_sandbox_created()

        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = session.log_metrics(step=100)

        assert result is False
        assert session._reporter._sandboxes_created == 1


class TestSessionAutoTracking:
    """Tests for automatic exec tracking via sandbox completion callback."""

    def test_auto_tracking_reports_to_session_reporter(self) -> None:
        """Test that sandbox exec completion reports to session's reporter."""
        session = Session(report_to=["wandb"])
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        from aviato._types import ProcessResult

        result = ProcessResult(stdout="", stderr="", returncode=0)
        sandbox._on_exec_complete(result, None)

        assert session._reporter._executions == 1
        assert session._reporter._exec_completed_ok == 1

    def test_auto_tracking_nonzero_returncode(self) -> None:
        """Test that nonzero returncode reports COMPLETED_NONZERO to reporter."""
        session = Session(report_to=["wandb"])
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        from aviato._types import ProcessResult

        result = ProcessResult(stdout="", stderr="error", returncode=1)
        sandbox._on_exec_complete(result, None)

        assert session._reporter._executions == 1
        assert session._reporter._exec_completed_nonzero == 1
        assert session._reporter._exec_completed_ok == 0
        assert session._reporter._exec_failures == 0

    def test_auto_tracking_exception_reports_failure(self) -> None:
        """Test that exception reports FAILURE to reporter."""
        session = Session(report_to=["wandb"])
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        sandbox._on_exec_complete(None, Exception("fail"))

        assert session._reporter._executions == 1
        assert session._reporter._exec_failures == 1
        assert session._reporter._exec_completed_ok == 0
        assert session._reporter._exec_completed_nonzero == 0

    def test_auto_tracking_noop_without_reporter(self) -> None:
        """Test that sandbox exec completion is noop without reporter."""
        session = Session(report_to=[])
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        from aviato._types import ProcessResult

        result = ProcessResult(stdout="", stderr="", returncode=0)
        sandbox._on_exec_complete(result, None)


class TestSessionCloseMetrics:
    """Tests for Session close() metrics behavior."""

    @pytest.mark.asyncio
    async def test_close_logs_remaining_metrics(self) -> None:
        """Test session.close() logs remaining metrics."""
        session = Session(report_to=["wandb"])
        session._reporter.record_sandbox_created()
        mock_run = MagicMock()
        session._reporter._run = mock_run

        await session._close_async()

        mock_run.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_does_not_log_without_metrics(self) -> None:
        """Test session close does not log when no metrics."""
        session = Session(report_to=["wandb"])
        mock_run = MagicMock()
        session._reporter._run = mock_run

        await session._close_async()

        mock_run.log.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_preserves_metrics_on_failed_log(self) -> None:
        """Test session close preserves metrics when log() fails (no run)."""
        session = Session(report_to=["wandb"])
        session._reporter.record_sandbox_created()

        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            await session._close_async()

        assert session._reporter._sandboxes_created == 1

    @pytest.mark.asyncio
    async def test_close_does_not_log_without_reporter(self) -> None:
        """Test session close works without reporter."""
        session = Session(report_to=[])

        await session._close_async()


class TestSessionGetMetrics:
    """Tests for Session.get_metrics method."""

    def test_get_metrics_returns_empty_without_reporter(self) -> None:
        """Test get_metrics returns empty dict when report_to=[]."""
        session = Session(report_to=[])
        assert session.get_metrics() == {}

    def test_get_metrics_delegates_to_reporter(self) -> None:
        """Test get_metrics returns reporter metrics after recording."""
        from aviato._wandb import WandbReporter

        session = Session(report_to=["wandb"])
        assert isinstance(session._reporter, WandbReporter)

        session._reporter.record_sandbox_created()
        session._reporter.record_sandbox_created()

        metrics = session.get_metrics()
        assert metrics["aviato/sandboxes_created"] == 2
        assert metrics["aviato/executions"] == 0


class TestSessionInitReporterWarning:
    """Tests for Session._init_reporter warning on mixed report_to values."""

    def test_mixed_report_to_creates_reporter_and_warns(self) -> None:
        """Test report_to with wandb + unsupported creates reporter and logs warning."""
        from aviato._wandb import WandbReporter

        with patch("aviato._session.logger") as mock_logger:
            session = Session(report_to=["wandb", "unsupported"])

        assert isinstance(session._reporter, WandbReporter)
        mock_logger.warning.assert_called_once()
        warning_args = mock_logger.warning.call_args[0]
        assert "unsupported" in str(warning_args)
