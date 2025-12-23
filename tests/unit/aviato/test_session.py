"""Unit tests for aviato._session module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aviato import Sandbox, Session


class TestSessionCreate:
    """Tests for Session.create method."""

    def test_create_returns_sandbox(self) -> None:
        """Test session.create returns a Sandbox instance."""
        session = Session()
        sandbox = session.create(command="sleep", args=["infinity"])

        assert isinstance(sandbox, Sandbox)


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
        sandbox1 = session.create(command="sleep", args=["infinity"])
        sandbox2 = session.create(command="sleep", args=["infinity"])

        sandbox1.stop = AsyncMock()
        sandbox2.stop = AsyncMock()

        await session.close()

        sandbox1.stop.assert_called_once()
        sandbox2.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandbox_deregisters_on_stop(self) -> None:
        """Test sandbox is deregistered from session when stopped."""
        from unittest.mock import AsyncMock, MagicMock

        session = Session()
        sandbox = session.create(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "test-sandbox-id"

        sandbox._client = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.exit_code = 0
        sandbox._client.stop = AsyncMock(return_value=mock_response)
        sandbox._client.close = AsyncMock()

        assert id(sandbox) in session._sandboxes

        await sandbox.stop()

        assert id(sandbox) not in session._sandboxes

    @pytest.mark.asyncio
    async def test_close_attempts_all_sandboxes_on_partial_failure(self) -> None:
        """Test session.close() attempts to stop all sandboxes even if some fail."""
        from unittest.mock import AsyncMock

        from aviato.exceptions import SandboxError

        session = Session()
        sandbox1 = session.create(command="sleep", args=["infinity"])
        sandbox2 = session.create(command="sleep", args=["infinity"])
        sandbox3 = session.create(command="sleep", args=["infinity"])

        sandbox1.stop = AsyncMock()
        sandbox2.stop = AsyncMock(side_effect=Exception("Network error"))
        sandbox3.stop = AsyncMock()

        with pytest.raises(SandboxError, match="Failed to stop 1 sandbox"):
            await session.close()

        sandbox1.stop.assert_called_once()
        sandbox2.stop.assert_called_once()
        sandbox3.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Test calling close() multiple times is safe."""
        session = Session()

        # Should not raise on repeated calls
        await session.close()
        await session.close()


class TestSessionFromSandbox:
    """Tests for creating sessions via Sandbox.session()."""

    def test_sandbox_session_returns_session(self) -> None:
        """Test Sandbox.session returns a Session."""
        session = Sandbox.session()
        assert isinstance(session, Session)


class TestSessionFunctionDecorator:
    """Tests for Session.function() decorator."""

    def test_function_decorator_returns_async_callable(self) -> None:
        """Test @session.function() returns an async function that preserves name."""
        import asyncio

        session = Session()

        @session.function()
        def my_function(x: int, y: int) -> int:
            return x + y

        assert asyncio.iscoroutinefunction(my_function)
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
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = AsyncMock()

        mock_exec_result = MagicMock()
        mock_exec_result.returncode = 0
        mock_exec_result.stderr = ""
        mock_sandbox.exec = AsyncMock(return_value=mock_exec_result)

        result_json = json.dumps(5).encode()
        mock_sandbox.read_file = AsyncMock(return_value=result_json)

        with patch.object(session, "create", return_value=mock_sandbox):
            result = await add(2, 3)

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
        mock_sandbox.sandbox_id = "test-sandbox-id"
        mock_sandbox.write_file = AsyncMock()

        mock_exec_result = MagicMock()
        mock_exec_result.returncode = 0
        mock_exec_result.stderr = ""
        mock_sandbox.exec = AsyncMock(return_value=mock_exec_result)

        result_json = json.dumps(50).encode()
        mock_sandbox.read_file = AsyncMock(return_value=result_json)

        with patch.object(session, "create", return_value=mock_sandbox):
            result = await compute_with_closure(5)

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

    def test_create_with_valid_kwargs(self) -> None:
        """Test Session.create accepts valid kwargs."""
        session = Session()
        sandbox = session.create(
            command="echo",
            args=["hello"],
            resources={"cpu": "100m"},
            ports=[{"container_port": 8080}],
        )
        assert sandbox._start_kwargs["resources"] == {"cpu": "100m"}
        assert sandbox._start_kwargs["ports"] == [{"container_port": 8080}]

    def test_create_with_invalid_kwargs(self) -> None:
        """Test Session.create rejects invalid kwargs."""
        session = Session()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            session.create(
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
        assert callable(add)

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.list = AsyncMock(
                    return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
                )

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.list = AsyncMock(
                    return_value=atc_pb2.ListSandboxesResponse(sandboxes=[])
                )

                await session.list()

                call_args = mock_atc_instance.list.call_args[0][0]
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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.list = AsyncMock(
                    return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
                )

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.list = AsyncMock(
                    return_value=atc_pb2.ListSandboxesResponse(sandboxes=[mock_sandbox_info])
                )

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.get = AsyncMock(return_value=mock_response)

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.get = AsyncMock(return_value=mock_response)

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

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            with patch("aviato._sandbox.atc_connect.ATCServiceClient") as mock_atc_client:
                mock_atc_instance = MagicMock()
                mock_atc_client.return_value = mock_atc_instance
                mock_atc_instance.get = AsyncMock(return_value=mock_response)

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
