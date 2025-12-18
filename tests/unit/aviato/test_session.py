"""Unit tests for aviato._session module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aviato import Sandbox, SandboxDefaults, Serialization, Session


class TestSessionInit:
    """Tests for Session initialization."""

    def test_session_init_no_defaults(self) -> None:
        """Test Session can be initialized without defaults."""
        session = Session()

        assert session._defaults is not None  # Creates empty defaults
        assert session._sandboxes == {}

    def test_session_init_with_defaults(self) -> None:
        """Test Session can be initialized with defaults."""
        defaults = SandboxDefaults(container_image="python:3.10")
        session = Session(defaults)

        assert session._defaults == defaults


class TestSessionCreate:
    """Tests for Session.create method."""

    def test_create_returns_sandbox(self) -> None:
        """Test session.create returns a Sandbox instance."""
        session = Session()
        sandbox = session.create(command="sleep", args=["infinity"])

        assert isinstance(sandbox, Sandbox)
        assert sandbox._command == "sleep"
        assert sandbox._args == ["infinity"]

    def test_create_applies_session_defaults(self) -> None:
        """Test session.create applies session defaults."""
        defaults = SandboxDefaults(
            container_image="python:3.10",
            request_timeout_seconds=60.0,
        )
        session = Session(defaults)

        sandbox = session.create(command="sleep", args=["infinity"])

        assert sandbox._container_image == "python:3.10"
        assert sandbox._request_timeout_seconds == 60.0

    def test_create_registers_sandbox(self) -> None:
        """Test session.create registers the sandbox."""
        session = Session()

        sandbox = session.create(command="sleep", args=["infinity"])

        assert id(sandbox) in session._sandboxes

    def test_create_explicit_overrides_defaults(self) -> None:
        """Test explicit args in create override session defaults."""
        defaults = SandboxDefaults(container_image="python:3.10")
        session = Session(defaults)

        sandbox = session.create(
            command="sleep",
            args=["infinity"],
            container_image="python:3.12",
        )

        assert sandbox._container_image == "python:3.12"


class TestSessionContextManager:
    """Tests for Session context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enters(self) -> None:
        """Test Session enters context successfully."""
        async with Session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_empty(self) -> None:
        """Test Session cleanup with no sandboxes."""
        async with Session() as session:
            pass  # No sandboxes created

        assert session._sandboxes == {}


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

        await session.close()
        await session.close()

        assert session._sandboxes == {}


class TestSessionFromSandbox:
    """Tests for creating sessions via Sandbox.session()."""

    def test_sandbox_session_returns_session(self) -> None:
        """Test Sandbox.session returns a Session."""
        session = Sandbox.session()

        assert isinstance(session, Session)

    def test_sandbox_session_with_defaults(self) -> None:
        """Test Sandbox.session accepts defaults."""
        defaults = SandboxDefaults(container_image="python:3.10")
        session = Sandbox.session(defaults)

        assert session._defaults == defaults


class TestSessionFunctionDecorator:
    """Tests for Session.function() decorator."""

    def test_function_decorator_with_parens(self) -> None:
        """Test @session.function() decorator with parentheses."""
        session = Session()

        @session.function()
        def add(x: int, y: int) -> int:
            return x + y

        import asyncio

        assert asyncio.iscoroutinefunction(add)

    def test_function_decorator_preserves_name(self) -> None:
        """Test decorator preserves function name."""
        session = Session()

        @session.function()
        def my_unique_function_name(x: int) -> int:
            return x

        assert my_unique_function_name.__name__ == "my_unique_function_name"

    def test_function_decorator_accepts_serialization(self) -> None:
        """Test decorator accepts serialization parameter."""
        session = Session()

        @session.function(serialization=Serialization.PICKLE)
        def process(data: list[int]) -> int:
            return sum(data)

        import asyncio

        assert asyncio.iscoroutinefunction(process)

    def test_function_decorator_accepts_container_image(self) -> None:
        """Test decorator accepts container_image override."""
        session = Session()

        @session.function(container_image="python:3.12")
        def compute(x: int) -> int:
            return x * 2

        import asyncio

        assert asyncio.iscoroutinefunction(compute)

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

    def test_function_decorator_uses_session_defaults_temp_dir(self) -> None:
        """Test decorator uses session defaults for temp_dir."""
        defaults = SandboxDefaults(temp_dir="/custom/tmp")
        session = Session(defaults)

        @session.function()
        def func(x: int) -> int:
            return x

        import asyncio

        assert asyncio.iscoroutinefunction(func)
