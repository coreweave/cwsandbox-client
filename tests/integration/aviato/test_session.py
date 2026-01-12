"""Integration tests for aviato._session module.

These tests require a running Aviato backend.
Set AVIATO_BASE_URL and AVIATO_API_KEY environment variables before running.
"""

import pytest

from aviato import Sandbox, SandboxDefaults, Serialization

GLOBAL_CONSTANT = 42


@pytest.mark.asyncio
async def test_session_create_sandbox(sandbox_defaults: SandboxDefaults) -> None:
    """Test creating sandbox via session."""
    async with Sandbox.session(sandbox_defaults) as session:
        sandbox = session.create()

        async with sandbox:
            assert sandbox.sandbox_id is not None

            result = await sandbox.exec(["echo", "from session"])
            assert result.returncode == 0


@pytest.mark.asyncio
async def test_session_multiple_sandboxes(sandbox_defaults: SandboxDefaults) -> None:
    """Test session managing multiple sandboxes."""
    async with Sandbox.session(sandbox_defaults) as session:
        sb1 = session.create()
        sb2 = session.create()

        async with sb1, sb2:
            r1 = await sb1.exec(["echo", "sandbox-1"])
            r2 = await sb2.exec(["echo", "sandbox-2"])

            assert r1.stdout.strip() == "sandbox-1"
            assert r2.stdout.strip() == "sandbox-2"


@pytest.mark.asyncio
async def test_session_function_pickle(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function execution with pickle serialization."""
    async with Sandbox.session(sandbox_defaults) as session:

        @session.function()
        def add(x: int, y: int) -> int:
            return x + y

        result = await add(2, 3)

        assert result == 5


@pytest.mark.asyncio
async def test_session_function_json(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function execution with JSON serialization."""
    async with Sandbox.session(sandbox_defaults) as session:

        @session.function(serialization=Serialization.JSON)
        def create_dict(key: str, value: int) -> dict[str, int]:
            return {key: value}

        result = await create_dict("test", 42)

        assert result == {"test": 42}


@pytest.mark.asyncio
async def test_session_function_with_closure(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function with closure variables."""
    multiplier = 10

    async with Sandbox.session(sandbox_defaults) as session:

        @session.function()
        def multiply(x: int) -> int:
            return x * multiplier

        result = await multiply(5)

        assert result == 50


@pytest.mark.asyncio
async def test_session_function_raises_exception(sandbox_defaults: SandboxDefaults) -> None:
    """Test that exceptions from sandbox functions are properly propagated."""
    from aviato.exceptions import SandboxExecutionError

    async with Sandbox.session(sandbox_defaults) as session:

        @session.function()
        def raises_value_error() -> None:
            raise ValueError("test error message")

        with pytest.raises(SandboxExecutionError) as exc_info:
            await raises_value_error()

        assert exc_info.value.exec_result is not None
        assert exc_info.value.exec_result.returncode != 0


@pytest.mark.asyncio
async def test_session_function_with_global_variables(sandbox_defaults: SandboxDefaults) -> None:
    """Test function execution with module-level global variables."""
    async with Sandbox.session(sandbox_defaults) as session:

        @session.function()
        def use_global() -> int:
            return GLOBAL_CONSTANT * 2

        result = await use_global()

        assert result == 84


@pytest.mark.asyncio
async def test_session_create_after_close_raises(sandbox_defaults: SandboxDefaults) -> None:
    """Test creating sandbox after session.close() raises SandboxError."""
    from aviato.exceptions import SandboxError

    session = Sandbox.session(sandbox_defaults)

    await session.close()

    with pytest.raises(SandboxError) as exc_info:
        session.create()

    assert "closed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_session_close_stops_orphaned_sandboxes(sandbox_defaults: SandboxDefaults) -> None:
    """Test session.close() stops sandboxes that weren't manually stopped.

    TODO: Convert to sync test when Session gets sync context manager support
    in branch 04-session-function. Currently async because Session only has
    async context manager in this branch.
    """
    async with Sandbox.session(sandbox_defaults) as session:
        sandbox = session.create()

        # Start sandbox in async context - must use __aenter__ directly since
        # Session is async-only in this branch
        await sandbox.__aenter__()

        assert sandbox.sandbox_id is not None
        assert session.sandbox_count == 1
        # Intentionally NOT calling __aexit__ - sandbox is orphaned

    # Session close should have stopped the orphaned sandbox
    assert session.sandbox_count == 0


@pytest.mark.asyncio
async def test_session_function_pickle_complex_types(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function with complex types using pickle serialization."""
    async with Sandbox.session(sandbox_defaults) as session:

        @session.function(serialization=Serialization.PICKLE)
        def process_nested(data: dict) -> dict:
            return {
                "processed": True,
                "original": data,
                "computed": data["value"] * 2,
            }

        result = await process_nested({"value": 21, "nested": {"key": "val"}})

        assert result["processed"] is True
        assert result["computed"] == 42
        assert result["original"]["nested"]["key"] == "val"
