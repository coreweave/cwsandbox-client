# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Integration tests for aviato._session module.

These tests require a running Aviato backend.
Set AVIATO_BASE_URL and AVIATO_API_KEY environment variables before running.
"""

import pytest

from aviato import Sandbox, SandboxDefaults, Serialization

GLOBAL_CONSTANT = 42


def test_session_create_sandbox(sandbox_defaults: SandboxDefaults) -> None:
    """Test creating sandbox via session."""
    with Sandbox.session(sandbox_defaults) as session:
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        # sandbox_id is assigned on first operation (deferred-start)
        result = sandbox.exec(["echo", "from session"]).result()
        assert result.returncode == 0
        assert sandbox.sandbox_id is not None


def test_session_multiple_sandboxes(sandbox_defaults: SandboxDefaults) -> None:
    """Test session managing multiple sandboxes."""
    with Sandbox.session(sandbox_defaults) as session:
        sb1 = session.sandbox(command="sleep", args=["infinity"])
        sb2 = session.sandbox(command="sleep", args=["infinity"])

        r1 = sb1.exec(["echo", "sandbox-1"]).result()
        r2 = sb2.exec(["echo", "sandbox-2"]).result()

        assert r1.stdout.strip() == "sandbox-1"
        assert r2.stdout.strip() == "sandbox-2"


def test_session_function_pickle(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function execution with pickle serialization."""
    with Sandbox.session(sandbox_defaults) as session:

        @session.function(serialization=Serialization.PICKLE)
        def add(x: int, y: int) -> int:
            return x + y

        result = add.remote(2, 3).result()

        assert result == 5


def test_session_function_json(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function execution with JSON serialization."""
    with Sandbox.session(sandbox_defaults) as session:

        @session.function(serialization=Serialization.JSON)
        def create_dict(key: str, value: int) -> dict[str, int]:
            return {key: value}

        result = create_dict.remote("test", 42).result()

        assert result == {"test": 42}


def test_session_function_with_closure(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function with closure variables."""
    multiplier = 10

    with Sandbox.session(sandbox_defaults) as session:

        @session.function()
        def multiply(x: int) -> int:
            return x * multiplier

        result = multiply.remote(5).result()

        assert result == 50


def test_session_function_raises_exception(sandbox_defaults: SandboxDefaults) -> None:
    """Test that exceptions from sandbox functions are properly propagated."""
    from aviato.exceptions import SandboxExecutionError

    with Sandbox.session(sandbox_defaults) as session:

        @session.function()
        def raises_value_error() -> None:
            raise ValueError("test error message")

        with pytest.raises(SandboxExecutionError) as exc_info:
            raises_value_error.remote().result()

        assert exc_info.value.exec_result is not None
        assert exc_info.value.exec_result.returncode != 0


def test_session_function_with_global_variables(sandbox_defaults: SandboxDefaults) -> None:
    """Test function execution with module-level global variables."""
    with Sandbox.session(sandbox_defaults) as session:

        @session.function()
        def use_global() -> int:
            return GLOBAL_CONSTANT * 2

        result = use_global.remote().result()

        assert result == 84


def test_session_sandbox_after_close_raises(sandbox_defaults: SandboxDefaults) -> None:
    """Test creating sandbox after session.close() raises SandboxError."""
    from aviato.exceptions import SandboxError

    session = Sandbox.session(sandbox_defaults)

    session.close().result()

    with pytest.raises(SandboxError) as exc_info:
        session.sandbox()

    assert "closed" in str(exc_info.value).lower()


def test_session_close_stops_orphaned_sandboxes(sandbox_defaults: SandboxDefaults) -> None:
    """Test session.close() stops sandboxes that weren't manually stopped."""
    with Sandbox.session(sandbox_defaults) as session:
        sandbox = session.sandbox(command="sleep", args=["infinity"])
        sandbox.wait()  # trigger auto-start so sandbox has an ID to stop

        assert sandbox.sandbox_id is not None
        assert session.sandbox_count == 1
        # Intentionally NOT calling __aexit__ - sandbox is orphaned

    # Session close should have stopped the orphaned sandbox
    assert session.sandbox_count == 0


def test_session_function_pickle_complex_types(sandbox_defaults: SandboxDefaults) -> None:
    """Test session function with complex types using pickle serialization."""
    with Sandbox.session(sandbox_defaults) as session:

        @session.function(serialization=Serialization.PICKLE)
        def process_nested(data: dict) -> dict:
            return {
                "processed": True,
                "original": data,
                "computed": data["value"] * 2,
            }

        result = process_nested.remote({"value": 21, "nested": {"key": "val"}}).result()

        assert result["processed"] is True
        assert result["computed"] == 42
        assert result["original"]["nested"]["key"] == "val"


# Async context manager tests


@pytest.mark.asyncio
async def test_session_async_context_manager(sandbox_defaults: SandboxDefaults) -> None:
    """Test async context manager for Session.

    Verifies that 'async with Session(...)' properly:
    1. Creates and manages sandboxes
    2. Allows await on exec() and file operations
    3. Properly cleans up (stops all sandboxes) on exit

    This is a regression test for event loop routing bugs where __aexit__
    directly awaited async operations instead of routing through _LoopManager.
    """
    from aviato import Session

    async with Session(sandbox_defaults) as session:
        # Create sandbox through session
        sandbox = session.sandbox(command="sleep", args=["infinity"])

        # Use await pattern (not .result()) - first exec triggers auto-start
        result = await sandbox.exec(["echo", "async session"])
        assert result.returncode == 0
        assert result.stdout.strip() == "async session"
        assert sandbox.sandbox_id is not None

        # Verify session.list() works with await
        sandboxes = await session.list()
        assert len(sandboxes) >= 1

    # After exiting, all session sandboxes should be cleaned up
    assert session.sandbox_count == 0


# List include_stopped tests


def test_session_list_include_stopped(sandbox_defaults: SandboxDefaults) -> None:
    """Test session.list(include_stopped=True) returns terminal sandboxes.

    Creates a sandbox via session, lets it complete, then verifies that
    session.list(include_stopped=True) returns it while the default
    session.list() excludes it.
    """
    import time
    import uuid

    unique_tag = f"e2e-session-stopped-{uuid.uuid4().hex[:8]}"
    defaults = sandbox_defaults.with_overrides(tags=sandbox_defaults.merge_tags([unique_tag]))

    with Sandbox.session(defaults) as session:
        sandbox = session.sandbox(command="echo", args=["hello"])
        sandbox.wait_until_complete(timeout=60.0).result()
        sandbox_id = sandbox.sandbox_id
        assert sandbox_id is not None

    # Session is closed, sandbox is stopped. Wait for the status to propagate.
    time.sleep(5)

    # Use a fresh session to list with the same tags
    with Sandbox.session(defaults) as session:
        # Default list should exclude the stopped sandbox (eventually)
        for _ in range(15):
            active = session.list().result()
            ids = [sb.sandbox_id for sb in active]
            if sandbox_id not in ids:
                break
            time.sleep(2)

        # include_stopped should include it.
        # The status may not yet reflect the final terminal state, so we only
        # assert the sandbox is returned — not a specific status.
        found = False
        for _ in range(15):
            all_sandboxes = session.list(include_stopped=True).result()
            for sb in all_sandboxes:
                if sb.sandbox_id == sandbox_id:
                    found = True
                    break
            if found:
                break
            time.sleep(1)

        assert found, f"Stopped sandbox {sandbox_id} not found with include_stopped=True"


def test_session_list_terminal_status_filter(sandbox_defaults: SandboxDefaults) -> None:
    """Test that session.list with a terminal status filter returns stopped sandboxes.

    A terminal status filter automatically widens the search,
    even without include_stopped=True.

    The sandbox is created outside the session context manager to avoid
    session.close() calling stop(), which would change the status from
    COMPLETED to TERMINATED.
    """
    import time
    import uuid

    unique_tag = f"e2e-session-status-filter-{uuid.uuid4().hex[:8]}"
    defaults = sandbox_defaults.with_overrides(tags=sandbox_defaults.merge_tags([unique_tag]))

    # Create sandbox directly (not via session) so it reaches COMPLETED
    # without being stopped by session cleanup.
    sandbox = Sandbox.run("echo", "hello", defaults=defaults, tags=[unique_tag])
    sandbox_id = sandbox.sandbox_id
    assert sandbox_id is not None
    sandbox.wait_until_complete(timeout=60.0).result()

    # Use session.list() with terminal status filter to find it
    with Sandbox.session(defaults) as session:
        found = False
        for _ in range(15):
            sandboxes = session.list(status="completed").result()
            for sb in sandboxes:
                if sb.sandbox_id == sandbox_id:
                    assert sb.status == "completed"
                    found = True
                    break
            if found:
                break
            time.sleep(1)

        assert found, f"Sandbox {sandbox_id} not found with status='completed' filter via session"
