# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Unit tests for aviato._loop_manager module."""

import asyncio
import gc
import threading
import time
import weakref
from unittest.mock import AsyncMock, MagicMock

import pytest

from aviato._loop_manager import _LoopManager


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the _LoopManager singleton before and after each test."""
    _LoopManager._reset_for_testing()
    yield
    _LoopManager._reset_for_testing()


class TestLoopManagerSingleton:
    """Tests for _LoopManager singleton behavior."""

    def test_get_returns_singleton(self) -> None:
        """Test _LoopManager.get() returns the same instance."""
        manager1 = _LoopManager.get()
        manager2 = _LoopManager.get()

        assert manager1 is manager2

    def test_get_is_thread_safe(self) -> None:
        """Test _LoopManager.get() is thread-safe with concurrent access."""
        instances: list[_LoopManager] = []
        errors: list[Exception] = []

        def get_instance() -> None:
            try:
                instances.append(_LoopManager.get())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(instances) == 10
        # All instances should be the same singleton
        assert all(inst is instances[0] for inst in instances)


class TestLoopManagerGetLoop:
    """Tests for _LoopManager.get_loop() method."""

    def test_get_loop_creates_loop(self) -> None:
        """Test get_loop() creates an event loop."""
        manager = _LoopManager.get()
        loop = manager.get_loop()

        assert isinstance(loop, asyncio.AbstractEventLoop)

    def test_get_loop_returns_same_loop(self) -> None:
        """Test get_loop() returns the same loop on subsequent calls."""
        manager = _LoopManager.get()
        loop1 = manager.get_loop()
        loop2 = manager.get_loop()

        assert loop1 is loop2

    def test_get_loop_creates_daemon_thread(self) -> None:
        """Test get_loop() creates a daemon thread."""
        manager = _LoopManager.get()
        manager.get_loop()

        assert manager._thread is not None
        assert manager._thread.daemon is True
        assert manager._thread.is_alive()

    def test_get_loop_thread_name(self) -> None:
        """Test the background thread has the expected name."""
        manager = _LoopManager.get()
        manager.get_loop()

        assert manager._thread is not None
        assert manager._thread.name == "aviato-event-loop"

    def test_loop_is_running(self) -> None:
        """Test the event loop is running in the background thread."""
        manager = _LoopManager.get()
        loop = manager.get_loop()

        # Give the thread a moment to start
        time.sleep(0.01)

        assert loop.is_running()


class TestLoopManagerRunSync:
    """Tests for _LoopManager.run_sync() method."""

    def test_run_sync_executes_coroutine(self) -> None:
        """Test run_sync() executes a coroutine and returns result."""
        manager = _LoopManager.get()

        async def async_add(a: int, b: int) -> int:
            return a + b

        result = manager.run_sync(async_add(2, 3))

        assert result == 5

    def test_run_sync_blocks_until_complete(self) -> None:
        """Test run_sync() blocks until coroutine completes."""
        manager = _LoopManager.get()
        completed = False

        async def slow_operation() -> str:
            nonlocal completed
            await asyncio.sleep(0.1)
            completed = True
            return "done"

        result = manager.run_sync(slow_operation())

        assert result == "done"
        assert completed is True

    def test_run_sync_propagates_exceptions(self) -> None:
        """Test run_sync() propagates exceptions from coroutine."""
        manager = _LoopManager.get()

        async def failing_operation() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            manager.run_sync(failing_operation())

    def test_run_sync_raises_from_daemon_thread(self) -> None:
        """Test run_sync() raises RuntimeError if called from daemon thread."""
        manager = _LoopManager.get()
        error_raised = threading.Event()
        error_message: str | None = None

        async def call_run_sync_from_daemon() -> None:
            nonlocal error_message
            coro = asyncio.sleep(0.01)
            try:
                # This should raise because we're in the daemon thread
                manager.run_sync(coro)
            except RuntimeError as e:
                error_message = str(e)
                error_raised.set()
            finally:
                coro.close()  # Close coroutine to prevent unawaited warning

        manager.run_sync(call_run_sync_from_daemon())

        assert error_raised.wait(timeout=1.0)
        assert error_message is not None
        assert "daemon thread" in error_message


class TestLoopManagerRunAsync:
    """Tests for _LoopManager.run_async() method."""

    def test_run_async_returns_future_immediately(self) -> None:
        """Test run_async() returns a Future immediately."""
        import concurrent.futures

        manager = _LoopManager.get()
        started = threading.Event()

        async def slow_operation() -> str:
            started.set()
            await asyncio.sleep(0.5)
            return "done"

        future = manager.run_async(slow_operation())

        assert isinstance(future, concurrent.futures.Future)
        # Future should be returned before operation completes
        started.wait(timeout=0.1)
        assert not future.done()

    def test_run_async_future_contains_result(self) -> None:
        """Test run_async() Future contains the result when done."""
        manager = _LoopManager.get()

        async def async_multiply(a: int, b: int) -> int:
            return a * b

        future = manager.run_async(async_multiply(4, 5))
        result = future.result(timeout=1.0)

        assert result == 20

    def test_run_async_future_propagates_exceptions(self) -> None:
        """Test run_async() Future propagates exceptions."""
        manager = _LoopManager.get()

        async def failing_operation() -> None:
            raise ValueError("async error")

        future = manager.run_async(failing_operation())

        with pytest.raises(ValueError, match="async error"):
            future.result(timeout=1.0)


class TestLoopManagerThreadReuse:
    """Tests for thread reuse across operations."""

    def test_thread_reused_across_operations(self) -> None:
        """Test the same thread is used for multiple operations."""
        manager = _LoopManager.get()

        async def get_thread_id() -> int:
            return threading.current_thread().ident or 0

        thread_id1 = manager.run_sync(get_thread_id())
        thread_id2 = manager.run_sync(get_thread_id())
        thread_id3 = manager.run_sync(get_thread_id())

        assert thread_id1 == thread_id2 == thread_id3
        assert thread_id1 != threading.current_thread().ident


class TestLoopManagerSessionTracking:
    """Tests for session registration and tracking."""

    def test_register_session(self) -> None:
        """Test register_session() adds session to tracking set."""
        manager = _LoopManager.get()
        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        manager.register_session(mock_session)

        assert mock_session in manager._sessions

    def test_sessions_are_weakly_referenced(self) -> None:
        """Test registered sessions can be garbage collected."""
        manager = _LoopManager.get()
        mock_session = MagicMock()
        weak_ref = weakref.ref(mock_session)

        manager.register_session(mock_session)
        assert mock_session in manager._sessions

        # Delete the session and force GC
        del mock_session
        gc.collect()

        # Session should be garbage collected
        assert weak_ref() is None

    def test_cleanup_all_closes_sessions(self) -> None:
        """Test cleanup_all() closes all registered sessions."""
        manager = _LoopManager.get()

        mock_session1 = MagicMock()
        mock_session1.close = AsyncMock()
        mock_session2 = MagicMock()
        mock_session2.close = AsyncMock()

        manager.register_session(mock_session1)
        manager.register_session(mock_session2)

        manager.cleanup_all()

        mock_session1.close.assert_called_once()
        mock_session2.close.assert_called_once()

    def test_cleanup_all_handles_empty_sessions(self) -> None:
        """Test cleanup_all() handles empty session set gracefully."""
        manager = _LoopManager.get()

        # Should not raise
        manager.cleanup_all()

    def test_cleanup_all_continues_on_error(self) -> None:
        """Test cleanup_all() continues cleaning up even if a session fails."""
        manager = _LoopManager.get()

        mock_session1 = MagicMock()
        mock_session1.close = AsyncMock(side_effect=Exception("cleanup error"))
        mock_session2 = MagicMock()
        mock_session2.close = AsyncMock()

        manager.register_session(mock_session1)
        manager.register_session(mock_session2)

        # Should not raise, but should log the error
        manager.cleanup_all()

        # Both sessions should have close() called
        mock_session1.close.assert_called_once()
        mock_session2.close.assert_called_once()


class TestLoopManagerConcurrency:
    """Tests for concurrent operation handling."""

    def test_concurrent_run_sync_calls(self) -> None:
        """Test multiple run_sync() calls from different threads."""
        manager = _LoopManager.get()
        results: list[int] = []
        errors: list[Exception] = []

        async def compute(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        def worker(x: int) -> None:
            try:
                result = manager.run_sync(compute(x))
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert sorted(results) == [0, 2, 4, 6, 8]

    def test_concurrent_run_async_calls(self) -> None:
        """Test multiple run_async() calls complete correctly."""
        manager = _LoopManager.get()

        async def compute(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3

        futures = [manager.run_async(compute(i)) for i in range(5)]
        results = [f.result(timeout=1.0) for f in futures]

        assert sorted(results) == [0, 3, 6, 9, 12]
