# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Background event loop manager for sync/async hybrid API.

This module provides a singleton _LoopManager that manages a background daemon thread
running an asyncio event loop. This enables synchronous code to execute async operations
without requiring users to manage their own event loop.

The daemon thread approach provides several benefits:
- Works seamlessly in Jupyter notebooks without nest_asyncio
- Independent of any user-managed event loop
- Allows reliable cleanup via atexit and signal handlers
- Simple API: users call .result() to block, await for async
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import weakref
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from cwsandbox._session import Session

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _LoopManager:
    """Singleton manager for background asyncio event loop.

    Provides infrastructure for the sync/async hybrid API by running async operations
    in a dedicated daemon thread while exposing blocking interfaces for sync callers.

    Usage:
        manager = _LoopManager.get()

        # Run coroutine and block until complete
        result = manager.run_sync(some_async_operation())

        # Run coroutine and get Future immediately
        future = manager.run_async(some_async_operation())
        # ... do other work ...
        result = future.result()

    Thread Safety:
        The singleton instance is created lazily and protected by a lock.
        The event loop runs in a dedicated daemon thread named "cwsandbox-event-loop".

    Cleanup:
        The thread is daemon=True, so Python automatically kills it on process exit.
        Sessions are tracked in a WeakSet to allow garbage collection.
    """

    _instance: _LoopManager | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the loop manager.

        Do not call directly - use _LoopManager.get() to obtain the singleton.
        """
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sessions: weakref.WeakSet[Session] = weakref.WeakSet()

    @classmethod
    def get(cls) -> _LoopManager:
        """Get the singleton _LoopManager instance.

        Creates the instance on first call. Thread-safe.

        Returns:
            The singleton _LoopManager instance.
        """
        if cls._instance is None:
            with cls._instance_lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the background event loop, creating it if necessary.

        On first call, creates a new event loop and starts a daemon thread
        to run it. Subsequent calls return the same loop.

        Returns:
            The asyncio event loop running in the background thread.
        """
        if self._loop is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._loop is None:
                    self._loop = asyncio.new_event_loop()
                    self._thread = threading.Thread(
                        target=self._run_loop,
                        name="cwsandbox-event-loop",
                        daemon=True,
                    )
                    self._thread.start()
                    logger.debug("Started background event loop thread")
        return self._loop

    def _run_loop(self) -> None:
        """Run the event loop forever in the background thread.

        This method is the target of the daemon thread. It sets the event loop
        for the thread and runs it until the process exits.
        """
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Execute a coroutine and block until it completes.

        Schedules the coroutine on the background event loop and blocks the
        calling thread until the result is available.

        Args:
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine.

        Raises:
            RuntimeError: If called from the daemon thread (would deadlock).
            Exception: Any exception raised by the coroutine.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError(
                "Cannot call run_sync from daemon thread - this would deadlock. "
                "Use run_async() instead or restructure your code to avoid "
                "blocking from within async context."
            )
        future = asyncio.run_coroutine_threadsafe(coro, self.get_loop())
        return future.result()

    def run_async(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Execute a coroutine and return a Future immediately.

        Schedules the coroutine on the background event loop and returns a
        concurrent.futures.Future that can be used to retrieve the result later.

        Args:
            coro: The coroutine to execute.

        Returns:
            A Future that will contain the result when the coroutine completes.
        """
        return asyncio.run_coroutine_threadsafe(coro, self.get_loop())

    def register_session(self, session: Session) -> None:
        """Register a session for tracking.

        Registered sessions are tracked in a WeakSet, allowing them to be
        garbage collected when no other references exist.

        Args:
            session: The session to register.
        """
        self._sessions.add(session)

    def cleanup_all(self) -> None:
        """Clean up all registered sessions.

        Stops all sandboxes in all registered sessions. Called during
        process shutdown to ensure sandboxes are cleaned up.

        Note: This method blocks until cleanup is complete.
        """
        sessions = list(self._sessions)
        if not sessions:
            return

        async def _cleanup() -> None:
            for session in sessions:
                try:
                    await session.close()
                except Exception:
                    logger.exception("Error closing session during cleanup")

        try:
            self.run_sync(_cleanup())
        except Exception:
            logger.exception("Error during cleanup_all")

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Reset the singleton for testing purposes.

        This method is intended for use in tests only. It stops the background
        loop and clears the singleton instance.
        """
        if cls._instance is not None:
            instance = cls._instance
            if instance._loop is not None:
                instance._loop.call_soon_threadsafe(instance._loop.stop)
                if instance._thread is not None:
                    instance._thread.join(timeout=1.0)
            cls._instance = None
