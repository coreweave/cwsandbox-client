# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from aviato.exceptions import SandboxExecutionError

if TYPE_CHECKING:
    from aviato._loop_manager import _LoopManager

T = TypeVar("T")


class OperationRef(Generic[T]):
    """Generic ref for async operations with lazy result retrieval.

    OperationRef wraps a concurrent.futures.Future and provides a unified interface
    for both synchronous and asynchronous result retrieval. This enables the
    sync/async hybrid API where operations return immediately and results are
    retrieved lazily.

    Type Parameters:
        T: The type of the result this operation will return.

    Examples:
        Synchronous usage:
        ```python
        ref = sandbox.read_file("/path/to/file")  # Returns OperationRef[bytes]
        data = ref.result()  # Block until complete
        ```

        With timeout:
        ```python
        try:
            data = ref.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            print("Operation timed out")
        ```

        Async usage:
        ```python
        data = await ref  # Awaitable in async context
        ```
    """

    def __init__(self, future: concurrent.futures.Future[T]) -> None:
        """Initialize with a concurrent.futures.Future.

        Args:
            future: The underlying future that will contain the result.
        """
        self._future = future

    def result(self, timeout: float | None = None) -> T:
        """Block until the result is ready and return it.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The result of the operation.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires before completion.
            concurrent.futures.CancelledError: If the operation was cancelled.
            Exception: Any exception raised by the operation.
        """
        return self._future.result(timeout)

    def __await__(self) -> Generator[Any, None, T]:
        """Make this ref awaitable for async contexts.

        Bridges the concurrent.futures.Future to asyncio, allowing the ref
        to be awaited in async code.

        Returns:
            Generator that yields the result when complete.

        Example:
            ```python
            async def example():
                ref = sandbox.read_file("/path")
                data = await ref  # Works in async context
            ```
        """
        return asyncio.wrap_future(self._future).__await__()


class Serialization(str, Enum):
    """Serialization modes for sandbox function execution."""

    PICKLE = "pickle"
    JSON = "json"


class ExecOutcome(StrEnum):
    """Outcome classification for exec() calls.

    Taxonomy:
    - COMPLETED_OK: returncode == 0
    - COMPLETED_NONZERO: returncode != 0 (process completed but returned error)
    - FAILURE: SandboxTimeoutError, cancellation, transport failures
    """

    COMPLETED_OK = "completed_ok"
    COMPLETED_NONZERO = "completed_nonzero"
    FAILURE = "failure"


@dataclass(frozen=True)
class NetworkOptions:
    """Network configuration for sandbox ingress/egress."""

    ingress_mode: str | None = None
    exposed_ports: tuple[int, ...] | None = None
    egress_mode: str | None = None

    def __post_init__(self) -> None:
        # Normalize list to tuple for immutability
        if isinstance(self.exposed_ports, list):
            object.__setattr__(self, "exposed_ports", tuple(self.exposed_ports))
        # Normalize empty to None
        if self.exposed_ports is not None and len(self.exposed_ports) == 0:
            object.__setattr__(self, "exposed_ports", None)


@dataclass
class ProcessResult:
    """Result from a completed streaming exec operation.

    Contains both the raw bytes and decoded strings for stdout/stderr,
    along with the exit code and original command.

    Attributes:
        stdout: Decoded stdout as UTF-8 string
        stderr: Decoded stderr as UTF-8 string
        returncode: Exit code from the command (0 = success)
        stdout_bytes: Raw stdout bytes
        stderr_bytes: Raw stderr bytes
        command: The command that was executed

    Examples:
        ```python
        result = process.result()
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
        ```
    """

    stdout: str
    stderr: str
    returncode: int
    stdout_bytes: bytes = field(default=b"")
    stderr_bytes: bytes = field(default=b"")
    command: list[str] = field(default_factory=list)


class StreamReader:
    """Sync and async iterable for streaming output.

    StreamReader wraps an asyncio.Queue and provides both synchronous and
    asynchronous iteration interfaces. This enables streaming output to be
    consumed in both sync and async contexts.

    The stream uses None as a sentinel value to signal end-of-stream.
    Exception instances in the queue are re-raised to the consumer.

    Examples:
        Synchronous iteration:
        ```python
        for line in process.stdout:
            print(line)
        ```

        Asynchronous iteration:
        ```python
        async for line in process.stdout:
            print(line)
        ```
    """

    def __init__(
        self, queue: asyncio.Queue[str | Exception | None], loop_manager: _LoopManager
    ) -> None:
        """Initialize with a queue and loop manager.

        Args:
            queue: The asyncio.Queue to read from. Supports string items,
                None as end-of-stream sentinel, and Exception instances
                which are re-raised to the consumer.
            loop_manager: The _LoopManager for executing async operations.
        """
        self._queue = queue
        self._loop_manager = loop_manager
        self._exhausted = False

    def __iter__(self) -> StreamReader:
        """Return self as iterator for sync iteration."""
        return self

    def __next__(self) -> str:
        """Get next item from stream (blocking).

        Returns:
            The next line from the stream.

        Raises:
            StopIteration: When the stream is exhausted (None sentinel).
            Exception: Re-raised if an Exception instance is in the queue.
        """
        if self._exhausted:
            raise StopIteration
        item = self._loop_manager.run_sync(self._queue.get())
        if item is None:
            self._exhausted = True
            raise StopIteration
        if isinstance(item, Exception):
            self._exhausted = True
            raise item
        return item

    def __aiter__(self) -> StreamReader:
        """Return self as async iterator for async iteration."""
        return self

    async def __anext__(self) -> str:
        """Get next item from stream (async).

        Returns:
            The next line from the stream.

        Raises:
            StopAsyncIteration: When the stream is exhausted (None sentinel).
            Exception: Re-raised if an Exception instance is in the queue.
        """
        if self._exhausted:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is None:
            self._exhausted = True
            raise StopAsyncIteration
        if isinstance(item, Exception):
            self._exhausted = True
            raise item
        return item


class StreamWriter:
    """Sync and async writer for streaming input to a process.

    StreamWriter wraps a bounded asyncio.Queue and provides both synchronous and
    asynchronous write interfaces. This enables streaming input to be sent in
    both sync and async contexts.

    The stream uses None as a sentinel value to signal end-of-stream (EOF).
    The queue is bounded (~16 items for ~1MB with 64KB chunks) to provide
    backpressure.

    Examples:
        Synchronous write:
        ```python
        process.stdin.write(b"data").result()
        process.stdin.writeline("hello").result()
        process.stdin.close().result()
        ```

        Asynchronous write:
        ```python
        await process.stdin.write(b"data")
        await process.stdin.writeline("hello")
        await process.stdin.close()
        ```
    """

    QUEUE_SIZE = 16  # ~1MB with 64KB chunks

    def __init__(self, queue: asyncio.Queue[bytes | None], loop_manager: _LoopManager) -> None:
        """Initialize with a queue and loop manager.

        Args:
            queue: The bounded asyncio.Queue to write to.
            loop_manager: The _LoopManager for executing async operations.
        """
        self._queue = queue
        self._loop_manager = loop_manager
        self._closed = False
        self._exception: BaseException | None = None

    @property
    def closed(self) -> bool:
        """True if close() has been called."""
        return self._closed

    def _check_writable(self) -> None:
        """Check if the stream is writable.

        Raises:
            SandboxExecutionError: If the stream is closed or has failed.
        """
        if self._exception is not None:
            raise SandboxExecutionError(
                "Cannot write to stdin: stream has failed"
            ) from self._exception
        if self._closed:
            raise SandboxExecutionError("Cannot write to stdin: stream is closed")

    def write(self, data: bytes) -> OperationRef[None]:
        """Write raw bytes to the stream.

        Queues the data to be sent to the process stdin. Blocks (via OperationRef.result())
        if the queue is full, providing backpressure.

        Args:
            data: The bytes to write.

        Returns:
            An OperationRef that completes when the data is queued.

        Raises:
            SandboxExecutionError: If the stream is closed or has failed.
        """
        self._check_writable()

        async def _write() -> None:
            await self._queue.put(data)

        future = self._loop_manager.run_async(_write())
        return OperationRef(future)

    def writeline(self, text: str, encoding: str = "utf-8") -> OperationRef[None]:
        """Write a line of text to the stream.

        Encodes the text, appends a newline, and queues it for sending.

        Args:
            text: The text to write.
            encoding: The text encoding to use. Defaults to "utf-8".

        Returns:
            An OperationRef that completes when the data is queued.

        Raises:
            SandboxExecutionError: If the stream is closed or has failed.
        """
        data = (text + "\n").encode(encoding)
        return self.write(data)

    def close(self) -> OperationRef[None]:
        """Close the stream, sending EOF sentinel.

        The EOF sentinel is queued at the end, so pending writes complete first.
        Multiple calls to close() are idempotent and return immediately.

        Returns:
            An OperationRef that completes when EOF is queued.
        """
        if self._closed:
            # Idempotent: return immediately-completed operation
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return OperationRef(future)

        self._closed = True

        async def _close() -> None:
            await self._queue.put(None)

        future = self._loop_manager.run_async(_close())
        return OperationRef(future)

    def set_exception(self, exception: BaseException) -> None:
        """Store an exception to be raised on subsequent writes.

        Called internally when the stream fails (e.g., process exits).

        Args:
            exception: The exception to store.
        """
        self._exception = exception


class Process(OperationRef[ProcessResult]):
    """Handle for a running process with streaming stdout/stderr.

    Process inherits from OperationRef[ProcessResult] and adds streaming
    capabilities and process-specific methods. It wraps an async operation
    that executes a command in a sandbox.

    The process's output streams (stdout, stderr) can be iterated either
    synchronously or asynchronously. The result() method blocks until
    completion and returns the full ProcessResult.

    Attributes:
        stdout: StreamReader for standard output
        stderr: StreamReader for standard error
        stdin: StreamWriter for standard input (None if stdin streaming is disabled)

    Examples:
        Basic execution with result:
        ```python
        process = sandbox.exec(["echo", "hello"])
        result = process.result()
        print(result.stdout)  # hello
        ```

        Streaming output:
        ```python
        process = sandbox.exec(["python", "-c", "print('line1'); print('line2')"])
        for line in process.stdout:
            print(f"Got: {line}")
        ```

        Async streaming:
        ```python
        async for line in process.stdout:
            print(f"Got: {line}")
        ```

        Waiting with timeout:
        ```python
        try:
            exit_code = process.wait(timeout=10.0)
        except concurrent.futures.TimeoutError:
            process.cancel()
        ```
    """

    def __init__(
        self,
        future: concurrent.futures.Future[ProcessResult],
        command: list[str],
        stdout: StreamReader,
        stderr: StreamReader,
        stdin: StreamWriter | None = None,
        stats_callback: Callable[[ProcessResult | None, BaseException | None], None] | None = None,
    ) -> None:
        """Initialize with a future and stream readers.

        Args:
            future: Future that will contain the ProcessResult when complete.
            command: The command being executed.
            stdout: StreamReader for stdout.
            stderr: StreamReader for stderr.
            stdin: StreamWriter for stdin, or None if stdin streaming is disabled.
            stats_callback: Optional callback invoked once when result is available.
                Called with (result, None) on success or (None, exception) on failure.
        """
        super().__init__(future)
        self._command = command
        self._returncode: int | None = None
        self._result: ProcessResult | None = None
        self._exception: BaseException | None = None
        self.stdout = stdout
        self.stderr = stderr
        self.stdin = stdin
        self._stats_callback = stats_callback
        self._stats_recorded = False
        self._stats_lock = threading.Lock()

        # Ensure stats are recorded even if user only streams without calling result()
        if stats_callback is not None:
            future.add_done_callback(self._on_future_done)

    def poll(self) -> int | None:
        """Check if the process has completed without blocking.

        Returns:
            The exit code if the process has completed, None otherwise.
        """
        if self._future.done():
            self._ensure_result()
            return self._returncode
        return None

    def wait(self, timeout: float | None = None) -> int:
        """Block until the process completes.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The process exit code.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires.
            concurrent.futures.CancelledError: If the operation was cancelled.
            Exception: Any exception from the execution.
        """
        self._ensure_result(timeout)
        if self._exception is not None:
            raise self._exception
        assert self._returncode is not None
        return self._returncode

    def result(self, timeout: float | None = None) -> ProcessResult:
        """Block until complete and return the full ProcessResult.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            The ProcessResult containing stdout, stderr, and exit code.

        Raises:
            concurrent.futures.TimeoutError: If timeout expires.
            concurrent.futures.CancelledError: If the operation was cancelled.
            Exception: Any exception from the execution.
        """
        self._ensure_result(timeout)
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result

    @property
    def returncode(self) -> int | None:
        """The process exit code, or None if not yet complete."""
        return self._returncode

    @property
    def command(self) -> list[str]:
        """The command that was executed."""
        return self._command

    def cancel(self) -> bool:
        """Attempt to cancel the process.

        Returns:
            True if successfully cancelled, False otherwise.
        """
        return self._future.cancel()

    def _ensure_result(self, timeout: float | None = None) -> None:
        """Ensure result is available, fetching if necessary.

        Args:
            timeout: Maximum seconds to wait.
        """
        if self._result is None and self._exception is None:
            try:
                self._result = self._future.result(timeout)
                self._returncode = self._result.returncode
                self._record_stats()
            except concurrent.futures.TimeoutError:
                # Do not cache timeouts: allow callers to retry with a longer timeout.
                raise
            except Exception as e:
                self._exception = e
                self._record_stats()

    def _record_stats(self) -> None:
        """Record stats via callback exactly once.

        Thread-safe: uses lock to prevent double-counting when callback
        and result() race on different threads. The callback is invoked
        inside the lock to guarantee that when the main thread's
        _record_stats() returns (seeing _stats_recorded=True), the
        callback has already completed. Without this, a race exists
        where _on_future_done sets the flag but hasn't called the
        callback yet, causing result() to return before metrics update.
        """
        if self._stats_callback is None:
            return

        with self._stats_lock:
            if self._stats_recorded:
                return
            self._stats_recorded = True
            self._stats_callback(self._result, self._exception)

    def _on_future_done(self, future: concurrent.futures.Future[ProcessResult]) -> None:
        """Callback invoked when future completes, ensures stats are recorded.

        This handles the case where users only stream stdout/stderr without
        calling result()/wait()/await.
        """
        if self._stats_recorded:
            return

        try:
            result = future.result()
            self._result = result
            self._returncode = result.returncode
        except Exception as e:
            self._exception = e

        self._record_stats()

    def __await__(self) -> Generator[Any, None, ProcessResult]:
        """Make this process awaitable for async contexts.

        Ensures stats are recorded when awaited in async code, including
        on failure.

        Returns:
            Generator that yields the ProcessResult when complete.
        """

        async def _await_and_record() -> ProcessResult:
            try:
                result = await asyncio.wrap_future(self._future)
                self._result = result
                self._returncode = result.returncode
                self._record_stats()
                return result
            except Exception as e:
                self._exception = e
                self._record_stats()
                raise

        return _await_and_record().__await__()
