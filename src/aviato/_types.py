from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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
            ref = sandbox.read_file("/path/to/file")  # Returns OperationRef[bytes]
            data = ref.result()  # Block until complete

        With timeout:
            try:
                data = ref.result(timeout=5.0)
            except concurrent.futures.TimeoutError:
                print("Operation timed out")

        Async usage:
            data = await ref  # Awaitable in async context
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
            async def example():
                ref = sandbox.read_file("/path")
                data = await ref  # Works in async context
        """
        return asyncio.wrap_future(self._future).__await__()


class Serialization(str, Enum):
    """Serialization modes for sandbox function execution."""

    PICKLE = "pickle"
    JSON = "json"


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
        result = process.result()
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
    """

    stdout: str
    stderr: str
    returncode: int
    stdout_bytes: bytes = field(default=b"")
    stderr_bytes: bytes = field(default=b"")
    command: list[str] = field(default_factory=list)


# Original ExecResult for backward compatibility with _sandbox.py from main.
# TODO: Remove in 03-sandbox-core branch (PR-20) when _sandbox.py is updated to use ProcessResult.
@dataclass
class ExecResult:
    """Result from a completed sandbox exec operation (legacy).

    This class is kept for backward compatibility with _sandbox.py.
    New code should use ProcessResult instead.
    """

    stdout_bytes: bytes
    stderr_bytes: bytes
    returncode: int
    command: list[str] = field(default_factory=list)

    @property
    def stdout(self) -> str:
        """Decode stdout as UTF-8."""
        return self.stdout_bytes.decode("utf-8", errors="replace")

    @property
    def stderr(self) -> str:
        """Decode stderr as UTF-8."""
        return self.stderr_bytes.decode("utf-8", errors="replace")


class StreamReader:
    """Sync and async iterable for streaming output.

    StreamReader wraps an asyncio.Queue and provides both synchronous and
    asynchronous iteration interfaces. This enables streaming output to be
    consumed in both sync and async contexts.

    The stream uses None as a sentinel value to signal end-of-stream.

    Examples:
        Synchronous iteration:
            for line in process.stdout:
                print(line)

        Asynchronous iteration:
            async for line in process.stdout:
                print(line)
    """

    def __init__(self, queue: asyncio.Queue[str | None], loop_manager: _LoopManager) -> None:
        """Initialize with a queue and loop manager.

        Args:
            queue: The asyncio.Queue to read from.
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
            The next string from the stream.

        Raises:
            StopIteration: When the stream is exhausted.
        """
        if self._exhausted:
            raise StopIteration

        line = self._loop_manager.run_sync(self._queue.get())
        if line is None:  # Sentinel for end-of-stream
            self._exhausted = True
            raise StopIteration
        return line

    def __aiter__(self) -> StreamReader:
        """Return self as async iterator for async iteration."""
        return self

    async def __anext__(self) -> str:
        """Get next item from stream (async).

        Returns:
            The next string from the stream.

        Raises:
            StopAsyncIteration: When the stream is exhausted.
        """
        if self._exhausted:
            raise StopAsyncIteration
        line = await self._queue.get()
        if line is None:  # Sentinel for end-of-stream
            self._exhausted = True
            raise StopAsyncIteration
        return line


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

    Examples:
        Basic execution with result:
            process = sandbox.exec(["echo", "hello"])
            result = process.result()
            print(result.stdout)  # hello

        Streaming output:
            process = sandbox.exec(["python", "-c", "print('line1'); print('line2')"])
            for line in process.stdout:
                print(f"Got: {line}")

        Async streaming:
            async for line in process.stdout:
                print(f"Got: {line}")

        Waiting with timeout:
            try:
                exit_code = process.wait(timeout=10.0)
            except concurrent.futures.TimeoutError:
                process.cancel()
    """

    def __init__(
        self,
        future: concurrent.futures.Future[ProcessResult],
        command: list[str],
        stdout: StreamReader,
        stderr: StreamReader,
    ) -> None:
        """Initialize with a future and stream readers.

        Args:
            future: Future that will contain the ProcessResult when complete.
            command: The command being executed.
            stdout: StreamReader for stdout.
            stderr: StreamReader for stderr.
        """
        super().__init__(future)
        self._command = command
        self._returncode: int | None = None
        self._result: ProcessResult | None = None
        self._exception: BaseException | None = None
        self.stdout = stdout
        self.stderr = stderr

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
            except concurrent.futures.TimeoutError:
                # Do not cache timeouts: allow callers to retry with a longer timeout.
                raise
            except Exception as e:
                self._exception = e
