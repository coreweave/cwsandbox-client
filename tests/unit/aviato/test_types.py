"""Unit tests for aviato._types module."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock

import pytest

from aviato._types import OperationRef, Process, ProcessResult, Serialization, StreamReader


class TestOperationRef:
    """Tests for OperationRef generic class."""

    def test_operation_ref_get_returns_result(self) -> None:
        """Test get() blocks and returns the result."""
        future: Future[str] = Future()
        future.set_result("hello")
        ref: OperationRef[str] = OperationRef(future)

        assert ref.get() == "hello"

    def test_operation_ref_get_with_bytes(self) -> None:
        """Test OperationRef[bytes] works for read_file() use case."""
        future: Future[bytes] = Future()
        future.set_result(b"file contents")
        ref: OperationRef[bytes] = OperationRef(future)

        assert ref.get() == b"file contents"

    def test_operation_ref_get_with_none(self) -> None:
        """Test OperationRef[None] works for write_file() use case."""
        future: Future[None] = Future()
        future.set_result(None)
        ref: OperationRef[None] = OperationRef(future)

        assert ref.get() is None

    def test_operation_ref_get_with_timeout(self) -> None:
        """Test get() with timeout raises TimeoutError when not complete."""
        future: Future[str] = Future()
        ref: OperationRef[str] = OperationRef(future)

        with pytest.raises(FuturesTimeoutError):
            ref.get(timeout=0.01)

    def test_operation_ref_get_timeout_success(self) -> None:
        """Test get() with timeout succeeds when result available."""
        future: Future[str] = Future()
        future.set_result("completed")
        ref: OperationRef[str] = OperationRef(future)

        assert ref.get(timeout=1.0) == "completed"

    def test_operation_ref_get_raises_exception(self) -> None:
        """Test get() raises the exception from the operation."""
        future: Future[str] = Future()
        future.set_exception(ValueError("something went wrong"))
        ref: OperationRef[str] = OperationRef(future)

        with pytest.raises(ValueError, match="something went wrong"):
            ref.get()

    @pytest.mark.asyncio
    async def test_operation_ref_await(self) -> None:
        """Test OperationRef is awaitable in async context."""
        # Create a future and set result in a thread pool
        with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: "async result")
            ref: OperationRef[str] = OperationRef(future)

            result = await ref
            assert result == "async result"

    @pytest.mark.asyncio
    async def test_operation_ref_await_with_exception(self) -> None:
        """Test await raises exception from the operation."""

        def raise_error() -> str:
            raise ValueError("async error")

        with ThreadPoolExecutor() as executor:
            future = executor.submit(raise_error)
            ref: OperationRef[str] = OperationRef(future)

            with pytest.raises(ValueError, match="async error"):
                await ref

    def test_operation_ref_with_executor(self) -> None:
        """Test OperationRef works with ThreadPoolExecutor futures."""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: 42)
            ref: OperationRef[int] = OperationRef(future)

            result = ref.get(timeout=5.0)
            assert result == 42


class TestSerialization:
    """Tests for Serialization enum."""

    def test_serialization_is_string_enum(self) -> None:
        """Test Serialization members can be used as strings.

        This is important because the enum values may be used in string contexts
        (e.g., logging, error messages) and should work seamlessly.
        """
        assert isinstance(Serialization.PICKLE, str)
        assert isinstance(Serialization.JSON, str)

    def test_serialization_members_exist(self) -> None:
        """Test expected serialization modes are available.

        Users depend on these members existing - if we remove one,
        their code breaks.
        """
        assert hasattr(Serialization, "PICKLE")
        assert hasattr(Serialization, "JSON")


class TestProcessResult:
    """Tests for ProcessResult dataclass."""

    def test_process_result_creation(self) -> None:
        """Test ProcessResult can be created with required fields."""
        result = ProcessResult(stdout="hello", stderr="", returncode=0)

        assert result.stdout == "hello"
        assert result.stderr == ""
        assert result.returncode == 0

    def test_process_result_with_all_fields(self) -> None:
        """Test ProcessResult stores all fields including bytes and command."""
        result = ProcessResult(
            stdout="hello",
            stderr="error",
            returncode=1,
            stdout_bytes=b"hello",
            stderr_bytes=b"error",
            command=["echo", "hello"],
        )

        assert result.stdout == "hello"
        assert result.stderr == "error"
        assert result.returncode == 1
        assert result.stdout_bytes == b"hello"
        assert result.stderr_bytes == b"error"
        assert result.command == ["echo", "hello"]

    def test_process_result_defaults(self) -> None:
        """Test ProcessResult has correct defaults for optional fields."""
        result = ProcessResult(stdout="out", stderr="err", returncode=0)

        assert result.stdout_bytes == b""
        assert result.stderr_bytes == b""
        assert result.command == []


class TestStreamReader:
    """Tests for StreamReader class."""

    def _create_mock_loop_manager(self) -> MagicMock:
        """Create a mock _LoopManager for testing."""
        mock = MagicMock()

        # Mock run_sync to directly await the coroutine in the test thread
        def run_sync_impl(coro):
            # Get event loop or create one
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock.run_sync.side_effect = run_sync_impl
        return mock

    def test_stream_reader_sync_iteration(self) -> None:
        """Test StreamReader works with sync for loop."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        queue.put_nowait("line1")
        queue.put_nowait("line2")
        queue.put_nowait(None)  # Sentinel

        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        lines = list(reader)
        assert lines == ["line1", "line2"]

    def test_stream_reader_stops_on_sentinel(self) -> None:
        """Test StreamReader stops iteration on None sentinel."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        queue.put_nowait("only_line")
        queue.put_nowait(None)  # Sentinel

        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        lines = list(reader)
        assert lines == ["only_line"]

    def test_stream_reader_exhausted_raises_stop_iteration(self) -> None:
        """Test exhausted StreamReader raises StopIteration immediately."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        queue.put_nowait(None)  # Immediately exhausted

        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        # Exhaust the reader
        list(reader)

        # Further iteration should raise immediately
        with pytest.raises(StopIteration):
            next(reader)

    @pytest.mark.asyncio
    async def test_stream_reader_async_iteration(self) -> None:
        """Test StreamReader works with async for loop."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        await queue.put("async_line1")
        await queue.put("async_line2")
        await queue.put(None)  # Sentinel

        mock_manager = MagicMock()
        reader = StreamReader(queue, mock_manager)

        lines = [line async for line in reader]
        assert lines == ["async_line1", "async_line2"]

    @pytest.mark.asyncio
    async def test_stream_reader_async_stops_on_sentinel(self) -> None:
        """Test StreamReader async iteration stops on None sentinel."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        await queue.put("single")
        await queue.put(None)

        mock_manager = MagicMock()
        reader = StreamReader(queue, mock_manager)

        lines = [line async for line in reader]
        assert lines == ["single"]

    @pytest.mark.asyncio
    async def test_stream_reader_async_exhausted_raises_stop(self) -> None:
        """Test exhausted StreamReader raises StopAsyncIteration."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        await queue.put(None)

        mock_manager = MagicMock()
        reader = StreamReader(queue, mock_manager)

        # Exhaust the reader
        _ = [line async for line in reader]

        # Further iteration should raise immediately
        with pytest.raises(StopAsyncIteration):
            await reader.__anext__()


class TestProcess:
    """Tests for Process class."""

    def _create_mock_stream_reader(self) -> StreamReader:
        """Create a mock StreamReader for testing."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        queue.put_nowait(None)  # Empty stream
        mock_manager = MagicMock()
        return StreamReader(queue, mock_manager)

    def test_process_poll_returns_none_when_running(self) -> None:
        """Test poll() returns None while process is running."""
        future: Future[ProcessResult] = Future()
        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()

        process = Process(future, ["echo", "hello"], stdout, stderr)

        assert process.poll() is None

    def test_process_poll_returns_exit_code_when_done(self) -> None:
        """Test poll() returns exit code when process is complete."""
        future: Future[ProcessResult] = Future()
        result = ProcessResult(stdout="output", stderr="", returncode=42)
        future.set_result(result)

        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["echo"], stdout, stderr)

        assert process.poll() == 42

    def test_process_wait_returns_exit_code(self) -> None:
        """Test wait() blocks and returns exit code."""
        future: Future[ProcessResult] = Future()
        result = ProcessResult(stdout="", stderr="", returncode=0)
        future.set_result(result)

        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["true"], stdout, stderr)

        assert process.wait() == 0

    def test_process_wait_with_timeout(self) -> None:
        """Test wait() times out when not complete."""
        future: Future[ProcessResult] = Future()
        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["sleep"], stdout, stderr)

        with pytest.raises(FuturesTimeoutError):
            process.wait(timeout=0.01)

    def test_process_result_returns_process_result(self) -> None:
        """Test result() blocks and returns ProcessResult."""
        future: Future[ProcessResult] = Future()
        expected = ProcessResult(stdout="hello", stderr="", returncode=0, command=["echo", "hello"])
        future.set_result(expected)

        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["echo", "hello"], stdout, stderr)

        result = process.result()
        assert result.stdout == "hello"
        assert result.returncode == 0

    def test_process_result_raises_stored_exception(self) -> None:
        """Test result() raises exception from the execution."""
        future: Future[ProcessResult] = Future()
        future.set_exception(ValueError("execution failed"))

        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["bad"], stdout, stderr)

        with pytest.raises(ValueError, match="execution failed"):
            process.result()

    def test_process_result_with_timeout(self) -> None:
        """Test result() times out when not complete."""
        future: Future[ProcessResult] = Future()
        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["sleep"], stdout, stderr)

        with pytest.raises(FuturesTimeoutError):
            process.result(timeout=0.01)

    def test_process_returncode_property(self) -> None:
        """Test returncode property reflects completion status."""
        future: Future[ProcessResult] = Future()
        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["cmd"], stdout, stderr)

        # Before completion
        assert process.returncode is None

        # After completion
        future.set_result(ProcessResult(stdout="", stderr="", returncode=5))
        process.poll()  # Trigger result fetch
        assert process.returncode == 5

    def test_process_command_property(self) -> None:
        """Test command property returns the command."""
        future: Future[ProcessResult] = Future()
        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        command = ["python", "-c", "print('hi')"]
        process = Process(future, command, stdout, stderr)

        assert process.command == command

    def test_process_cancel(self) -> None:
        """Test cancel() cancels the underlying future."""
        future: Future[ProcessResult] = Future()
        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["long"], stdout, stderr)

        result = process.cancel()
        assert result is True

    def test_process_cancel_completed_fails(self) -> None:
        """Test cancel() returns False for completed process."""
        future: Future[ProcessResult] = Future()
        future.set_result(ProcessResult(stdout="", stderr="", returncode=0))
        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["done"], stdout, stderr)

        result = process.cancel()
        assert result is False

    @pytest.mark.asyncio
    async def test_process_await(self) -> None:
        """Test Process is awaitable in async context."""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: ProcessResult(stdout="awaited", stderr="", returncode=0)
            )
            stdout = self._create_mock_stream_reader()
            stderr = self._create_mock_stream_reader()
            process = Process(future, ["await"], stdout, stderr)

            result = await process
            assert result.stdout == "awaited"

    @pytest.mark.asyncio
    async def test_process_await_with_exception(self) -> None:
        """Test await raises exception from the process."""

        def raise_error() -> ProcessResult:
            raise ValueError("async process error")

        with ThreadPoolExecutor() as executor:
            future = executor.submit(raise_error)
            stdout = self._create_mock_stream_reader()
            stderr = self._create_mock_stream_reader()
            process = Process(future, ["fail"], stdout, stderr)

            with pytest.raises(ValueError, match="async process error"):
                await process

    def test_process_wait_raises_exception(self) -> None:
        """Test wait() raises stored exception."""
        future: Future[ProcessResult] = Future()
        future.set_exception(RuntimeError("process died"))

        stdout = self._create_mock_stream_reader()
        stderr = self._create_mock_stream_reader()
        process = Process(future, ["crash"], stdout, stderr)

        with pytest.raises(RuntimeError, match="process died"):
            process.wait()
