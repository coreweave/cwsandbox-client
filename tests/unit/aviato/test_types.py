# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Unit tests for aviato._types module."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock

import pytest

from aviato._types import (
    NetworkOptions,
    OperationRef,
    Process,
    ProcessResult,
    Serialization,
    StreamReader,
    StreamWriter,
)
from aviato.exceptions import SandboxExecutionError


class TestOperationRef:
    """Tests for OperationRef generic class."""

    def test_operation_ref_result_returns_result(self) -> None:
        """Test result() blocks and returns the result."""
        future: Future[str] = Future()
        future.set_result("hello")
        ref: OperationRef[str] = OperationRef(future)

        assert ref.result() == "hello"

    def test_operation_ref_result_with_bytes(self) -> None:
        """Test OperationRef[bytes] works for read_file() use case."""
        future: Future[bytes] = Future()
        future.set_result(b"file contents")
        ref: OperationRef[bytes] = OperationRef(future)

        assert ref.result() == b"file contents"

    def test_operation_ref_result_with_none(self) -> None:
        """Test OperationRef[None] works for write_file() use case."""
        future: Future[None] = Future()
        future.set_result(None)
        ref: OperationRef[None] = OperationRef(future)

        assert ref.result() is None

    def test_operation_ref_result_with_timeout(self) -> None:
        """Test result() with timeout raises TimeoutError when not complete."""
        future: Future[str] = Future()
        ref: OperationRef[str] = OperationRef(future)

        with pytest.raises(FuturesTimeoutError):
            ref.result(timeout=0.01)

    def test_operation_ref_result_timeout_success(self) -> None:
        """Test result() with timeout succeeds when result available."""
        future: Future[str] = Future()
        future.set_result("completed")
        ref: OperationRef[str] = OperationRef(future)

        assert ref.result(timeout=1.0) == "completed"

    def test_operation_ref_result_raises_exception(self) -> None:
        """Test result() raises the exception from the operation."""
        future: Future[str] = Future()
        future.set_exception(ValueError("something went wrong"))
        ref: OperationRef[str] = OperationRef(future)

        with pytest.raises(ValueError, match="something went wrong"):
            ref.result()

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

            result = ref.result(timeout=5.0)
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


class TestNetworkOptions:
    """Tests for NetworkOptions dataclass."""

    def test_default_values_all_none(self) -> None:
        """Test NetworkOptions defaults all fields to None."""
        opts = NetworkOptions()
        assert opts.ingress_mode is None
        assert opts.exposed_ports is None
        assert opts.egress_mode is None

    def test_setting_ingress_mode_only(self) -> None:
        """Test setting only ingress_mode."""
        opts = NetworkOptions(ingress_mode="public")
        assert opts.ingress_mode == "public"
        assert opts.exposed_ports is None
        assert opts.egress_mode is None

    def test_setting_egress_mode_only(self) -> None:
        """Test setting only egress_mode."""
        opts = NetworkOptions(egress_mode="internet")
        assert opts.ingress_mode is None
        assert opts.exposed_ports is None
        assert opts.egress_mode == "internet"

    def test_setting_exposed_ports_as_tuple(self) -> None:
        """Test setting exposed_ports as tuple preserves it."""
        opts = NetworkOptions(exposed_ports=(8080, 443))
        assert opts.exposed_ports == (8080, 443)

    def test_exposed_ports_list_normalizes_to_tuple(self) -> None:
        """Test exposed_ports list is normalized to tuple for immutability."""
        opts = NetworkOptions(exposed_ports=[8080, 443])  # type: ignore[arg-type]
        assert opts.exposed_ports == (8080, 443)
        assert isinstance(opts.exposed_ports, tuple)

    def test_empty_exposed_ports_normalizes_to_none(self) -> None:
        """Test empty exposed_ports sequence normalizes to None."""
        opts = NetworkOptions(exposed_ports=())
        assert opts.exposed_ports is None

        opts_list = NetworkOptions(exposed_ports=[])  # type: ignore[arg-type]
        assert opts_list.exposed_ports is None

    def test_frozen_immutability(self) -> None:
        """Test NetworkOptions is frozen (immutable)."""
        opts = NetworkOptions(ingress_mode="public")
        with pytest.raises(AttributeError):
            opts.ingress_mode = "internal"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Test NetworkOptions equality comparison."""
        opts1 = NetworkOptions(ingress_mode="public", exposed_ports=(8080,))
        opts2 = NetworkOptions(ingress_mode="public", exposed_ports=(8080,))
        opts3 = NetworkOptions(ingress_mode="internal", exposed_ports=(8080,))

        assert opts1 == opts2
        assert opts1 != opts3

    def test_all_fields_set(self) -> None:
        """Test setting all fields together."""
        opts = NetworkOptions(
            ingress_mode="public",
            exposed_ports=(8080, 443, 22),
            egress_mode="internet",
        )
        assert opts.ingress_mode == "public"
        assert opts.exposed_ports == (8080, 443, 22)
        assert opts.egress_mode == "internet"

    def test_repr_and_hash(self) -> None:
        """Test NetworkOptions has reasonable repr and is hashable (frozen dataclass)."""
        opts = NetworkOptions(ingress_mode="public")
        # Should be hashable since it's frozen
        hash(opts)
        # Should have a reasonable repr
        assert "NetworkOptions" in repr(opts)
        assert "public" in repr(opts)


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

    def test_stream_reader_sync_exception_propagation(self) -> None:
        """Test StreamReader re-raises exceptions from the queue."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        queue.put_nowait("line1")
        queue.put_nowait(RuntimeError("stream error"))
        queue.put_nowait(None)

        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        # First item succeeds
        assert next(reader) == "line1"

        # Second item raises the exception
        with pytest.raises(RuntimeError, match="stream error"):
            next(reader)

        # Reader is exhausted after exception
        with pytest.raises(StopIteration):
            next(reader)

    @pytest.mark.asyncio
    async def test_stream_reader_async_exception_propagation(self) -> None:
        """Test StreamReader re-raises exceptions in async iteration."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        await queue.put("line1")
        await queue.put(ValueError("async stream error"))
        await queue.put(None)

        mock_manager = MagicMock()
        reader = StreamReader(queue, mock_manager)

        # First item succeeds
        line = await reader.__anext__()
        assert line == "line1"

        # Second item raises the exception
        with pytest.raises(ValueError, match="async stream error"):
            await reader.__anext__()

        # Reader is exhausted after exception
        with pytest.raises(StopAsyncIteration):
            await reader.__anext__()


class TestStreamWriter:
    """Tests for StreamWriter class."""

    def _create_mock_loop_manager(self) -> MagicMock:
        """Create a mock _LoopManager for testing."""
        mock = MagicMock()

        def run_async_impl(coro):
            """Execute coroutine and return a Future with the result."""
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(coro)
                future: Future[None] = Future()
                future.set_result(result)
                return future
            finally:
                loop.close()

        mock.run_async.side_effect = run_async_impl
        return mock

    def test_write_queues_data_correctly(self) -> None:
        """Test write() queues bytes data to the underlying queue."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.write(b"hello world")
        ref.result()

        assert queue.get_nowait() == b"hello world"

    def test_writeline_encodes_and_adds_newline(self) -> None:
        """Test writeline() encodes text and appends newline."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.writeline("hello")
        ref.result()

        assert queue.get_nowait() == b"hello\n"

    def test_writeline_custom_encoding(self) -> None:
        """Test writeline() uses specified encoding."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.writeline("hello", encoding="ascii")
        ref.result()

        assert queue.get_nowait() == b"hello\n"

    def test_close_sets_closed_property(self) -> None:
        """Test close() sets the closed property to True."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        assert writer.closed is False
        ref = writer.close()
        ref.result()
        assert writer.closed is True

    def test_multiple_writes_maintain_fifo_order(self) -> None:
        """Test multiple writes are queued in FIFO order."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        writer.write(b"first").result()
        writer.write(b"second").result()
        writer.write(b"third").result()

        assert queue.get_nowait() == b"first"
        assert queue.get_nowait() == b"second"
        assert queue.get_nowait() == b"third"

    def test_close_is_idempotent(self) -> None:
        """Test close() is idempotent - multiple calls are safe."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        # First close
        ref1 = writer.close()
        ref1.result()
        assert writer.closed is True

        # Second close should succeed without error
        ref2 = writer.close()
        ref2.result()
        assert writer.closed is True

        # Only one sentinel should be in queue
        assert queue.get_nowait() is None
        assert queue.empty()

    def test_write_after_close_raises_exception(self) -> None:
        """Test write() after close() raises SandboxExecutionError."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        writer.close().result()

        with pytest.raises(SandboxExecutionError, match="stream is closed"):
            writer.write(b"data")

    def test_writeline_after_close_raises_exception(self) -> None:
        """Test writeline() after close() raises SandboxExecutionError."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        writer.close().result()

        with pytest.raises(SandboxExecutionError, match="stream is closed"):
            writer.writeline("text")

    def test_close_queues_sentinel_after_pending_writes(self) -> None:
        """Test close() queues EOF sentinel after pending data."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        writer.write(b"data1").result()
        writer.write(b"data2").result()
        writer.close().result()

        # Data should come first, then sentinel
        assert queue.get_nowait() == b"data1"
        assert queue.get_nowait() == b"data2"
        assert queue.get_nowait() is None

    def test_set_exception_causes_write_to_fail(self) -> None:
        """Test write() after set_exception raises SandboxExecutionError."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        original_error = RuntimeError("process exited")
        writer.set_exception(original_error)

        with pytest.raises(SandboxExecutionError, match="stream has failed") as exc_info:
            writer.write(b"data")

        # Verify chained exception
        assert exc_info.value.__cause__ is original_error

    def test_set_exception_causes_writeline_to_fail(self) -> None:
        """Test writeline() after set_exception raises SandboxExecutionError."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        writer.set_exception(RuntimeError("process died"))

        with pytest.raises(SandboxExecutionError, match="stream has failed"):
            writer.writeline("text")

    def test_exception_takes_precedence_over_closed(self) -> None:
        """Test exception is raised even if stream is also closed."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        writer.set_exception(RuntimeError("process error"))
        writer.close().result()

        # Should raise the exception, not the closed error
        with pytest.raises(SandboxExecutionError, match="stream has failed"):
            writer.write(b"data")

    def test_write_returns_operation_ref(self) -> None:
        """Test write() returns OperationRef[None]."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.write(b"data")

        assert isinstance(ref, OperationRef)
        result = ref.result()
        assert result is None

    def test_writeline_returns_operation_ref(self) -> None:
        """Test writeline() returns OperationRef[None]."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.writeline("text")

        assert isinstance(ref, OperationRef)
        result = ref.result()
        assert result is None

    def test_close_returns_operation_ref(self) -> None:
        """Test close() returns OperationRef[None]."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.close()

        assert isinstance(ref, OperationRef)
        result = ref.result()
        assert result is None

    def _create_async_mock_loop_manager(self) -> MagicMock:
        """Create a mock _LoopManager for async test contexts.

        This mock uses asyncio.get_event_loop() instead of creating a new loop,
        which is necessary when tests run inside an existing async context.
        """
        mock = MagicMock()

        def run_async_impl(coro):
            """Execute coroutine using running loop and return Future."""
            loop = asyncio.get_event_loop()
            task = loop.create_task(coro)
            future: Future[None] = Future()

            def on_done(t):
                if t.exception():
                    future.set_exception(t.exception())
                else:
                    future.set_result(t.result())

            task.add_done_callback(on_done)
            return future

        mock.run_async.side_effect = run_async_impl
        return mock

    @pytest.mark.asyncio
    async def test_write_awaitable_in_async_context(self) -> None:
        """Test write() returns awaitable OperationRef in async context."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_async_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.write(b"async data")
        result = await ref

        assert result is None
        assert queue.get_nowait() == b"async data"

    @pytest.mark.asyncio
    async def test_writeline_awaitable_in_async_context(self) -> None:
        """Test writeline() returns awaitable OperationRef in async context."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_async_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.writeline("async text")
        result = await ref

        assert result is None
        assert queue.get_nowait() == b"async text\n"

    @pytest.mark.asyncio
    async def test_close_awaitable_in_async_context(self) -> None:
        """Test close() returns awaitable OperationRef in async context."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_async_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.close()
        result = await ref

        assert result is None
        assert writer.closed is True

    def test_queue_size_constant(self) -> None:
        """Test StreamWriter has expected queue size constant."""
        assert StreamWriter.QUEUE_SIZE == 16

    def test_closed_property_initially_false(self) -> None:
        """Test closed property is False before close() is called."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        assert writer.closed is False

    def test_write_empty_bytes(self) -> None:
        """Test write() handles empty bytes."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.write(b"")
        ref.result()

        assert queue.get_nowait() == b""

    def test_writeline_empty_string(self) -> None:
        """Test writeline() handles empty string (still adds newline)."""
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        writer = StreamWriter(queue, mock_manager)

        ref = writer.writeline("")
        ref.result()

        assert queue.get_nowait() == b"\n"


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
