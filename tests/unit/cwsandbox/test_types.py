# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._types module."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock

import pytest

from cwsandbox._types import (
    CidrBlock,
    EgressRule,
    Endpoint,
    EndpointAuth,
    EndpointKind,
    IngressRule,
    NetworkOptions,
    ObjectStorageAccess,
    ObjectStoragePermission,
    OperationRef,
    PortRange,
    Process,
    ProcessResult,
    RegisteredVolumeOptions,
    ScratchVolumeOptions,
    SecurityContext,
    Service,
    ServiceProtocol,
    ServiceVisibility,
    StorageMedium,
    StreamReader,
    StreamWriter,
    TerminalResult,
    TerminalSession,
)
from cwsandbox.exceptions import SandboxExecutionError


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


class TestEgressRule:
    """Tests for create-time DNS-name egress grants."""

    def test_normalizes_dns_name(self) -> None:
        rule = EgressRule(dns_name="  PyPI.org ")
        assert rule.dns_name == "pypi.org"

    def test_wildcard_allowed(self) -> None:
        assert EgressRule(dns_name="*.pypi.org").dns_name == "*.pypi.org"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            EgressRule(dns_name="   ")

    def test_star_ceiling_rejected(self) -> None:
        with pytest.raises(ValueError, match="policy ceiling"):
            EgressRule(dns_name="*")

    @pytest.mark.parametrize(
        "name",
        [
            "*.*.pypi.org",
            "*pypi.org",
            "foo.*.com",
            "pypi.org:443",
            "*.",
            "*example.com",
            "**.example.com",
            "foo.*.example.com",
            "-bad.example.com",
        ],
    )
    def test_invalid_grammar_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="DNS-1123 subdomain"):
            EgressRule(dns_name=name)

    def test_frozen(self) -> None:
        rule = EgressRule(dns_name="pypi.org")
        with pytest.raises(AttributeError):
            rule.dns_name = "example.com"  # type: ignore[misc]

    def test_dns_name_except_rejected(self) -> None:
        with pytest.raises(ValueError, match="policy-only"):
            EgressRule(dns_name="pypi.org", dns_name_except=["files.pypi.org"])

    def test_dns_name_allows_omitted_or_443_ports(self) -> None:
        assert EgressRule(dns_name="pypi.org").ports is None
        assert EgressRule(dns_name="pypi.org", ports=[443]).ports == (PortRange(port=443),)
        tcp = PortRange(port=443, protocol="TCP")
        assert EgressRule(dns_name="pypi.org", ports=[tcp]).ports == (tcp,)

    def test_dns_name_rejects_non_https_ports(self) -> None:
        with pytest.raises(ValueError, match="TCP 443"):
            EgressRule(dns_name="pypi.org", ports=[80])
        with pytest.raises(ValueError, match="TCP 443"):
            EgressRule(dns_name="pypi.org", ports=[443, 80])
        with pytest.raises(ValueError, match="TCP 443"):
            EgressRule(dns_name="pypi.org", ports=[PortRange(port=443, protocol="UDP")])

    def test_from_proto_drops_policy_except_and_invalid_dns_ports(self) -> None:
        from cwsandbox._proto import sandbox_pb2
        from cwsandbox._spec import egress_rule_from_proto

        proto = sandbox_pb2.EgressRule(dns_name="pypi.org")
        proto.dns_name_except.append("files.pypi.org")
        proto.ports.add(port=80)
        rule = egress_rule_from_proto(proto)
        assert rule.dns_name == "pypi.org"
        assert rule.dns_name_except is None
        assert rule.ports is None


class TestNetworkOptions:
    """Tests for v1 NetworkOptions (deny flags and hostname grants)."""

    def test_default_values_all_none(self) -> None:
        opts = NetworkOptions()
        assert opts.deny_egress is None
        assert opts.deny_ingress is None
        assert opts.egress is None

    def test_deny_flags(self) -> None:
        opts = NetworkOptions(deny_egress=True, deny_ingress=False)
        assert opts.deny_egress is True
        assert opts.deny_ingress is False

    def test_frozen_immutability(self) -> None:
        opts = NetworkOptions(deny_egress=True)
        with pytest.raises(AttributeError):
            opts.deny_egress = False  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        opts1 = NetworkOptions(deny_egress=True)
        opts2 = NetworkOptions(deny_egress=True)
        opts3 = NetworkOptions(deny_ingress=True)
        assert opts1 == opts2
        assert opts1 != opts3
        hash(opts1)
        assert "NetworkOptions" in repr(opts1)

    def test_egress_coerces_dicts_and_lists(self) -> None:
        opts = NetworkOptions(egress=[{"dns_name": "pypi.org"}, EgressRule(dns_name="*.pypi.org")])
        assert opts.egress == (
            EgressRule(dns_name="pypi.org"),
            EgressRule(dns_name="*.pypi.org"),
        )

    def test_deny_egress_with_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            NetworkOptions(deny_egress=True, egress=[EgressRule(dns_name="pypi.org")])

    def test_egress_rejects_bare_string(self) -> None:
        with pytest.raises(TypeError, match="sequence"):
            NetworkOptions(egress="pypi.org")  # type: ignore[arg-type]


class TestService:
    """Tests for typed Service ports."""

    def test_service_visibility_coercion(self) -> None:
        svc = Service(port=8080, visibility="public")
        assert svc.visibility == ServiceVisibility.PUBLIC

    def test_invalid_port(self) -> None:
        with pytest.raises(ValueError, match="1-65535"):
            Service(port=0)

    def test_endpoint_required_kind_and_auth(self) -> None:
        with pytest.raises(TypeError):
            Endpoint()  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="auth is required"):
            Endpoint(kind=EndpointKind.HTTPS)
        ep = Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN)
        assert ep.kind == EndpointKind.HTTPS
        assert ep.auth == EndpointAuth.OPEN
        assert ep.request_timeout_seconds is None

    def test_tls_passthrough_omits_auth_and_timeout(self) -> None:
        ep = Endpoint(kind=EndpointKind.TLS_PASSTHROUGH)
        assert ep.kind == EndpointKind.TLS_PASSTHROUGH
        assert ep.auth is None
        assert ep.request_timeout_seconds is None
        coerced = Endpoint(kind="tls_passthrough")
        assert coerced.kind == EndpointKind.TLS_PASSTHROUGH

    def test_tls_passthrough_rejects_auth(self) -> None:
        with pytest.raises(ValueError, match="auth must be unset"):
            Endpoint(kind=EndpointKind.TLS_PASSTHROUGH, auth=EndpointAuth.OPEN)

    def test_tls_passthrough_rejects_timeout(self) -> None:
        with pytest.raises(ValueError, match="request_timeout_seconds must be unset"):
            Endpoint(kind=EndpointKind.TLS_PASSTHROUGH, request_timeout_seconds=15)
        with pytest.raises(ValueError, match="request_timeout_seconds must be unset"):
            Endpoint(kind=EndpointKind.TLS_PASSTHROUGH, request_timeout_seconds=0)

    def test_service_tls_nested_dict_rejects_auth_and_timeout(self) -> None:
        with pytest.raises(ValueError, match="auth must be unset"):
            Service(
                port=8443,
                visibility="public",
                endpoint={"kind": "tls_passthrough", "auth": "open"},
            )
        with pytest.raises(ValueError, match="request_timeout_seconds must be unset"):
            Service(
                port=8443,
                visibility="public",
                endpoint={"kind": "tls_passthrough", "request_timeout_seconds": 15},
            )

    def test_endpoint_string_coercion(self) -> None:
        ep = Endpoint(kind="https", auth="open")
        assert ep.kind == EndpointKind.HTTPS
        assert ep.auth == EndpointAuth.OPEN

    def test_endpoint_unknown_auth_string(self) -> None:
        with pytest.raises(ValueError):
            Endpoint(kind="https", auth="token")

    def test_service_endpoint_nested_dict(self) -> None:
        svc = Service(
            port=8080,
            visibility="public",
            endpoint={"kind": "https", "auth": "open"},
        )
        assert isinstance(svc.endpoint, Endpoint)
        assert svc.endpoint.kind == EndpointKind.HTTPS
        assert svc.endpoint.auth == EndpointAuth.OPEN
        assert svc.endpoint.request_timeout_seconds is None

    def test_service_endpoint_nested_dict_with_timeout(self) -> None:
        svc = Service(
            port=8080,
            visibility="public",
            endpoint={
                "kind": "https",
                "auth": "open",
                "request_timeout_seconds": 120,
            },
        )
        assert isinstance(svc.endpoint, Endpoint)
        assert svc.endpoint.request_timeout_seconds == 120

    def test_endpoint_request_timeout_seconds_allows_int(self) -> None:
        zero = Endpoint(kind="https", auth="open", request_timeout_seconds=0)
        typical = Endpoint(kind="https", auth="open", request_timeout_seconds=120)
        # Range is enforced by the API; the client must not reject a later bump.
        below_current_min = Endpoint(kind="https", auth="open", request_timeout_seconds=14)
        above_current_max = Endpoint(kind="https", auth="open", request_timeout_seconds=901)
        assert zero.request_timeout_seconds == 0
        assert typical.request_timeout_seconds == 120
        assert below_current_min.request_timeout_seconds == 14
        assert above_current_max.request_timeout_seconds == 901

    @pytest.mark.parametrize("bad_value", [15.0, "120", True])
    def test_endpoint_request_timeout_seconds_rejects_non_int(self, bad_value: object) -> None:
        with pytest.raises(TypeError, match="request_timeout_seconds"):
            Endpoint(
                kind="https",
                auth="open",
                request_timeout_seconds=bad_value,  # type: ignore[arg-type]
            )

    def test_service_endpoint_requires_public(self) -> None:
        with pytest.raises(ValueError, match="PUBLIC"):
            Service(
                port=8080,
                endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN),
            )
        with pytest.raises(ValueError, match="PUBLIC"):
            Service(
                port=8080,
                visibility=ServiceVisibility.PRIVATE,
                endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN),
            )

    def test_service_endpoint_rejects_non_tcp_protocol(self) -> None:
        with pytest.raises(ValueError, match="TCP"):
            Service(
                port=8080,
                visibility=ServiceVisibility.PUBLIC,
                protocol=ServiceProtocol.UDP,
                endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN),
            )
        with pytest.raises(ValueError, match="TCP"):
            Service(
                port=8080,
                visibility=ServiceVisibility.PUBLIC,
                protocol="sctp",
                endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN),
            )

    def test_service_endpoint_allows_unset_or_tcp_protocol(self) -> None:
        unset = Service(
            port=8080,
            visibility=ServiceVisibility.PUBLIC,
            endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN),
        )
        tcp = Service(
            port=8080,
            visibility=ServiceVisibility.PUBLIC,
            protocol=ServiceProtocol.TCP,
            endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN),
        )
        assert unset.protocol is None
        assert tcp.protocol == ServiceProtocol.TCP

    def test_service_omitted_endpoint_stays_none(self) -> None:
        svc = Service(port=8080, visibility=ServiceVisibility.PUBLIC)
        assert svc.endpoint is None


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
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        queue.put_nowait("line1")
        queue.put_nowait("line2")
        queue.put_nowait(None)  # Sentinel

        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        lines = list(reader)
        assert lines == ["line1", "line2"]

    def test_stream_reader_stops_on_sentinel(self) -> None:
        """Test StreamReader stops iteration on None sentinel."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        queue.put_nowait("only_line")
        queue.put_nowait(None)  # Sentinel

        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        lines = list(reader)
        assert lines == ["only_line"]

    def test_stream_reader_exhausted_raises_stop_iteration(self) -> None:
        """Test exhausted StreamReader raises StopIteration immediately."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
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
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
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
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        await queue.put("single")
        await queue.put(None)

        mock_manager = MagicMock()
        reader = StreamReader(queue, mock_manager)

        lines = [line async for line in reader]
        assert lines == ["single"]

    @pytest.mark.asyncio
    async def test_stream_reader_async_exhausted_raises_stop(self) -> None:
        """Test exhausted StreamReader raises StopAsyncIteration."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
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

    def test_stream_reader_close_marks_exhausted(self) -> None:
        """Test close() marks the reader as exhausted."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        queue.put_nowait("line1")
        queue.put_nowait(None)

        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        reader.close()
        with pytest.raises(StopIteration):
            next(reader)

    def test_stream_reader_close_calls_cancel(self) -> None:
        """Test close() invokes the cancel callback."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        cancel = MagicMock()
        reader = StreamReader(queue, mock_manager, cancel=cancel)

        reader.close()
        cancel.assert_called_once()

    def test_stream_reader_close_without_cancel(self) -> None:
        """Test close() is safe when no cancel callback is set."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        reader = StreamReader(queue, mock_manager)

        reader.close()  # Should not raise
        assert reader._exhausted

    def test_stream_reader_close_idempotent(self) -> None:
        """Test close() can be called multiple times safely."""
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        mock_manager = self._create_mock_loop_manager()
        cancel = MagicMock()
        reader = StreamReader(queue, mock_manager, cancel=cancel)

        reader.close()
        reader.close()
        cancel.assert_called_once()  # cancel only fires on the first close()


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
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
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


class TestTerminalResult:
    """Tests for TerminalResult dataclass."""

    def test_creation(self) -> None:
        """Test TerminalResult can be created with fields."""
        result = TerminalResult(returncode=0, command=["bash"])
        assert result.returncode == 0
        assert result.command == ["bash"]

    def test_defaults(self) -> None:
        """Test TerminalResult has correct defaults."""
        result = TerminalResult(returncode=1)
        assert result.command == []

    def test_frozen(self) -> None:
        """Test TerminalResult is immutable."""
        result = TerminalResult(returncode=0)
        with pytest.raises(AttributeError):
            result.returncode = 1  # type: ignore[misc]


class TestTerminalSession:
    """Tests for TerminalSession class."""

    def _create_session(
        self,
        *,
        future: Future[TerminalResult] | None = None,
        command: list[str] | None = None,
    ) -> TerminalSession:
        """Create a TerminalSession with sensible defaults for testing."""
        if future is None:
            future = Future()
        if command is None:
            command = ["/bin/bash"]

        output_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        output_queue.put_nowait(None)
        stdin_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        resize_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()

        mock_manager = MagicMock()
        output = StreamReader(output_queue, mock_manager)
        stdin = StreamWriter(stdin_queue, mock_manager)

        return TerminalSession(
            future=future,
            command=command,
            output=output,
            stdin=stdin,
            resize_queue=resize_queue,
        )

    def test_is_operation_ref(self) -> None:
        """TerminalSession is an OperationRef[TerminalResult]."""
        session = self._create_session()
        assert isinstance(session, OperationRef)

    def test_result_returns_terminal_result(self) -> None:
        """Test result() blocks and returns TerminalResult."""
        future: Future[TerminalResult] = Future()
        expected = TerminalResult(returncode=0, command=["/bin/bash"])
        future.set_result(expected)

        session = self._create_session(future=future)
        result = session.result()
        assert result.returncode == 0
        assert result.command == ["/bin/bash"]

    def test_wait_returns_exit_code(self) -> None:
        """Test wait() returns the exit code."""
        future: Future[TerminalResult] = Future()
        future.set_result(TerminalResult(returncode=42))

        session = self._create_session(future=future)
        assert session.wait() == 42

    def test_returncode_none_while_active(self) -> None:
        """Test returncode is None while session is active."""
        session = self._create_session()
        assert session.returncode is None

    def test_returncode_after_completion(self) -> None:
        """Test returncode reflects exit code after completion."""
        future: Future[TerminalResult] = Future()
        future.set_result(TerminalResult(returncode=7))

        session = self._create_session(future=future)
        assert session.returncode == 7

    def test_command_property(self) -> None:
        """Test command property returns the executed command."""
        session = self._create_session(command=["/bin/zsh"])
        assert session.command == ["/bin/zsh"]

    def test_result_raises_exception(self) -> None:
        """Test result() raises exception from the session."""
        future: Future[TerminalResult] = Future()
        future.set_exception(RuntimeError("session failed"))

        session = self._create_session(future=future)
        with pytest.raises(RuntimeError, match="session failed"):
            session.result()

    def test_result_with_timeout(self) -> None:
        """Test result() times out when session is still active."""
        session = self._create_session()
        with pytest.raises(FuturesTimeoutError):
            session.result(timeout=0.01)

    def test_resize_after_exit_raises(self) -> None:
        """Test resize() raises when session has ended."""
        future: Future[TerminalResult] = Future()
        future.set_result(TerminalResult(returncode=0))

        session = self._create_session(future=future)
        with pytest.raises(SandboxExecutionError, match="terminal session has ended"):
            session.resize(80, 24)

    def test_resize_enqueues_dimensions(self) -> None:
        """Test resize() enqueues (width, height) on the resize queue."""
        from unittest.mock import patch

        future: Future[TerminalResult] = Future()
        resize_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()

        output_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        output_queue.put_nowait(None)
        stdin_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        mock_manager = MagicMock()

        session = TerminalSession(
            future=future,
            command=["/bin/bash"],
            output=StreamReader(output_queue, mock_manager),
            stdin=StreamWriter(stdin_queue, mock_manager),
            resize_queue=resize_queue,
        )

        mock_loop = MagicMock()
        with patch("cwsandbox._loop_manager._LoopManager") as mock_lm_cls:
            mock_lm_cls.get.return_value.get_loop.return_value = mock_loop
            session.resize(120, 40)

        mock_loop.call_soon_threadsafe.assert_called_once_with(resize_queue.put_nowait, (120, 40))

    def test_stdin_always_present(self) -> None:
        """Test stdin is always present (not Optional)."""
        session = self._create_session()
        assert session.stdin is not None
        assert isinstance(session.stdin, StreamWriter)

    def test_fetch_result_caches_exception(self) -> None:
        """Test _fetch_result caches exception and re-raises on subsequent calls."""
        future: Future[TerminalResult] = Future()
        future.set_exception(RuntimeError("session crashed"))

        session = self._create_session(future=future)
        with pytest.raises(RuntimeError, match="session crashed"):
            session.result()

        # Second call should re-raise the cached exception without touching the future
        with pytest.raises(RuntimeError, match="session crashed"):
            session.result()

    @pytest.mark.asyncio
    async def test_await_in_async_context(self) -> None:
        """Test TerminalSession is awaitable in async context."""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: TerminalResult(returncode=0, command=["/bin/bash"]))
            output_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
            output_queue.put_nowait(None)
            stdin_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            resize_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
            mock_manager = MagicMock()

            session = TerminalSession(
                future=future,
                command=["/bin/bash"],
                output=StreamReader(output_queue, mock_manager),
                stdin=StreamWriter(stdin_queue, mock_manager),
                resize_queue=resize_queue,
            )

            result = await session
            assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_await_with_exception(self) -> None:
        """Test await raises exception from the session."""

        def raise_error() -> TerminalResult:
            raise ValueError("async session error")

        with ThreadPoolExecutor() as executor:
            future = executor.submit(raise_error)
            output_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
            output_queue.put_nowait(None)
            stdin_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            resize_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
            mock_manager = MagicMock()

            session = TerminalSession(
                future=future,
                command=["/bin/bash"],
                output=StreamReader(output_queue, mock_manager),
                stdin=StreamWriter(stdin_queue, mock_manager),
                resize_queue=resize_queue,
            )

            with pytest.raises(ValueError, match="async session error"):
                await session


class TestSecurityContext:
    def test_validates_and_coerces(self) -> None:
        ctx = SecurityContext(
            run_as_user=1000,
            privileged=True,
            capabilities_add=["SYS_PTRACE"],
        )
        assert ctx.run_as_user == 1000
        assert ctx.privileged is True
        assert ctx.capabilities_add == ("SYS_PTRACE",)

    def test_rejects_negative_uid(self) -> None:
        with pytest.raises(ValueError, match="unsigned"):
            SecurityContext(run_as_user=-1)


class TestRegisteredVolumeOptions:
    def test_validates_sub_path(self) -> None:
        vol = RegisteredVolumeOptions(
            name="data",
            volume_id="vol-1",
            mount_path="/data",
            sub_path="runs/1",
        )
        assert vol.sub_path == "runs/1"

    def test_rejects_absolute_sub_path(self) -> None:
        with pytest.raises(ValueError, match="relative"):
            RegisteredVolumeOptions(
                name="data",
                volume_id="vol-1",
                mount_path="/data",
                sub_path="/abs",
            )


class TestScratchVolumeOptionsExtras:
    def test_medium_and_sub_path(self) -> None:
        vol = ScratchVolumeOptions(
            name="tmp",
            mount_path="/tmp/vol",
            medium="memory",
            sub_path="nested",
            read_only=True,
        )
        assert vol.medium == StorageMedium.MEMORY
        assert vol.sub_path == "nested"
        assert vol.read_only is True


class TestObjectStorageAccess:
    def test_requires_buckets(self) -> None:
        with pytest.raises(ValueError, match="buckets"):
            ObjectStorageAccess()

    def test_coerces_permission(self) -> None:
        access = ObjectStorageAccess(buckets=["a"], permission="read_write")
        assert access.permission == ObjectStoragePermission.READ_WRITE

    def test_rejects_bare_string_buckets(self) -> None:
        with pytest.raises(TypeError, match="bare string"):
            ObjectStorageAccess(buckets="team-data")  # type: ignore[arg-type]

    def test_rejects_empty_or_non_string_bucket_entries(self) -> None:
        with pytest.raises(ValueError, match="entries cannot be empty"):
            ObjectStorageAccess(buckets=[""])
        with pytest.raises(TypeError, match="must be strings"):
            ObjectStorageAccess(buckets=[1])  # type: ignore[list-item]


class TestIngressAndExpandedEgress:
    def test_egress_cidr_and_any(self) -> None:
        rule = EgressRule(cidr="10.0.0.0/8", ports=[443])
        assert rule.cidr == CidrBlock(cidr="10.0.0.0/8")
        assert rule.ports == (PortRange(port=443),)

    def test_egress_requires_one_destination(self) -> None:
        with pytest.raises(ValueError, match="exactly one destination"):
            EgressRule()

    def test_ingress_any(self) -> None:
        rule = IngressRule(any=True)
        assert rule.any is True

    def test_network_ingress_coerces_dicts(self) -> None:
        opts = NetworkOptions(ingress=[{"any": True}])
        assert opts.ingress == (IngressRule(any=True),)
