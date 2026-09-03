# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for sandbox-scoped direct data-plane connections."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from cryptography import x509
from google.protobuf import any_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from google.rpc import error_details_pb2, status_pb2

from cwsandbox._direct import (
    DirectDataPlaneClient,
    DirectDataPlaneUnavailable,
    _CredentialBundle,
    _DirectChannelPool,
)
from cwsandbox._proto import sandbox_pb2
from cwsandbox._sandbox import Sandbox
from cwsandbox._types import DataPlaneMode
from cwsandbox.exceptions import SandboxUnavailableError


class _FakeChannel:
    def __init__(self) -> None:
        self.channel_ready = AsyncMock()
        self.close = AsyncMock()


class _DirectRpcError(grpc.RpcError):
    def __init__(
        self,
        *,
        reason: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._trailing: list[tuple[str, bytes]] = []
        if reason is not None:
            info = error_details_pb2.ErrorInfo(
                reason=reason,
                domain="cwsandbox.com",
                metadata=metadata or {},
            )
            detail = any_pb2.Any()
            detail.Pack(info)
            status = status_pb2.Status(code=14, message="runner unavailable", details=[detail])
            self._trailing.append(("grpc-status-details-bin", status.SerializeToString()))

    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.UNAVAILABLE

    def details(self) -> str:
        return "runner unavailable"

    def trailing_metadata(self) -> list[tuple[str, bytes]]:
        return self._trailing


class _DirectAioRpcError(grpc.aio.AioRpcError):
    def __init__(
        self,
        code: grpc.StatusCode,
        *,
        reason: str | None = None,
        details: str = "transient failure",
    ) -> None:
        self._code = code
        self._details = details
        self._trailing: list[tuple[str, bytes]] = []
        if reason is not None:
            info = error_details_pb2.ErrorInfo(
                reason=reason,
                domain="cwsandbox.com",
            )
            detail = any_pb2.Any()
            detail.Pack(info)
            status = status_pb2.Status(code=code.value[0], message=details)
            status.details.append(detail)
            self._trailing.append(("grpc-status-details-bin", status.SerializeToString()))

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def initial_metadata(self) -> tuple[()]:
        return ()

    def trailing_metadata(self) -> list[tuple[str, bytes]]:
        return self._trailing


class _FakeExecCall:
    def __init__(
        self,
        responses: list[sandbox_pb2.ExecStreamResponse] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._responses = responses or []
        self._error = error
        self._index = 0
        self.cancel = MagicMock()
        self.add_done_callback = MagicMock()

    def __aiter__(self) -> _FakeExecCall:
        return self

    async def __anext__(self) -> sandbox_pb2.ExecStreamResponse:
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        raise StopAsyncIteration


def _connection_response(*permissions: int) -> sandbox_pb2.SandboxConnection:
    expires_at = Timestamp()
    expires_at.FromDatetime(datetime.now(UTC) + timedelta(hours=2))
    return sandbox_pb2.SandboxConnection(
        endpoint_uri="https://runner.example.test:9443",
        endpoint_id="runner-1",
        client_certificate_chain_pem=b"client certificate",
        server_ca_bundle_pem=b"runner CA",
        expires_at=expires_at,
        transport=sandbox_pb2.SANDBOX_DATA_TRANSPORT_DIRECT_MTLS,
        protocol=sandbox_pb2.SANDBOX_DATA_PROTOCOL_CONNECT_H2_V1,
        granted_permissions=permissions,
    )


def _bundle(index: int) -> _CredentialBundle:
    return _CredentialBundle(
        cache_key=f"sandbox-{index}",
        target=f"runner-{index}.example.test:9443",
        channel_credentials=MagicMock(),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        granted_permissions=frozenset({sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC}),
    )


@pytest.mark.asyncio
async def test_idle_channel_pool_is_bounded_for_one_thousand_sandboxes() -> None:
    """Sequential use of 1,000 sandboxes retains only the configured idle cap."""
    pool = _DirectChannelPool(max_idle_channels=64)
    channels: list[_FakeChannel] = []

    def make_channel(*_args: object, **_kwargs: object) -> _FakeChannel:
        channel = _FakeChannel()
        channels.append(channel)
        return channel

    with (
        patch("cwsandbox._direct.create_channel", side_effect=make_channel),
        patch("cwsandbox._direct.sandbox_data_plane_pb2_grpc.SandboxDataPlaneServiceStub"),
    ):
        for index in range(1_000):
            lease = await pool.acquire(_bundle(index), timeout=1)
            await lease.release()

    assert len(pool._entries) == 64
    assert sum(channel.close.await_count for channel in channels) == 936


@pytest.mark.asyncio
async def test_pool_does_not_evict_active_streams() -> None:
    pool = _DirectChannelPool(max_idle_channels=1)
    channels: list[_FakeChannel] = []

    def make_channel(*_args: object, **_kwargs: object) -> _FakeChannel:
        channel = _FakeChannel()
        channels.append(channel)
        return channel

    with (
        patch("cwsandbox._direct.create_channel", side_effect=make_channel),
        patch("cwsandbox._direct.sandbox_data_plane_pb2_grpc.SandboxDataPlaneServiceStub"),
    ):
        active = await pool.acquire(_bundle(0), timeout=1)
        for index in (1, 2):
            idle = await pool.acquire(_bundle(index), timeout=1)
            await idle.release()

        assert "sandbox-0" in pool._entries
        assert channels[0].close.await_count == 0
        await active.release()


@pytest.mark.asyncio
async def test_discarded_active_channel_is_not_reused() -> None:
    pool = _DirectChannelPool()
    channels: list[_FakeChannel] = []

    def make_channel(*_args: object, **_kwargs: object) -> _FakeChannel:
        channel = _FakeChannel()
        channels.append(channel)
        return channel

    with (
        patch("cwsandbox._direct.create_channel", side_effect=make_channel),
        patch("cwsandbox._direct.sandbox_data_plane_pb2_grpc.SandboxDataPlaneServiceStub"),
    ):
        stale = await pool.acquire(_bundle(0), timeout=1)
        await stale.discard()
        replacement = await pool.acquire(_bundle(0), timeout=1)

        assert len(channels) == 2
        assert channels[0].close.await_count == 0
        await replacement.release()
        await stale.release()
        channels[0].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_acquires_wait_for_shared_channel_readiness() -> None:
    pool = _DirectChannelPool()
    channel = _FakeChannel()
    ready = asyncio.Event()

    async def wait_until_ready() -> None:
        await ready.wait()

    channel.channel_ready.side_effect = wait_until_ready
    with (
        patch("cwsandbox._direct.create_channel", return_value=channel),
        patch("cwsandbox._direct.sandbox_data_plane_pb2_grpc.SandboxDataPlaneServiceStub"),
    ):
        first = asyncio.create_task(pool.acquire(_bundle(0), timeout=1))
        await asyncio.sleep(0)
        second = asyncio.create_task(pool.acquire(_bundle(0), timeout=1))
        await asyncio.sleep(0)

        assert not first.done()
        assert not second.done()
        ready.set()
        first_lease, second_lease = await asyncio.gather(first, second)
        await first_lease.release()
        await second_lease.release()


@pytest.mark.asyncio
async def test_connect_sends_signed_csr_and_caches_credentials() -> None:
    response = _connection_response(sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC)
    control_stub = MagicMock()
    control_stub.ConnectSandbox = AsyncMock(return_value=response)
    lease = MagicMock()
    client = DirectDataPlaneClient()

    with (
        patch("cwsandbox._direct.grpc.ssl_channel_credentials", return_value=MagicMock()),
        patch("cwsandbox._direct._CHANNEL_POOL.acquire", AsyncMock(return_value=lease)),
    ):
        first = await client.acquire(
            control_stub=control_stub,
            sandbox_id="sandbox-1",
            auth_metadata=(("authorization", "Bearer redacted"),),
            permission=sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC,
            request_timeout=5,
        )
        second = await client.acquire(
            control_stub=control_stub,
            sandbox_id="sandbox-1",
            auth_metadata=(("authorization", "Bearer redacted"),),
            permission=sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC,
            request_timeout=5,
        )

    assert first is lease
    assert second is lease
    control_stub.ConnectSandbox.assert_awaited_once()
    request = control_stub.ConnectSandbox.await_args.args[0]
    assert request.sandbox_id == "sandbox-1"
    assert x509.load_der_x509_csr(request.csr_der).is_signature_valid
    assert list(request.requested_permissions) == [sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC]


@pytest.mark.asyncio
async def test_credentials_are_scoped_and_cached_by_permission() -> None:
    control_stub = MagicMock()
    control_stub.ConnectSandbox = AsyncMock(
        side_effect=lambda request, **_kwargs: _connection_response(
            request.requested_permissions[0]
        )
    )
    client = DirectDataPlaneClient()

    with (
        patch("cwsandbox._direct.grpc.ssl_channel_credentials", return_value=MagicMock()),
        patch("cwsandbox._direct._CHANNEL_POOL.acquire", AsyncMock(return_value=MagicMock())),
    ):
        for permission in (
            sandbox_pb2.SANDBOX_DATA_PERMISSION_READ_FILE,
            sandbox_pb2.SANDBOX_DATA_PERMISSION_WRITE_FILE,
            sandbox_pb2.SANDBOX_DATA_PERMISSION_READ_FILE,
        ):
            await client.acquire(
                control_stub=control_stub,
                sandbox_id="sandbox-1",
                auth_metadata=(),
                permission=permission,
                request_timeout=5,
            )

    assert control_stub.ConnectSandbox.await_count == 2
    assert [
        list(call.args[0].requested_permissions)
        for call in control_stub.ConnectSandbox.await_args_list
    ] == [
        [sandbox_pb2.SANDBOX_DATA_PERMISSION_READ_FILE],
        [sandbox_pb2.SANDBOX_DATA_PERMISSION_WRITE_FILE],
    ]


@pytest.mark.asyncio
async def test_channel_failure_defers_cached_credential_retry() -> None:
    control_stub = MagicMock()
    control_stub.ConnectSandbox = AsyncMock(
        return_value=_connection_response(sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC)
    )
    acquire = AsyncMock(side_effect=TimeoutError)
    client = DirectDataPlaneClient()

    with (
        patch("cwsandbox._direct.grpc.ssl_channel_credentials", return_value=MagicMock()),
        patch("cwsandbox._direct._CHANNEL_POOL.acquire", acquire),
    ):
        for _ in range(2):
            with pytest.raises(DirectDataPlaneUnavailable):
                await client.acquire(
                    control_stub=control_stub,
                    sandbox_id="sandbox-1",
                    auth_metadata=(),
                    permission=sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC,
                    request_timeout=5,
                )

    control_stub.ConnectSandbox.assert_awaited_once()
    acquire.assert_awaited_once()
    assert 0 < acquire.await_args.kwargs["timeout"] <= 1


@pytest.mark.asyncio
async def test_strict_mode_keeps_longer_connect_timeout() -> None:
    control_stub = MagicMock()
    control_stub.ConnectSandbox = AsyncMock(
        return_value=_connection_response(sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC)
    )
    acquire = AsyncMock(return_value=MagicMock())
    client = DirectDataPlaneClient()

    with (
        patch("cwsandbox._direct.grpc.ssl_channel_credentials", return_value=MagicMock()),
        patch("cwsandbox._direct._CHANNEL_POOL.acquire", acquire),
    ):
        await client.acquire(
            control_stub=control_stub,
            sandbox_id="sandbox-1",
            auth_metadata=(),
            permission=sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC,
            request_timeout=300,
            strict=True,
        )

    assert acquire.await_args.kwargs["timeout"] == 10


@pytest.mark.asyncio
async def test_non_awaitable_connect_rpc_is_treated_as_unavailable() -> None:
    control_stub = MagicMock()
    client = DirectDataPlaneClient()

    with pytest.raises(DirectDataPlaneUnavailable, match="does not support"):
        await client.acquire(
            control_stub=control_stub,
            sandbox_id="sandbox-1",
            auth_metadata=(),
            permission=sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC,
            request_timeout=5,
        )


def _running_sandbox(mode: DataPlaneMode) -> Sandbox:
    sandbox = Sandbox(data_plane_mode=mode)
    sandbox._sandbox_id = "sandbox-1"
    sandbox._stub = MagicMock()
    sandbox._auth_metadata = (("authorization", "Bearer gateway-only"),)
    return sandbox


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_gateway_without_forwarding_direct_credentials() -> None:
    sandbox = _running_sandbox(DataPlaneMode.AUTO)
    sandbox._direct_data_plane.acquire = AsyncMock(
        side_effect=DirectDataPlaneUnavailable("not reachable")
    )
    gateway_stub = MagicMock()

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        patch.object(sandbox, "_get_or_create_streaming_channel", AsyncMock()),
        patch("cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub", return_value=gateway_stub),
    ):
        prepared = await sandbox._prepare_streaming_call()

    assert prepared.stub is gateway_stub
    assert prepared.metadata == (("authorization", "Bearer gateway-only"),)


@pytest.mark.asyncio
async def test_direct_call_uses_mtls_without_bearer_metadata() -> None:
    sandbox = _running_sandbox(DataPlaneMode.AUTO)
    direct_stub = MagicMock()
    lease = MagicMock(stub=direct_stub)
    lease.release = AsyncMock()
    sandbox._direct_data_plane.acquire = AsyncMock(return_value=lease)

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
    ):
        prepared = await sandbox._prepare_streaming_call()

    assert prepared.stub is direct_stub
    assert prepared.metadata == ()
    await prepared.release()
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_direct_mode_does_not_fall_back() -> None:
    sandbox = _running_sandbox(DataPlaneMode.DIRECT)
    sandbox._direct_data_plane.acquire = AsyncMock(
        side_effect=DirectDataPlaneUnavailable("not reachable")
    )

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        pytest.raises(SandboxUnavailableError, match="Direct data-plane access is unavailable"),
    ):
        await sandbox._prepare_streaming_call()


@pytest.mark.asyncio
async def test_retiring_direct_unary_discards_channel_and_retries_once() -> None:
    sandbox = _running_sandbox(DataPlaneMode.DIRECT)
    first_stub = MagicMock()
    first_stub.ReadFile = AsyncMock(
        side_effect=_DirectRpcError(reason="CWSANDBOX_RUNNER_SHARD_RETIRING")
    )
    second_stub = MagicMock()
    second_stub.ReadFile = AsyncMock(return_value=sandbox_pb2.ReadFileResponse(content=b"ok"))
    first_lease = MagicMock(stub=first_stub)
    first_lease.release = AsyncMock()
    first_lease.discard = AsyncMock()
    second_lease = MagicMock(stub=second_stub)
    second_lease.release = AsyncMock()
    second_lease.discard = AsyncMock()
    sandbox._direct_data_plane.acquire = AsyncMock(side_effect=[first_lease, second_lease])

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
    ):
        content = await sandbox._read_file_unary_async("/tmp/file", 5)

    assert content == b"ok"
    assert sandbox._direct_data_plane.acquire.await_count == 2
    first_lease.discard.assert_awaited_once()
    first_lease.release.assert_awaited_once_with(discard=True)
    second_lease.release.assert_awaited_once_with(discard=False)


@pytest.mark.parametrize(
    ("code", "reason", "details"),
    [
        pytest.param(
            grpc.StatusCode.UNAVAILABLE,
            "CWSANDBOX_RUNNER_SHARD_RETIRING",
            "Runner shard is retiring; retry the operation on the current endpoint",
            id="runner-shard-retiring-action-33802245090",
        ),
        pytest.param(
            grpc.StatusCode.UNAVAILABLE,
            None,
            "connection lost",
            id="bare-unavailable",
        ),
        pytest.param(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            None,
            "runner overloaded",
            id="resource-exhausted",
        ),
        pytest.param(
            grpc.StatusCode.INTERNAL,
            "CWSANDBOX_BACKEND_UNAVAILABLE",
            "backend unavailable",
            id="structured-backend-unavailable",
        ),
    ],
)
@pytest.mark.asyncio
async def test_direct_exec_retries_transient_error_before_first_response(
    code: grpc.StatusCode,
    reason: str | None,
    details: str,
) -> None:
    sandbox = _running_sandbox(DataPlaneMode.DIRECT)
    first_stub = MagicMock()
    first_stub.StreamExec = MagicMock(
        return_value=_FakeExecCall(error=_DirectAioRpcError(code, reason=reason, details=details))
    )
    second_stub = MagicMock()
    second_stub.StreamExec = MagicMock(
        return_value=_FakeExecCall(
            [
                sandbox_pb2.ExecStreamResponse(
                    output=sandbox_pb2.ExecStreamOutput(
                        stream=sandbox_pb2.ExecStreamOutput.STREAM_STDOUT,
                        data=b"ok\n",
                    )
                ),
                sandbox_pb2.ExecStreamResponse(exit=sandbox_pb2.ExecStreamExit(exit_code=0)),
            ]
        )
    )
    first_lease = MagicMock(stub=first_stub)
    first_lease.release = AsyncMock()
    first_lease.discard = AsyncMock()
    second_lease = MagicMock(stub=second_stub)
    second_lease.release = AsyncMock()
    second_lease.discard = AsyncMock()
    sandbox._direct_data_plane.acquire = AsyncMock(side_effect=[first_lease, second_lease])
    stdout_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
    stderr_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
    ):
        result = await sandbox._exec_streaming_async(["echo", "ok"], stdout_queue, stderr_queue)

    assert result.stdout == "ok\n"
    assert sandbox._direct_data_plane.acquire.await_count == 2
    first_lease.discard.assert_awaited_once()
    first_lease.release.assert_awaited_once_with(discard=True)
    assert await stdout_queue.get() == "ok\n"
    assert await stdout_queue.get() is None
    assert await stderr_queue.get() is None


@pytest.mark.asyncio
async def test_direct_exec_retries_transient_error_only_once() -> None:
    sandbox = _running_sandbox(DataPlaneMode.DIRECT)
    leases = []
    for _ in range(2):
        stub = MagicMock()
        stub.StreamExec = MagicMock(
            return_value=_FakeExecCall(error=_DirectAioRpcError(grpc.StatusCode.UNAVAILABLE))
        )
        lease = MagicMock(stub=stub)
        lease.release = AsyncMock()
        lease.discard = AsyncMock()
        leases.append(lease)
    sandbox._direct_data_plane.acquire = AsyncMock(side_effect=leases)
    stdout_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
    stderr_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        pytest.raises(SandboxUnavailableError),
    ):
        await sandbox._exec_streaming_async(["echo", "ok"], stdout_queue, stderr_queue)

    assert sandbox._direct_data_plane.acquire.await_count == 2
    leases[0].discard.assert_awaited_once()
    leases[0].release.assert_awaited_once_with(discard=True)
    assert stdout_queue.qsize() == 1
    assert await stdout_queue.get() is None
    assert await stderr_queue.get() is None


@pytest.mark.asyncio
async def test_direct_exec_retry_connection_failure_closes_streams() -> None:
    sandbox = _running_sandbox(DataPlaneMode.DIRECT)
    stub = MagicMock()
    stub.StreamExec = MagicMock(
        return_value=_FakeExecCall(error=_DirectAioRpcError(grpc.StatusCode.UNAVAILABLE))
    )
    lease = MagicMock(stub=stub)
    lease.release = AsyncMock()
    lease.discard = AsyncMock()
    sandbox._direct_data_plane.acquire = AsyncMock(
        side_effect=[lease, DirectDataPlaneUnavailable("replacement unavailable")]
    )
    stdout_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
    stderr_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        pytest.raises(SandboxUnavailableError, match="replacement unavailable"),
    ):
        await sandbox._exec_streaming_async(["echo", "ok"], stdout_queue, stderr_queue)

    assert await stdout_queue.get() is None
    assert await stderr_queue.get() is None


@pytest.mark.asyncio
async def test_direct_exec_does_not_retry_after_first_response() -> None:
    sandbox = _running_sandbox(DataPlaneMode.DIRECT)
    stub = MagicMock()
    stub.StreamExec = MagicMock(
        return_value=_FakeExecCall(
            [sandbox_pb2.ExecStreamResponse(ready=sandbox_pb2.ExecStreamReady())],
            error=_DirectAioRpcError(grpc.StatusCode.UNAVAILABLE),
        )
    )
    lease = MagicMock(stub=stub)
    lease.release = AsyncMock()
    lease.discard = AsyncMock()
    sandbox._direct_data_plane.acquire = AsyncMock(return_value=lease)
    stdout_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
    stderr_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        pytest.raises(SandboxUnavailableError),
    ):
        await sandbox._exec_streaming_async(["echo", "ok"], stdout_queue, stderr_queue)

    sandbox._direct_data_plane.acquire.assert_awaited_once()


@pytest.mark.asyncio
async def test_ambiguous_direct_unavailable_is_not_replayed() -> None:
    sandbox = _running_sandbox(DataPlaneMode.DIRECT)
    stub = MagicMock()
    stub.WriteFile = AsyncMock(side_effect=_DirectRpcError())
    lease = MagicMock(stub=stub)
    lease.release = AsyncMock()
    lease.discard = AsyncMock()
    sandbox._direct_data_plane.acquire = AsyncMock(return_value=lease)

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        pytest.raises(SandboxUnavailableError),
    ):
        await sandbox._write_file_unary_async("/tmp/file", b"payload", 5)

    sandbox._direct_data_plane.acquire.assert_awaited_once()
    lease.discard.assert_awaited_once()
    lease.release.assert_awaited_once_with(discard=True)


def test_log_wrong_shard_detection_requires_structured_owner_metadata() -> None:
    from cwsandbox._sandbox import _is_log_session_wrong_shard_error

    wrong_shard = _DirectRpcError(
        reason="CWSANDBOX_BACKEND_UNAVAILABLE",
        metadata={"local_shard_id": "shard-2", "owner_shard_id": "shard-1"},
    )
    generic = _DirectRpcError(reason="CWSANDBOX_BACKEND_UNAVAILABLE")

    assert _is_log_session_wrong_shard_error(wrong_shard)
    assert not _is_log_session_wrong_shard_error(generic)


@pytest.mark.asyncio
async def test_gateway_mode_never_requests_direct_credentials() -> None:
    sandbox = _running_sandbox(DataPlaneMode.GATEWAY)
    sandbox._direct_data_plane.acquire = AsyncMock()

    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        patch.object(sandbox, "_get_or_create_streaming_channel", AsyncMock()),
        patch("cwsandbox._sandbox.sandbox_pb2_grpc.SandboxServiceStub"),
    ):
        await sandbox._prepare_streaming_call()

    sandbox._direct_data_plane.acquire.assert_not_awaited()


class _FakeLogCall:
    def __init__(self, frames: list[object]) -> None:
        self._frames = frames
        self._index = 0
        self.cancel = MagicMock()
        self.add_done_callback = MagicMock()

    def __aiter__(self) -> _FakeLogCall:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._frames):
            raise StopAsyncIteration
        frame = self._frames[self._index]
        self._index += 1
        return frame


@pytest.mark.asyncio
async def test_stream_logs_resume_cancels_and_releases_direct_lease() -> None:
    """Follow resume must tear down the prior direct call before reconnecting."""
    sandbox = _running_sandbox(DataPlaneMode.AUTO)
    interrupt = sandbox_pb2.LogEntry(
        error=sandbox_pb2.LogStreamError(code="STREAM_INTERRUPTED", message="restart")
    )
    complete = sandbox_pb2.LogEntry(data=b"ok\n", log_session_id="s1", next_log_offset=3)
    calls = [_FakeLogCall([interrupt]), _FakeLogCall([complete])]
    stub = MagicMock()
    stub.StreamLogs = MagicMock(side_effect=calls)
    lease = MagicMock(stub=stub)
    lease.release = AsyncMock()
    sandbox._direct_data_plane.acquire = AsyncMock(return_value=lease)

    output_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
    with (
        patch.object(sandbox, "_ensure_started_async", AsyncMock()),
        patch.object(sandbox, "_wait_until_running_async", AsyncMock()),
        patch.object(sandbox, "_ensure_client", AsyncMock()),
        patch("cwsandbox._sandbox.asyncio.sleep", AsyncMock()),
    ):
        await sandbox._stream_logs_async(output_queue, follow=True)

    assert calls[0].cancel.call_count == 1
    assert calls[1].cancel.call_count == 1
    assert lease.release.await_count == 2
    assert sandbox._direct_data_plane.acquire.await_count == 2
