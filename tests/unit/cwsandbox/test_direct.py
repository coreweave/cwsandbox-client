# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for sandbox-scoped direct data-plane connections."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography import x509
from google.protobuf.timestamp_pb2 import Timestamp

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
