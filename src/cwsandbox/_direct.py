# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Sandbox-scoped direct data-plane credentials and channel leasing."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from cwsandbox._network import create_channel, parse_grpc_target
from cwsandbox._proto import sandbox_data_plane_pb2_grpc, sandbox_pb2

_DIRECT_AUTO_TIMEOUT_SECONDS = 1.0
_DIRECT_CONNECT_TIMEOUT_SECONDS = 10.0
_DIRECT_CREDENTIAL_RPC_TIMEOUT_SECONDS = 5.0
_DIRECT_RETRY_COOLDOWN_SECONDS = 30.0
_DIRECT_EXPIRY_SKEW = timedelta(seconds=30)
_MAX_IDLE_DIRECT_CHANNELS = 64

_FALLBACK_CONNECT_CODES = {
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.FAILED_PRECONDITION,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.UNIMPLEMENTED,
}
_DIRECT_READINESS_RETRY_CODES = {grpc.StatusCode.UNAVAILABLE}


class DirectDataPlaneUnavailable(Exception):
    """The direct path is unavailable and AUTO mode may use the gateway."""


class DirectDataPlanePermissionUnavailable(Exception):
    """The issued certificate did not grant a requested operation."""


@dataclass(frozen=True)
class _CredentialBundle:
    cache_key: str
    target: str
    channel_credentials: grpc.ChannelCredentials
    expires_at: datetime
    granted_permissions: frozenset[int]


@dataclass
class _PoolEntry:
    channel: grpc.aio.Channel
    readiness: asyncio.Future[None]
    active_leases: int = 0
    discard_when_idle: bool = False
    closed: bool = False


class _DirectChannelLease:
    def __init__(
        self,
        pool: _DirectChannelPool,
        cache_key: str,
        entry: _PoolEntry,
    ) -> None:
        self._pool = pool
        self._cache_key = cache_key
        self._entry = entry
        self._released = False
        self.stub = sandbox_data_plane_pb2_grpc.SandboxDataPlaneServiceStub(entry.channel)  # type: ignore[no-untyped-call]

    async def release(self, *, discard: bool = False) -> None:
        if self._released:
            return
        self._released = True
        await self._pool.release(self._cache_key, self._entry, discard=discard)

    async def discard(self) -> None:
        """Remove this channel from the reusable pool, even after release."""

        await self._pool.discard(self._cache_key, self._entry)


class _DirectChannelPool:
    """Process-wide bounded cache of idle, sandbox-scoped mTLS channels."""

    def __init__(self, max_idle_channels: int = _MAX_IDLE_DIRECT_CHANNELS) -> None:
        self._max_idle_channels = max_idle_channels
        self._entries: OrderedDict[str, _PoolEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        bundle: _CredentialBundle,
        *,
        timeout: float,
    ) -> _DirectChannelLease:
        async with self._lock:
            entry = self._entries.get(bundle.cache_key)
            if entry is None:
                channel = create_channel(
                    bundle.target,
                    True,
                    credentials=bundle.channel_credentials,
                )
                entry = _PoolEntry(
                    channel=channel,
                    readiness=asyncio.ensure_future(channel.channel_ready()),
                )
                entry.readiness.add_done_callback(_consume_future_exception)
                self._entries[bundle.cache_key] = entry
            entry.active_leases += 1
            self._entries.move_to_end(bundle.cache_key)

        lease = _DirectChannelLease(self, bundle.cache_key, entry)
        try:
            await asyncio.wait_for(asyncio.shield(entry.readiness), timeout=timeout)
        except BaseException:
            await lease.release(discard=True)
            raise
        return lease

    async def release(
        self,
        cache_key: str,
        entry: _PoolEntry,
        *,
        discard: bool = False,
    ) -> None:
        to_close: list[grpc.aio.Channel] = []
        async with self._lock:
            entry.active_leases = max(0, entry.active_leases - 1)
            entry.discard_when_idle = entry.discard_when_idle or discard
            if discard and self._entries.get(cache_key) is entry:
                self._entries.pop(cache_key, None)
            elif self._entries.get(cache_key) is entry:
                self._entries.move_to_end(cache_key)
            if entry.active_leases == 0 and entry.discard_when_idle:
                if self._entries.get(cache_key) is entry:
                    self._entries.pop(cache_key, None)
                if not entry.closed:
                    entry.closed = True
                    to_close.append(entry.channel)
            to_close.extend(self._evict_idle_locked())
        await _close_channels(to_close)

    async def discard(self, cache_key: str, entry: _PoolEntry | None = None) -> None:
        to_close: list[grpc.aio.Channel] = []
        async with self._lock:
            if entry is None:
                entry = self._entries.get(cache_key)
            if entry is None:
                return
            entry.discard_when_idle = True
            if self._entries.get(cache_key) is entry:
                self._entries.pop(cache_key, None)
            if entry.active_leases == 0 and not entry.closed:
                entry.closed = True
                to_close.append(entry.channel)
        await _close_channels(to_close)

    def _evict_idle_locked(self) -> list[grpc.aio.Channel]:
        idle_count = sum(entry.active_leases == 0 for entry in self._entries.values())
        if idle_count <= self._max_idle_channels:
            return []

        evicted: list[grpc.aio.Channel] = []
        for cache_key, entry in list(self._entries.items()):
            if idle_count <= self._max_idle_channels:
                break
            if entry.active_leases:
                continue
            self._entries.pop(cache_key, None)
            entry.closed = True
            evicted.append(entry.channel)
            idle_count -= 1
        return evicted


async def _close_channels(channels: list[grpc.aio.Channel]) -> None:
    for channel in channels:
        await channel.close(grace=None)


def _consume_future_exception(future: asyncio.Future[None]) -> None:
    if not future.cancelled():
        future.exception()


_CHANNEL_POOL = _DirectChannelPool()


class DirectDataPlaneClient:
    """Lazily issues credentials and leases a direct channel for one sandbox."""

    def __init__(self) -> None:
        self._credentials: dict[int, _CredentialBundle] = {}
        self._lock = asyncio.Lock()
        self._retry_at = 0.0

    async def acquire(
        self,
        *,
        control_stub: Any,
        sandbox_id: str,
        auth_metadata: tuple[tuple[str, str], ...],
        permission: int,
        request_timeout: float,
        strict: bool = False,
    ) -> _DirectChannelLease:
        auto_deadline = (
            None
            if strict
            else time.monotonic() + min(request_timeout, _DIRECT_AUTO_TIMEOUT_SECONDS)
        )
        bundle = await self._ensure_credentials(
            control_stub=control_stub,
            sandbox_id=sandbox_id,
            auth_metadata=auth_metadata,
            permission=permission,
            request_timeout=request_timeout,
            strict=strict,
            deadline=auto_deadline,
        )
        if permission not in bundle.granted_permissions:
            raise DirectDataPlanePermissionUnavailable(
                f"The direct data-plane certificate does not grant permission {permission}"
            )

        connect_timeout = min(request_timeout, _DIRECT_CONNECT_TIMEOUT_SECONDS)
        if auto_deadline is not None:
            connect_timeout = max(0.0, auto_deadline - time.monotonic())
        if connect_timeout <= 0:
            self._defer_retry()
            raise DirectDataPlaneUnavailable(
                f"Timed out connecting to the direct data-plane endpoint {bundle.target}"
            )
        try:
            lease = await _CHANNEL_POOL.acquire(bundle, timeout=connect_timeout)
        except TimeoutError as exc:
            self._defer_retry()
            raise DirectDataPlaneUnavailable(
                f"Timed out connecting to the direct data-plane endpoint {bundle.target}"
            ) from exc
        except grpc.RpcError as exc:
            self._defer_retry()
            raise DirectDataPlaneUnavailable(
                f"Could not connect to the direct data-plane endpoint {bundle.target}"
            ) from exc
        self._retry_at = 0.0
        return lease

    async def _ensure_credentials(
        self,
        *,
        control_stub: Any,
        sandbox_id: str,
        auth_metadata: tuple[tuple[str, str], ...],
        permission: int,
        request_timeout: float,
        strict: bool,
        deadline: float | None,
    ) -> _CredentialBundle:
        if not strict and time.monotonic() < self._retry_at:
            raise DirectDataPlaneUnavailable("Direct data-plane retry is temporarily deferred")
        now = datetime.now(UTC)
        credentials = self._credentials.get(permission)
        if credentials is not None and credentials.expires_at > now + _DIRECT_EXPIRY_SKEW:
            return credentials

        async with self._lock:
            if not strict and time.monotonic() < self._retry_at:
                raise DirectDataPlaneUnavailable("Direct data-plane retry is temporarily deferred")
            now = datetime.now(UTC)
            credentials = self._credentials.get(permission)
            if credentials is not None and credentials.expires_at > now + _DIRECT_EXPIRY_SKEW:
                return credentials

            old_cache_key = credentials.cache_key if credentials is not None else None
            private_key = ec.generate_private_key(ec.SECP256R1())
            csr = (
                x509.CertificateSigningRequestBuilder()
                .subject_name(x509.Name([]))
                .sign(private_key, hashes.SHA256())
            )
            request = sandbox_pb2.ConnectSandboxRequest(
                sandbox_id=sandbox_id,
                csr_der=csr.public_bytes(serialization.Encoding.DER),
                requested_permissions=[cast(sandbox_pb2.SandboxDataPermission, permission)],
            )
            retry_deadline = time.monotonic() + min(
                request_timeout,
                _DIRECT_CONNECT_TIMEOUT_SECONDS,
            )
            retry_delay = 0.2
            while True:
                try:
                    rpc_timeout = min(request_timeout, _DIRECT_CREDENTIAL_RPC_TIMEOUT_SECONDS)
                    if strict:
                        rpc_timeout = min(
                            rpc_timeout,
                            max(0.1, retry_deadline - time.monotonic()),
                        )
                    elif deadline is not None:
                        rpc_timeout = max(0.0, deadline - time.monotonic())
                        if rpc_timeout <= 0:
                            self._defer_retry()
                            raise DirectDataPlaneUnavailable(
                                "Direct data-plane credential request timed out"
                            )
                    pending_response = control_stub.ConnectSandbox(
                        request,
                        timeout=rpc_timeout,
                        metadata=auth_metadata,
                    )
                    if not inspect.isawaitable(pending_response):
                        self._defer_retry()
                        raise DirectDataPlaneUnavailable(
                            "The API does not support direct data-plane connections"
                        )
                    response = await pending_response
                    break
                except grpc.RpcError as exc:
                    if exc.code() not in _FALLBACK_CONNECT_CODES:
                        raise
                    if (
                        strict
                        and exc.code() in _DIRECT_READINESS_RETRY_CODES
                        and time.monotonic() < retry_deadline
                    ):
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 1.0)
                        continue
                    self._defer_retry()
                    raise DirectDataPlaneUnavailable(
                        "The direct data-plane endpoint is not currently available"
                    ) from exc

            try:
                bundle = _credential_bundle(private_key, sandbox_id, response)
            except DirectDataPlaneUnavailable:
                self._defer_retry()
                raise
            self._credentials[permission] = bundle
            if old_cache_key is not None and old_cache_key != bundle.cache_key:
                await _CHANNEL_POOL.discard(old_cache_key)
            return bundle

    async def close(self) -> None:
        async with self._lock:
            credentials = tuple(self._credentials.values())
            self._credentials.clear()
            self._retry_at = 0.0
        for bundle in credentials:
            await _CHANNEL_POOL.discard(bundle.cache_key)

    def _defer_retry(self) -> None:
        self._retry_at = time.monotonic() + _DIRECT_RETRY_COOLDOWN_SECONDS


def _credential_bundle(
    private_key: ec.EllipticCurvePrivateKey,
    sandbox_id: str,
    response: Any,
) -> _CredentialBundle:
    if response.transport != sandbox_pb2.SANDBOX_DATA_TRANSPORT_DIRECT_MTLS:
        raise DirectDataPlaneUnavailable("The server returned an unsupported data-plane transport")
    if response.protocol != sandbox_pb2.SANDBOX_DATA_PROTOCOL_CONNECT_H2_V1:
        raise DirectDataPlaneUnavailable("The server returned an unsupported data-plane protocol")

    try:
        target, is_secure = parse_grpc_target(response.endpoint_uri)
    except ValueError as exc:
        raise DirectDataPlaneUnavailable(
            "The server returned an invalid direct data-plane endpoint"
        ) from exc
    if not is_secure:
        raise DirectDataPlaneUnavailable("The direct data-plane endpoint must use HTTPS")

    try:
        expires_at = response.expires_at.ToDatetime(tzinfo=UTC)
    except (OverflowError, ValueError) as exc:
        raise DirectDataPlaneUnavailable(
            "The server returned an invalid direct data-plane certificate expiry"
        ) from exc
    if expires_at <= datetime.now(UTC) + _DIRECT_EXPIRY_SKEW:
        raise DirectDataPlaneUnavailable("The direct data-plane certificate is already expiring")

    certificate_chain = bytes(response.client_certificate_chain_pem)
    server_ca_bundle = bytes(response.server_ca_bundle_pem)
    if not certificate_chain:
        raise DirectDataPlaneUnavailable("The server returned no client certificate")

    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    channel_credentials = grpc.ssl_channel_credentials(
        root_certificates=server_ca_bundle or None,
        private_key=private_key_pem,
        certificate_chain=certificate_chain,
    )
    certificate_fingerprint = hashlib.sha256(certificate_chain).hexdigest()
    cache_key = f"{response.endpoint_id}:{sandbox_id}:{certificate_fingerprint}"
    return _CredentialBundle(
        cache_key=cache_key,
        target=target,
        channel_credentials=channel_credentials,
        expires_at=expires_at,
        granted_permissions=frozenset(
            int(permission) for permission in response.granted_permissions
        ),
    )
