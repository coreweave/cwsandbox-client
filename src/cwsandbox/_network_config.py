# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""NetworkConfig CRUD (NetworkConfigService)."""

from __future__ import annotations

import builtins
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import grpc
from google.protobuf import field_mask_pb2

from cwsandbox._auth import AuthConfig, resolve_auth_metadata
from cwsandbox._defaults import (
    DEFAULT_BASE_URL,
    DEFAULT_POLL_RETRY_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from cwsandbox._error_info import (
    CWSANDBOX_NETWORK_CONFIG_NOT_FOUND,
    is_not_found,
    parse_error_info,
)
from cwsandbox._loop_manager import _LoopManager
from cwsandbox._network import create_channel, paginate_async, parse_grpc_target
from cwsandbox._proto import network_config_pb2, network_config_pb2_grpc
from cwsandbox._spec import network_from_proto, network_to_proto
from cwsandbox._types import NetworkOptions, OperationRef
from cwsandbox.exceptions import CWSandboxError, SandboxRequestTimeoutError

# Mutable NetworkOptions fields, in proto order. Each maps to one
# ``spec.<field>`` update-mask path.
_MUTABLE_FIELDS = ("egress", "ingress", "deny_egress", "deny_ingress", "dns_egress")
_MASK_PATHS = frozenset({"*"} | {f"spec.{field}" for field in _MUTABLE_FIELDS})


class NetworkConfig:
    """Desired network policy for one sandbox, editable while it runs.

    A NetworkConfig holds a ``NetworkOptions`` document. Bind it to a
    sandbox with ``Sandbox.run(network_config_id=...)``; while bound and
    RUNNING, ``update()`` applies the change live. An unbound config only
    stores the document. A config binds to at most one live sandbox.

    Sandboxes created without ``network_config_id`` get a platform-managed
    config (``platform_managed=True``, ID prefix ``nc-``). It is deleted
    when the sandbox ends. Configs you create persist and can be reused
    once their sandbox ends.

    Attributes:
        network_config_id: Org-unique ID.
        network: Stored desired document. Every field is populated, so it
            can be edited with ``dataclasses.replace`` and passed back to
            ``update()``.
        sandbox_id: Last sandbox this config was bound to, or None.
        platform_managed: True when the platform minted this config.
        etag: Opaque version for guarded updates (``update(etag=...)``).
        created_at: Creation time.
        updated_at: Last update time.
    """

    def __init__(
        self,
        *,
        network_config_id: str,
        network: NetworkOptions | None = None,
        sandbox_id: str | None = None,
        platform_managed: bool = False,
        etag: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> None:
        self.network_config_id = network_config_id
        self.network = network if network is not None else NetworkOptions()
        self.sandbox_id = sandbox_id
        self.platform_managed = platform_managed
        self.etag = etag
        self.created_at = created_at
        self.updated_at = updated_at
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._auth = auth

    def __repr__(self) -> str:
        parts = [f"network_config_id={self.network_config_id!r}"]
        if self.sandbox_id:
            parts.append(f"sandbox_id={self.sandbox_id!r}")
        if self.platform_managed:
            parts.append("platform_managed=True")
        return f"NetworkConfig({', '.join(parts)})"

    @classmethod
    def create(
        cls,
        network_config_id: str,
        *,
        network: NetworkOptions | Mapping[str, Any] | None = None,
        request_id: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[NetworkConfig]:
        """Create an unbound NetworkConfig.

        Args:
            network_config_id: Caller-chosen ID (lowercase letters, digits,
                and hyphens, starting with a letter; 3-63 characters). The
                ``nc-`` prefix is reserved for platform-managed configs.
            network: Initial document. Omit for the policy default.
            request_id: Idempotency key. Generated when omitted.

        Returns:
            OperationRef[NetworkConfig]: the stored config.
        """
        future = _LoopManager.get().run_async(
            cls._create_async(
                network_config_id,
                network=network,
                request_id=request_id,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                auth=auth,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _create_async(
        cls,
        network_config_id: str,
        *,
        network: NetworkOptions | Mapping[str, Any] | None,
        request_id: str | None,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> NetworkConfig:
        if not network_config_id:
            raise ValueError("network_config_id cannot be empty")
        options = _coerce_network(network) if network is not None else NetworkOptions()
        request = network_config_pb2.CreateNetworkConfigRequest(
            network_config=network_config_pb2.NetworkConfig(
                network_config_id=network_config_id,
                spec=network_to_proto(options),
            ),
            request_id=request_id or str(uuid.uuid4()),
        )
        response = await _call_network_config_rpc(
            "CreateNetworkConfig",
            request,
            operation="Create network config",
            network_config_id=network_config_id,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )
        return _network_config_from_proto(
            cast(network_config_pb2.NetworkConfig, response),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )

    @classmethod
    def get(
        cls,
        network_config_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[NetworkConfig]:
        """Fetch a NetworkConfig by ID."""
        future = _LoopManager.get().run_async(
            cls._get_async(
                network_config_id,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                auth=auth,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _get_async(
        cls,
        network_config_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> NetworkConfig:
        if not network_config_id:
            raise ValueError("network_config_id cannot be empty")
        request = network_config_pb2.GetNetworkConfigRequest(network_config_id=network_config_id)
        response = await _call_network_config_rpc(
            "GetNetworkConfig",
            request,
            operation="Get network config",
            network_config_id=network_config_id,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )
        return _network_config_from_proto(
            cast(network_config_pb2.NetworkConfig, response),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )

    @classmethod
    def list(
        cls,
        *,
        platform_managed: bool | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[builtins.list[NetworkConfig]]:
        """List the organization's NetworkConfigs (auto-paginated).

        Args:
            platform_managed: True for platform-managed configs only, False
                for caller-created only, None for both.
        """
        future = _LoopManager.get().run_async(
            cls._list_async(
                platform_managed=platform_managed,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                auth=auth,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _list_async(
        cls,
        *,
        platform_managed: bool | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> builtins.list[NetworkConfig]:
        request = network_config_pb2.ListNetworkConfigsRequest()
        if platform_managed is not None:
            request.platform_managed = platform_managed
        protos = await _list_network_config_rpc(
            request,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )
        return [
            _network_config_from_proto(
                proto, base_url=base_url, timeout_seconds=timeout_seconds, auth=auth
            )
            for proto in protos
        ]

    def update(
        self,
        network: NetworkOptions | Mapping[str, Any],
        *,
        update_mask: Sequence[str] | None = None,
        etag: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[NetworkConfig]:
        """Change the stored document; applies live when bound to a sandbox.

        Only the fields named by the update mask change. By default the mask
        is every field set (not None) on ``network``, so
        ``NetworkOptions(egress=[])`` clears egress to the policy default
        and leaves ingress alone. Leave ``dns_egress`` as None to keep the
        current DNS mode; ``DNSEgressMode.UNSPECIFIED`` resets it to the
        policy default.

        When bound, the call waits for a CREATING sandbox to reach RUNNING
        and returns after the runner applied the change.

        Args:
            network: Fields to write.
            update_mask: Explicit paths (``"egress"`` or ``"spec.egress"``;
                ``"*"`` replaces all five fields). An empty list is a no-op
                that returns the current config.
            etag: Reject the update if the config changed since this etag
                was read (``NetworkConfigEtagMismatchError``). Pass
                ``config.etag`` to guard a read-modify-write. Omitted means
                last writer wins.

        Returns:
            OperationRef[NetworkConfig]: this object, refreshed.

        Raises:
            ValueError: No field set on ``network`` and no ``update_mask``,
                or an unknown mask path.
        """
        future = _LoopManager.get().run_async(
            self._update_async(
                network,
                update_mask=update_mask,
                etag=etag,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                auth=auth,
            )
        )
        return OperationRef(future)

    async def _update_async(
        self,
        network: NetworkOptions | Mapping[str, Any],
        *,
        update_mask: Sequence[str] | None,
        etag: str | None,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> NetworkConfig:
        effective_auth = auth if auth is not None else self._auth
        effective_base_url = base_url or self._base_url
        effective_timeout = timeout_seconds or self._timeout_seconds
        updated = await _update_network_config_async(
            self.network_config_id,
            network,
            update_mask=update_mask,
            etag=etag,
            base_url=effective_base_url,
            timeout_seconds=effective_timeout,
            auth=effective_auth,
        )
        self._copy_from(updated)
        return self

    def delete(
        self,
        *,
        allow_missing: bool = False,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[None]:
        """Delete an unbound NetworkConfig.

        Raises ``NetworkConfigInUseError`` while a live sandbox is bound.
        Set ``allow_missing=True`` to succeed when the config is already gone.
        """
        future = _LoopManager.get().run_async(
            self._delete_async(
                allow_missing=allow_missing,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                auth=auth,
            )
        )
        return OperationRef(future)

    async def _delete_async(
        self,
        *,
        allow_missing: bool,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> None:
        request = network_config_pb2.DeleteNetworkConfigRequest(
            network_config_id=self.network_config_id,
            allow_missing=allow_missing,
        )
        await _call_network_config_rpc(
            "DeleteNetworkConfig",
            request,
            operation="Delete network config",
            network_config_id=self.network_config_id,
            base_url=base_url or self._base_url,
            timeout_seconds=timeout_seconds or self._timeout_seconds,
            auth=auth if auth is not None else self._auth,
            not_found_ok=allow_missing,
            reconcile_not_found_on_retry=True,
        )

    def _copy_from(self, other: NetworkConfig) -> None:
        self.network_config_id = other.network_config_id
        self.network = other.network
        self.sandbox_id = other.sandbox_id
        self.platform_managed = other.platform_managed
        self.etag = other.etag
        self.created_at = other.created_at
        self.updated_at = other.updated_at
        self._base_url = other._base_url
        self._timeout_seconds = other._timeout_seconds
        self._auth = other._auth


def _coerce_network(network: NetworkOptions | Mapping[str, Any]) -> NetworkOptions:
    if isinstance(network, NetworkOptions):
        return network
    if isinstance(network, Mapping):
        return NetworkOptions(**network)
    raise TypeError(f"network must be NetworkOptions or dict, got {type(network).__name__}")


def _resolve_update_mask(network: NetworkOptions, update_mask: Sequence[str] | None) -> list[str]:
    """Return explicit ``spec.*`` mask paths for an update.

    The SDK always sends a mask. The backend's omitted-mask rule (populated
    fields only) cannot clear a repeated field, so an explicit path is how
    ``egress=[]`` reaches the server as "clear".
    """
    if update_mask is None:
        paths = [
            f"spec.{field}" for field in _MUTABLE_FIELDS if getattr(network, field) is not None
        ]
        if not paths:
            raise ValueError(
                "update needs at least one NetworkOptions field set, or an explicit update_mask"
            )
        return paths
    if isinstance(update_mask, str):
        raise TypeError("update_mask must be a sequence of paths, not a string")
    paths = []
    for raw in update_mask:
        path = raw if raw == "*" or raw.startswith("spec.") else f"spec.{raw}"
        if path not in _MASK_PATHS:
            allowed = ", ".join(sorted(_MASK_PATHS))
            raise ValueError(f"unknown update_mask path {raw!r}; expected one of: {allowed}")
        if path not in paths:
            paths.append(path)
    return paths


async def _update_network_config_async(
    network_config_id: str,
    network: NetworkOptions | Mapping[str, Any],
    *,
    update_mask: Sequence[str] | None,
    etag: str | None,
    base_url: str | None,
    timeout_seconds: float | None,
    auth: AuthConfig | None,
) -> NetworkConfig:
    """UpdateNetworkConfig by ID. Shared by ``NetworkConfig.update`` and Sandbox."""
    if not network_config_id:
        raise ValueError("network_config_id cannot be empty")
    options = _coerce_network(network)
    paths = _resolve_update_mask(options, update_mask)
    config = network_config_pb2.NetworkConfig(
        network_config_id=network_config_id,
        spec=network_to_proto(options),
    )
    if etag:
        config.etag = etag
    request = network_config_pb2.UpdateNetworkConfigRequest(
        network_config_id=network_config_id,
        network_config=config,
        update_mask=field_mask_pb2.FieldMask(paths=paths),
    )
    response = await _call_network_config_rpc(
        "UpdateNetworkConfig",
        request,
        operation="Update network config",
        network_config_id=network_config_id,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        auth=auth,
        # A bound update may wait server-side for RUNNING, bounded by the
        # request deadline. A client deadline is that ceiling, not a blip.
        non_retryable=(SandboxRequestTimeoutError,),
    )
    return _network_config_from_proto(
        cast(network_config_pb2.NetworkConfig, response),
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        auth=auth,
    )


def _network_config_from_proto(
    proto: network_config_pb2.NetworkConfig,
    *,
    base_url: str | None,
    timeout_seconds: float | None,
    auth: AuthConfig | None = None,
) -> NetworkConfig:
    status = proto.status
    return NetworkConfig(
        network_config_id=proto.network_config_id,
        network=network_from_proto(proto.spec),
        sandbox_id=status.sandbox_id or None,
        platform_managed=status.platform_managed,
        etag=proto.etag or None,
        created_at=_timestamp(status, "create_time"),
        updated_at=_timestamp(status, "update_time"),
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        auth=auth,
    )


def _timestamp(message: Any, field_name: str) -> datetime | None:
    if not message.HasField(field_name):
        return None
    result = getattr(message, field_name).ToDatetime(tzinfo=UTC)
    return result if isinstance(result, datetime) else None


def _stub_for(
    base_url: str | None,
    auth: AuthConfig | None,
) -> tuple[grpc.aio.Channel, Any, tuple[tuple[str, str], ...]]:
    effective_base_url = (
        base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
    target, is_secure = parse_grpc_target(effective_base_url)
    channel = create_channel(target, is_secure)
    stub = network_config_pb2_grpc.NetworkConfigServiceStub(channel)  # type: ignore[no-untyped-call]
    return channel, stub, auth_metadata


async def _call_network_config_rpc(
    method_name: str,
    request: Any,
    *,
    operation: str,
    network_config_id: str,
    base_url: str | None,
    timeout_seconds: float | None,
    auth: AuthConfig | None = None,
    non_retryable: tuple[type[CWSandboxError], ...] = (),
    not_found_ok: bool = False,
    reconcile_not_found_on_retry: bool = False,
) -> Any:
    from cwsandbox._sandbox import _retry_transient_rpc, _translate_rpc_error

    timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS
    channel, stub, auth_metadata = _stub_for(base_url, auth)
    attempts = {"n": 0}
    try:

        async def _attempt() -> Any:
            attempts["n"] += 1
            method = getattr(stub, method_name)
            try:
                return await method(request, timeout=timeout, metadata=auth_metadata)
            except grpc.RpcError as e:
                parsed = parse_error_info(e)
                if is_not_found(e, parsed, CWSANDBOX_NETWORK_CONFIG_NOT_FOUND) and (
                    not_found_ok or (reconcile_not_found_on_retry and attempts["n"] > 1)
                ):
                    return None
                raise _translate_rpc_error(
                    e, operation=operation, network_config_id=network_config_id
                ) from e

        return await _retry_transient_rpc(
            _attempt,
            budget_seconds=DEFAULT_POLL_RETRY_BUDGET_SECONDS,
            operation=operation,
            non_retryable=non_retryable,
        )
    finally:
        await channel.close(grace=None)


async def _list_network_config_rpc(
    request: network_config_pb2.ListNetworkConfigsRequest,
    *,
    base_url: str | None,
    timeout_seconds: float | None,
    auth: AuthConfig | None = None,
) -> list[Any]:
    from cwsandbox._sandbox import _retry_transient_rpc, _translate_rpc_error

    timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS
    channel, stub, auth_metadata = _stub_for(base_url, auth)
    try:

        async def _attempt() -> list[Any]:
            fresh = network_config_pb2.ListNetworkConfigsRequest()
            fresh.CopyFrom(request)
            try:
                return await paginate_async(
                    stub.ListNetworkConfigs,
                    fresh,
                    "network_configs",
                    auth_metadata,
                    timeout,
                    operation="List network configs",
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, operation="List network configs") from e

        return await _retry_transient_rpc(
            _attempt,
            budget_seconds=DEFAULT_POLL_RETRY_BUDGET_SECONDS,
            operation="List network configs",
        )
    finally:
        await channel.close(grace=None)
