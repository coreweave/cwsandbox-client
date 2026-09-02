# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Registered Volume CRUD (VolumeService)."""

from __future__ import annotations

import builtins
import os
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

import grpc
from google.protobuf import field_mask_pb2

from cwsandbox._auth import AuthConfig, resolve_auth_metadata
from cwsandbox._defaults import (
    DEFAULT_BASE_URL,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_RETRY_BUDGET_SECONDS,
    DEFAULT_POLL_RPC_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from cwsandbox._error_info import CWSANDBOX_VOLUME_NOT_FOUND, is_not_found, parse_error_info
from cwsandbox._loop_manager import _LoopManager
from cwsandbox._network import create_channel, paginate_async, parse_grpc_target
from cwsandbox._proto import volume_pb2, volume_pb2_grpc
from cwsandbox._types import OperationRef, _validate_sub_path
from cwsandbox.exceptions import (
    CWSandboxError,
    SandboxRequestTimeoutError,
    VolumeError,
    VolumeWaitTimeoutError,
)

_monotonic = time.monotonic

_STATE_FROM_PROTO = {
    volume_pb2.VOLUME_STATE_UNSPECIFIED: "unspecified",
    volume_pb2.VOLUME_STATE_VALIDATING: "validating",
    volume_pb2.VOLUME_STATE_READY: "ready",
    volume_pb2.VOLUME_STATE_ERROR: "error",
    volume_pb2.VOLUME_STATE_DELETING: "deleting",
}
_STATE_TO_PROTO = {value: key for key, value in _STATE_FROM_PROTO.items()}

_LOCALITY_FROM_PROTO = {
    volume_pb2.VOLUME_LOCALITY_UNSPECIFIED: "unspecified",
    volume_pb2.VOLUME_LOCALITY_CLUSTER_LOCAL: "cluster_local",
    volume_pb2.VOLUME_LOCALITY_GLOBAL: "global",
}


class VolumeState(StrEnum):
    """Lifecycle state of a registered Volume."""

    UNSPECIFIED = "unspecified"
    VALIDATING = "validating"
    READY = "ready"
    ERROR = "error"
    DELETING = "deleting"


class VolumeLocality(StrEnum):
    """Whether a volume pins scheduling to one cluster."""

    UNSPECIFIED = "unspecified"
    CLUSTER_LOCAL = "cluster_local"
    GLOBAL = "global"


class PvcVolumeSource:
    """Existing PersistentVolumeClaim referenced by a registered Volume.

    Attributes:
        runner_id: Runner whose cluster holds the PVC. Also the scheduling pin.
        namespace: Kubernetes namespace of the claim.
        claim_name: PVC name.
        sub_path: Optional volume-level root inside the claim.
    """

    def __init__(
        self,
        *,
        runner_id: str,
        namespace: str,
        claim_name: str,
        sub_path: str | None = None,
    ) -> None:
        if not runner_id:
            raise ValueError("PvcVolumeSource.runner_id cannot be empty")
        if not namespace:
            raise ValueError("PvcVolumeSource.namespace cannot be empty")
        if not claim_name:
            raise ValueError("PvcVolumeSource.claim_name cannot be empty")
        self.runner_id = runner_id
        self.namespace = namespace
        self.claim_name = claim_name
        self.sub_path = _validate_sub_path(sub_path, field="PvcVolumeSource.sub_path")


class Volume:
    """An org-registered shared volume.

    Create returns immediately in ``VALIDATING``; poll with
    ``wait_until_ready()`` or ``get()`` until ``READY`` / ``ERROR``.
    Mount via ``RegisteredVolumeOptions`` on ``Sandbox.run(volumes=...)``.
    """

    def __init__(
        self,
        *,
        volume_id: str,
        pvc: PvcVolumeSource | None = None,
        read_only: bool = False,
        description: str = "",
        state: VolumeState = VolumeState.UNSPECIFIED,
        state_reason: str = "",
        locality: VolumeLocality = VolumeLocality.UNSPECIFIED,
        access_modes: tuple[str, ...] = (),
        capacity: str = "",
        attached_sandbox_count: int = 0,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        last_validated_at: datetime | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> None:
        self.volume_id = volume_id
        self.pvc = pvc
        self.read_only = read_only
        self.description = description
        self.state = state
        self.state_reason = state_reason
        self.locality = locality
        self.access_modes = access_modes
        self.capacity = capacity
        self.attached_sandbox_count = attached_sandbox_count
        self.created_at = created_at
        self.updated_at = updated_at
        self.last_validated_at = last_validated_at
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._auth = auth

    def __repr__(self) -> str:
        parts = [f"volume_id={self.volume_id!r}", f"state={self.state.value}"]
        if self.locality != VolumeLocality.UNSPECIFIED:
            parts.append(f"locality={self.locality.value}")
        if self.state_reason:
            parts.append(f"state_reason={self.state_reason!r}")
        return f"Volume({', '.join(parts)})"

    @classmethod
    def create(
        cls,
        volume_id: str,
        *,
        pvc: PvcVolumeSource,
        read_only: bool = False,
        description: str = "",
        request_id: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[Volume]:
        """Register a Volume. Returns immediately in ``VALIDATING``."""
        future = _LoopManager.get().run_async(
            cls._create_async(
                volume_id,
                pvc=pvc,
                read_only=read_only,
                description=description,
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
        volume_id: str,
        *,
        pvc: PvcVolumeSource,
        read_only: bool,
        description: str,
        request_id: str | None,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> Volume:
        if not volume_id:
            raise ValueError("volume_id cannot be empty")
        proto = volume_pb2.Volume(
            volume_id=volume_id,
            spec=volume_pb2.VolumeSpec(
                pvc=volume_pb2.PvcVolumeSource(
                    runner_id=pvc.runner_id,
                    namespace=pvc.namespace,
                    claim_name=pvc.claim_name,
                    sub_path=pvc.sub_path or "",
                ),
                read_only=read_only,
                description=description,
            ),
        )
        request = volume_pb2.CreateVolumeRequest(
            volume=proto,
            request_id=request_id or str(uuid.uuid4()),
        )
        response = await _call_volume_rpc(
            "CreateVolume",
            request,
            operation="Create volume",
            volume_id=volume_id,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )
        return _volume_from_proto(
            cast(volume_pb2.Volume, response),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )

    @classmethod
    def get(
        cls,
        volume_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[Volume]:
        """Fetch a registered Volume by ID."""
        future = _LoopManager.get().run_async(
            cls._get_async(volume_id, base_url=base_url, timeout_seconds=timeout_seconds, auth=auth)
        )
        return OperationRef(future)

    @classmethod
    async def _get_async(
        cls,
        volume_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
        retry_budget_seconds: float | None = None,
        non_retryable: tuple[type[CWSandboxError], ...] = (),
    ) -> Volume:
        if not volume_id:
            raise ValueError("volume_id cannot be empty")
        request = volume_pb2.GetVolumeRequest(volume_id=volume_id)
        response = await _call_volume_rpc(
            "GetVolume",
            request,
            operation="Get volume",
            volume_id=volume_id,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
            retry_budget_seconds=retry_budget_seconds,
            non_retryable=non_retryable,
        )
        return _volume_from_proto(
            cast(volume_pb2.Volume, response),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )

    @classmethod
    def list(
        cls,
        *,
        states: Sequence[VolumeState | str] | None = None,
        runner_ids: Sequence[str] | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[builtins.list[Volume]]:
        """List registered Volumes for the organization (auto-paginated)."""
        future = _LoopManager.get().run_async(
            cls._list_async(
                states=states,
                runner_ids=runner_ids,
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
        states: Sequence[VolumeState | str] | None = None,
        runner_ids: Sequence[str] | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> builtins.list[Volume]:
        request = volume_pb2.ListVolumesRequest()
        if states:
            for state in states:
                value = VolumeState(state) if isinstance(state, str) else state
                request.states.append(_STATE_TO_PROTO[value.value])
        if runner_ids:
            request.runner_ids.extend(runner_ids)
        protos = await _list_volume_rpc(
            request,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            auth=auth,
        )
        return [
            _volume_from_proto(proto, base_url=base_url, timeout_seconds=timeout_seconds, auth=auth)
            for proto in protos
        ]

    def update(
        self,
        *,
        description: str,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[Volume]:
        """Replace the volume description."""
        future = _LoopManager.get().run_async(
            self._update_async(
                description=description,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                auth=auth,
            )
        )
        return OperationRef(future)

    async def _update_async(
        self,
        *,
        description: str,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> Volume:
        request = volume_pb2.UpdateVolumeRequest(
            volume_id=self.volume_id,
            volume=volume_pb2.Volume(
                volume_id=self.volume_id,
                spec=volume_pb2.VolumeSpec(description=description),
            ),
            update_mask=field_mask_pb2.FieldMask(paths=["spec.description"]),
        )
        effective_auth = auth if auth is not None else self._auth
        effective_base_url = base_url or self._base_url
        effective_timeout = timeout_seconds or self._timeout_seconds
        response = await _call_volume_rpc(
            "UpdateVolume",
            request,
            operation="Update volume",
            volume_id=self.volume_id,
            base_url=effective_base_url,
            timeout_seconds=effective_timeout,
            auth=effective_auth,
        )
        updated = _volume_from_proto(
            cast(volume_pb2.Volume, response),
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
        force: bool = False,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[Volume]:
        """Deregister the Volume. Does not delete the backing PVC."""
        future = _LoopManager.get().run_async(
            self._delete_async(
                allow_missing=allow_missing,
                force=force,
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
        force: bool,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> Volume:
        request = volume_pb2.DeleteVolumeRequest(
            volume_id=self.volume_id,
            allow_missing=allow_missing,
            force=force,
        )
        effective_auth = auth if auth is not None else self._auth
        effective_base_url = base_url or self._base_url
        effective_timeout = timeout_seconds or self._timeout_seconds
        response = await _call_volume_rpc(
            "DeleteVolume",
            request,
            operation="Delete volume",
            volume_id=self.volume_id,
            base_url=effective_base_url,
            timeout_seconds=effective_timeout,
            auth=effective_auth,
            not_found_ok=allow_missing,
            reconcile_not_found_on_retry=True,
        )
        if response is None:
            return self
        updated = _volume_from_proto(
            response,
            base_url=effective_base_url,
            timeout_seconds=effective_timeout,
            auth=effective_auth,
        )
        self._copy_from(updated)
        return self

    def validate(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[Volume]:
        """Trigger an on-demand re-validation of the backing source."""
        future = _LoopManager.get().run_async(
            self._validate_async(base_url=base_url, timeout_seconds=timeout_seconds, auth=auth)
        )
        return OperationRef(future)

    async def _validate_async(
        self,
        *,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> Volume:
        request = volume_pb2.ValidateVolumeRequest(volume_id=self.volume_id)
        effective_auth = auth if auth is not None else self._auth
        effective_base_url = base_url or self._base_url
        effective_timeout = timeout_seconds or self._timeout_seconds
        response = await _call_volume_rpc(
            "ValidateVolume",
            request,
            operation="Validate volume",
            volume_id=self.volume_id,
            base_url=effective_base_url,
            timeout_seconds=effective_timeout,
            auth=effective_auth,
        )
        updated = _volume_from_proto(
            cast(volume_pb2.Volume, response),
            base_url=effective_base_url,
            timeout_seconds=effective_timeout,
            auth=effective_auth,
        )
        self._copy_from(updated)
        return self

    def wait_until_ready(
        self,
        timeout: float = 60.0,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        auth: AuthConfig | None = None,
    ) -> OperationRef[Volume]:
        """Poll GetVolume until READY. Raises on ERROR or timeout."""
        future = _LoopManager.get().run_async(
            self._wait_until_ready_async(
                timeout=timeout,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                auth=auth,
            )
        )
        return OperationRef(future)

    async def _wait_until_ready_async(
        self,
        *,
        timeout: float,
        base_url: str | None,
        timeout_seconds: float | None,
        auth: AuthConfig | None,
    ) -> Volume:
        if not (timeout > 0):
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        deadline = _monotonic() + timeout
        poll_rpc_timeout = timeout_seconds or DEFAULT_POLL_RPC_TIMEOUT_SECONDS
        effective_base_url = base_url or self._base_url
        effective_auth = auth if auth is not None else self._auth

        def _timeout() -> VolumeWaitTimeoutError:
            return VolumeWaitTimeoutError(
                f"Timed out waiting for volume '{self.volume_id}' to become READY",
                volume_id=self.volume_id,
            )

        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                raise _timeout()
            rpc_timeout = min(poll_rpc_timeout, remaining)
            retry_budget = min(DEFAULT_POLL_RETRY_BUDGET_SECONDS, remaining)
            ceiling_hit = rpc_timeout >= remaining
            try:
                current = await Volume._get_async(
                    self.volume_id,
                    base_url=effective_base_url,
                    timeout_seconds=rpc_timeout,
                    auth=effective_auth,
                    retry_budget_seconds=retry_budget,
                    non_retryable=(SandboxRequestTimeoutError,) if ceiling_hit else (),
                )
            except SandboxRequestTimeoutError as exc:
                if ceiling_hit or _monotonic() >= deadline:
                    raise _timeout() from exc
                raise
            if _monotonic() >= deadline:
                raise _timeout()
            self._copy_from(current)
            if current.state == VolumeState.READY:
                return self
            if current.state == VolumeState.ERROR:
                raise VolumeError(
                    f"Volume '{self.volume_id}' entered ERROR: {current.state_reason}",
                    volume_id=self.volume_id,
                )
            if current.state == VolumeState.DELETING:
                raise VolumeError(
                    f"Volume '{self.volume_id}' is deleting",
                    volume_id=self.volume_id,
                )
            remaining = deadline - _monotonic()
            if remaining <= 0:
                raise _timeout()
            await _sleep(min(DEFAULT_POLL_INTERVAL_SECONDS, remaining))

    def _copy_from(self, other: Volume) -> None:
        self.volume_id = other.volume_id
        self.pvc = other.pvc
        self.read_only = other.read_only
        self.description = other.description
        self.state = other.state
        self.state_reason = other.state_reason
        self.locality = other.locality
        self.access_modes = other.access_modes
        self.capacity = other.capacity
        self.attached_sandbox_count = other.attached_sandbox_count
        self.created_at = other.created_at
        self.updated_at = other.updated_at
        self.last_validated_at = other.last_validated_at
        self._base_url = other._base_url
        self._timeout_seconds = other._timeout_seconds
        self._auth = other._auth


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _volume_from_proto(
    proto: volume_pb2.Volume,
    *,
    base_url: str | None,
    timeout_seconds: float | None,
    auth: AuthConfig | None = None,
) -> Volume:
    pvc = None
    if proto.HasField("spec") and proto.spec.HasField("pvc"):
        source = proto.spec.pvc
        pvc = PvcVolumeSource(
            runner_id=source.runner_id,
            namespace=source.namespace,
            claim_name=source.claim_name,
            sub_path=source.sub_path or None,
        )
    status = proto.status
    return Volume(
        volume_id=proto.volume_id,
        pvc=pvc,
        read_only=proto.spec.read_only if proto.HasField("spec") else False,
        description=proto.spec.description if proto.HasField("spec") else "",
        state=VolumeState(_STATE_FROM_PROTO.get(status.state, "unspecified")),
        state_reason=status.state_reason,
        locality=VolumeLocality(_LOCALITY_FROM_PROTO.get(status.locality, "unspecified")),
        access_modes=tuple(status.access_modes),
        capacity=status.capacity,
        attached_sandbox_count=status.attached_sandbox_count,
        created_at=_timestamp(status, "create_time"),
        updated_at=_timestamp(status, "update_time"),
        last_validated_at=_timestamp(status, "last_validated_time"),
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        auth=auth,
    )


def _timestamp(message: Any, field_name: str) -> datetime | None:
    if not message.HasField(field_name):
        return None
    result = getattr(message, field_name).ToDatetime(tzinfo=UTC)
    return result if isinstance(result, datetime) else None


async def _call_volume_rpc(
    method_name: str,
    request: Any,
    *,
    operation: str,
    volume_id: str,
    base_url: str | None,
    timeout_seconds: float | None,
    auth: AuthConfig | None = None,
    retry_budget_seconds: float | None = None,
    non_retryable: tuple[type[CWSandboxError], ...] = (),
    not_found_ok: bool = False,
    reconcile_not_found_on_retry: bool = False,
) -> volume_pb2.Volume | None:
    from cwsandbox._sandbox import _retry_transient_rpc, _translate_rpc_error

    effective_base_url = (
        base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS
    retry_budget = (
        retry_budget_seconds
        if retry_budget_seconds is not None
        else DEFAULT_POLL_RETRY_BUDGET_SECONDS
    )
    auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
    target, is_secure = parse_grpc_target(effective_base_url)
    channel = create_channel(target, is_secure)
    stub = volume_pb2_grpc.VolumeServiceStub(channel)  # type: ignore[no-untyped-call]
    attempts = {"n": 0}
    try:

        async def _attempt() -> volume_pb2.Volume | None:
            attempts["n"] += 1
            method = getattr(stub, method_name)
            try:
                return cast(
                    volume_pb2.Volume,
                    await method(request, timeout=timeout, metadata=auth_metadata),
                )
            except grpc.RpcError as e:
                parsed = parse_error_info(e)
                if is_not_found(e, parsed, CWSANDBOX_VOLUME_NOT_FOUND) and (
                    not_found_ok or (reconcile_not_found_on_retry and attempts["n"] > 1)
                ):
                    return None
                raise _translate_rpc_error(e, operation=operation, volume_id=volume_id) from e

        return await _retry_transient_rpc(
            _attempt,
            budget_seconds=retry_budget,
            operation=operation,
            non_retryable=non_retryable,
        )
    finally:
        await channel.close(grace=None)


async def _list_volume_rpc(
    request: volume_pb2.ListVolumesRequest,
    *,
    base_url: str | None,
    timeout_seconds: float | None,
    auth: AuthConfig | None = None,
) -> list[Any]:
    from cwsandbox._sandbox import _retry_transient_rpc, _translate_rpc_error

    effective_base_url = (
        base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    timeout = timeout_seconds or DEFAULT_REQUEST_TIMEOUT_SECONDS
    auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
    target, is_secure = parse_grpc_target(effective_base_url)
    channel = create_channel(target, is_secure)
    stub = volume_pb2_grpc.VolumeServiceStub(channel)  # type: ignore[no-untyped-call]
    try:

        async def _attempt() -> list[Any]:
            fresh = volume_pb2.ListVolumesRequest()
            fresh.CopyFrom(request)
            try:
                return await paginate_async(
                    stub.ListVolumes,
                    fresh,
                    "volumes",
                    auth_metadata,
                    timeout,
                    operation="List volumes",
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, operation="List volumes") from e

        return await _retry_transient_rpc(
            _attempt,
            budget_seconds=DEFAULT_POLL_RETRY_BUDGET_SECONDS,
            operation="List volumes",
        )
    finally:
        await channel.close(grace=None)
