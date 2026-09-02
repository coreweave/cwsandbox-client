# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

from __future__ import annotations

import asyncio
import builtins
import contextlib
import inspect
import logging
import math
import os
import random
import shlex
import threading
import time
import uuid
import warnings
import weakref
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

import grpc
import grpc.aio
from google.protobuf import timestamp_pb2

from cwsandbox._auth import AuthConfig, resolve_auth_metadata
from cwsandbox._defaults import (
    DEFAULT_BASE_URL,
    DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS,
    DEFAULT_FILE_OPERATION_CAP_BYTES,
    DEFAULT_FSS_RETRY_BUDGET_SECONDS,
    DEFAULT_FSS_STOP_CLIENT_SLACK_SECONDS,
    DEFAULT_FSS_STOP_TIMEOUT_SECONDS,
    DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    DEFAULT_MAX_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_BACKOFF_FACTOR,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_POLL_RETRY_BUDGET_SECONDS,
    DEFAULT_POLL_RPC_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    MAX_AUTO_FALLBACK_BYTES,
    MAX_FILE_UNARY_BYTES,
    MAX_LINE_BUFFER_BYTES,
    STAT_INTEGRITY_TIMEOUT_SECONDS,
    STDIN_CHUNK_SIZE,
    STREAMING_OUTPUT_QUEUE_SIZE,
    STREAMING_READ_STDERR_CAP_BYTES,
    STREAMING_RESPONSE_QUEUE_SIZE,
    STREAMING_RESUME_BACKOFF_SECONDS,
    STREAMING_RESUME_MAX_ATTEMPTS,
    STREAMING_RESUME_MAX_BACKOFF_SECONDS,
    STREAMING_WRITE_CHUNK_SIZE,
    TRUNCATION_CHECK_MIN_BYTES,
    SandboxDefaults,
    _normalize_tags,
    _resolve_selector,
    _validate_poll_config,
)
from cwsandbox._direct import (
    DirectDataPlaneClient,
    DirectDataPlanePermissionUnavailable,
    DirectDataPlaneUnavailable,
)
from cwsandbox._error_info import (
    CWSANDBOX_BACKEND_UNAVAILABLE,
    CWSANDBOX_COMMAND_TIMEOUT,
    CWSANDBOX_ERROR_DOMAIN,
    CWSANDBOX_FILE_IO_FAILED,
    CWSANDBOX_FILE_IS_DIRECTORY,
    CWSANDBOX_FILE_NOT_FOUND,
    CWSANDBOX_FILE_TOO_LARGE,
    CWSANDBOX_FILE_TRUNCATED,
    CWSANDBOX_FSS_BUCKET_MISMATCH,
    CWSANDBOX_FSS_NOT_FOUND,
    CWSANDBOX_FSS_NOT_READY,
    CWSANDBOX_FSS_NOT_SUPPORTED,
    CWSANDBOX_FSS_QUOTA_EXCEEDED,
    CWSANDBOX_FSS_SIZE_EXCEEDED,
    CWSANDBOX_FSS_WAIT_TIMEOUT,
    CWSANDBOX_INVALID_REQUEST,
    CWSANDBOX_RUNNER_SHARD_RETIRING,
    CWSANDBOX_SANDBOX_NOT_FOUND,
    CWSANDBOX_VOLUME_BACKEND_NOT_FOUND,
    CWSANDBOX_VOLUME_IN_USE,
    CWSANDBOX_VOLUME_NOT_FOUND,
    CWSANDBOX_VOLUME_NOT_READY,
    CWSANDBOX_VOLUME_NOT_SNAPSHOTTABLE,
    CWSANDBOX_VOLUME_PLACEMENT_CONFLICT,
    CWSANDBOX_VOLUME_QUOTA_EXCEEDED,
    CWSANDBOX_VOLUME_RUNNER_INELIGIBLE,
    CWSANDBOX_VOLUME_RUNNER_UNAVAILABLE,
    CWSANDBOX_VOLUME_TYPE_NOT_SUPPORTED,
    FILE_ERROR_REASONS,
    SNAPSHOT_INTERNAL_REASONS,
    SNAPSHOT_TRANSIENT_REASONS,
    SPILLOVER_BLOCKED_REASONS,
    SPILLOVER_ELIGIBLE_REASONS,
    STREAM_BACKPRESSURE,
    STREAM_TRUNCATED,
    UNAVAILABLE_REASONS,
    is_not_found,
    parse_error_info,
)
from cwsandbox._loop_manager import _LoopManager
from cwsandbox._network import (
    create_channel,
    paginate_async,
    parse_grpc_target,
    translate_grpc_error,
)
from cwsandbox._proto import (
    sandbox_pb2,
    sandbox_pb2_grpc,
    settings_pb2,
    settings_pb2_grpc,
)
from cwsandbox._resources import normalize_resources
from cwsandbox._spec import (
    coerce_security_context,
    egress_rule_from_proto,
    ingress_rule_from_proto,
    network_to_proto,
    object_storage_to_proto,
    scratch_volume_to_proto,
    security_context_to_proto,
    volume_mount_to_proto,
    volumes_to_proto,
)
from cwsandbox._types import (
    Container,
    DataPlaneMode,
    EgressRule,
    Endpoint,
    EndpointAuth,
    EndpointKind,
    ExecOutcome,
    FileSystemSnapshot,
    FileSystemSnapshotBucketConfig,
    FileSystemSnapshotBucketMode,
    FileSystemSnapshotOptions,
    FileSystemSnapshotStatus,
    FileSystemSnapshotTrigger,
    HttpsEndpointStatus,
    ImagePullCredentials,
    IngressRule,
    NetworkOptions,
    ObjectStorageAccess,
    OperationRef,
    PlacementMode,
    PlacementSpillover,
    Process,
    ProcessResult,
    RegisteredVolumeOptions,
    ResourceOptions,
    ScratchVolumeOptions,
    Secret,
    SecurityContext,
    Service,
    ServiceProtocol,
    ServiceVisibility,
    StreamReader,
    StreamWriter,
    TerminalResult,
    TerminalSession,
    VolumeMount,
    _coerce_container,
    _coerce_object_storage_access,
    _coerce_volume_mount,
    _unique_secrets_by_env_var,
    _validate_containers,
)
from cwsandbox.exceptions import (
    CWSandboxAuthenticationError,
    CWSandboxError,
    SandboxCommandTimeoutError,
    SandboxError,
    SandboxExecutionError,
    SandboxFailedError,
    SandboxFileError,
    SandboxNotFoundError,
    SandboxNotRunningError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
    SandboxSnapshotError,
    SandboxStreamBackpressureError,
    SandboxStreamTruncatedError,
    SandboxTerminalStateUnavailableError,
    SandboxTerminatedError,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SnapshotBackendThrottledError,
    SnapshotBucketMismatchError,
    SnapshotNotFoundError,
    SnapshotNotReadyError,
    SnapshotNotSupportedError,
    SnapshotOnStopConflictError,
    SnapshotQuotaExceededError,
    SnapshotSizeExceededError,
    SnapshotWaitTimeoutError,
    VolumeBackendNotFoundError,
    VolumeError,
    VolumeInUseError,
    VolumeNotFoundError,
    VolumeNotReadyError,
    VolumeNotSnapshottableError,
    VolumePlacementConflictError,
    VolumeQuotaExceededError,
    VolumeRunnerIneligibleError,
    VolumeRunnerUnavailableError,
    VolumeTypeNotSupportedError,
)

if TYPE_CHECKING:
    import concurrent.futures

    from cwsandbox._session import Session

logger = logging.getLogger(__name__)


@dataclass
class _PreparedDataPlaneCall:
    """A data-plane stub plus the metadata and optional direct-channel lease."""

    stub: Any
    metadata: tuple[tuple[str, str], ...]
    direct_lease: Any | None = None
    _released: bool = False

    async def release(self, *, discard: bool = False) -> None:
        if self._released:
            return
        self._released = True
        if self.direct_lease is not None:
            await self.direct_lease.release(discard=discard)

    @property
    def is_direct(self) -> bool:
        return self.direct_lease is not None

    async def discard(self) -> None:
        if self.direct_lease is not None:
            await self.direct_lease.discard()

    def release_when_done(self, call: Any) -> None:
        """Hold a direct channel until a streaming RPC reaches a terminal state."""

        if self.direct_lease is None:
            return

        async def _release(done_call: Any) -> None:
            discard = False
            try:
                code = done_call.code()
                if inspect.isawaitable(code):
                    code = await code
                discard = code == grpc.StatusCode.UNAVAILABLE
            except Exception:
                pass
            if discard:
                await self.discard()
            await self.release(discard=discard)

        call.add_done_callback(lambda done_call: asyncio.create_task(_release(done_call)))


class SandboxStatus(StrEnum):
    """Sandbox lifecycle status values.

    Lifecycle: CREATING -> RUNNING -> TERMINATING -> COMPLETED | FAILED

    Attributes:
        PENDING: Sandbox has been accepted but not yet scheduled.
        CREATING: Sandbox container is being created.
        RUNNING: Sandbox is running and ready for operations.
        PAUSED: Sandbox is paused (resources may be reclaimed).
        TERMINATING: Sandbox is draining through its grace period before exit.
        COMPLETED: Sandbox exited normally (check ``returncode``).
        FAILED: Sandbox failed to start or encountered a fatal error.
        TERMINATED: Sandbox was stopped via ``stop()`` or timeout (deprecated
            in favor of the TERMINATING -> COMPLETED/FAILED flow, but still
            emitted by older backends).
        UNSPECIFIED: Status is unknown or not yet reported by the backend.
    """

    RUNNING = "running"
    CREATING = "creating"
    PENDING = "pending"
    PAUSED = "paused"
    TERMINATING = "terminating"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
    UNSPECIFIED = "unspecified"

    @classmethod
    def from_proto(cls, proto_status: int) -> SandboxStatus:
        """Convert protobuf State enum to SandboxStatus."""
        try:
            proto_name = sandbox_pb2.State.Name(proto_status)
            enum_name = proto_name.replace("STATE_", "")
            return cls[enum_name]
        except (ValueError, KeyError):
            logger.warning("Unknown sandbox status %s, treating as UNSPECIFIED", proto_status)
            return cls.UNSPECIFIED

    def to_proto(self) -> int:
        """Convert SandboxStatus to protobuf State enum."""
        proto_name = f"STATE_{self.name}"
        return sandbox_pb2.State.Value(proto_name)


@dataclass(frozen=True, kw_only=True)
class ContainerStatus:
    """Observed state of one container on a sandbox.

    Sandbox ``status`` / ``returncode`` stay primary-owned. This row is the
    kubelet's view of that named container.

    Attributes:
        name: Container name.
        state: Observed lifecycle state of this container.
        exit_code: Exit code once the container is in a terminal state.
            None while the container is still running.
        restart_count: Times the container has restarted.
    """

    name: str
    state: SandboxStatus
    exit_code: int | None = None
    restart_count: int = 0


def _fss_status_from_proto(value: int) -> FileSystemSnapshotStatus:
    """Convert a proto SnapshotState enum to the SDK enum."""
    try:
        name = sandbox_pb2.SnapshotState.Name(value).replace("SNAPSHOT_STATE_", "")
        return FileSystemSnapshotStatus[name]
    except (ValueError, KeyError):
        logger.warning("Unknown snapshot status %s, treating as UNSPECIFIED", value)
        return FileSystemSnapshotStatus.UNSPECIFIED


def _fss_trigger_from_proto(value: int) -> FileSystemSnapshotTrigger:
    """Convert a proto SnapshotTrigger enum to the SDK enum."""
    try:
        name = sandbox_pb2.SnapshotTrigger.Name(value).replace("SNAPSHOT_TRIGGER_", "")
        if name == "ON_DELETE":
            return FileSystemSnapshotTrigger.ON_DELETE
        return FileSystemSnapshotTrigger[name]
    except (ValueError, KeyError):
        logger.warning("Unknown snapshot trigger %s, treating as UNSPECIFIED", value)
        return FileSystemSnapshotTrigger.UNSPECIFIED


def _fss_bucket_mode_from_proto(value: int) -> FileSystemSnapshotBucketMode:
    """Convert a proto FileSystemSnapshotBucketMode enum to the SDK enum."""
    try:
        name = settings_pb2.FileSystemSnapshotBucketMode.Name(value).replace(
            "FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_", ""
        )
        return FileSystemSnapshotBucketMode[name]
    except (ValueError, KeyError):
        logger.warning("Unknown snapshot bucket mode %s, treating as UNSPECIFIED", value)
        return FileSystemSnapshotBucketMode.UNSPECIFIED


def _proto_timestamp_to_datetime(message: Any, field_name: str) -> datetime | None:
    """Return a UTC datetime for a set proto Timestamp field, else None."""
    if not message.HasField(field_name):
        return None
    result = getattr(message, field_name).ToDatetime(tzinfo=UTC)
    return result if isinstance(result, datetime) else None


def _snapshot_from_proto(proto: sandbox_pb2.FileSystemSnapshot) -> FileSystemSnapshot:
    """Convert a proto FileSystemSnapshot to the SDK dataclass."""
    return FileSystemSnapshot(
        file_system_snapshot_id=proto.file_system_snapshot_id,
        status=_fss_status_from_proto(proto.state),
        status_reason=proto.state_reason,
        size_bytes=proto.size_bytes,
        source_sandbox_id=proto.source_sandbox_id,
        trigger=_fss_trigger_from_proto(proto.trigger),
        request_id=proto.request_id,
        object_bucket=proto.object_bucket,
        source_volume_name=proto.source_volume_name,
        created_at=_proto_timestamp_to_datetime(proto, "create_time"),
        updated_at=_proto_timestamp_to_datetime(proto, "updated_at"),
        completed_at=_proto_timestamp_to_datetime(proto, "complete_time"),
    )


def _bucket_config_from_proto(
    proto: settings_pb2.FileSystemSnapshotBucketConfig,
) -> FileSystemSnapshotBucketConfig:
    """Convert a proto FileSystemSnapshotBucketConfig to the SDK dataclass."""
    return FileSystemSnapshotBucketConfig(
        mode=_fss_bucket_mode_from_proto(proto.mode),
        bucket_name=proto.bucket_name,
        region=proto.region,
        effective_bucket_name=proto.effective_bucket_name,
    )


@dataclass(frozen=True)
class _SandboxView:
    """Adapt a v1 Sandbox resource to poll/lifecycle field accessors."""

    _sandbox: sandbox_pb2.Sandbox

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.sandbox_id

    @property
    def sandbox_status(self) -> int:
        return self._sandbox.status.state

    @property
    def exit_code(self) -> int:
        return self._sandbox.status.exit_code

    @property
    def runner_id(self) -> str:
        return self._sandbox.status.runner_id

    @property
    def runner_group_id(self) -> str:
        return self._sandbox.status.runner_group_id

    @property
    def status_reason(self) -> str:
        return self._sandbox.status.state_reason

    @property
    def started_at_time(self) -> Any:
        if self._sandbox.status.HasField("start_time"):
            return self._sandbox.status.start_time
        return None

    def HasField(self, field_name: str) -> bool:
        if field_name == "exit_code":
            return self._sandbox.status.HasField("exit_code")
        if field_name in ("start_time", "create_time", "end_time", "started_at_time"):
            key = "start_time" if field_name == "started_at_time" else field_name
            return self._sandbox.status.HasField(key)
        if field_name in ("status", "spec"):
            return self._sandbox.HasField(field_name)
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)


def _as_sandbox_view(value: Any) -> _SandboxView:
    """Normalize Get/List sandbox messages to ``_SandboxView``.

    Duck-typed stand-ins used by unit tests (SimpleNamespace/MagicMock that
    already expose ``sandbox_status``) are returned unchanged.
    """
    if isinstance(value, _SandboxView):
        return value
    if isinstance(value, sandbox_pb2.Sandbox):
        return _SandboxView(value)
    # Non-proto stand-ins already expose the poll accessors.
    if hasattr(value, "sandbox_status"):
        return cast(_SandboxView, value)
    return _SandboxView(value)


DEFAULT_SCRATCH_VOLUME_NAME = "workspace"


def _coerce_file_system_snapshot(
    value: FileSystemSnapshotOptions | Mapping[str, Any] | None,
) -> FileSystemSnapshotOptions | None:
    """Coerce convenience FSS options (maps to a named scratch volume)."""
    if value is None:
        return None
    if isinstance(value, FileSystemSnapshotOptions):
        return value
    if isinstance(value, Mapping):
        return FileSystemSnapshotOptions(**value)
    raise TypeError(
        "file_system_snapshot must be FileSystemSnapshotOptions, dict, or None, "
        f"got {type(value).__name__}"
    )


def _scratch_from_fss_options(
    opts: FileSystemSnapshotOptions,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return (SandboxVolume kwargs, VolumeMount kwargs or None) for convenience FSS."""
    scratch: dict[str, Any] = {}
    if opts.size is not None:
        scratch["size"] = opts.size
    if opts.file_system_snapshot_id is not None:
        scratch["restore_from_snapshot_id"] = opts.file_system_snapshot_id
    volume = {"name": opts.name or DEFAULT_SCRATCH_VOLUME_NAME, "scratch": scratch}
    if not opts.mount_path:
        return volume, None
    mount = {"volume": volume["name"], "mount_path": opts.mount_path}
    return volume, mount


_CONTAINERS_EXCLUSIVE_KWARGS = (
    "container_image",
    "command",
    "args",
    "resources",
    "mounted_files",
    "secrets",
    "image_pull_credentials",
    "environment_variables",
)


def _normalize_container_target(container: str | None) -> str | None:
    if container is None:
        return None
    stripped = container.strip()
    return stripped or None


def _set_request_container(message: Any, container: str | None) -> None:
    target = _normalize_container_target(container)
    if target:
        message.container = target


def _has_compute_resources(resources: ResourceOptions | None) -> bool:
    return resources is not None and bool(resources.requests or resources.limits or resources.gpu)


def _validate_container_compute(containers: Sequence[Container]) -> None:
    count = len(containers)
    for row in containers:
        is_primary = row.primary or count == 1
        resources = normalize_resources(row.resources)
        if count > 1 and not _has_compute_resources(resources):
            label = row.name or "<unnamed>"
            raise ValueError(
                f"container {label!r} requires resources when more than one container is specified"
            )
        if resources is not None and resources.gpu and not is_primary:
            raise ValueError("GPU is only allowed on the primary container")


def _containers_conflict_message(conflicts: Sequence[str]) -> str:
    listed = ", ".join(conflicts)
    return f"containers= is mutually exclusive with single-container kwargs ({listed})"


async def _create_snapshot_via_stub(
    stub: sandbox_pb2_grpc.SandboxServiceStub,
    sandbox_id: str,
    *,
    request_id: str | None,
    wait_for_ready: bool,
    auth_metadata: tuple[tuple[str, str], ...],
    timeout: float,
    scratch_volume_name: str | None = None,
) -> str:
    """Call CreateFileSystemSnapshot on ``stub``; return the new snapshot ID.

    When ``wait_for_ready`` is True, poll GetFileSystemSnapshot until READY/FAILED.
    """
    request = sandbox_pb2.CreateFileSystemSnapshotRequest(sandbox_id=sandbox_id)
    if request_id:
        request.request_id = request_id
    if scratch_volume_name:
        request.scratch_volume_name = scratch_volume_name
    try:
        response = await stub.CreateFileSystemSnapshot(
            request, timeout=timeout, metadata=auth_metadata
        )
    except grpc.RpcError as e:
        raise _translate_rpc_error(
            e, sandbox_id=sandbox_id, operation="Create file-system snapshot"
        ) from e
    snapshot_id = str(response.file_system_snapshot_id)
    if not wait_for_ready:
        return snapshot_id

    await _wait_for_snapshot_via_stub(
        stub,
        snapshot_id,
        auth_metadata=auth_metadata,
        timeout=timeout,
    )
    return snapshot_id


async def _wait_for_snapshot_via_stub(
    stub: sandbox_pb2_grpc.SandboxServiceStub,
    snapshot_id: str,
    *,
    auth_metadata: tuple[tuple[str, str], ...],
    timeout: float,
) -> None:
    """Poll an existing snapshot until READY or FAILED."""
    deadline = time.monotonic() + max(0.0, timeout)
    interval = DEFAULT_POLL_INTERVAL_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SnapshotWaitTimeoutError(
                f"Timed out waiting for snapshot {snapshot_id} to become ready",
                file_system_snapshot_id=snapshot_id,
            )
        get_timeout = min(remaining, DEFAULT_POLL_RPC_TIMEOUT_SECONDS)

        async def _get_once(timeout: float = get_timeout) -> FileSystemSnapshot:
            return await _get_snapshot_via_stub(
                stub,
                snapshot_id,
                auth_metadata=auth_metadata,
                timeout=timeout,
            )

        snap = await _retry_transient_rpc(
            _get_once,
            budget_seconds=min(remaining, DEFAULT_FSS_RETRY_BUDGET_SECONDS),
            operation="Wait for file-system snapshot",
        )
        if snap.status == FileSystemSnapshotStatus.READY:
            return
        if snap.status == FileSystemSnapshotStatus.FAILED:
            raise SandboxSnapshotError(
                f"Snapshot {snapshot_id} failed: {snap.status_reason or 'unknown error'}",
                file_system_snapshot_id=snapshot_id,
            )
        await asyncio.sleep(interval)
        interval = min(interval * DEFAULT_POLL_BACKOFF_FACTOR, DEFAULT_MAX_POLL_INTERVAL_SECONDS)


async def _get_snapshot_via_stub(
    stub: sandbox_pb2_grpc.SandboxServiceStub,
    file_system_snapshot_id: str,
    *,
    auth_metadata: tuple[tuple[str, str], ...],
    timeout: float,
) -> FileSystemSnapshot:
    """Call GetFileSystemSnapshot on ``stub``; return the snapshot record."""
    request = sandbox_pb2.GetFileSystemSnapshotRequest(
        file_system_snapshot_id=file_system_snapshot_id
    )
    try:
        proto = await stub.GetFileSystemSnapshot(request, timeout=timeout, metadata=auth_metadata)
    except grpc.RpcError as e:
        raise _translate_rpc_error(
            e,
            operation="Get file-system snapshot",
            file_system_snapshot_id=file_system_snapshot_id,
        ) from e
    return _snapshot_from_proto(proto)


def _validate_cwd(cwd: str | None) -> None:
    """Validate cwd parameter for exec().

    Args:
        cwd: Working directory path to validate

    Raises:
        ValueError: If cwd is empty string or not an absolute path
    """
    if cwd is None:
        return
    if not cwd:
        raise ValueError("cwd cannot be empty string")
    if not cwd.startswith("/"):
        raise ValueError(f"cwd must be an absolute path, got: {cwd!r}")


def _coerce_bytes_chunk(chunk: Any) -> bytes:
    """Coerce a caller-supplied iterator chunk to bytes for gRPC transmission.

    Accepts bytes (identity), bytearray, or memoryview. Anything else raises
    TypeError; without this guard, ``bytes(int)`` would silently produce NUL
    padding and corrupt the upload.
    """
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, (bytearray, memoryview)):
        return bytes(chunk)
    raise TypeError(
        f"streaming source must yield bytes-like objects (bytes, bytearray, "
        f"or memoryview); got {type(chunk).__name__}"
    )


# Sentinel marking normal exhaustion of a synchronous source iterator across the
# executor boundary. ``StopIteration`` cannot propagate out of a Future, so the
# executor task returns this instead.
_SYNC_ITER_DONE = object()


def _next_coerced_chunk(iterator: Iterator[bytes]) -> bytes | object:
    """Pull and coerce the next chunk from a sync iterator (runs in an executor).

    Returns the coerced ``bytes`` chunk, or ``_SYNC_ITER_DONE`` on normal
    exhaustion. A non-bytes-like chunk raises ``TypeError`` (the documented
    write_file_streaming contract) right here in the worker thread; the Future
    carries it back so the caller re-raises it unchanged.
    """
    try:
        chunk = next(iterator)
    except StopIteration:
        return _SYNC_ITER_DONE
    return _coerce_bytes_chunk(chunk)


async def _iter_sync_source_in_executor(source: Iterable[bytes]) -> AsyncIterator[bytes]:
    """Yield from a *synchronous* iterable without blocking the event loop.

    A caller-supplied sync iterable (a file handle, an NFS/FUSE read, a network
    generator) can block on ``next()``; driving it inline on the shared
    background loop would stall every other operation: heartbeats, other
    sandboxes' RPCs. Each ``next()`` is instead run in the default executor, so
    the blocking call parks an executor thread rather than the event loop.

    No prefetch buffer is needed: the generator advances the iterator exactly
    one step per item the consumer pulls, so a slow downstream naturally paces a
    fast source and nothing runs ahead. The downstream exec stdin path applies
    its own bounded-queue backpressure. A non-bytes-like chunk raises
    ``TypeError`` from the executor, propagating out unchanged (the documented
    write_file_streaming contract).
    """
    loop = asyncio.get_running_loop()
    iterator = iter(source)
    while True:
        item = await loop.run_in_executor(None, _next_coerced_chunk, iterator)
        if item is _SYNC_ITER_DONE:
            return
        assert isinstance(item, bytes)
        yield item


def _wrap_command_with_cwd(command: Sequence[str], cwd: str) -> list[str]:
    """Wrap command with shell cd to change working directory.

    Args:
        command: Original command and arguments (must not be empty)
        cwd: Absolute path for working directory

    Returns:
        Wrapped command: ["/bin/sh", "-c", "cd /path && exec cmd arg1 arg2"]

    Raises:
        ValueError: If command is empty
    """
    if not command:
        raise ValueError("Command cannot be empty when wrapping with cwd")
    escaped_cwd = shlex.quote(cwd)
    escaped_command = " ".join(shlex.quote(arg) for arg in command)
    return ["/bin/sh", "-c", f"cd {escaped_cwd} && exec {escaped_command}"]


def _translate_snapshot_reason(
    reason: str,
    *,
    details: str,
    operation: str,
    file_system_snapshot_id: str | None,
    metadata: Mapping[str, str] | None,
    retry_delay: timedelta | None,
) -> CWSandboxError | None:
    """Map a trusted FSS ``CWSANDBOX_FSS_*`` reason to a typed exception.

    Returns ``None`` when ``reason`` is not a known FSS reason, so the caller
    can fall through to status-code mapping. The ``file_system_snapshot_id`` is attached
    only to ``SandboxSnapshotError`` variants; the transient and wait-timeout
    classes inherit non-snapshot parents and do not carry it.
    """
    snapshot_classes: dict[str, type[SandboxSnapshotError]] = {
        CWSANDBOX_FSS_NOT_FOUND: SnapshotNotFoundError,
        CWSANDBOX_FSS_NOT_READY: SnapshotNotReadyError,
        CWSANDBOX_FSS_NOT_SUPPORTED: SnapshotNotSupportedError,
        CWSANDBOX_FSS_SIZE_EXCEEDED: SnapshotSizeExceededError,
        CWSANDBOX_FSS_QUOTA_EXCEEDED: SnapshotQuotaExceededError,
        CWSANDBOX_FSS_BUCKET_MISMATCH: SnapshotBucketMismatchError,
    }
    cls = snapshot_classes.get(reason)
    if cls is not None:
        return cls(
            f"{operation} failed ({reason}): {details}",
            file_system_snapshot_id=file_system_snapshot_id,
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if reason in SNAPSHOT_INTERNAL_REASONS:
        return SandboxSnapshotError(
            f"{operation} failed ({reason}): {details}",
            file_system_snapshot_id=file_system_snapshot_id,
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if reason == CWSANDBOX_FSS_WAIT_TIMEOUT:
        return SnapshotWaitTimeoutError(
            f"{operation} timed out waiting for snapshot ready ({reason}): {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if reason in SNAPSHOT_TRANSIENT_REASONS:
        return SnapshotBackendThrottledError(
            f"Snapshot backend throttled ({reason}): {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    return None


def _translate_volume_reason(
    reason: str,
    *,
    details: str,
    operation: str,
    volume_id: str | None,
    metadata: Mapping[str, str] | None,
    retry_delay: timedelta | None,
) -> CWSandboxError | None:
    """Map a trusted ``CWSANDBOX_VOLUME_*`` reason to a typed exception."""
    volume_classes: dict[str, type[VolumeError]] = {
        CWSANDBOX_VOLUME_NOT_FOUND: VolumeNotFoundError,
        CWSANDBOX_VOLUME_NOT_READY: VolumeNotReadyError,
        CWSANDBOX_VOLUME_PLACEMENT_CONFLICT: VolumePlacementConflictError,
        CWSANDBOX_VOLUME_TYPE_NOT_SUPPORTED: VolumeTypeNotSupportedError,
        CWSANDBOX_VOLUME_NOT_SNAPSHOTTABLE: VolumeNotSnapshottableError,
        CWSANDBOX_VOLUME_RUNNER_INELIGIBLE: VolumeRunnerIneligibleError,
        CWSANDBOX_VOLUME_BACKEND_NOT_FOUND: VolumeBackendNotFoundError,
        CWSANDBOX_VOLUME_IN_USE: VolumeInUseError,
        CWSANDBOX_VOLUME_QUOTA_EXCEEDED: VolumeQuotaExceededError,
    }
    cls = volume_classes.get(reason)
    if cls is not None:
        return cls(
            f"{operation} failed ({reason}): {details}",
            volume_id=volume_id,
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if reason == CWSANDBOX_VOLUME_RUNNER_UNAVAILABLE:
        return VolumeRunnerUnavailableError(
            f"{operation} failed ({reason}): {details}",
            volume_id=volume_id,
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    return None


def _is_runner_shard_retiring_error(error: grpc.RpcError) -> bool:
    parsed = parse_error_info(error)
    return (
        parsed is not None
        and parsed.domain == CWSANDBOX_ERROR_DOMAIN
        and parsed.reason == CWSANDBOX_RUNNER_SHARD_RETIRING
    )


def _is_unavailable_rpc_error(error: grpc.RpcError) -> bool:
    return error.code() == grpc.StatusCode.UNAVAILABLE


def _is_log_session_wrong_shard_error(error: grpc.RpcError) -> bool:
    parsed = parse_error_info(error)
    return (
        parsed is not None
        and parsed.domain == CWSANDBOX_ERROR_DOMAIN
        and parsed.reason == CWSANDBOX_BACKEND_UNAVAILABLE
        and bool(parsed.metadata.get("owner_shard_id"))
        and bool(parsed.metadata.get("local_shard_id"))
    )


def _translate_rpc_error(
    e: grpc.RpcError,
    *,
    sandbox_id: str | None = None,
    operation: str = "operation",
    filepath: str | None = None,
    file_system_snapshot_id: str | None = None,
    volume_id: str | None = None,
) -> CWSandboxError:
    """Translate gRPC RpcError to appropriate CWSandbox exception.

    Resolves the exception class in this priority order:

    1. If AIP-193 ``ErrorInfo`` is present with a matching ``domain`` and the
       ``reason`` matches a known ``CWSANDBOX_*`` string, use the reason-
       specific mapping (file ops, sandbox-not-found, timeout, unavailable).
       Reason mapping only applies when the ErrorInfo ``domain`` matches
       ``CWSANDBOX_ERROR_DOMAIN``. This is a namespace gate, not a peer-
       identity check: any peer with a valid TLS certificate can set the
       domain field, but distinct AIP-193 services (``google.rpc.*``,
       third-party sidecars, service-mesh-injected details) typically use
       their own domain, so the gate prevents accidental collisions with
       reason strings emitted by other AIP-193 services in the gRPC pipe.
       Peer-identity trust comes from the TLS trust chain, not this check.
    2. Otherwise fall through to gRPC status code mapping (NOT_FOUND,
       CANCELLED, DEADLINE_EXCEEDED, UNAVAILABLE).
    3. Otherwise delegate to the shared transport-level translator.

    Any parsed ``ErrorInfo`` / ``RetryInfo`` fields are attached to the
    returned exception regardless of which branch picks the class - the
    ``reason`` attribute is populated even when the domain does not match,
    so callers that want to inspect raw server metadata still can.

    For ``SandboxFileError``, ``filepath`` is resolved from the caller's
    ``filepath`` kwarg first, then falls back to
    ``ErrorInfo.metadata["filepath"]`` if the backend provided one. The
    explicit kwarg always wins so client-local context survives even
    when the backend drops metadata.

    Args:
        e: The gRPC RpcError to translate.
        sandbox_id: Optional sandbox ID for context in error messages.
        operation: Description of the operation that failed.
        filepath: Optional file path for file-op callers; used as fallback
            target for ``SandboxFileError.filepath``.

    Returns:
        An appropriate CWSandbox exception.
    """
    code = e.code()
    details = e.details() or str(e)
    parsed = parse_error_info(e)
    reason = parsed.reason if parsed is not None else None
    metadata = parsed.metadata if parsed is not None else None
    retry_delay = parsed.retry_delay if parsed is not None else None
    domain_trusted = parsed is not None and parsed.domain == CWSANDBOX_ERROR_DOMAIN

    if domain_trusted and reason is not None:
        parsed_metadata = parsed.metadata if parsed is not None else {}
        if reason in FILE_ERROR_REASONS:
            effective_filepath = (
                filepath if filepath is not None else parsed_metadata.get("filepath")
            )
            return SandboxFileError(
                f"File operation failed ({reason}): {details}",
                filepath=effective_filepath,
                reason=reason,
                metadata=metadata,
                retry_delay=retry_delay,
            )
        if reason == CWSANDBOX_SANDBOX_NOT_FOUND:
            return SandboxNotFoundError(
                f"Sandbox '{sandbox_id}' not found" if sandbox_id else details,
                sandbox_id=sandbox_id,
                reason=reason,
                metadata=metadata,
                retry_delay=retry_delay,
            )
        if reason == CWSANDBOX_COMMAND_TIMEOUT:
            return SandboxCommandTimeoutError(
                f"{operation} timed out: {details}",
                reason=reason,
                metadata=metadata,
                retry_delay=retry_delay,
            )
        if reason in UNAVAILABLE_REASONS:
            return SandboxUnavailableError(
                f"Service unavailable: {details}",
                reason=reason,
                metadata=metadata,
                retry_delay=retry_delay,
            )
        # File-system snapshot (FSS) reasons. The transient ones subclass
        # SandboxUnavailableError so the poll loop treats them as retryable;
        # the rest are terminal SandboxSnapshotError variants.
        snapshot_exc = _translate_snapshot_reason(
            reason,
            details=details,
            operation=operation,
            file_system_snapshot_id=file_system_snapshot_id,
            metadata=metadata,
            retry_delay=retry_delay,
        )
        if snapshot_exc is not None:
            return snapshot_exc
        volume_exc = _translate_volume_reason(
            reason,
            details=details,
            operation=operation,
            volume_id=volume_id or (parsed_metadata.get("volume_id") if parsed_metadata else None),
            metadata=metadata,
            retry_delay=retry_delay,
        )
        if volume_exc is not None:
            return volume_exc

    if code == grpc.StatusCode.NOT_FOUND:
        # An FSS operation carries a snapshot ID, not a sandbox ID. Map a bare
        # NOT_FOUND (no AIP-193 FSS reason, e.g. an older backend or a proxy that
        # dropped the metadata) to the documented SnapshotNotFoundError so callers
        # catching it still work.
        if file_system_snapshot_id is not None:
            return SnapshotNotFoundError(
                f"File-system snapshot '{file_system_snapshot_id}' not found",
                file_system_snapshot_id=file_system_snapshot_id,
                reason=reason,
                metadata=metadata,
                retry_delay=retry_delay,
            )
        if volume_id is not None:
            return VolumeNotFoundError(
                f"Volume '{volume_id}' not found",
                volume_id=volume_id,
                reason=reason,
                metadata=metadata,
                retry_delay=retry_delay,
            )
        return SandboxNotFoundError(
            f"Sandbox '{sandbox_id}' not found" if sandbox_id else details,
            sandbox_id=sandbox_id,
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if code == grpc.StatusCode.CANCELLED:
        return SandboxNotRunningError(
            f"{operation} was cancelled"
            + (f" (sandbox {sandbox_id} connection closed)" if sandbox_id else ""),
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return SandboxRequestTimeoutError(
            f"{operation} timed out: {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if code == grpc.StatusCode.UNAVAILABLE:
        return SandboxUnavailableError(
            f"Service unavailable: {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return SandboxResourceExhaustedError(
            f"{operation} resource exhausted: {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    return translate_grpc_error(
        e,
        operation=operation,
        fallback_cls=SandboxError,
        parsed=parsed,
    )


def _normalize_placement_spillover(
    value: PlacementSpillover | str | None,
) -> PlacementSpillover:
    """Normalize spillover config; ``None`` means ``STRICT``."""
    if value is None:
        return PlacementSpillover.STRICT
    if isinstance(value, str):
        return PlacementSpillover(value.lower())
    return value


def _resolve_placement_for_spillover(
    placement_mode: PlacementMode | None,
    spillover: PlacementSpillover,
    *,
    from_template: bool,
) -> PlacementMode | None:
    """Validate spillover against placement_mode/template; resolve attempt-1 mode.

    For non-``STRICT`` spillover, an unset/unspecified primary is filled in as
    the spillover's first mode (CKS or serverless). Templates only allow
    ``STRICT``.
    """
    if from_template and spillover != PlacementSpillover.STRICT:
        raise ValueError(
            f"placement_spillover must be STRICT for template sandboxes (got {spillover.value!r})"
        )
    if spillover == PlacementSpillover.STRICT:
        return placement_mode

    primary = placement_mode
    if primary == PlacementMode.UNSPECIFIED:
        primary = None

    if spillover == PlacementSpillover.CKS_THEN_SERVERLESS:
        if primary == PlacementMode.SERVERLESS:
            raise ValueError(
                "placement_spillover='cks_then_serverless' requires "
                "placement_mode=cks (or unset); got serverless"
            )
        return PlacementMode.CKS if primary is None else primary

    if spillover == PlacementSpillover.SERVERLESS_THEN_CKS:
        if primary == PlacementMode.CKS:
            raise ValueError(
                "placement_spillover='serverless_then_cks' requires "
                "placement_mode=serverless (or unset); got cks"
            )
        return PlacementMode.SERVERLESS if primary is None else primary

    return placement_mode


def _is_spillover_eligible(exc: Exception) -> bool:
    """True when a CreateSandbox failure may trigger one alternate-mode retry.

    Spillable when the primary mode cannot satisfy the request, identified
    by AIP-193 reasons in ``SPILLOVER_ELIGIBLE_REASONS``. Never spills on
    serverless product gates, auth, ``INVALID_ARGUMENT``, or bare
    ``RESOURCE_EXHAUSTED`` without a recognized reason.
    """
    reason = getattr(exc, "reason", None)
    if reason in SPILLOVER_BLOCKED_REASONS:
        return False
    return reason in SPILLOVER_ELIGIBLE_REASONS


def _create_attempt_definitely_rejected(exc: Exception) -> bool:
    """True when a CreateSandbox error means the server did not commit a sandbox.

    Restore the original request id only for an allowlisted reject. Transport
    failures (UNAVAILABLE, DEADLINE_EXCEEDED, INTERNAL, UNKNOWN, CANCELLED)
    and bare ``SandboxError`` without a reason keep the spilled id so a later
    ``start()`` retries the same create.
    """
    if _is_spillover_eligible(exc):
        return True
    reason = getattr(exc, "reason", None)
    if reason in SPILLOVER_BLOCKED_REASONS:
        return True
    if isinstance(exc, (CWSandboxAuthenticationError, SandboxNotFoundError)):
        return True
    if isinstance(exc, (SandboxUnavailableError, SandboxTimeoutError, SandboxNotRunningError)):
        return False
    if isinstance(exc, SandboxResourceExhaustedError) and reason:
        return True
    if isinstance(exc, SandboxError) and reason:
        return True
    return False


def _volume_source_is_scratch(volume: sandbox_pb2.SandboxVolume) -> bool:
    """True when the volume is scratch (explicit or proto3 default).

    Proto3 optional oneof: an unset source serializes as scratch.
    ``HasField("scratch")`` is false for that default, so treat
    "not a named volume" as scratch.
    """
    return not volume.HasField("volume_id")


def _scratch_names_from_volumes(volumes: Sequence[Any]) -> tuple[str, ...]:
    """Collect scratch volume names; skip registered-volume mounts."""
    names: list[str] = []
    for vol in volumes:
        if isinstance(vol, RegisteredVolumeOptions):
            continue
        if isinstance(vol, ScratchVolumeOptions):
            names.append(vol.name)
            continue
        if isinstance(vol, Mapping):
            if "volume_id" in vol:
                continue
            name = vol.get("name", "")
            if name:
                names.append(str(name))
    return tuple(names)


_PollErrorClassification = Literal["retryable", "fatal"]


# Maximum time to honor for a server-hinted retry_delay (AIP-193 RetryInfo).
# Ensures one hinted sleep cannot consume the entire retry budget in a
# single sleep - the remaining budget is also a ceiling, so a misconfigured
# server emitting a large hint still only stalls the poll by at most
# min(hint, budget, 10s).
MAX_POLL_RETRY_HINTED_DELAY_SECONDS: float = 10.0

# Bounded retry budget for post-stop NOT_FOUND responses. The backend
# persists terminal state for stopped sandboxes, so Get should return
# COMPLETED or FAILED. NOT_FOUND here is expected only in a narrow race
# between the backend's terminal-state write and our next poll, or in
# backend-rollout skew; retrying briefly lets the backend converge. If
# NOT_FOUND persists past this budget, SandboxTerminalStateUnavailableError
# is raised so the caller sees the ambiguity explicitly.
NOT_FOUND_AFTER_STOP_RETRY_BUDGET_SECONDS: float = 2.0

# Bounded grace re-poll for COMPLETED responses that lack an exit code. The
# runner reports exit codes on a batched status flush (~5s cadence), so a Get
# can observe COMPLETED before the report carrying the code lands in the
# backend. Once the poll loop latches the terminal state, returncode is
# frozen, so take up to EXIT_CODE_GRACE_POLLS extra polls first. Bounded
# because "no code" is also a legitimate permanent state (older gateways,
# gateway-initiated stops, containers that never ran).
EXIT_CODE_GRACE_POLLS: int = 2
EXIT_CODE_GRACE_POLL_INTERVAL_SECONDS: float = 2.0
# Per-RPC timeout for the grace re-poll's single unretried Get. Deliberately
# short and separate from the primary poll's 15s/30s retry envelope: the
# grace poll is best-effort enrichment of an answer already in hand.
EXIT_CODE_GRACE_RPC_TIMEOUT_SECONDS: float = 2.0


_RETRYABLE_POLL_EXCEPTIONS: tuple[type[CWSandboxError], ...] = (
    SandboxUnavailableError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
)


# In-band error codes the server may emit on a streaming response.  These
# are application-level (the gRPC call itself succeeds); the codes describe
# server-side outcomes for the streaming session.  Mirrors the documented
# error contract in sandbox_pb2.pyi (LogStreamError.code).
#
# Per the wire contract every LogStreamError is terminal — the server will
# not send further frames on the same call.  The client's recovery action
# is dictated by the code:
#
#   SESSION_NOT_FOUND / REPLAY_GAP / RUNNER_UNAVAILABLE / RUNNER_DRAINING
#       reconnect with a FRESH init (no resume_session_id / resume_offset)
#       to pick up the live tail from the current head.
#   STREAM_INTERRUPTED
#       reconnect and RESUME (keep session_id / offset). The server still
#       holds the session; this is emitted on routine restarts/deploys.
#   INVALID_RESUME_OFFSET
#       terminal, no retry — the echoed offset is corrupt and reconnecting
#       with the same state would just reproduce the failure.
#   SANDBOX_NOT_FOUND / PERMISSION_DENIED / other unknown codes
#       terminal, no retry — surface to the caller as a SandboxError.
_STREAMING_SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
_STREAMING_REPLAY_GAP = "REPLAY_GAP"
_STREAMING_INVALID_RESUME_OFFSET = "INVALID_RESUME_OFFSET"
_STREAMING_RUNNER_UNAVAILABLE = "RUNNER_UNAVAILABLE"
_STREAMING_RUNNER_DRAINING = "RUNNER_DRAINING"
_STREAMING_INTERRUPTED = "STREAM_INTERRUPTED"

# Codes that the wire contract says are transient — the client should drop
# its resume state and reconnect with a fresh init.  Membership in this set
# is the only thing that controls fresh-reinit behavior; the dispatcher
# below treats every other documented code as terminal.
_STREAMING_FRESH_REINIT_CODES: frozenset[str] = frozenset(
    {
        _STREAMING_SESSION_NOT_FOUND,
        _STREAMING_REPLAY_GAP,
        _STREAMING_RUNNER_UNAVAILABLE,
        _STREAMING_RUNNER_DRAINING,
    }
)


def _exec_stream_error(message: str, code: str | None) -> SandboxExecutionError:
    """Build the typed exception for a terminal ``ExecStreamError``.

    ``STREAM_BACKPRESSURE`` means the output stream was ended early because it
    was not being read fast enough to keep up with the command's output, so
    some output was lost. Surface it as ``SandboxStreamBackpressureError`` (a
    subclass of ``SandboxExecutionError``) with guidance the caller can act on,
    rather than an opaque exec failure.

    ``STREAM_TRUNCATED`` means the command ran to completion but some of its
    output was lost in transit. Surface it as ``SandboxStreamTruncatedError``
    with guidance (use a file for large output; re-run only if idempotent).

    For both, the code is carried on ``.stream_code`` (a streaming-channel
    code), not ``.reason`` (the AIP-193 ErrorInfo namespace). Every other code
    stays a plain ``SandboxExecutionError`` carrying the raw ``reason``.
    """
    if code == STREAM_BACKPRESSURE:
        return SandboxStreamBackpressureError(
            "Output stream ended early because it was not being read fast "
            "enough to keep up with the command's output; some output was "
            "lost. If you do slow work between reads, move it off the read "
            "loop (drain into a fast local sink such as a file, then process "
            "afterward) and use read_file_streaming / write_file_streaming for "
            "large files. If the destination is itself slow (rate-limited API, "
            "slow disk) and cannot keep up no matter how tight the loop, split "
            "the work into smaller transfers. Retrying the same pattern will "
            "hit this again.",
            stream_code=code,
        )
    if code == STREAM_TRUNCATED:
        return SandboxStreamTruncatedError(
            "The command completed but some of its output was lost in transit, "
            "so the output you received is incomplete. For large output, write "
            "it to a file and retrieve the file (read_file_streaming) instead "
            "of streaming over stdout. Re-running may truncate again and may "
            "have side effects, so re-run only if the command is idempotent.",
            stream_code=code,
        )
    return SandboxExecutionError(
        f"Exec stream error: {message}",
        reason=code or None,
    )


# gRPC status codes that indicate a transient transport-level failure where
# a resume attempt makes sense.  DEADLINE_EXCEEDED is intentionally excluded
# — it reflects a real client timeout that the caller asked for, not a
# server-side blip.  CANCELLED is excluded because it is overwhelmingly a
# client- or server-initiated signal (sandbox.stop(), call.cancel() during
# shutdown, intentional teardown) — retrying it just burns the resume
# budget on a session that is being torn down on purpose.  NOT_FOUND /
# PERMISSION_DENIED / INVALID_ARGUMENT are terminal and must not be
# retried.
_STREAMING_RESUMABLE_STATUS_CODES: frozenset[grpc.StatusCode] = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNKNOWN,
    }
)


def _is_resumable_transport_error(exc: BaseException) -> bool:
    """Return True if a streaming gRPC error is worth attempting to resume.

    Only the transport-level codes that typically map to a gateway pod
    restart or a transient network blip qualify.  Anything else, including
    DEADLINE_EXCEEDED (caller-requested timeout), NOT_FOUND, PERMISSION_DENIED,
    and INVALID_ARGUMENT, propagates as-is.

    Note: this classifier intentionally diverges from ``_translate_rpc_error``,
    the canonical translator that consults AIP-193 ``ErrorInfo`` reasons
    (e.g. ``CWSANDBOX_*``) when deciding the typed-exception class.  The
    streaming retry loop runs on the hot path during transient gateway
    churn, so it dispatches on the raw ``grpc.StatusCode`` only and skips
    the metadata parse.  The current set of streaming errors the backend
    emits has no AIP-193 reason payload, so the simpler check is correct
    today.  If the backend ever attaches a ``CWSANDBOX_*`` reason to a
    streaming error (e.g. a hypothetical ``CWSANDBOX_RUNNER_TERMINATED``
    on ``INTERNAL``), this classifier should be updated to consult
    ``_translate_rpc_error`` first and dispatch on the resulting typed
    exception, matching the pattern in ``_classify_poll_error``.
    """
    if not isinstance(exc, grpc.aio.AioRpcError):
        return False
    return exc.code() in _STREAMING_RESUMABLE_STATUS_CODES


def _classify_poll_error(exc: CWSandboxError) -> _PollErrorClassification:
    """Classify a translated poll exception as retryable or fatal.

    NOT_FOUND is always fatal regardless of the reason or transport code
    that produced it - callers that receive ``SandboxNotFoundError`` have an
    authoritative "gone" signal and must not retry it at the poll level.
    """
    if isinstance(exc, SandboxNotFoundError):
        return "fatal"
    if isinstance(exc, _RETRYABLE_POLL_EXCEPTIONS):
        return "retryable"
    return "fatal"


_T = TypeVar("_T")


async def _retry_transient_rpc(
    attempt: Callable[[], Awaitable[_T]],
    *,
    budget_seconds: float,
    operation: str,
    non_retryable: tuple[type[CWSandboxError], ...] = (),
) -> _T:
    """Run ``attempt`` with bounded retry on transient CWSandbox errors.

    ``attempt`` performs exactly one RPC try and must raise a *translated*
    ``CWSandboxError`` on failure (i.e. wrap the stub call and
    ``_translate_rpc_error``). Only classes in ``_RETRYABLE_POLL_EXCEPTIONS``
    are retried - transient unavailability, request-deadline, resource
    exhaustion, and FSS backend-throttling (which subclasses
    ``SandboxUnavailableError``). Every other error, including ``NOT_FOUND``
    and ``FAILED_PRECONDITION``, is fatal and re-raised on the first attempt.

    ``non_retryable`` lists exception classes to treat as fatal even when they
    would otherwise be retryable. Use it when the per-attempt timeout *is* the
    operation's ceiling, so a deadline is the ceiling being hit rather than a
    transient blip - retrying would just re-spend the full (large) timeout and
    overrun the ceiling (the loop bounds the inter-attempt gap, not attempt
    duration).

    ``budget_seconds`` caps wall-clock time spent *retrying*; it never delays
    the first attempt. On exhaustion the last translated exception is re-raised
    unchanged. AIP-193 ``RetryInfo`` hints are honored; otherwise the backoff
    uses the same decorrelated jitter as the status-poll loop.
    """
    retry_deadline: float | None = None
    last_exc: CWSandboxError | None = None
    prev_sleep = DEFAULT_POLL_INTERVAL_SECONDS
    attempts = 0

    while True:
        try:
            return await attempt()
        except CWSandboxError as exc:
            last_exc = exc
            if (
                isinstance(exc, non_retryable)
                or _classify_poll_error(exc) != "retryable"
                or budget_seconds <= 0
            ):
                raise

            # First retryable failure starts the budget timer.
            if retry_deadline is None:
                retry_deadline = time.monotonic() + budget_seconds

            attempts += 1
            now = time.monotonic()
            if now >= retry_deadline:
                logger.debug(
                    "FSS retry budget exhausted for %s after %d attempt(s)",
                    operation,
                    attempts,
                )
                raise
            remaining = retry_deadline - now
            hinted_delay = exc.retry_delay.total_seconds() if exc.retry_delay else None
            if hinted_delay is not None and hinted_delay > 0:
                sleep_for = min(hinted_delay, remaining, MAX_POLL_RETRY_HINTED_DELAY_SECONDS)
            else:
                base = DEFAULT_POLL_INTERVAL_SECONDS
                cap = DEFAULT_MAX_POLL_INTERVAL_SECONDS
                jitter_ceiling = max(
                    base, min(cap, prev_sleep * DEFAULT_POLL_BACKOFF_FACTOR, remaining)
                )
                sleep_for = min(random.uniform(base, jitter_ceiling), remaining)
            logger.debug(
                "FSS retry for %s: sleep=%.2fs remaining=%.2fs attempt=%d",
                operation,
                sleep_for,
                remaining,
                attempts,
            )
        await asyncio.sleep(sleep_for)
        prev_sleep = sleep_for
        # A long hinted delay can exhaust the budget while we slept; re-raise
        # rather than issuing an attempt guaranteed to overrun the ceiling.
        assert retry_deadline is not None
        if time.monotonic() >= retry_deadline:
            assert last_exc is not None
            raise last_exc


# ---------------------------------------------------------------------------
# Lifecycle state types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NotStarted:
    cancelled: bool = False


@dataclass(frozen=True)
class _Starting:
    sandbox_id: str
    status: SandboxStatus = SandboxStatus.PENDING


@dataclass(frozen=True)
class _Running:
    sandbox_id: str
    status: SandboxStatus = SandboxStatus.RUNNING
    runner_id: str | None = None
    runner_group_id: str | None = None
    started_at: datetime | None = None


@dataclass(frozen=True)
class _Stopping:
    sandbox_id: str
    status: SandboxStatus = SandboxStatus.TERMINATING
    runner_id: str | None = None
    runner_group_id: str | None = None
    started_at: datetime | None = None


@dataclass(frozen=True)
class _Terminal:
    sandbox_id: str
    status: SandboxStatus
    returncode: int | None = None
    runner_id: str | None = None
    runner_group_id: str | None = None
    started_at: datetime | None = None


_LifecycleState = _NotStarted | _Starting | _Running | _Stopping | _Terminal


class _SandboxInfoLike(Protocol):
    """Structural type for protobuf sandbox info responses.

    Required fields are always present. Optional fields are guarded by
    hasattr/getattr in _apply_sandbox_info. The main-process exit code
    arrives as proto3 ``optional int32 exit_code``; its presence is checked
    via ``HasField("exit_code")`` (with a getattr fallback for non-proto
    stand-ins that lack presence tracking).
    """

    @property
    def sandbox_id(self) -> Any: ...
    @property
    def sandbox_status(self) -> Any: ...
    @property
    def exit_code(self) -> int: ...
    def HasField(self, field_name: str) -> bool: ...


_RUNNING_STATUSES = frozenset({SandboxStatus.RUNNING, SandboxStatus.PAUSED})
_TERMINAL_STATUSES = frozenset(
    {SandboxStatus.COMPLETED, SandboxStatus.FAILED, SandboxStatus.TERMINATED}
)


def _exit_code_from_info(info: _SandboxInfoLike) -> int | None:
    """Exit code from a sandbox info response, None when absent.

    The backend reports it as proto3 ``optional int32 exit_code``: HasField
    distinguishes "exited 0" from "not reported" (older gateways, gateway-
    initiated stops, and sandboxes whose container never ran omit the field
    entirely).
    """
    try:
        if info.HasField("exit_code"):
            return info.exit_code
        return None
    except (ValueError, AttributeError):
        # ValueError: proto message whose descriptor lacks the field (stubs
        # vendored before exit_code existed). AttributeError: non-proto
        # stand-ins (tests) without HasField.
        return getattr(info, "exit_code", None)


def _lifecycle_state_from_info(
    *,
    sandbox_id: str,
    status: SandboxStatus,
    runner_id: str | None = None,
    runner_group_id: str | None = None,
    started_at: datetime | None = None,
    returncode: int | None = None,
) -> _LifecycleState:
    """Build a lifecycle state from sandbox info fields.

    Used by _from_sandbox_info and _apply_sandbox_info (poll/query).
    """
    if status in _RUNNING_STATUSES:
        return _Running(
            sandbox_id=sandbox_id,
            status=status,
            runner_id=runner_id,
            runner_group_id=runner_group_id,
            started_at=started_at,
        )
    if status == SandboxStatus.TERMINATING:
        return _Stopping(
            sandbox_id=sandbox_id,
            status=status,
            runner_id=runner_id,
            runner_group_id=runner_group_id,
            started_at=started_at,
        )
    if status in _TERMINAL_STATUSES:
        return _Terminal(
            sandbox_id=sandbox_id,
            status=status,
            returncode=returncode,
            runner_id=runner_id,
            runner_group_id=runner_group_id,
            started_at=started_at,
        )
    return _Starting(sandbox_id=sandbox_id, status=status)


class Sandbox:
    """CWSandbox client with sync/async hybrid API.

    All methods return immediately and can be used in both sync and async contexts.
    Operations are executed in a background event loop managed by _LoopManager.

    Examples:
        Factory method:
        ```python
        sb = Sandbox.run("echo", "hello")  # Returns immediately
        result = sb.exec(["echo", "more"]).result()  # Block for result
        sb.stop().result()  # Block for completion
        ```

        Context manager (recommended):
        ```python
        with Sandbox.run("sleep", "infinity") as sb:
            result = sb.exec(["echo", "hello"]).result()
        # Automatically stopped on exit
        ```

        Async context manager:
        ```python
        async with Sandbox.run("sleep", "infinity") as sb:
            result = await sb.exec(["echo", "hello"])
        ```

    Attributes:
        sandbox_id: Unique identifier for this sandbox.
        status: Cached status from last API call.
        runner_id: Runner ID where sandbox is running.
        returncode: Exit code if sandbox completed.
        started_at: When sandbox started running.
        dns_egress_names: Hostnames granted at create (from status.effective_egress).
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        defaults: SandboxDefaults | None = None,
        auth: AuthConfig | None = None,
        container_image: str | None = None,
        tags: list[str] | None = None,
        base_url: str | None = None,
        request_timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
        max_lifetime_seconds: float | None = None,
        profile_ids: list[str] | None = None,
        profile_names: list[str] | None = None,
        runner_ids: list[str] | None = None,
        resources: ResourceOptions | dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        network: NetworkOptions | dict[str, Any] | None = None,
        file_system_snapshot: FileSystemSnapshotOptions | dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
        placement_mode: PlacementMode | str | None = None,
        placement_spillover: PlacementSpillover | str | None = None,
        services: list[Service] | tuple[Service, ...] | None = None,
        volumes: (
            list[ScratchVolumeOptions | RegisteredVolumeOptions | dict[str, Any]]
            | tuple[ScratchVolumeOptions | RegisteredVolumeOptions | dict[str, Any], ...]
            | None
        ) = None,
        template_id: str | None = None,
        image_pull_credentials: ImagePullCredentials | dict[str, Any] | None = None,
        runtime_class: str | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
        working_dir: str | None = None,
        object_storage_access: ObjectStorageAccess | dict[str, Any] | None = None,
        environment_variables: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        secrets: Sequence[Secret | dict[str, Any]] | None = None,
        data_plane_mode: DataPlaneMode | str | None = None,
        containers: Sequence[Container | Mapping[str, Any]] | None = None,
        _session: Session | None = None,
    ) -> None:
        """Initialize a sandbox (does not start it).

        Args:
            command: Optional command to run in the sandbox
            args: Optional arguments for the command
            defaults: Optional SandboxDefaults to apply
            auth: Authentication mode or provider. Explicit values override
                ``defaults.auth`` for this sandbox instance.
            container_image: Container image to use (default: python:3.11)
            tags: Optional tags for the sandbox
            base_url: API URL (default: CWSANDBOX_BASE_URL env or localhost)
            request_timeout_seconds: Timeout for API requests (client-side, default: 300s)
            poll_retry_budget_seconds: Wall-clock budget for retrying transient
                errors on the sandbox-status poll loop (default: 30s). Set to 0
                to disable retry.
            poll_rpc_timeout_seconds: Per-call timeout for poll Get RPCs
                (default: 15s). Separate from request_timeout_seconds.
            max_lifetime_seconds: Max sandbox lifetime (server-side). If not set,
                the backend controls the default.
            profile_ids: Removed in 1.x; passing a value raises ``TypeError``.
            profile_names: Removed in 1.x; passing a value raises ``TypeError``.
            runner_ids: Optional CKS runner pin (incompatible with serverless
                and with ``placement_spillover='serverless_then_cks'``).
            resources: Resource configuration. Accepts ResourceOptions for separate
                requests/limits, or a flat dict for backward-compatible Guaranteed QoS.
            mounted_files: Files to mount into the sandbox at startup. Each dict
                should have ``mount_path`` (str) and ``file_content`` (bytes).
                Note: Mounted files are read-only at runtime. To modify a file,
                use ``sandbox.write_file()`` after the sandbox is running.
            s3_mount: Removed in 1.x; passing a value raises ``TypeError``.
            ports: Removed in 1.x; use ``services=[Service(...)]`` instead.
            network: ``NetworkOptions`` (or dict) with deny flags and optional
                create-time hostname grants (``egress=[EgressRule(dns_name=...)]``).
                Port exposure uses ``services=``.
            placement_mode: ``PlacementMode`` or string (``serverless`` / ``cks``).
            placement_spillover: ``PlacementSpillover`` or string. Default
                ``strict`` (no create retry). Non-strict modes retry CreateSandbox
                once on the alternate mode when the primary cannot place the
                request. ``serverless_then_cks`` cannot be combined with
                ``runner_ids``. Template sandboxes require ``strict``.
            services: Typed service ports (``Service`` list/tuple).
            volumes: Scratch or registered volumes (``ScratchVolumeOptions``,
                ``RegisteredVolumeOptions``, or a ``volume_id`` dict).
            runtime_class: Optional runtime-class pin (e.g. ``"gvisor"``).
            security_context: In-guest privilege for the primary container.
            working_dir: Working directory for the primary container command.
            object_storage_access: Temporary object-storage credentials.
            file_system_snapshot: Convenience single-mount FSS options
                (``FileSystemSnapshotOptions`` or dict). Prefer ``volumes=`` for
                multi-volume setups. Requires the organization to be enabled for FSS.
            max_timeout_seconds: Removed in 1.x; use ``request_timeout_seconds``.
            environment_variables: Environment variables to inject into the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive config only.
            annotations: Kubernetes pod annotations for the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive metadata only.
            secrets: Secrets to inject as environment variables at create time.
                Merged with defaults (defaults first, then this list).
            data_plane_mode: Transport policy for exec, logs, and file operations.
                Defaults to ``SandboxDefaults.data_plane_mode``.
            containers: Multi-container spec. Mutually exclusive with
                ``container_image``, ``command``/``args``, ``resources``,
                ``mounted_files``, ``secrets``, ``image_pull_credentials``,
                ``environment_variables``, ``security_context``, and
                ``working_dir``. This list replaces those single-container
                fields, including the same names on ``SandboxDefaults``.
                Put secrets, env, and working_dir on each ``Container``.
        """
        if network is not None:
            if isinstance(network, dict):
                network = NetworkOptions(**network)
            elif not isinstance(network, NetworkOptions):
                raise TypeError(
                    f"network must be NetworkOptions, dict, or None, got {type(network).__name__}"
                )

        self._defaults = defaults or SandboxDefaults()
        self._session = _session
        self._auth = auth if auth is not None else self._defaults.auth

        from_template = template_id is not None
        effective_containers = (
            containers
            if containers is not None
            else (None if from_template else self._defaults.containers)
        )
        using_containers = effective_containers is not None
        if effective_containers is not None:
            user_containers = _validate_containers(
                tuple(_coerce_container(row) for row in effective_containers)
            )
            _validate_container_compute(user_containers)
            conflicts = [
                name
                for name, value in (
                    ("container_image", container_image),
                    ("command", command),
                    ("args", args),
                    ("resources", resources),
                    ("mounted_files", mounted_files),
                    ("secrets", secrets),
                    ("image_pull_credentials", image_pull_credentials),
                    ("environment_variables", environment_variables),
                    ("security_context", security_context),
                    ("working_dir", working_dir),
                )
                if value is not None
            ]
            if conflicts:
                raise TypeError(_containers_conflict_message(conflicts))
        else:
            user_containers = None

        # Template creates stay sparse for spec-owned fields (env, annotations,
        # command): SDK defaults would otherwise replace the template. Tags
        # still merge so session.list()/adopt can find the sandbox after a crash.
        if using_containers:
            self._command: str | None = None
            self._args: list[str] | None = None
            self._container_image: str | None = None
        else:
            self._command = command if from_template else command or self._defaults.command
            self._args = (
                args if args is not None else (None if from_template else list(self._defaults.args))
            )
            self._container_image = (
                container_image
                if from_template
                else container_image or self._defaults.container_image
            )
        self._base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or self._defaults.base_url
        ).rstrip("/")
        self._request_timeout_seconds = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else self._defaults.request_timeout_seconds
        )
        effective_data_plane_mode = (
            data_plane_mode if data_plane_mode is not None else self._defaults.data_plane_mode
        )
        self._data_plane_mode = (
            DataPlaneMode(effective_data_plane_mode.lower())
            if isinstance(effective_data_plane_mode, str)
            else effective_data_plane_mode
        )
        self._direct_data_plane = DirectDataPlaneClient()
        self._poll_retry_budget_seconds = (
            poll_retry_budget_seconds
            if poll_retry_budget_seconds is not None
            else self._defaults.poll_retry_budget_seconds
        )
        self._poll_rpc_timeout_seconds = (
            poll_rpc_timeout_seconds
            if poll_rpc_timeout_seconds is not None
            else self._defaults.poll_rpc_timeout_seconds
        )
        _validate_poll_config(
            self._poll_retry_budget_seconds,
            self._poll_rpc_timeout_seconds,
        )
        self._max_lifetime_seconds = max_lifetime_seconds
        if not from_template and self._max_lifetime_seconds is None:
            self._max_lifetime_seconds = self._defaults.max_lifetime_seconds

        self._tags: list[str] | None = self._defaults.merge_tags(tags)
        self._environment_variables = (
            {}
            if using_containers
            else (
                dict(environment_variables or {})
                if from_template
                else self._defaults.merge_environment_variables(environment_variables)
            )
        )
        self._annotations = (
            dict(annotations or {})
            if from_template
            else self._defaults.merge_annotations(annotations)
        )

        if profile_ids is not None or profile_names is not None:
            raise TypeError(
                "profile_ids/profile_names were removed in cwsandbox 1.x; "
                "use placement_mode, runner_ids, and discovery capabilities"
            )
        self._runner_ids = _resolve_selector(
            runner_ids, None if from_template else self._defaults.runner_ids
        )

        self._start_kwargs: dict[str, Any] = {}
        self._create_request_id: str | None = None
        self._placement_mode: PlacementMode | None = None
        self._placement_spillover: PlacementSpillover = PlacementSpillover.STRICT
        self._services: list[Service] | None = None
        self._template_id: str | None = None
        self._image_pull_credentials: ImagePullCredentials | None = None
        self._runtime_class: str | None = None
        self._security_context: SecurityContext | None = None
        self._working_dir: str | None = None
        self._object_storage_access: ObjectStorageAccess | None = None
        self._effective_runtime_class: str | None = None
        self._attached_volume_ids: tuple[str, ...] = ()
        self._effective_egress: tuple[EgressRule, ...] = ()
        self._effective_ingress: tuple[IngressRule, ...] = ()
        self._scratch_volume_names: tuple[str, ...] = ()
        self._service_urls: tuple[tuple[int, str, str], ...] = ()
        self._service_endpoints: tuple[HttpsEndpointStatus, ...] = ()
        self._dns_egress_names: tuple[str, ...] = ()
        self._file_system_snapshot_ids: tuple[str, ...] = ()
        self._spec_containers: tuple[Container, ...] = ()
        self._container_statuses: tuple[ContainerStatus, ...] = ()
        # Use explicit resources or fall back to defaults, then normalize
        effective_resources = (
            None
            if using_containers
            else (resources if resources is not None or from_template else self._defaults.resources)
        )
        normalized = normalize_resources(effective_resources)
        if normalized is not None:
            self._start_kwargs["resources"] = normalized
        if mounted_files is not None:
            self._start_kwargs["mounted_files"] = mounted_files
        if s3_mount is not None:
            raise TypeError("s3_mount was removed in cwsandbox 1.x")
        if ports is not None:
            raise TypeError("ports was removed in cwsandbox 1.x")
        # Use explicit network or fall back to defaults
        effective_network = (
            network if network is not None or from_template else self._defaults.network
        )
        if effective_network is not None:
            self._start_kwargs["network"] = effective_network
        # Use explicit file-system snapshot mount or fall back to defaults.
        effective_fss = (
            file_system_snapshot
            if file_system_snapshot is not None
            else (None if from_template else self._defaults.file_system_snapshot)
        )
        effective_fss = _coerce_file_system_snapshot(effective_fss)
        if effective_fss is not None:
            self._start_kwargs["file_system_snapshot"] = effective_fss
        if max_timeout_seconds is not None:
            raise TypeError(
                "max_timeout_seconds was removed in cwsandbox 1.x; use request_timeout_seconds"
            )
        if placement_mode is not None:
            if isinstance(placement_mode, str):
                placement_mode = PlacementMode(placement_mode.lower())
            self._placement_mode = placement_mode
        elif not from_template and self._defaults.placement_mode is not None:
            pm = self._defaults.placement_mode
            self._placement_mode = PlacementMode(pm.lower()) if isinstance(pm, str) else pm
        if placement_spillover is not None:
            self._placement_spillover = _normalize_placement_spillover(placement_spillover)
        elif not from_template:
            self._placement_spillover = _normalize_placement_spillover(
                self._defaults.placement_spillover
            )
        else:
            self._placement_spillover = PlacementSpillover.STRICT
        self._placement_mode = _resolve_placement_for_spillover(
            self._placement_mode,
            self._placement_spillover,
            from_template=from_template,
        )
        if self._placement_spillover == PlacementSpillover.SERVERLESS_THEN_CKS and self._runner_ids:
            raise ValueError(
                "placement_spillover='serverless_then_cks' cannot be combined with "
                "runner_ids: the first create attempt is serverless, which rejects "
                "runner pins. Use placement_spillover='cks_then_serverless' or omit "
                "runner_ids."
            )
        if services is not None:
            self._services = [Service(**s) if isinstance(s, dict) else s for s in services]
        elif not from_template and self._defaults.services is not None:
            self._services = list(self._defaults.services)
        if volumes is not None:
            self._start_kwargs["volumes"] = list(volumes)
            self._scratch_volume_names = _scratch_names_from_volumes(volumes)
        elif not from_template and self._defaults.volumes is not None:
            self._start_kwargs["volumes"] = list(self._defaults.volumes)
            self._scratch_volume_names = _scratch_names_from_volumes(self._defaults.volumes)
        if template_id is not None:
            self._template_id = template_id
        if image_pull_credentials is not None:
            if isinstance(image_pull_credentials, dict):
                image_pull_credentials = ImagePullCredentials(**image_pull_credentials)
            self._image_pull_credentials = image_pull_credentials
        effective_runtime_class = (
            runtime_class
            if runtime_class is not None or from_template
            else self._defaults.runtime_class
        )
        self._runtime_class = effective_runtime_class or None
        effective_security_context = (
            security_context
            if security_context is not None or from_template
            else (None if using_containers else self._defaults.security_context)
        )
        self._security_context = coerce_security_context(effective_security_context)
        effective_working_dir = (
            working_dir
            if working_dir is not None or from_template
            else (None if using_containers else self._defaults.working_dir)
        )
        self._working_dir = effective_working_dir or None
        effective_osa = (
            object_storage_access
            if object_storage_access is not None or from_template
            else self._defaults.object_storage_access
        )
        self._object_storage_access = _coerce_object_storage_access(effective_osa)
        if using_containers:
            assert user_containers is not None
            self._start_kwargs["containers"] = list(user_containers)
        inherited_secrets: list[Secret] = (
            [] if from_template or using_containers else list(self._defaults.secrets or ())
        )
        merged_secrets = inherited_secrets + [
            Secret(**s) if isinstance(s, dict) else s for s in (secrets or ())
        ]
        if merged_secrets:
            self._start_kwargs["secrets"] = list(_unique_secrets_by_env_var(merged_secrets))

        self._channel: grpc.aio.Channel | None = None
        self._stub: sandbox_pb2_grpc.SandboxServiceStub | None = None
        self._auth_metadata: tuple[tuple[str, str], ...] = ()
        self._auth_metadata_resolved = False
        self._streaming_channel: grpc.aio.Channel | None = None
        self._streaming_channel_lock = asyncio.Lock()
        self._sandbox_id: str | None = None
        self._start_lock = asyncio.Lock()

        # Updated when the server reports CWSANDBOX_FILE_TOO_LARGE with
        # max_size_bytes; lets the client use the cluster's actual cap on
        # subsequent file operations.
        self._observed_file_op_cap_bytes: int | None = None
        self._streaming_fallback_warned: bool = False

        self._state: _LifecycleState = _NotStarted()

        # Shared polling task for _wait_until_running_async deduplication
        self._running_task: asyncio.Task[None] | None = None
        self._running_lock = asyncio.Lock()

        # Shared polling task for _wait_until_complete_async deduplication
        self._complete_task: asyncio.Task[SandboxStatus] | None = None
        self._complete_lock = asyncio.Lock()

        # Terminal response held by an in-flight exit-code grace re-poll.
        # The grace window defers the terminal latch, so a waiter whose
        # deadline expires mid-window latches this instead of raising a
        # spurious SandboxTimeoutError for an already-observed completion.
        self._grace_pending_response: _SandboxView | None = None

        # Shared stop task so repeated stop() calls join the same operation
        self._stop_task: asyncio.Task[None] | None = None
        self._stop_lock = asyncio.Lock()
        self._stop_owned: bool = False
        # Whether the in-flight shared stop task (when _stop_task is set) was
        # created for a snapshot-on-stop. Read under _stop_lock to decide
        # whether a later snapshot-on-stop caller can safely join it.
        self._stop_snapshot_requested: bool = False
        # Set when a caller invokes stop(missing_ok=True) on a sandbox that
        # is already draining (observe-only path). Widens the NOT_FOUND
        # retry gate in _do_poll_complete so the observe-only waiter treats
        # NOT_FOUND as a backend race (retry briefly for authoritative
        # terminal state) rather than propagating SandboxNotFoundError.
        self._missing_ok_observe: bool = False

        self._status_updated_at: datetime | None = None
        self._exposed_ports: tuple[tuple[int, str], ...] | None = None
        self._resource_limits: dict[str, str] | None = None
        self._resource_requests: dict[str, str] | None = None
        self._resource_gpu: dict[str, Any] | None = None
        # Snapshot ID produced by stop(snapshot_on_stop=True), set when the
        # Stop response reports it. None until then.
        self._file_system_snapshot_id: str | None = None

        # Execution statistics for metrics (protected by _exec_stats_lock)
        self._exec_stats_lock = threading.Lock()
        self._exec_count = 0
        self._exec_completed_ok = 0
        self._exec_completed_nonzero = 0
        self._exec_failures = 0

        # Startup timing for metrics
        self._start_accepted_at: float | None = None
        self._startup_recorded: bool = False

        # Get the singleton loop manager for sync/async bridging
        self._loop_manager = _LoopManager.get()

    @classmethod
    def run(
        cls,
        *args: str,
        container_image: str | None = None,
        defaults: SandboxDefaults | None = None,
        auth: AuthConfig | None = None,
        request_timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
        max_lifetime_seconds: float | None = None,
        tags: list[str] | None = None,
        profile_ids: list[str] | None = None,
        profile_names: list[str] | None = None,
        runner_ids: list[str] | None = None,
        resources: ResourceOptions | dict[str, Any] | None = None,
        mounted_files: list[dict[str, Any]] | None = None,
        s3_mount: dict[str, Any] | None = None,
        ports: list[dict[str, Any]] | None = None,
        network: NetworkOptions | dict[str, Any] | None = None,
        file_system_snapshot: FileSystemSnapshotOptions | dict[str, Any] | None = None,
        max_timeout_seconds: int | None = None,
        placement_mode: PlacementMode | str | None = None,
        placement_spillover: PlacementSpillover | str | None = None,
        services: list[Service] | tuple[Service, ...] | None = None,
        volumes: (
            list[ScratchVolumeOptions | RegisteredVolumeOptions | dict[str, Any]]
            | tuple[ScratchVolumeOptions | RegisteredVolumeOptions | dict[str, Any], ...]
            | None
        ) = None,
        template_id: str | None = None,
        image_pull_credentials: ImagePullCredentials | dict[str, Any] | None = None,
        runtime_class: str | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
        working_dir: str | None = None,
        object_storage_access: ObjectStorageAccess | dict[str, Any] | None = None,
        environment_variables: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        secrets: Sequence[Secret | dict[str, Any]] | None = None,
        data_plane_mode: DataPlaneMode | str | None = None,
        containers: Sequence[Container | Mapping[str, Any]] | None = None,
    ) -> Sandbox:
        """Create and start a sandbox, return immediately once backend accepts.

        Does NOT wait for RUNNING status. Use .wait() to block until ready.
        If positional args are provided, the first is the command and the rest
        are its arguments. If no args are provided, uses a shell-trapped
        keep-alive default that responds to SIGTERM on stop.

        Args:
            *args: Optional command and arguments (e.g., "echo", "hello", "world").
                If omitted, uses default command from SandboxDefaults.
            container_image: Container image to use
            defaults: Optional SandboxDefaults to apply
            auth: Authentication mode or provider. Overrides ``defaults.auth``.
            request_timeout_seconds: Timeout for API requests (client-side)
            poll_retry_budget_seconds: Wall-clock budget for retrying transient
                errors on the sandbox-status poll loop (default: 30s). Set to
                0 to disable retry.
            poll_rpc_timeout_seconds: Per-call timeout for poll Get RPCs
                (default: 15s). Separate from request_timeout_seconds.
            max_lifetime_seconds: Max sandbox lifetime (server-side)
            tags: Optional tags for the sandbox
            profile_ids: Removed in 1.x; passing a value raises ``TypeError``.
            profile_names: Removed in 1.x; passing a value raises ``TypeError``.
            runner_ids: Optional CKS runner pin (incompatible with serverless
                and with ``placement_spillover='serverless_then_cks'``).
            resources: Resource configuration. Accepts ResourceOptions for separate
                requests/limits, or a flat dict for backward-compatible Guaranteed QoS.
            mounted_files: Files to mount into the sandbox at startup. Each dict
                should have ``mount_path`` (str) and ``file_content`` (bytes).
                Note: Mounted files are read-only at runtime. To modify a file,
                use ``sandbox.write_file()`` after the sandbox is running.
            s3_mount: Removed in 1.x; passing a value raises ``TypeError``.
            ports: Removed in 1.x; use ``services=[Service(...)]`` instead.
            network: ``NetworkOptions`` (or dict) with deny flags and optional
                create-time hostname grants (``egress=[EgressRule(dns_name=...)]``).
                Port exposure uses ``services=``.
            placement_mode: ``PlacementMode`` or string (``serverless`` / ``cks``).
            placement_spillover: ``PlacementSpillover`` or string. Default
                ``strict``. See ``Sandbox.__init__``.
            services: Typed service ports (``Service`` list/tuple).
            volumes: Scratch or registered volumes (``ScratchVolumeOptions``,
                ``RegisteredVolumeOptions``, or a ``volume_id`` dict).
            runtime_class: Optional runtime-class pin (e.g. ``"gvisor"``).
            security_context: In-guest privilege for the primary container.
            working_dir: Working directory for the primary container command.
            object_storage_access: Temporary object-storage credentials.
            file_system_snapshot: Convenience single-mount FSS options
                (``FileSystemSnapshotOptions`` or dict). Prefer ``volumes=`` for
                multi-volume setups.
            max_timeout_seconds: Removed in 1.x; use ``request_timeout_seconds``.
            environment_variables: Environment variables to inject into the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive config only.
            annotations: Kubernetes pod annotations for the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive metadata only.
            secrets: Secrets to inject as environment variables at create time.
                Merged with defaults (defaults first, then this list).
            data_plane_mode: Transport policy for exec, logs, and file operations.
                ``auto`` prefers direct mTLS with gateway fallback.
            containers: Multi-container spec. Mutually exclusive with
                positional command/args and with ``container_image``,
                ``resources``, ``mounted_files``, ``secrets``,
                ``image_pull_credentials``, ``environment_variables``,
                ``security_context``, and ``working_dir``. This list
                replaces those single-container fields, including the
                same names on ``SandboxDefaults``. Put secrets, env,
                and working_dir on each ``Container``.
        Returns:
            A Sandbox instance (start request sent, but may still be starting)

        Examples:
            ```python
            # Using defaults (shell-trapped keep-alive)
            sb = Sandbox.run()

            # Fire and forget style
            sb = Sandbox.run("echo", "hello")
            # sb.sandbox_id is set, but sandbox may still be starting

            # Wait for ready if needed
            sb = Sandbox.run("sleep", "infinity").wait()
            result = sb.exec(["echo", "hello"]).result()

            # Or use context manager for automatic cleanup
            with Sandbox.run("sleep", "infinity") as sb:
                result = sb.exec(["echo", "hello"]).result()
            ```
        """
        if network is not None:
            if isinstance(network, dict):
                network = NetworkOptions(**network)
            elif not isinstance(network, NetworkOptions):
                raise TypeError(
                    f"network must be NetworkOptions, dict, or None, got {type(network).__name__}"
                )

        command = args[0] if args else None
        cmd_args = list(args[1:]) if len(args) > 1 else None

        sandbox = cls(
            command=command,
            args=cmd_args,
            container_image=container_image,
            defaults=defaults,
            auth=auth,
            request_timeout_seconds=request_timeout_seconds,
            poll_retry_budget_seconds=poll_retry_budget_seconds,
            poll_rpc_timeout_seconds=poll_rpc_timeout_seconds,
            max_lifetime_seconds=max_lifetime_seconds,
            tags=tags,
            profile_ids=profile_ids,
            profile_names=profile_names,
            runner_ids=runner_ids,
            resources=resources,
            mounted_files=mounted_files,
            s3_mount=s3_mount,
            ports=ports,
            network=network,
            file_system_snapshot=file_system_snapshot,
            max_timeout_seconds=max_timeout_seconds,
            placement_mode=placement_mode,
            placement_spillover=placement_spillover,
            services=services,
            volumes=volumes,
            template_id=template_id,
            image_pull_credentials=image_pull_credentials,
            runtime_class=runtime_class,
            security_context=security_context,
            working_dir=working_dir,
            object_storage_access=object_storage_access,
            environment_variables=environment_variables,
            annotations=annotations,
            secrets=secrets,
            data_plane_mode=data_plane_mode,
            containers=containers,
        )
        logger.debug("Creating sandbox with command: %s", command)
        sandbox.start().result()
        return sandbox

    @classmethod
    def run_from_template(
        cls,
        template_id: str,
        /,
        *args: str,
        command: str | None = None,
        defaults: SandboxDefaults | None = None,
        **kwargs: Any,
    ) -> Sandbox:
        """Create a sandbox from an organization template (CreateSandboxFromTemplate).

        Args:
            template_id: Organization-scoped template UUID.
            *args: Optional command args override. Honored without ``command``.
                Sparse single-container overlays still require ``container_image``.
            command: Optional command override. Requires ``container_image``
                unless ``containers=`` replaces the whole list.
            defaults: Optional SandboxDefaults.
            **kwargs: Same advanced create kwargs as ``run()``. Passing
                ``container_image`` replaces the whole template container
                (command, args, env, files, resources, name ``main``); there
                is no fetch-and-merge. ``containers=`` is a full list replace
                and does not require ``container_image``. Other container-field
                overlays (``command``, ``args``, ``environment_variables``,
                ``secrets``, ``resources``, ``mounted_files``, ``volumes``,
                ``file_system_snapshot``, ``image_pull_credentials``,
                ``security_context``, ``working_dir``) require
                ``container_image`` unless ``containers=`` replaces the list:
                the API replaces the whole container list and rejects a
                sparse patch. Session/default tags are merged and sent as a
                replace-on-presence override so ``list()``/``adopt`` can find
                the sandbox; environment variables and annotations are not
                merged (template-owned).

        Returns:
            Sandbox handle (lazy-start).
        """
        if not template_id:
            raise ValueError("template_id must not be empty")
        kwargs = dict(kwargs)
        kwargs["template_id"] = template_id
        # Do not route args-only overrides through run(*args): run() treats the
        # first positional as the command, which would replace the template
        # command with an argument value.
        if command is not None:
            return cls.run(command, *args, defaults=defaults, **kwargs)
        sandbox = cls(
            command=None,
            args=list(args) if args else None,
            defaults=defaults,
            **kwargs,
        )
        sandbox.start().result()
        return sandbox

    @classmethod
    def session(
        cls,
        defaults: SandboxDefaults | Mapping[str, Any] | None = None,
        *,
        auth: AuthConfig | None = None,
    ) -> Session:
        """Create a session for managing multiple sandboxes.

        Sessions provide:
        - Shared configuration via defaults
        - Automatic cleanup of orphaned sandboxes
        - Function execution via @session.function() decorator

        Args:
            defaults: Optional defaults to apply to sandboxes created via session
            auth: Authentication strategy, resolved headers, or provider. Overrides
                ``defaults.auth`` when provided.

        Returns:
            A Session instance

        Examples:
            ```python
            session = Sandbox.session(defaults)
            sb = session.create(command="sleep", args=["infinity"])

            @session.function()
            def compute(x, y):
                return x + y

            await session.close()
            ```
        """
        from cwsandbox._session import Session

        return Session(defaults, auth=auth)

    @classmethod
    def _from_sandbox_info(
        cls,
        info: _SandboxInfoLike,
        *,
        base_url: str,
        timeout_seconds: float,
        poll_retry_budget_seconds: float = DEFAULT_POLL_RETRY_BUDGET_SECONDS,
        poll_rpc_timeout_seconds: float = DEFAULT_POLL_RPC_TIMEOUT_SECONDS,
        data_plane_mode: DataPlaneMode | str = DataPlaneMode.AUTO,
        auth: AuthConfig | None = None,
        auth_metadata: tuple[tuple[str, str], ...] | None = None,
    ) -> Sandbox:
        """Create a Sandbox instance from a protobuf sandbox info response."""
        info = _as_sandbox_view(info)
        sandbox = cls.__new__(cls)
        sandbox._sandbox_id = str(info.sandbox_id)
        sandbox._status_updated_at = datetime.now(UTC)
        sandbox._base_url = base_url
        sandbox._request_timeout_seconds = timeout_seconds
        sandbox._poll_retry_budget_seconds = poll_retry_budget_seconds
        sandbox._poll_rpc_timeout_seconds = poll_rpc_timeout_seconds
        _validate_poll_config(
            sandbox._poll_retry_budget_seconds,
            sandbox._poll_rpc_timeout_seconds,
        )
        # Not applicable for discovered sandboxes
        sandbox._command = None
        sandbox._args = None
        sandbox._container_image = None
        sandbox._tags = None
        sandbox._max_lifetime_seconds = None
        sandbox._runner_ids = None
        sandbox._environment_variables = {}
        sandbox._annotations = {}
        sandbox._channel = None
        sandbox._stub = None
        sandbox._auth = auth
        sandbox._auth_metadata = auth_metadata or ()
        sandbox._auth_metadata_resolved = auth_metadata is not None
        sandbox._streaming_channel = None
        sandbox._streaming_channel_lock = asyncio.Lock()
        sandbox._data_plane_mode = (
            DataPlaneMode(data_plane_mode.lower())
            if isinstance(data_plane_mode, str)
            else data_plane_mode
        )
        sandbox._direct_data_plane = DirectDataPlaneClient()
        sandbox._observed_file_op_cap_bytes = None
        sandbox._streaming_fallback_warned = False
        sandbox._session = None
        sandbox._defaults = SandboxDefaults(auth=auth)
        sandbox._start_kwargs = {}
        sandbox._create_request_id = None
        sandbox._placement_mode = None
        sandbox._placement_spillover = PlacementSpillover.STRICT
        sandbox._services = None
        sandbox._template_id = None
        sandbox._image_pull_credentials = None
        sandbox._runtime_class = None
        sandbox._security_context = None
        sandbox._working_dir = None
        sandbox._object_storage_access = None
        sandbox._effective_runtime_class = None
        sandbox._attached_volume_ids = ()
        sandbox._effective_egress = ()
        sandbox._effective_ingress = ()
        sandbox._scratch_volume_names = (
            tuple(
                volume.name
                for volume in info._sandbox.spec.volumes
                if _volume_source_is_scratch(volume)
            )
            if isinstance(info, _SandboxView)
            else ()
        )
        sandbox._service_urls = ()
        sandbox._service_endpoints = ()
        sandbox._dns_egress_names = ()
        sandbox._file_system_snapshot_id = None
        sandbox._file_system_snapshot_ids = ()
        sandbox._spec_containers = ()
        sandbox._container_statuses = ()
        sandbox._observed_file_op_cap_bytes = None
        sandbox._streaming_fallback_warned = False
        sandbox._start_lock = asyncio.Lock()
        sandbox._running_task = None
        sandbox._running_lock = asyncio.Lock()
        sandbox._complete_task = None
        sandbox._complete_lock = asyncio.Lock()
        sandbox._grace_pending_response = None
        sandbox._stop_task = None
        sandbox._stop_lock = asyncio.Lock()
        sandbox._stop_owned = False
        sandbox._missing_ok_observe = False
        sandbox._loop_manager = _LoopManager.get()
        sandbox._exposed_ports = None
        sandbox._resource_limits = None
        sandbox._resource_requests = None
        sandbox._resource_gpu = None
        # Exec stats (protected by _exec_stats_lock)
        sandbox._exec_stats_lock = threading.Lock()
        sandbox._exec_count = 0
        sandbox._exec_completed_ok = 0
        sandbox._exec_completed_nonzero = 0
        sandbox._exec_failures = 0
        sandbox._start_accepted_at = None
        sandbox._startup_recorded = True

        status = SandboxStatus.from_proto(info.sandbox_status)
        started_at = (
            info.started_at_time.ToDatetime()
            if hasattr(info, "started_at_time") and info.started_at_time
            else None
        )
        sandbox._state = _lifecycle_state_from_info(
            sandbox_id=str(info.sandbox_id),
            status=status,
            runner_id=getattr(info, "runner_id", None) or None,
            runner_group_id=getattr(info, "runner_group_id", None) or None,
            started_at=started_at,
        )
        if isinstance(info, _SandboxView):
            sandbox._apply_status_echo(info)
        return sandbox

    @classmethod
    def list(
        cls,
        *,
        tags: list[str] | None = None,
        status: str | None = None,
        profile_ids: list[str] | None = None,
        profile_names: list[str] | None = None,
        runner_ids: list[str] | None = None,
        show_terminated: bool = False,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
        data_plane_mode: DataPlaneMode | str = DataPlaneMode.AUTO,
        volume_ids: builtins.list[str] | tuple[str, ...] | None = None,
    ) -> OperationRef[builtins.list[Sandbox]]:
        """List existing sandboxes with optional filters.

        Returns OperationRef that resolves to Sandbox instances usable for
        operations like exec(), stop(), get_status(), read_file(), write_file().

        By default, only active (non-terminal) sandboxes are returned.
        Set ``show_terminated=True`` to widen the search to include terminal
        sandboxes (completed, failed, terminated).
        A terminal status filter (e.g. ``status="completed"``) also widens
        the search automatically.

        Args:
            tags: Filter by tags (sandboxes must have ALL specified tags)
            status: Filter by status ("running", "completed", "failed", etc.)
            profile_ids: Removed in 1.x; passing a value raises ``TypeError``.
            profile_names: Removed in 1.x; passing a value raises ``TypeError``.
            runner_ids: Filter by runner IDs
            volume_ids: Filter to sandboxes attached to these registered Volume IDs
            show_terminated: If True, include terminal sandboxes (completed,
                failed, terminated). Defaults to False.
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default)
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s)
            poll_retry_budget_seconds: Wall-clock budget for retrying transient
                errors on the sandbox-status poll loop (default: 30s). Set to 0
                to disable retry. Applied to returned Sandbox instances.
            poll_rpc_timeout_seconds: Per-call timeout for poll Get RPCs
                (default: 15s). Separate from ``timeout_seconds``. Applied to
                returned Sandbox instances.
            data_plane_mode: Transport policy applied to returned sandboxes.

        Returns:
            OperationRef[list[Sandbox]]: Use .result() to block for results,
            or await directly in async contexts.

        Examples:
            ```python
            # Sync usage - active sandboxes only (default)
            sandboxes = Sandbox.list(tags=["my-batch-job"]).result()
            for sb in sandboxes:
                print(f"{sb.sandbox_id}: {sb.status}")
                sb.stop().result()

            # Include stopped sandboxes
            all_sandboxes = Sandbox.list(
                tags=["my-batch-job"], show_terminated=True
            ).result()

            # Async usage
            sandboxes = await Sandbox.list(status="running")
            for sb in sandboxes:
                result = await sb.exec(["echo", "hello"])
            ```
        """
        future = _LoopManager.get().run_async(
            cls._list_async(
                tags=tags,
                status=status,
                profile_ids=profile_ids,
                profile_names=profile_names,
                runner_ids=runner_ids,
                show_terminated=show_terminated,
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
                poll_retry_budget_seconds=poll_retry_budget_seconds,
                poll_rpc_timeout_seconds=poll_rpc_timeout_seconds,
                data_plane_mode=data_plane_mode,
                volume_ids=volume_ids,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _list_async(
        cls,
        *,
        tags: builtins.list[str] | None = None,
        status: str | None = None,
        profile_ids: builtins.list[str] | None = None,
        profile_names: builtins.list[str] | None = None,
        runner_ids: builtins.list[str] | None = None,
        show_terminated: bool = False,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
        data_plane_mode: DataPlaneMode | str = DataPlaneMode.AUTO,
        volume_ids: builtins.list[str] | tuple[str, ...] | None = None,
    ) -> builtins.list[Sandbox]:
        """Internal async: List existing sandboxes with optional filters."""
        normalized_tags = _normalize_tags(tags)
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        effective_poll_retry_budget = (
            poll_retry_budget_seconds
            if poll_retry_budget_seconds is not None
            else DEFAULT_POLL_RETRY_BUDGET_SECONDS
        )
        effective_poll_rpc_timeout = (
            poll_rpc_timeout_seconds
            if poll_rpc_timeout_seconds is not None
            else DEFAULT_POLL_RPC_TIMEOUT_SECONDS
        )
        _validate_poll_config(effective_poll_retry_budget, effective_poll_rpc_timeout)

        status_enum = None
        if status is not None:
            status_enum = SandboxStatus(status)

        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request_kwargs: dict[str, Any] = {}
            if normalized_tags:
                request_kwargs["tags"] = list(normalized_tags)
            if status_enum:
                request_kwargs["state"] = status_enum.to_proto()
            if profile_ids is not None or profile_names is not None:
                raise TypeError("profile_ids/profile_names were removed in cwsandbox 1.x")
            if runner_ids is not None:
                request_kwargs["runner_ids"] = runner_ids
            if volume_ids:
                request_kwargs["volume_ids"] = list(volume_ids)

            if show_terminated:
                request_kwargs["show_terminated"] = True
            request = sandbox_pb2.ListSandboxesRequest(**request_kwargs)
            try:
                sandbox_infos = await paginate_async(
                    stub.ListSandboxes,
                    request,
                    "sandboxes",
                    auth_metadata,
                    timeout,
                    operation="List sandboxes",
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, operation="List sandboxes") from e

            return [
                cls._from_sandbox_info(
                    _as_sandbox_view(sb),
                    base_url=effective_base_url,
                    timeout_seconds=timeout,
                    poll_retry_budget_seconds=effective_poll_retry_budget,
                    poll_rpc_timeout_seconds=effective_poll_rpc_timeout,
                    data_plane_mode=data_plane_mode,
                    auth=auth,
                    auth_metadata=auth_metadata,
                )
                for sb in sandbox_infos
            ]
        finally:
            await channel.close(grace=None)

    @classmethod
    def from_id(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
        data_plane_mode: DataPlaneMode | str = DataPlaneMode.AUTO,
    ) -> OperationRef[Sandbox]:
        """Attach to an existing sandbox by ID.

        Creates a Sandbox instance connected to an existing sandbox,
        allowing operations like exec(), stop(), get_status(), etc.

        Args:
            sandbox_id: The ID of the existing sandbox
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default)
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s)
            poll_retry_budget_seconds: Wall-clock budget for retrying transient
                errors on the sandbox-status poll loop (default: 30s). Set to 0
                to disable retry. Applied to the returned Sandbox instance.
            poll_rpc_timeout_seconds: Per-call timeout for poll Get RPCs
                (default: 15s). Separate from ``timeout_seconds``. Applied to
                the returned Sandbox instance.
            data_plane_mode: Transport policy applied to the returned sandbox.

        Returns:
            OperationRef[Sandbox]: Use .result() to block for the Sandbox instance,
            or await directly in async contexts.

        Raises:
            SandboxNotFoundError: If sandbox doesn't exist

        Examples:
            ```python
            # Sync usage
            sb = Sandbox.from_id("sandbox-abc123").result()
            result = sb.exec(["python", "-c", "print('hello')"]).result()
            sb.stop().result()

            # Async usage
            sb = await Sandbox.from_id("sandbox-abc123")
            result = await sb.exec(["python", "-c", "print('hello')"])
            ```
        """
        future = _LoopManager.get().run_async(
            cls._from_id_async(
                sandbox_id,
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
                poll_retry_budget_seconds=poll_retry_budget_seconds,
                poll_rpc_timeout_seconds=poll_rpc_timeout_seconds,
                data_plane_mode=data_plane_mode,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _from_id_async(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
        data_plane_mode: DataPlaneMode | str = DataPlaneMode.AUTO,
    ) -> Sandbox:
        """Internal async: Attach to an existing sandbox by ID."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        effective_poll_retry_budget = (
            poll_retry_budget_seconds
            if poll_retry_budget_seconds is not None
            else DEFAULT_POLL_RETRY_BUDGET_SECONDS
        )
        effective_poll_rpc_timeout = (
            poll_rpc_timeout_seconds
            if poll_rpc_timeout_seconds is not None
            else DEFAULT_POLL_RPC_TIMEOUT_SECONDS
        )
        _validate_poll_config(effective_poll_retry_budget, effective_poll_rpc_timeout)

        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request = sandbox_pb2.GetSandboxRequest(sandbox_id=sandbox_id)
            try:
                response = _as_sandbox_view(
                    await stub.GetSandbox(request, timeout=timeout, metadata=auth_metadata)
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, sandbox_id=sandbox_id, operation="Get sandbox") from e

            return cls._from_sandbox_info(
                response,
                base_url=effective_base_url,
                timeout_seconds=timeout,
                poll_retry_budget_seconds=effective_poll_retry_budget,
                poll_rpc_timeout_seconds=effective_poll_rpc_timeout,
                data_plane_mode=data_plane_mode,
                auth=auth,
                auth_metadata=auth_metadata,
            )
        finally:
            await channel.close(grace=None)

    @classmethod
    def delete(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> OperationRef[None]:
        """Delete a sandbox by ID without creating a Sandbox instance.

        This is a convenience method for cleanup scenarios where you
        don't need to perform other operations on the sandbox.

        Args:
            sandbox_id: The sandbox ID to delete
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default)
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s)
            missing_ok: If True, suppress SandboxNotFoundError when sandbox
                doesn't exist.

        Returns:
            OperationRef[None]: Use .result() to block until complete.
            Raises SandboxNotFoundError if not found (unless missing_ok=True),
            SandboxError if deletion failed.

        Raises:
            SandboxNotFoundError: If sandbox doesn't exist and missing_ok=False
            SandboxError: If deletion failed for other reasons

        Examples:
            ```python
            # Sync usage
            Sandbox.delete("sandbox-abc123").result()

            # Ignore if already deleted
            Sandbox.delete("sandbox-abc123", missing_ok=True).result()

            # Async usage
            await Sandbox.delete("sandbox-abc123")
            ```
        """
        future = _LoopManager.get().run_async(
            cls._delete_async(
                sandbox_id,
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
                missing_ok=missing_ok,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _delete_async(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> None:
        """Internal async: Delete a sandbox by ID."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )

        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request = sandbox_pb2.DeleteSandboxRequest(sandbox_id=sandbox_id)
            try:
                await stub.DeleteSandbox(request, timeout=timeout, metadata=auth_metadata)
            except grpc.RpcError as e:
                parsed = parse_error_info(e)
                if missing_ok and is_not_found(e, parsed, CWSANDBOX_SANDBOX_NOT_FOUND):
                    return
                raise _translate_rpc_error(
                    e, sandbox_id=sandbox_id, operation="Delete sandbox"
                ) from e
        finally:
            await channel.close(grace=None)

    @classmethod
    def get_snapshot(
        cls,
        file_system_snapshot_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> OperationRef[FileSystemSnapshot]:
        """Fetch a file-system snapshot (FSS) record by ID.

        Snapshots are org-scoped: any snapshot owned by your organization is
        visible, regardless of which sandbox created it.

        Args:
            file_system_snapshot_id: The snapshot ID to fetch.
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s).

        Returns:
            OperationRef[FileSystemSnapshot]: Use .result() to block or await.
            Raises SnapshotNotFoundError if the snapshot does not exist.

        Examples:
            ```python
            snap = Sandbox.get_snapshot("fss-abc123").result()
            print(snap.status, snap.size_bytes)
            ```
        """
        future = _LoopManager.get().run_async(
            cls._get_snapshot_async(
                file_system_snapshot_id,
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _get_snapshot_async(
        cls,
        file_system_snapshot_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> FileSystemSnapshot:
        """Internal async: fetch a snapshot record by ID."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]
        try:
            return await _retry_transient_rpc(
                lambda: _get_snapshot_via_stub(
                    stub,
                    file_system_snapshot_id,
                    auth_metadata=auth_metadata,
                    timeout=timeout,
                ),
                budget_seconds=DEFAULT_FSS_RETRY_BUDGET_SECONDS,
                operation="Get file-system snapshot",
            )
        finally:
            await channel.close(grace=None)

    @classmethod
    def list_snapshots(
        cls,
        *,
        source_sandbox_id: str | None = None,
        status: FileSystemSnapshotStatus | str | None = None,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> OperationRef[builtins.list[FileSystemSnapshot]]:
        """List file-system snapshots (FSS) for the organization.

        Snapshots are org-scoped and the listing is auto-paginated. The
        ``source_sandbox_id`` and ``status`` filters are applied client-side
        (the backend list RPC does not filter), so all snapshots are fetched
        before filtering.

        Args:
            source_sandbox_id: If set, only snapshots captured from this sandbox.
            status: If set, only snapshots in this status (FileSystemSnapshotStatus
                or its string value).
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s).

        Returns:
            OperationRef[list[FileSystemSnapshot]]: Use .result() to block or await.

        Examples:
            ```python
            # All ready snapshots from a given sandbox
            snaps = Sandbox.list_snapshots(
                source_sandbox_id=sb.sandbox_id,
                status=FileSystemSnapshotStatus.READY,
            ).result()
            ```
        """
        future = _LoopManager.get().run_async(
            cls._list_snapshots_async(
                source_sandbox_id=source_sandbox_id,
                status=status,
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _list_snapshots_async(
        cls,
        *,
        source_sandbox_id: str | None = None,
        status: FileSystemSnapshotStatus | str | None = None,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> builtins.list[FileSystemSnapshot]:
        """Internal async: list snapshots with optional client-side filters."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        status_filter = FileSystemSnapshotStatus(status) if status is not None else None

        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]
        try:

            async def _attempt() -> builtins.list[Any]:
                # Build the request inside the attempt: paginate_async mutates
                # page_token in place, so a retry must start from a fresh
                # request (page 1) rather than resuming from the last token.
                request = sandbox_pb2.ListFileSystemSnapshotsRequest()
                try:
                    return await paginate_async(
                        stub.ListFileSystemSnapshots,
                        request,
                        "file_system_snapshots",
                        auth_metadata,
                        timeout,
                        operation="List file-system snapshots",
                    )
                except grpc.RpcError as e:
                    raise _translate_rpc_error(e, operation="List file-system snapshots") from e

            protos = await _retry_transient_rpc(
                _attempt,
                budget_seconds=DEFAULT_FSS_RETRY_BUDGET_SECONDS,
                operation="List file-system snapshots",
            )

            snapshots = [_snapshot_from_proto(p) for p in protos]
        finally:
            await channel.close(grace=None)

        if source_sandbox_id is not None:
            snapshots = [s for s in snapshots if s.source_sandbox_id == source_sandbox_id]
        if status_filter is not None:
            snapshots = [s for s in snapshots if s.status == status_filter]
        return snapshots

    @classmethod
    def delete_snapshot(
        cls,
        file_system_snapshot_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> OperationRef[None]:
        """Delete a file-system snapshot (FSS) by ID.

        Deleting a snapshot does not affect sandboxes already restored from it.

        Args:
            file_system_snapshot_id: The snapshot ID to delete.
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s).
            missing_ok: If True, suppress SnapshotNotFoundError when the snapshot
                doesn't exist (already deleted).

        Returns:
            OperationRef[None]: Use .result() to block or await.
            Raises SnapshotNotFoundError if not found (unless missing_ok=True).

        Examples:
            ```python
            Sandbox.delete_snapshot("fss-abc123").result()
            Sandbox.delete_snapshot("fss-abc123", missing_ok=True).result()
            ```
        """
        future = _LoopManager.get().run_async(
            cls._delete_snapshot_async(
                file_system_snapshot_id,
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
                missing_ok=missing_ok,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _delete_snapshot_async(
        cls,
        file_system_snapshot_id: str,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> None:
        """Internal async: delete a snapshot by ID."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]
        try:
            request = sandbox_pb2.DeleteFileSystemSnapshotRequest(
                file_system_snapshot_id=file_system_snapshot_id
            )
            attempts = {"n": 0}

            async def _attempt() -> None:
                attempts["n"] += 1
                try:
                    await stub.DeleteFileSystemSnapshot(
                        request, timeout=timeout, metadata=auth_metadata
                    )
                except grpc.RpcError as e:
                    parsed = parse_error_info(e)
                    # NOT_FOUND is success when missing_ok, or on a retry: an
                    # earlier attempt likely committed the delete before its
                    # response was lost to a transient failure. For DELETE the
                    # postcondition (snapshot gone) is satisfied either way.
                    if is_not_found(e, parsed, CWSANDBOX_FSS_NOT_FOUND) and (
                        missing_ok or attempts["n"] > 1
                    ):
                        return
                    raise _translate_rpc_error(
                        e,
                        operation="Delete file-system snapshot",
                        file_system_snapshot_id=file_system_snapshot_id,
                    ) from e
                # v1 returns the deleted FileSystemSnapshot resource (or empty
                # on allow_missing races handled above).

            await _retry_transient_rpc(
                _attempt,
                budget_seconds=DEFAULT_FSS_RETRY_BUDGET_SECONDS,
                operation="Delete file-system snapshot",
            )
        finally:
            await channel.close(grace=None)

    @classmethod
    def get_snapshot_bucket_config(
        cls,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> OperationRef[FileSystemSnapshotBucketConfig]:
        """Fetch the organization's FSS object-storage bucket configuration.

        Args:
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s).

        Returns:
            OperationRef[FileSystemSnapshotBucketConfig]: Use .result() or await.

        Examples:
            ```python
            cfg = Sandbox.get_snapshot_bucket_config().result()
            print(cfg.mode, cfg.effective_bucket_name)
            ```
        """
        future = _LoopManager.get().run_async(
            cls._get_snapshot_bucket_config_async(
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _get_snapshot_bucket_config_async(
        cls,
        *,
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> FileSystemSnapshotBucketConfig:
        """Internal async: fetch the org's FSS bucket configuration."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = settings_pb2_grpc.SettingsServiceStub(channel)  # type: ignore[no-untyped-call]
        try:
            request = settings_pb2.GetFileSystemSnapshotBucketConfigRequest()

            async def _attempt() -> FileSystemSnapshotBucketConfig:
                try:
                    proto = await stub.GetFileSystemSnapshotBucketConfig(
                        request, timeout=timeout, metadata=auth_metadata
                    )
                except grpc.RpcError as e:
                    raise _translate_rpc_error(
                        e, operation="Get file-system snapshot bucket config"
                    ) from e
                return _bucket_config_from_proto(proto)

            return await _retry_transient_rpc(
                _attempt,
                budget_seconds=DEFAULT_FSS_RETRY_BUDGET_SECONDS,
                operation="Get file-system snapshot bucket config",
            )
        finally:
            await channel.close(grace=None)

    @classmethod
    def set_snapshot_bucket_config(
        cls,
        *,
        bucket_name: str,
        region: str = "",
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> OperationRef[FileSystemSnapshotBucketConfig]:
        """Set the organization's FSS object-storage bucket configuration.

        Provide a ``bucket_name`` to use a bring-your-own bucket; pass an empty
        string to revert to the CoreWeave-managed bucket. This is an
        admin-gated operation.

        Args:
            bucket_name: Bucket to archive snapshots to. Empty string reverts to
                the CoreWeave-managed bucket.
            region: Bucket region (required by some providers for BYO buckets).
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
            auth: Authentication strategy, resolved headers, or provider for this request.
            timeout_seconds: Request timeout (default: 300s).

        Returns:
            OperationRef[FileSystemSnapshotBucketConfig]: The updated config.

        Examples:
            ```python
            # Bring-your-own bucket
            Sandbox.set_snapshot_bucket_config(
                bucket_name="my-org-fss", region="us-east-1"
            ).result()

            # Revert to CoreWeave-managed
            Sandbox.set_snapshot_bucket_config(bucket_name="").result()
            ```
        """
        future = _LoopManager.get().run_async(
            cls._set_snapshot_bucket_config_async(
                bucket_name=bucket_name,
                region=region,
                base_url=base_url,
                auth=auth,
                timeout_seconds=timeout_seconds,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _set_snapshot_bucket_config_async(
        cls,
        *,
        bucket_name: str,
        region: str = "",
        base_url: str | None = None,
        auth: AuthConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> FileSystemSnapshotBucketConfig:
        """Internal async: set the org's FSS bucket configuration."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        auth_metadata = resolve_auth_metadata(auth, base_url=effective_base_url)
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = settings_pb2_grpc.SettingsServiceStub(channel)  # type: ignore[no-untyped-call]
        try:
            request = settings_pb2.UpdateFileSystemSnapshotBucketConfigRequest(
                file_system_snapshot_bucket_config=settings_pb2.FileSystemSnapshotBucketConfig(
                    bucket_name=bucket_name,
                    region=region,
                ),
            )

            async def _attempt() -> FileSystemSnapshotBucketConfig:
                try:
                    proto = await stub.UpdateFileSystemSnapshotBucketConfig(
                        request, timeout=timeout, metadata=auth_metadata
                    )
                except grpc.RpcError as e:
                    raise _translate_rpc_error(
                        e, operation="Set file-system snapshot bucket config"
                    ) from e
                return _bucket_config_from_proto(proto)

            return await _retry_transient_rpc(
                _attempt,
                budget_seconds=DEFAULT_FSS_RETRY_BUDGET_SECONDS,
                operation="Set file-system snapshot bucket config",
            )
        finally:
            await channel.close(grace=None)

    @property
    def sandbox_id(self) -> str | None:
        """The unique sandbox ID, or None if not yet started."""
        if not isinstance(self._state, _NotStarted):
            return self._state.sandbox_id
        return self._sandbox_id

    @property
    def returncode(self) -> int | None:
        """Exit code if sandbox has completed, None if still running.

        Use wait() to block until the sandbox completes.

        May be None even for a COMPLETED sandbox when the backend did not
        record an exit code: older gateways without exit-code support,
        containers that never started (e.g. image pull failures), and
        completions whose terminal status was recorded before the runner's
        exit-code report arrived.

        This is the container's real exit code: a sandbox stopped while its
        main process was still running reports the code produced by the
        stopping signal (typically 143/SIGTERM or 137/SIGKILL) unless the
        process handles SIGTERM and exits on its own.
        """
        if isinstance(self._state, _Terminal):
            return self._state.returncode
        return None

    @property
    def runner_id(self) -> str | None:
        """Runner where sandbox is running, or None if not started."""
        if isinstance(self._state, (_Running, _Stopping, _Terminal)):
            return self._state.runner_id
        return None

    @property
    def status(self) -> SandboxStatus | None:
        """Last known status of the sandbox.

        This is the cached status from the most recent API interaction.

        Returns None only for sandboxes that haven't been started yet.

        Note: This value may be stale. Check status_updated_at for when it
        was last fetched. For guaranteed fresh status, use
        `await sandbox.get_status()` which always hits the API.
        """
        match self._state:
            case _NotStarted():
                return None
            case (
                _Starting(status=s) | _Running(status=s) | _Stopping(status=s) | _Terminal(status=s)
            ):
                return s

    @property
    def status_updated_at(self) -> datetime | None:
        """Timestamp when status was last confirmed.

        For terminal sandboxes, this is updated on each get_status() call
        without an API round-trip since terminal states are immutable.

        Returns None only for sandboxes that haven't been started yet.
        """
        return self._status_updated_at

    @property
    def started_at(self) -> datetime | None:
        """Timestamp when the sandbox was started.

        Populated after start() completes or when obtained via list()/from_id().
        None only for sandboxes that haven't been started yet.
        """
        if isinstance(self._state, (_Running, _Stopping, _Terminal)):
            return self._state.started_at
        return None

    @property
    def runner_group_id(self) -> str | None:
        """Runner group ID where the sandbox is running."""
        if isinstance(self._state, (_Running, _Stopping, _Terminal)):
            return self._state.runner_group_id
        return None

    @property
    def service_urls(self) -> tuple[tuple[int, str, str], ...]:
        """Per-service URLs assigned by the backend.

        Each entry is ``(port, name, url)``. A URL can appear while CREATING
        or RUNNING. Empty until a URL is assigned, when none was requested, or
        after the sandbox stops. Assigned is not the same as the app listening.

        Custom-visibility services often have no URL from the API; the SDK
        does not invent one. Those services still appear in
        ``exposed_ports`` when visibility is not ``UNSPECIFIED``.
        """
        return self._service_urls

    @property
    def service_endpoints(self) -> tuple[HttpsEndpointStatus, ...]:
        """HTTPS product endpoints echoed from create, Get, or list.

        Each entry includes the applied ``request_timeout_seconds`` (15 when
        create omitted or sent ``0``). ``url`` can be empty after the API
        suppresses it on a terminal sandbox; the timeout remains. Empty when
        no HTTPS product endpoint was requested or the response omitted one.
        """
        return self._service_endpoints

    @property
    def containers(self) -> tuple[Container, ...]:
        """Create-time container spec echoed from the sandbox resource.

        Empty until create/Get/list populates it. ``primary=True`` is filled
        on the inferred primary so a clone of this list is a valid create.
        """
        return self._spec_containers

    @property
    def container_statuses(self) -> tuple[ContainerStatus, ...]:
        """Per-container observed state. Sandbox ``status`` stays primary-owned."""
        return self._container_statuses

    @property
    def dns_egress_names(self) -> tuple[str, ...]:
        """Hostnames granted at create, echoed from ``status.effective_egress``.

        Empty until a create/Get/list response reports name rules, or when
        none were requested. Terminal sandboxes keep the last echoed names.
        """
        return self._dns_egress_names

    @property
    def effective_runtime_class(self) -> str | None:
        """Runtime class applied by the backend, echoed from status."""
        return self._effective_runtime_class

    @property
    def attached_volume_ids(self) -> tuple[str, ...]:
        """Registered Volume IDs attached to this sandbox, echoed from status."""
        return self._attached_volume_ids

    @property
    def effective_egress(self) -> tuple[EgressRule, ...]:
        """Effective egress rules echoed from ``status.effective_egress``."""
        return self._effective_egress

    @property
    def effective_ingress(self) -> tuple[IngressRule, ...]:
        """Effective ingress rules echoed from ``status.effective_ingress``."""
        return self._effective_ingress

    @property
    def exposed_ports(self) -> tuple[tuple[int, str], ...] | None:
        """Exposed ``(container_port, name)`` pairs derived from typed services.

        Populated from status service entries when URLs/ports are reported.
        ``None`` until the sandbox has reported services (or none were exposed).
        """
        return self._exposed_ports

    @property
    def resource_limits(self) -> dict[str, str] | None:
        """Resource limits from the start response, or None for discovered sandboxes."""
        return self._resource_limits

    @property
    def resource_requests(self) -> dict[str, str] | None:
        """Resource requests from the start response, or None for discovered sandboxes."""
        return self._resource_requests

    @property
    def resource_gpu(self) -> dict[str, Any] | None:
        """GPU config confirmed by the start response, or None for discovered sandboxes."""
        return self._resource_gpu

    @property
    def file_system_snapshot_id(self) -> str | None:
        """ID of the snapshot produced by ``stop(snapshot_on_stop=True)``.

        Populated once the stop OperationRef resolves and the backend reported a
        snapshot ID. None when no snapshot-on-stop was requested (or it produced
        none). Use ``snapshot()`` for mid-life snapshots, which return the record
        directly.
        """
        return self._file_system_snapshot_id

    @property
    def exec_stats(self) -> dict[str, int]:
        """Execution statistics for this sandbox.

        Returns:
            Dictionary with execution counts:

            - ``exec_count``: Total number of exec() calls
            - ``exec_completed_ok``: Execs that completed with returncode 0
            - ``exec_completed_nonzero``: Execs that completed with non-zero
              returncode (when check=False; with check=True, non-zero exits
              count as failures)
            - ``exec_failures``: Execs that failed with an exception (including
              SandboxExecutionError from check=True with non-zero exit)
        """
        with self._exec_stats_lock:
            return {
                "exec_count": self._exec_count,
                "exec_completed_ok": self._exec_completed_ok,
                "exec_completed_nonzero": self._exec_completed_nonzero,
                "exec_failures": self._exec_failures,
            }

    @property
    def _is_cancelled(self) -> bool:
        return isinstance(self._state, _NotStarted) and self._state.cancelled

    @property
    def _is_stopping(self) -> bool:
        """True when sandbox is in the TERMINATING grace period."""
        return isinstance(self._state, _Stopping)

    @property
    def _is_done(self) -> bool:
        """True when sandbox has reached a terminal state or was cancelled before start."""
        return isinstance(self._state, _Terminal) or self._is_cancelled

    def _raise_or_return_for_terminal(
        self, state: _Terminal, *, raise_on_termination: bool = True
    ) -> None:
        """Raise the appropriate error for FAILED/TERMINATED, or return for COMPLETED.

        Raises SandboxTerminatedError when raise_on_termination is True and either:
        - The backend reported legacy TERMINATED status (old backends), or
        - This client sent a successful Stop RPC (_stop_owned).

        Limitation: external kills (infrastructure, lifetime limits, other clients)
        that result in COMPLETED are not detectable as terminations until the
        backend provides termination_reason metadata.
        """
        if state.status == SandboxStatus.FAILED:
            raise SandboxFailedError(f"Sandbox {state.sandbox_id} failed")
        if state.status == SandboxStatus.TERMINATED and raise_on_termination:
            raise SandboxTerminatedError(f"Sandbox {state.sandbox_id} was terminated")
        if self._stop_owned and raise_on_termination:
            raise SandboxTerminatedError(f"Sandbox {state.sandbox_id} was terminated")

    def _apply_sandbox_info(
        self,
        info: _SandboxInfoLike,
        source: Literal["poll", "query"] = "poll",
    ) -> _LifecycleState:
        """Compute a new lifecycle state from a sandbox info/response protobuf.

        Guards against regressing from terminal or cancelled states.

        Args:
            info: Protobuf response with sandbox_status, runner_id,
                runner_group_id, started_at_time, and optionally an exit_code
                field (proto3 optional; presence checked via HasField).
            source: Controls returncode behavior:
                "poll" - set returncode (polling observed the exit)
                "query" - omit returncode (get_status/list/from_id)

        Returns:
            The new _LifecycleState (does NOT mutate self._state).
        """
        if isinstance(self._state, _Terminal):
            return self._state
        if self._is_cancelled:
            return self._state

        info = _as_sandbox_view(info)
        if isinstance(info, _SandboxView):
            self._apply_status_echo(info)
        status = SandboxStatus.from_proto(info.sandbox_status)
        # Polling: UNSPECIFIED means the sandbox exited cleanly
        if source == "poll" and status == SandboxStatus.UNSPECIFIED:
            status = SandboxStatus.COMPLETED

        # Guard: once in _Stopping, only allow forward transitions to _Terminal.
        # Stale poll responses reporting RUNNING/_Starting are rejected.
        if isinstance(self._state, _Stopping) and status not in _TERMINAL_STATUSES:
            if status != SandboxStatus.TERMINATING:
                logger.debug(
                    "Rejecting stale %s while in _Stopping for sandbox %s",
                    status,
                    self._state.sandbox_id,
                )
            return self._state

        if not isinstance(self._state, _NotStarted):
            sandbox_id = self._state.sandbox_id
        else:
            sandbox_id = getattr(self, "_sandbox_id", None) or str(info.sandbox_id)
        started_at = (
            info.started_at_time.ToDatetime()
            if hasattr(info, "started_at_time") and info.started_at_time
            else None
        )
        # returncode is only meaningful for completed sandboxes observed via
        # polling.
        returncode = None
        if source == "poll" and status == SandboxStatus.COMPLETED:
            returncode = _exit_code_from_info(info)

        new_state = _lifecycle_state_from_info(
            sandbox_id=sandbox_id,
            status=status,
            runner_id=getattr(info, "runner_id", None) or None,
            runner_group_id=getattr(info, "runner_group_id", None) or None,
            started_at=started_at,
            returncode=returncode,
        )

        return new_state

    def _on_exec_complete(
        self,
        result: ProcessResult | TerminalResult | None,
        exception: BaseException | None,
    ) -> None:
        """Record exec completion outcome for metrics.

        Args:
            result: The ProcessResult or TerminalResult if execution completed, None on failure
            exception: The exception if execution failed, None on success
        """
        with self._exec_stats_lock:
            if exception is not None:
                self._exec_failures += 1
                outcome = ExecOutcome.FAILURE
            elif result is not None:
                if result.returncode == 0:
                    self._exec_completed_ok += 1
                    outcome = ExecOutcome.COMPLETED_OK
                else:
                    self._exec_completed_nonzero += 1
                    outcome = ExecOutcome.COMPLETED_NONZERO
            else:
                return

        if self._session is not None:
            self._session._record_exec_outcome(outcome, self._sandbox_id)

    def __repr__(self) -> str:
        status_val = self.status
        if status_val is not None:
            status_str = status_val.value
        elif isinstance(self._state, _NotStarted):
            status_str = "not_started"
        else:
            status_str = "unknown"
        return f"<Sandbox id={self.sandbox_id} status={status_str}>"

    async def _get_status_async(self) -> SandboxStatus:
        """Internal async: Get the current status from the backend."""
        if isinstance(self._state, _Terminal):
            self._status_updated_at = datetime.now(UTC)
            return self._state.status
        # _Stopping is mutable (will transition to _Terminal), so always fetch
        if isinstance(self._state, _NotStarted):
            if self._state.cancelled:
                raise SandboxNotRunningError("Sandbox was cancelled before starting")
            raise SandboxNotRunningError("Sandbox has not been started")

        await self._ensure_client()
        assert self._stub is not None

        request = sandbox_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
        try:
            response = _as_sandbox_view(
                await self._stub.GetSandbox(
                    request,
                    timeout=self._poll_rpc_timeout_seconds,
                    metadata=self._auth_metadata,
                )
            )
        except grpc.RpcError as e:
            raise _translate_rpc_error(
                e, sandbox_id=self._sandbox_id, operation="Get status"
            ) from e

        self._state = self._apply_sandbox_info(response, source="query")
        self._status_updated_at = datetime.now(UTC)

        assert not isinstance(self._state, _NotStarted)
        return self._state.status

    def get_status(self) -> SandboxStatus:
        """Get the current status of the sandbox.

        For terminal sandboxes (COMPLETED/FAILED/TERMINATED), returns the cached
        status without an API call. For active sandboxes, fetches from backend.

        Returns:
            SandboxStatus enum value

        Raises:
            SandboxNotRunningError: If sandbox has not been started

        Examples:
            ```python
            sb = Sandbox.run("sleep", "10")
            status = sb.get_status()
            print(f"Sandbox is {status}")  # SandboxStatus.PENDING or RUNNING
            ```
        """
        return self._loop_manager.run_sync(self._get_status_async())

    # Context managers

    def __enter__(self) -> Sandbox:
        """Enter sync context manager.

        If sandbox not started, starts it. Returns self for use in with statement.
        """
        if self._sandbox_id is None:
            self.start().result()
        return self

    async def _cleanup_channels_async(self) -> None:
        """Close gRPC channels and deregister from session.

        Used by context managers when the sandbox already reached a terminal
        state (via polling) so stop() is unnecessary but local resources still
        need to be released.
        """
        if self._session is not None:
            self._session._deregister_sandbox(self)
        await self._direct_data_plane.close()
        if self._streaming_channel is not None:
            await self._streaming_channel.close(grace=None)
            self._streaming_channel = None
        if self._channel is not None:
            await self._channel.close(grace=None)
            self._channel = None
            self._stub = None

    def _cleanup_channels(self) -> None:
        """Sync wrapper for _cleanup_channels_async."""
        self._loop_manager.run_sync(self._cleanup_channels_async())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager, stopping the sandbox.

        If an exception is in flight, suppresses stop errors to avoid masking
        the original exception. Stop errors are logged as warnings.
        """
        if self._sandbox_id is None:
            return
        if self._is_done:
            try:
                self._cleanup_channels()
            except Exception as cleanup_error:
                if exc_val is not None:
                    logger.warning(
                        "Failed to clean up sandbox %s during exception handling: %s",
                        self._sandbox_id,
                        cleanup_error,
                    )
                else:
                    raise
            return
        try:
            self.stop().result()
        except Exception as stop_error:
            if exc_val is not None:
                logger.warning(
                    "Failed to stop sandbox %s during exception handling: %s",
                    self._sandbox_id,
                    stop_error,
                )
            else:
                raise

    async def __aenter__(self) -> Sandbox:
        """Enter async context manager.

        If sandbox not started, starts it. Returns self for use in async with.
        """
        if self._sandbox_id is None:
            future = self._loop_manager.run_async(self._start_async())
            await asyncio.wrap_future(future)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager, stopping the sandbox.

        If an exception is in flight, suppresses stop errors to avoid masking
        the original exception. Stop errors are logged as warnings.
        """
        if self._sandbox_id is None:
            return
        if self._is_done:
            try:
                await self._cleanup_channels_async()
            except Exception as cleanup_error:
                if exc_val is not None:
                    logger.warning(
                        "Failed to clean up sandbox %s during exception handling: %s",
                        self._sandbox_id,
                        cleanup_error,
                    )
                else:
                    raise
            return
        try:
            await self.stop()
        except Exception as stop_error:
            if exc_val is not None:
                logger.warning(
                    "Failed to stop sandbox %s during exception handling: %s",
                    self._sandbox_id,
                    stop_error,
                )
            else:
                raise

    def __del__(self) -> None:
        """Warn if sandbox was not properly stopped."""
        if hasattr(self, "_state") and isinstance(self._state, (_Starting, _Running, _Stopping)):
            warnings.warn(
                f"Sandbox {self._state.sandbox_id} was not stopped. "
                "Use 'sandbox.stop().result()' or the context manager pattern.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _ensure_client(self) -> None:
        """Ensure the gRPC channel and stub are initialized."""
        if self._channel is not None:
            return

        auth_metadata = (
            self._auth_metadata
            if self._auth_metadata_resolved
            else resolve_auth_metadata(self._auth, base_url=self._base_url)
        )
        target, is_secure = parse_grpc_target(self._base_url)
        channel = create_channel(target, is_secure)
        stub = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]
        self._channel = channel
        self._stub = stub
        self._auth_metadata = auth_metadata
        self._auth_metadata_resolved = True
        logger.debug("Initialized gRPC channel for %s", self._base_url)

    async def _get_or_create_streaming_channel(self) -> grpc.aio.Channel:
        """Get or create the cached streaming gRPC channel."""
        if self._streaming_channel is not None:
            return self._streaming_channel

        async with self._streaming_channel_lock:
            if self._streaming_channel is not None:
                return self._streaming_channel

            target, is_secure = parse_grpc_target(self._base_url)
            channel = create_channel(target, is_secure)

            try:
                await asyncio.wait_for(
                    channel.channel_ready(),
                    timeout=self._request_timeout_seconds,
                )
            except TimeoutError:
                await channel.close(grace=None)
                raise SandboxTimeoutError(
                    f"Timed out connecting to streaming service at {target}"
                ) from None

            self._streaming_channel = channel
            return channel

    async def _poll_until_stable(
        self,
        *,
        rpc_timeout_override: float | None = None,
    ) -> _SandboxView:
        """Poll sandbox status until a stable state is reached.

        Returns the response when sandbox reaches a stable state (RUNNING,
        PAUSED, COMPLETED, FAILED, TERMINATED, or UNSPECIFIED). Transient
        states like CREATING and PENDING are polled through. Polls
        indefinitely, relying on external cancellation via stop() or
        asyncio.wait_for.

        Args:
            rpc_timeout_override: Per-call override for the Get RPC timeout.
                When set, used instead of ``self._poll_rpc_timeout_seconds``.
                ``_poll_with_retry`` passes this to clamp each retried RPC to
                the remaining retry budget so a large per-call timeout cannot
                exceed the overall budget.

        Returns:
            The GetSandboxResponse with a stable status
        """
        if self._is_done:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox ID available")

        await self._ensure_client()
        assert self._stub is not None

        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS
        effective_rpc_timeout = (
            rpc_timeout_override
            if rpc_timeout_override is not None
            else self._poll_rpc_timeout_seconds
        )

        while True:
            if self._is_done or self._channel is None:
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} was stopped while polling"
                )

            request = sandbox_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
            try:
                response = _as_sandbox_view(
                    await self._stub.GetSandbox(
                        request,
                        timeout=effective_rpc_timeout,
                        metadata=self._auth_metadata,
                    )
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(
                    e, sandbox_id=self._sandbox_id, operation="Poll sandbox status"
                ) from e

            logger.debug(
                "Sandbox %s status: %s",
                self._sandbox_id,
                response.sandbox_status,
            )

            # Stable states - return for caller to handle
            if response.sandbox_status in (
                sandbox_pb2.STATE_RUNNING,
                sandbox_pb2.STATE_PAUSED,
                sandbox_pb2.STATE_TERMINATING,
                sandbox_pb2.STATE_COMPLETED,
                sandbox_pb2.STATE_FAILED,
                sandbox_pb2.STATE_TERMINATED,
                sandbox_pb2.STATE_UNSPECIFIED,
            ):
                return response

            # Transient states - keep polling
            await asyncio.sleep(poll_interval)
            poll_interval = min(
                poll_interval * DEFAULT_POLL_BACKOFF_FACTOR,
                DEFAULT_MAX_POLL_INTERVAL_SECONDS,
            )

    async def _poll_with_retry(self) -> _SandboxView:
        """Poll ``_poll_until_stable`` with bounded retry on transient errors.

        The retry budget (``poll_retry_budget_seconds``) caps wall-clock time
        spent retrying after a transient failure; it does not cap normal
        polling. See :attr:`SandboxDefaults.poll_retry_budget_seconds` for
        the full contract.

        Raises:
            SandboxNotFoundError: Fatal immediately; never retried.
            CWSandboxError: Any non-retryable exception from
                ``_poll_until_stable``. On budget exhaustion, the last
                translated exception is re-raised unchanged rather than
                wrapped.
        """
        # Retry state (deadline, prev sleep, attempts, last exception) is
        # local to this coroutine, not on ``self``. The shared _running_task
        # / _complete_task design lets multiple waiters await the same poll;
        # state on ``self`` would race between concurrent invocations and
        # leak budget across unrelated polls.
        #
        # Clamp the first RPC timeout to the retry budget so a single wedged
        # Get cannot stall longer than the budget ceiling. Do not start the
        # deadline timer yet: the budget is for retry bursts, and healthy
        # polling across transient states (CREATING, PENDING) must not
        # consume it. The timer starts on the first retryable failure below.
        rpc_timeout_override: float | None = None
        if self._poll_retry_budget_seconds > 0:
            rpc_timeout_override = min(
                self._poll_rpc_timeout_seconds,
                self._poll_retry_budget_seconds,
            )

        retry_deadline: float | None = None
        last_exc: CWSandboxError | None = None
        prev_sleep = DEFAULT_POLL_INTERVAL_SECONDS
        attempts = 0

        while True:
            try:
                return await self._poll_until_stable(
                    rpc_timeout_override=rpc_timeout_override,
                )
            except CWSandboxError as exc:
                last_exc = exc
                classification = _classify_poll_error(exc)
                if classification != "retryable":
                    raise
                if self._poll_retry_budget_seconds <= 0:
                    raise

                # First retryable failure: start the deadline timer.
                if retry_deadline is None:
                    retry_deadline = time.monotonic() + self._poll_retry_budget_seconds

                attempts += 1
                now = time.monotonic()
                if now >= retry_deadline:
                    logger.debug(
                        "poll retry budget exhausted for sandbox %s after %d attempt(s)",
                        self._sandbox_id,
                        attempts,
                    )
                    raise
                remaining = retry_deadline - now
                # AIP-193 RetryInfo hints are honored literally (the server
                # may already be jittering); otherwise use AWS-style
                # decorrelated jitter on the computed backoff to avoid
                # fleet-scale thundering herd during regional outages.
                hinted_delay = exc.retry_delay.total_seconds() if exc.retry_delay else None
                if hinted_delay is not None and hinted_delay > 0:
                    sleep_for = min(hinted_delay, remaining, MAX_POLL_RETRY_HINTED_DELAY_SECONDS)
                    source = "hinted"
                else:
                    base = DEFAULT_POLL_INTERVAL_SECONDS
                    cap = DEFAULT_MAX_POLL_INTERVAL_SECONDS
                    jitter_ceiling = max(
                        base,
                        min(cap, prev_sleep * DEFAULT_POLL_BACKOFF_FACTOR, remaining),
                    )
                    sleep_for = min(random.uniform(base, jitter_ceiling), remaining)
                    source = "computed-jittered"
                cause = exc.__cause__ if isinstance(exc.__cause__, grpc.RpcError) else None
                code = cause.code() if cause is not None else None
                logger.debug(
                    "poll retry for sandbox %s: code=%s sleep=%.2fs source=%s remaining=%.2fs",
                    self._sandbox_id,
                    code,
                    sleep_for,
                    source,
                    remaining,
                )
            await asyncio.sleep(sleep_for)
            prev_sleep = sleep_for
            # Re-check deadline after the sleep: a long hinted delay plus the
            # elapsed retry loop can exhaust the budget while we slept. Re-raise
            # the last translated exception rather than issuing an RPC that
            # would overrun the overall budget. The deadline is always set by
            # this point because the first retryable failure sets it above.
            assert retry_deadline is not None
            now = time.monotonic()
            if now >= retry_deadline:
                assert last_exc is not None
                raise last_exc
            # Clamp the next RPC timeout to whatever budget remains, so a
            # wedged Get cannot run past the overall ceiling. Floor at 0.1s
            # to avoid degenerate zero-timeout RPCs that would fail before
            # the gRPC stack even dispatches them.
            post_sleep_remaining = retry_deadline - now
            rpc_timeout_override = min(
                self._poll_rpc_timeout_seconds,
                max(0.1, post_sleep_remaining),
            )

    async def _ensure_started_async(self) -> None:
        """Ensure sandbox has been started, starting it if needed."""
        if self._sandbox_id is None:
            await self._start_async()

    async def _start_async(self) -> str:
        """Internal async: Send CreateSandbox to backend, return sandbox_id.

        Does NOT wait for RUNNING status. Idempotent - safe to call multiple times.
        Freezes one ``request_id`` across concurrent starts and ambiguous retries.
        """
        if self._sandbox_id is not None:
            return self._sandbox_id

        async with self._start_lock:
            if self._sandbox_id is not None:
                return self._sandbox_id
            if self._is_done:
                raise SandboxNotRunningError("Sandbox has been stopped")

            await self._ensure_client()
            assert self._stub is not None

            if not self._create_request_id:
                self._create_request_id = str(uuid.uuid4())

            template_id = self._start_kwargs.get("template_id") or self._template_id
            self._start_accepted_at = time.monotonic()

            if template_id:
                kwargs = dict(self._start_kwargs)
                kwargs.pop("template_id", None)
                request = self._build_create_from_template_request(
                    template_id=template_id,
                    request_id=self._create_request_id,
                    overrides_kwargs=kwargs,
                )
                logger.debug("Creating sandbox from template %s", template_id)
                try:
                    response = await self._stub.CreateSandboxFromTemplate(
                        request,
                        timeout=self._request_timeout_seconds,
                        metadata=self._auth_metadata,
                    )
                except grpc.RpcError as e:
                    raise _translate_rpc_error(e, operation="Create sandbox from template") from e
            else:
                response = await self._create_with_optional_spillover()

            view = _SandboxView(response)
            sandbox_id = str(view.sandbox_id)
            self._sandbox_id = sandbox_id
            self._status_updated_at = datetime.now(UTC)
            self._state = _Starting(sandbox_id=sandbox_id)
            self._apply_status_echo(view)
            logger.debug("Sandbox %s created (pending)", sandbox_id)
            return sandbox_id

    async def _create_with_optional_spillover(self) -> Any:
        """CreateSandbox with at most one placement-mode spillover retry.

        On a spillable primary failure (and non-``STRICT`` spillover), mints a
        new ``request_id``, flips ``_placement_mode`` to the alternate, clears
        ``runner_ids`` when spilling to serverless, and retries once. A
        definite attempt-2 reject restores the primary placement and request
        id so a later ``start()`` retries the caller's original mode. An
        ambiguous attempt-2 error (timeout, unavailability, bare resource
        exhaustion) keeps the spilled id and mode so a retry cannot create a
        second sandbox. Successful spill keeps the flipped state. Attempt-2
        failures raise with ``__cause__`` set to the primary error.
        """
        assert self._stub is not None
        assert self._create_request_id is not None

        original_placement_mode = self._placement_mode
        original_runner_ids = self._runner_ids
        original_create_request_id = self._create_request_id

        primary_error: Exception | None = None
        for attempt in (1, 2):
            request = self._build_create_request(
                request_id=self._create_request_id,
                start_kwargs=dict(self._start_kwargs),
            )
            logger.debug(
                "Creating sandbox with image %s (placement_mode=%s, attempt=%s)",
                self._container_image,
                self._placement_mode,
                attempt,
            )
            try:
                return await self._stub.CreateSandbox(
                    request,
                    timeout=self._request_timeout_seconds,
                    metadata=self._auth_metadata,
                )
            except grpc.RpcError as e:
                translated = _translate_rpc_error(e, operation="Create sandbox")
                if (
                    attempt == 1
                    and primary_error is None
                    and self._placement_spillover != PlacementSpillover.STRICT
                    and _is_spillover_eligible(translated)
                ):
                    primary_error = translated
                    alternate = (
                        PlacementMode.SERVERLESS
                        if self._placement_spillover == PlacementSpillover.CKS_THEN_SERVERLESS
                        else PlacementMode.CKS
                    )
                    if alternate == self._placement_mode:
                        # Already on the alternate (later start() after an
                        # ambiguous spill). Do not mint a new request_id.
                        raise translated from e
                    logger.warning(
                        "CreateSandbox failed with reason %s; spilling placement_mode "
                        "%s -> %s (new request_id)",
                        getattr(translated, "reason", None) or e.code().name,
                        self._placement_mode,
                        alternate,
                    )
                    self._placement_mode = alternate
                    if alternate == PlacementMode.SERVERLESS:
                        self._runner_ids = None
                    self._create_request_id = str(uuid.uuid4())
                    continue
                if primary_error is not None:
                    if _create_attempt_definitely_rejected(translated):
                        self._placement_mode = original_placement_mode
                        self._runner_ids = original_runner_ids
                        self._create_request_id = original_create_request_id
                    primary_reason = getattr(primary_error, "reason", None)
                    translated.add_note(
                        "Primary placement failed"
                        + (f" with {primary_reason}" if primary_reason else "")
                        + f": {primary_error}"
                    )
                    raise translated from primary_error
                raise translated from e
        raise AssertionError("unreachable: placement spillover loop exited without return")

    def _apply_resource_options(self, container: sandbox_pb2.Container, resources_opt: Any) -> None:
        if resources_opt is None:
            return
        if not isinstance(resources_opt, ResourceOptions):
            resources_opt = normalize_resources(resources_opt)
        if resources_opt is None:
            return
        reqs = self._resources_to_proto(resources_opt.requests, resources_opt.gpu)
        lims = self._resources_to_proto(resources_opt.limits, resources_opt.gpu)
        if reqs is not None or lims is not None:
            container.resource_requirements.CopyFrom(
                sandbox_pb2.ResourceRequirements(
                    requests=reqs or sandbox_pb2.Resources(),
                    limits=lims or sandbox_pb2.Resources(),
                )
            )

    @staticmethod
    def _apply_secrets(container: sandbox_pb2.Container, secrets: Any) -> None:
        if not secrets:
            return
        grouped: dict[str, list[dict[str, str]]] = {}
        for secret in secrets:
            if not isinstance(secret, Secret):
                secret = Secret(**secret)
            grouped.setdefault(secret.store, []).append(
                {
                    "path": secret.name,
                    "field": secret.field,
                    "env_var": secret.env_var or secret.name,
                }
            )
        for store, mappings in grouped.items():
            container.secret_stores.append(
                sandbox_pb2.SecretStoreReference(store_name=store, secrets=mappings)
            )

    @staticmethod
    def _apply_image_pull_credentials(container: sandbox_pb2.Container, ipc: Any) -> None:
        if ipc is None:
            return
        if isinstance(ipc, dict):
            ipc = ImagePullCredentials(**ipc)
        container.image_pull_credentials.CopyFrom(
            sandbox_pb2.ImagePullCredentials(
                registry=ipc.registry,
                credentials=sandbox_pb2.SecretSource(
                    store_name=ipc.store, path=ipc.name, field=ipc.field
                ),
            )
        )

    @staticmethod
    def _apply_mounted_files(container: sandbox_pb2.Container, mounted_files: Any) -> None:
        if not mounted_files:
            return
        entries = (
            (
                {"mount_path": path, "file_content": content}
                for path, content in mounted_files.items()
            )
            if isinstance(mounted_files, Mapping)
            else iter(mounted_files)
        )
        for entry in entries:
            path = entry["mount_path"]
            content = entry["file_content"]
            data = content if isinstance(content, (bytes, bytearray)) else str(content).encode()
            container.files.append(sandbox_pb2.FileMount(path=path, content=bytes(data)))

    @staticmethod
    def _volume_mount_to_proto(mount: VolumeMount) -> sandbox_pb2.VolumeMount:
        proto = sandbox_pb2.VolumeMount(volume=mount.volume, mount_path=mount.mount_path)
        if mount.read_only:
            proto.read_only = True
        if mount.sub_path:
            proto.sub_path = mount.sub_path
        return proto

    def _user_container_to_proto(self, row: Container) -> sandbox_pb2.Container:
        container = sandbox_pb2.Container(image=row.image)
        if row.name:
            container.name = row.name
        if row.command:
            container.command = row.command
        if row.args:
            container.args.extend(row.args)
        if row.environment_variables:
            container.environment_variables.update(row.environment_variables)
        if row.working_dir:
            container.working_dir = row.working_dir
        if row.primary:
            container.primary = True
        self._apply_resource_options(container, row.resources)
        self._apply_secrets(container, row.secrets)
        self._apply_image_pull_credentials(container, row.image_pull_credentials)
        self._apply_mounted_files(container, row.mounted_files)
        for mount in row.volume_mounts or ():
            container.volume_mounts.append(self._volume_mount_to_proto(_coerce_volume_mount(mount)))
        return container

    @staticmethod
    def _append_unique_mounts(
        container: sandbox_pb2.Container, mounts: Sequence[sandbox_pb2.VolumeMount]
    ) -> None:
        existing = {(mount.volume, mount.mount_path) for mount in container.volume_mounts}
        for mount in mounts:
            key = (mount.volume, mount.mount_path)
            if key not in existing:
                container.volume_mounts.append(mount)
                existing.add(key)

    @staticmethod
    def _primary_index(containers: Sequence[sandbox_pb2.Container]) -> int:
        for index, row in enumerate(containers):
            if row.HasField("primary") and row.primary:
                return index
        return 0

    @staticmethod
    def _container_to_partial(container: sandbox_pb2.Container) -> sandbox_pb2.PartialContainer:
        partial = sandbox_pb2.PartialContainer()
        partial.ParseFromString(container.SerializeToString())
        return partial

    def _build_create_request(
        self, *, request_id: str, start_kwargs: dict[str, Any]
    ) -> sandbox_pb2.CreateSandboxRequest:
        """Build a v1 CreateSandboxRequest from constructor/start kwargs."""
        user_containers = start_kwargs.pop("containers", None)
        proto_containers: list[sandbox_pb2.Container]
        if user_containers:
            proto_containers = [
                self._user_container_to_proto(_coerce_container(row)) for row in user_containers
            ]
            start_kwargs.pop("resources", None)
            start_kwargs.pop("secrets", None)
            start_kwargs.pop("mounted_files", None)
            start_kwargs.pop("image_pull_credentials", None)
        else:
            container = sandbox_pb2.Container(
                name="main",
                image=self._container_image,
                command=self._command,
                args=list(self._args or []),
            )
            if self._working_dir:
                container.working_dir = self._working_dir
            if self._security_context is not None:
                container.security_context.CopyFrom(
                    security_context_to_proto(self._security_context)
                )
            if self._environment_variables:
                container.environment_variables.update(self._environment_variables)
            self._apply_resource_options(container, start_kwargs.pop("resources", None))
            self._apply_secrets(container, start_kwargs.pop("secrets", None))
            self._apply_image_pull_credentials(
                container,
                start_kwargs.pop("image_pull_credentials", None) or self._image_pull_credentials,
            )
            self._apply_mounted_files(container, start_kwargs.pop("mounted_files", None))
            proto_containers = [container]

        volumes: list[sandbox_pb2.SandboxVolume] = []
        mounts: list[sandbox_pb2.VolumeMount] = []

        volumes_arg = start_kwargs.pop("volumes", None)
        if volumes_arg:
            volumes, mounts, scratch_names = volumes_to_proto(list(volumes_arg))
            self._scratch_volume_names = scratch_names

        fss_opts = start_kwargs.pop("file_system_snapshot", None)
        if fss_opts is not None:
            if not isinstance(fss_opts, FileSystemSnapshotOptions):
                fss_opts = FileSystemSnapshotOptions(**fss_opts)
            scratch_opts = fss_opts.to_scratch_volume()
            volumes.append(scratch_volume_to_proto(scratch_opts))
            if scratch_opts.mount_path:
                mounts.append(
                    volume_mount_to_proto(
                        scratch_opts.name,
                        scratch_opts.mount_path,
                        sub_path=scratch_opts.sub_path,
                        read_only=scratch_opts.read_only,
                    )
                )
            self._scratch_volume_names = tuple(
                list(self._scratch_volume_names) + [scratch_opts.name]
            )

        if proto_containers and mounts:
            self._append_unique_mounts(
                proto_containers[self._primary_index(proto_containers)], mounts
            )

        services_arg = start_kwargs.pop("services", None) or self._services
        services: list[sandbox_pb2.Service] = []
        if services_arg:
            for svc in services_arg:
                if isinstance(svc, dict):
                    svc = Service(**svc)
                if not isinstance(svc, Service):
                    raise TypeError(
                        f"services entries must be Service or dict, got {type(svc).__name__}"
                    )
                proto_svc = sandbox_pb2.Service(port=svc.port)
                if svc.name:
                    proto_svc.name = svc.name
                protocol = svc.protocol
                if isinstance(protocol, str):
                    protocol = ServiceProtocol(protocol.lower())
                if isinstance(protocol, ServiceProtocol):
                    proto_svc.protocol = cast(
                        sandbox_pb2.ServiceProtocol,
                        sandbox_pb2.ServiceProtocol.Value(f"SERVICE_PROTOCOL_{protocol.name}"),
                    )
                visibility = svc.visibility
                if isinstance(visibility, str):
                    visibility = ServiceVisibility(visibility.lower())
                if isinstance(visibility, ServiceVisibility):
                    proto_svc.visibility = cast(
                        sandbox_pb2.Visibility,
                        sandbox_pb2.Visibility.Value(f"VISIBILITY_{visibility.name}"),
                    )
                endpoint = svc.endpoint
                if isinstance(endpoint, dict):
                    endpoint = Endpoint(**endpoint)
                if isinstance(endpoint, Endpoint):
                    kind = endpoint.kind
                    auth = endpoint.auth
                    if isinstance(kind, str):
                        kind = EndpointKind(kind.lower())
                    if isinstance(auth, str):
                        auth = EndpointAuth(auth.lower())
                    proto_svc.endpoint.kind = cast(
                        sandbox_pb2.EndpointKind,
                        sandbox_pb2.EndpointKind.Value(f"ENDPOINT_KIND_{kind.name}"),
                    )
                    proto_svc.endpoint.auth = cast(
                        sandbox_pb2.EndpointAuth,
                        sandbox_pb2.EndpointAuth.Value(f"ENDPOINT_AUTH_{auth.name}"),
                    )
                    if endpoint.request_timeout_seconds:
                        proto_svc.endpoint.request_timeout_seconds = (
                            endpoint.request_timeout_seconds
                        )
                services.append(proto_svc)

        network = start_kwargs.pop("network", None)
        network_proto = None
        if network is not None:
            if isinstance(network, dict):
                network = NetworkOptions(**network)
            if not isinstance(network, NetworkOptions):
                raise TypeError(
                    f"network must be NetworkOptions, dict, or None, got {type(network).__name__}"
                )
            network_proto = network_to_proto(network)

        # Reject removed kwargs loudly.
        for removed in ("s3_mount", "max_timeout_seconds", "profile_ids", "profile_names", "ports"):
            if removed in start_kwargs:
                raise TypeError(
                    f"{removed!r} is not supported in cwsandbox 1.x; see the v1 migration notes"
                )

        placement = start_kwargs.pop("placement_mode", None) or self._placement_mode
        mode = sandbox_pb2.SANDBOX_MODE_UNSPECIFIED
        if placement is not None:
            if isinstance(placement, str):
                placement = PlacementMode(placement.lower())
            if placement == PlacementMode.SERVERLESS:
                mode = sandbox_pb2.SANDBOX_MODE_SERVERLESS
            elif placement == PlacementMode.CKS:
                mode = sandbox_pb2.SANDBOX_MODE_CKS

        runner_ids = self._runner_ids
        if runner_ids and mode == sandbox_pb2.SANDBOX_MODE_SERVERLESS:
            raise ValueError("runner_ids requires placement_mode=CKS")
        if runner_ids and mode == sandbox_pb2.SANDBOX_MODE_UNSPECIFIED:
            # Explicit CKS required by v1 when pinning runners.
            mode = sandbox_pb2.SANDBOX_MODE_CKS

        spec = sandbox_pb2.SandboxSpec(
            containers=proto_containers,
            volumes=volumes,
            services=services,
            tags=list(self._tags or []),
            mode=mode,
        )
        if self._max_lifetime_seconds is not None:
            spec.max_lifetime_seconds = int(self._max_lifetime_seconds)
        if runner_ids:
            spec.runner_ids.extend(runner_ids)
        if self._annotations:
            spec.annotations.update(self._annotations)
        if network_proto is not None:
            spec.network.CopyFrom(network_proto)
        if self._runtime_class:
            spec.runtime_class = self._runtime_class
        if self._object_storage_access is not None:
            spec.object_storage_access.CopyFrom(
                object_storage_to_proto(self._object_storage_access)
            )

        # Ignore unknown leftover keys that were consumed above; warn on leftovers.
        start_kwargs.pop("ports", None)
        if start_kwargs:
            logger.debug("Ignoring unhandled start kwargs: %s", sorted(start_kwargs))

        return sandbox_pb2.CreateSandboxRequest(
            sandbox=sandbox_pb2.Sandbox(spec=spec),
            request_id=request_id,
        )

    def _build_create_from_template_request(
        self,
        *,
        template_id: str,
        request_id: str,
        overrides_kwargs: dict[str, Any],
    ) -> sandbox_pb2.CreateSandboxFromTemplateRequest:
        """Build CreateSandboxFromTemplate with replace-on-presence overrides."""
        explicit_keys = set(overrides_kwargs)
        create_req = self._build_create_request(
            request_id=request_id, start_kwargs=dict(overrides_kwargs)
        )
        spec = create_req.sandbox.spec
        source_container = spec.containers[0] if spec.containers else sandbox_pb2.Container()
        containers_override = "containers" in explicit_keys
        container_field_overrides = (
            self._command is not None
            or self._args is not None
            or bool(self._environment_variables)
            or "mounted_files" in explicit_keys
            or bool({"volumes", "file_system_snapshot"} & explicit_keys)
            or "secrets" in explicit_keys
            or "resources" in explicit_keys
            or self._image_pull_credentials is not None
            or self._security_context is not None
            or self._working_dir is not None
            or containers_override
        )
        if container_field_overrides and self._container_image is None and not containers_override:
            raise TypeError(
                "replace-on-presence container overrides require container_image; "
                "the API replaces the whole container list and rejects sparse patches"
            )
        if containers_override:
            override_containers = [self._container_to_partial(row) for row in spec.containers]
        elif self._container_image is not None:
            container = sandbox_pb2.PartialContainer(
                name="main",
                image=self._container_image,
            )
            if self._command is not None:
                container.command = self._command
            if self._args is not None:
                container.args.extend(self._args)
            if self._environment_variables:
                container.environment_variables.update(self._environment_variables)
            if "mounted_files" in explicit_keys:
                container.files.extend(source_container.files)
            if {"volumes", "file_system_snapshot"} & explicit_keys:
                container.volume_mounts.extend(source_container.volume_mounts)
            if "secrets" in explicit_keys:
                container.secret_stores.extend(source_container.secret_stores)
            if "resources" in explicit_keys and source_container.HasField("resource_requirements"):
                container.resource_requirements.CopyFrom(source_container.resource_requirements)
            if source_container.HasField("image_pull_credentials"):
                container.image_pull_credentials.CopyFrom(source_container.image_pull_credentials)
            if self._working_dir:
                container.working_dir = self._working_dir
            if self._security_context is not None:
                container.security_context.CopyFrom(
                    security_context_to_proto(self._security_context)
                )
            override_containers = [container]
        else:
            override_containers = []

        overrides = sandbox_pb2.PartialSandboxSpec()
        if override_containers:
            overrides.containers.extend(override_containers)
        if {"volumes", "file_system_snapshot"} & explicit_keys:
            overrides.volumes.extend(spec.volumes)
        if self._services:
            overrides.services.extend(spec.services)
        if self._max_lifetime_seconds is not None:
            overrides.max_lifetime_seconds = int(self._max_lifetime_seconds)
        if self._tags:
            overrides.tags.extend(self._tags)
        if self._runner_ids:
            overrides.runner_ids.extend(self._runner_ids)
        if self._annotations:
            overrides.annotations.update(self._annotations)
        if spec.mode != sandbox_pb2.SANDBOX_MODE_UNSPECIFIED:
            overrides.mode = spec.mode
        if "network" in explicit_keys and spec.HasField("network"):
            overrides.network.CopyFrom(spec.network)
        if self._runtime_class:
            overrides.runtime_class = self._runtime_class
        if self._object_storage_access is not None:
            overrides.object_storage_access.CopyFrom(spec.object_storage_access)
        return sandbox_pb2.CreateSandboxFromTemplateRequest(
            template_id=template_id,
            overrides=overrides,
            request_id=request_id,
        )

    @staticmethod
    def _resources_to_proto(
        cpu_mem: dict[str, str] | None, gpu: dict[str, Any] | None
    ) -> sandbox_pb2.Resources | None:
        if not cpu_mem and not gpu:
            return None
        kwargs: dict[str, Any] = {}
        if cpu_mem:
            if "cpu" in cpu_mem:
                kwargs["cpu"] = cpu_mem["cpu"]
            if "memory" in cpu_mem:
                kwargs["memory"] = cpu_mem["memory"]
        if gpu:
            g: dict[str, Any] = {}
            if "count" in gpu:
                g["count"] = int(gpu["count"])
            if "type" in gpu:
                g["type"] = gpu["type"]
            if "memory_gb" in gpu:
                g["memory_gb"] = int(gpu["memory_gb"])
            if g:
                kwargs["gpu"] = g
        return sandbox_pb2.Resources(**kwargs) if kwargs else None

    def _apply_status_echo(self, view: _SandboxView) -> None:
        """Refresh property echoes from a v1 Sandbox resource (create/Get/list)."""
        self._scratch_volume_names = tuple(
            volume.name
            for volume in view._sandbox.spec.volumes
            if _volume_source_is_scratch(volume)
        )
        status = view._sandbox.status
        service_urls: list[tuple[int, str, str]] = []
        service_endpoints: list[HttpsEndpointStatus] = []
        for s in status.services:
            url = s.url or (s.endpoint.url if s.HasField("endpoint") else "")
            if url:
                service_urls.append((s.port, s.name, url))
            if (
                s.HasField("endpoint")
                and s.endpoint.kind == sandbox_pb2.ENDPOINT_KIND_HTTPS
                and s.endpoint.request_timeout_seconds > 0
            ):
                service_endpoints.append(
                    HttpsEndpointStatus(
                        port=s.port,
                        name=s.name,
                        kind=EndpointKind.HTTPS,
                        auth=EndpointAuth.OPEN,
                        url=s.endpoint.url,
                        request_timeout_seconds=s.endpoint.request_timeout_seconds,
                    )
                )
        self._service_urls = tuple(service_urls)
        self._service_endpoints = tuple(service_endpoints)
        self._dns_egress_names = tuple(
            rule.dns_name for rule in status.effective_egress if rule.dns_name
        )
        self._effective_runtime_class = status.effective_runtime_class or None
        self._attached_volume_ids = tuple(status.attached_volume_ids)
        egress_rules: list[EgressRule] = []
        for egress_proto in status.effective_egress:
            try:
                egress_rules.append(egress_rule_from_proto(egress_proto))
            except ValueError:
                continue
        self._effective_egress = tuple(egress_rules)
        ingress_rules: list[IngressRule] = []
        for ingress_proto in status.effective_ingress:
            try:
                ingress_rules.append(ingress_rule_from_proto(ingress_proto))
            except ValueError:
                continue
        self._effective_ingress = tuple(ingress_rules)
        if status.services:
            self._exposed_ports = tuple(
                (s.port, s.name)
                for s in status.services
                if s.visibility != sandbox_pb2.VISIBILITY_UNSPECIFIED
            )
        if status.HasField("effective_resource_requirements"):
            err = status.effective_resource_requirements
            gpu: dict[str, Any] | None = None
            if err.HasField("limits"):
                self._resource_limits = self._resources_from_proto(err.limits)
                gpu = self._gpu_from_proto(err.limits)
            if err.HasField("requests"):
                self._resource_requests = self._resources_from_proto(err.requests)
                if gpu is None:
                    gpu = self._gpu_from_proto(err.requests)
            self._resource_gpu = gpu
        elif status.HasField("effective_resources"):
            self._resource_limits = self._resources_from_proto(status.effective_resources)
            self._resource_requests = self._resource_limits
            self._resource_gpu = self._gpu_from_proto(status.effective_resources)
        echoed = tuple(self._container_from_proto(row) for row in view._sandbox.spec.containers)
        if echoed and not any(row.primary for row in echoed):
            echoed = (replace(echoed[0], primary=True),) + echoed[1:]
        self._spec_containers = echoed
        statuses: list[ContainerStatus] = []
        for row in status.container_statuses:
            state = SandboxStatus.from_proto(row.state)
            statuses.append(
                ContainerStatus(
                    name=row.name,
                    state=state,
                    exit_code=row.exit_code if state in _TERMINAL_STATUSES else None,
                    restart_count=row.restart_count,
                )
            )
        self._container_statuses = tuple(statuses)

    def _container_from_proto(self, proto: sandbox_pb2.Container) -> Container:
        resources: ResourceOptions | None = None
        if proto.HasField("resource_requirements"):
            rr = proto.resource_requirements
            gpu = None
            requests = self._resources_from_proto(rr.requests) if rr.HasField("requests") else None
            limits = self._resources_from_proto(rr.limits) if rr.HasField("limits") else None
            if rr.HasField("requests"):
                gpu = self._gpu_from_proto(rr.requests)
            if gpu is None and rr.HasField("limits"):
                gpu = self._gpu_from_proto(rr.limits)
            if requests or limits or gpu:
                resources = ResourceOptions(requests=requests, limits=limits, gpu=gpu)
        elif proto.HasField("resources"):
            cpu_mem = self._resources_from_proto(proto.resources)
            gpu = self._gpu_from_proto(proto.resources)
            if cpu_mem or gpu:
                resources = ResourceOptions(requests=cpu_mem, limits=cpu_mem, gpu=gpu)
        secrets = tuple(
            Secret(
                store=store.store_name,
                name=mapping.path,
                field=mapping.field,
                env_var=mapping.env_var or None,
            )
            for store in proto.secret_stores
            for mapping in store.secrets
        )
        mounts = tuple(
            VolumeMount(
                volume=mount.volume,
                mount_path=mount.mount_path,
                read_only=mount.read_only,
                sub_path=mount.sub_path or None,
            )
            for mount in proto.volume_mounts
        )
        files = tuple(
            {"mount_path": file_mount.path, "file_content": bytes(file_mount.content)}
            for file_mount in proto.files
        )
        ipc = None
        if proto.HasField("image_pull_credentials"):
            creds = proto.image_pull_credentials
            ipc = ImagePullCredentials(
                registry=creds.registry,
                store=creds.credentials.store_name,
                name=creds.credentials.path,
                field=creds.credentials.field,
            )
        return Container._from_observed(
            image=proto.image,
            name=proto.name or None,
            command=proto.command or None,
            args=tuple(proto.args) or None,
            environment_variables=dict(proto.environment_variables) or None,
            resources=resources,
            mounted_files=files or None,
            volume_mounts=mounts or None,
            secrets=secrets or None,
            working_dir=proto.working_dir or None,
            image_pull_credentials=ipc,
            primary=proto.HasField("primary") and proto.primary,
        )

    @staticmethod
    def _resources_from_proto(res: sandbox_pb2.Resources) -> dict[str, str] | None:
        d: dict[str, str] = {}
        if res.cpu:
            d["cpu"] = res.cpu
        if res.memory:
            d["memory"] = res.memory
        return d or None

    @staticmethod
    def _gpu_from_proto(res: sandbox_pb2.Resources) -> dict[str, Any] | None:
        if not res.HasField("gpu"):
            return None
        gpu = res.gpu
        d: dict[str, Any] = {}
        if gpu.count:
            d["count"] = gpu.count
        if gpu.type:
            d["type"] = gpu.type
        if gpu.memory_gb:
            d["memory_gb"] = gpu.memory_gb
        return d or None

    async def _get_sandbox_once(self, *, rpc_timeout: float) -> _SandboxView:
        """One Get RPC with error translation: no retries, no status loop."""
        await self._ensure_client()
        assert self._stub is not None
        request = sandbox_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
        try:
            response: _SandboxView = _as_sandbox_view(
                await self._stub.GetSandbox(
                    request, timeout=rpc_timeout, metadata=self._auth_metadata
                )
            )
            return response
        except grpc.RpcError as e:
            raise _translate_rpc_error(
                e, sandbox_id=self._sandbox_id, operation="Poll sandbox status"
            ) from e

    async def _grace_repoll_for_exit_code(self, response: _SandboxView) -> _SandboxView:
        """Briefly re-poll a COMPLETED response that lacks an exit code.

        Covers the runner's batch-flush lag: a Get can observe COMPLETED
        before the runner's terminal report carrying the exit code reaches
        the backend. The caller is about to latch the terminal state (which
        freezes returncode), so take up to EXIT_CODE_GRACE_POLLS extra polls
        first, returning early as soon as a code appears.

        Skipped when this client initiated the stop: gateway-initiated stops
        never stamp an exit code, so re-polling would only delay stop().

        Note: the grace gate deliberately keys on the raw proto COMPLETED.
        Poll-source UNSPECIFIED (mapped to COMPLETED by _apply_sandbox_info)
        is only emitted by backends that predate exit codes, so a grace
        window for it could never produce one.
        """
        published: _SandboxView | None = None
        try:
            for _ in range(EXIT_CODE_GRACE_POLLS):
                if response.sandbox_status != sandbox_pb2.STATE_COMPLETED:
                    return response
                if self._stop_owned or self._is_stopping or self._is_done:
                    return response
                if _exit_code_from_info(response) is not None:
                    return response
                logger.debug(
                    "Sandbox %s COMPLETED without exit code; grace re-poll",
                    self._sandbox_id,
                )
                # Publish the in-hand terminal response for the duration of
                # the window: a waiter whose asyncio.wait_for deadline expires
                # mid-grace latches this instead of raising a spurious
                # SandboxTimeoutError for a completion we already observed.
                self._grace_pending_response = published = response
                await asyncio.sleep(EXIT_CODE_GRACE_POLL_INTERVAL_SECONDS)
                # Re-check after the sleep: a concurrent stop() or a terminal
                # latch by another coroutine (get_status, a timed-out waiter)
                # during the window makes the re-poll a pointless RPC.
                if self._stop_owned or self._is_stopping or self._is_done:
                    return response
                try:
                    # Literally one unretried Get on a short timeout: a bonus
                    # poll must not borrow the primary poll's retry budget or
                    # its transient-status loop and stall a wait whose answer
                    # is already in hand.
                    bonus = await self._get_sandbox_once(
                        rpc_timeout=EXIT_CODE_GRACE_RPC_TIMEOUT_SECONDS
                    )
                except CWSandboxError as e:
                    # Best-effort enrichment: the caller already holds a
                    # terminal response, so a failed bonus poll (e.g. a
                    # concurrent delete returning NOT_FOUND) must not fail
                    # the wait.
                    logger.debug(
                        "Sandbox %s grace re-poll failed (%s); keeping in-hand terminal response",
                        self._sandbox_id,
                        type(e).__name__,
                    )
                    return response
                # Adopt the bonus response only when it corrects the terminal
                # status or delivers a code. A stale non-terminal read must
                # not un-observe an already-observed completion.
                if (
                    bonus.sandbox_status
                    in (
                        sandbox_pb2.STATE_COMPLETED,
                        sandbox_pb2.STATE_FAILED,
                        sandbox_pb2.STATE_TERMINATED,
                    )
                    or _exit_code_from_info(bonus) is not None
                ):
                    response = bonus
            return response
        finally:
            # Clear only this loop's own publication: a cancelled sibling
            # grace window (e.g. stop() cancelling _running_task) must not
            # wipe the response the other shared poll task published.
            if published is not None and self._grace_pending_response is published:
                self._grace_pending_response = None

    async def _do_poll_running(self) -> None:
        """Poll until sandbox reaches a stable state and update instance fields.

        Used as the body of the shared _running_task so multiple concurrent
        waiters share a single polling loop instead of each hitting the API.
        Polls indefinitely, relying on external cancellation via stop() for
        termination. Per-waiter timeouts allow individual waiters to give up
        without killing the shared poll (asyncio.shield protects the task).

        Raises directly on FAILED/TERMINATED so all waiters see the same
        exception. This differs from _do_poll_complete which returns
        SandboxStatus for per-waiter raise_on_termination control.
        """
        assert self._sandbox_id is not None
        response = await self._poll_with_retry()
        response = await self._grace_repoll_for_exit_code(response)

        self._state = self._apply_sandbox_info(response, source="poll")
        self._status_updated_at = datetime.now(UTC)

        if isinstance(self._state, _Running):
            if not self._startup_recorded and self._start_accepted_at is not None:
                startup_time = time.monotonic() - self._start_accepted_at
                self._startup_recorded = True
                if self._session is not None:
                    self._session._record_startup_time(startup_time)
            logger.debug("Sandbox %s is running", self._sandbox_id)
        elif isinstance(self._state, _Stopping):
            logger.info(
                "Sandbox %s entered TERMINATING during startup, draining to terminal",
                self._sandbox_id,
            )
        elif isinstance(self._state, _Terminal):
            if self._state.status == SandboxStatus.FAILED:
                raise SandboxFailedError(f"Sandbox {self._sandbox_id} failed to start")
            if self._state.status == SandboxStatus.TERMINATED:
                raise SandboxTerminatedError(f"Sandbox {self._sandbox_id} was terminated")
            logger.info("Sandbox %s completed during startup", self._sandbox_id)
        else:
            raise SandboxError(f"Unexpected sandbox status: {response.sandbox_status}")

    def _on_poll_task_done(self, task: asyncio.Task[None]) -> None:
        """Callback when _running_task completes.

        Retrieves and logs exceptions to prevent 'Task exception was never
        retrieved' warnings. Always clears the task reference so future
        waiters start a fresh poll instead of seeing a stale completed task.
        """
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.debug(
                "Polling task for sandbox %s failed: %s",
                self._sandbox_id,
                exc,
            )
        if self._running_task is task:
            self._running_task = None

    async def _wait_until_running_async(self, timeout: float | None = None) -> None:
        """Internal async: Wait until sandbox reaches RUNNING status.

        Multiple concurrent callers share a single polling task to avoid
        redundant GetSandbox API calls. asyncio.shield() prevents one
        caller's cancellation/timeout from killing the poll for others.

        Terminal states are handled without polling:
        - COMPLETED returns immediately (sandbox ran and finished).
        - FAILED raises SandboxFailedError.
        - TERMINATED raises SandboxTerminatedError.
        """
        if isinstance(self._state, _Terminal):
            self._raise_or_return_for_terminal(self._state)
            return
        if self._is_cancelled:
            raise SandboxNotRunningError(
                f"Sandbox {self._sandbox_id} was cancelled before starting"
            )
        if isinstance(self._state, _Running):
            return

        await self._ensure_started_async()
        effective_timeout = timeout if timeout is not None else self._request_timeout_seconds

        async with self._running_lock:
            if isinstance(self._state, _Terminal):
                self._raise_or_return_for_terminal(self._state)
                return
            if self._is_cancelled:
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} was cancelled before starting"
                )
            # Re-check after lock acquisition: another coroutine may have
            # completed polling between our first check and acquiring the lock.
            if isinstance(self._state, _Running):
                return
            if self._running_task is None:
                self._running_task = asyncio.create_task(self._do_poll_running())
                self._running_task.add_done_callback(self._on_poll_task_done)
            task = self._running_task

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=effective_timeout)
        except TimeoutError:
            # The shared poll task may be holding an already-observed terminal
            # response while the exit-code grace re-poll defers the latch. A
            # deadline expiring inside that window must not convert an
            # observed completion into a timeout: latch the in-hand response
            # and resolve through the normal terminal policy.
            pending = self._grace_pending_response
            if pending is not None:
                self._state = self._apply_sandbox_info(pending, source="poll")
                self._status_updated_at = datetime.now(UTC)
            if isinstance(self._state, _Terminal):
                self._raise_or_return_for_terminal(self._state)
                return
            raise SandboxTimeoutError(
                f"Sandbox {self._sandbox_id} did not become ready within {effective_timeout}s"
            ) from None
        except asyncio.CancelledError:
            if self._stop_owned:
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} has been stopped"
                ) from None
            if (
                isinstance(self._state, _Terminal)
                and self._state.status == SandboxStatus.TERMINATED
            ):
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} has been stopped"
                ) from None
            raise

    async def _retry_post_stop_not_found(self) -> _SandboxView:
        """Retry ``Get`` for a bounded budget after a post-stop NOT_FOUND.

        Raises:
            SandboxTerminalStateUnavailableError: NOT_FOUND persists past
                the retry budget.
        """
        sandbox_id = self._sandbox_id
        deadline = time.monotonic() + NOT_FOUND_AFTER_STOP_RETRY_BUDGET_SECONDS
        retry_interval = DEFAULT_POLL_INTERVAL_SECONDS
        while True:
            now = time.monotonic()
            if now >= deadline:
                logger.info(
                    "Sandbox %s: NOT_FOUND past %.1fs retry budget; "
                    "surfacing terminal-state ambiguity to caller",
                    sandbox_id,
                    NOT_FOUND_AFTER_STOP_RETRY_BUDGET_SECONDS,
                )
                raise SandboxTerminalStateUnavailableError(
                    f"Stop succeeded for sandbox {sandbox_id}, but backend "
                    f"did not report terminal state within "
                    f"{NOT_FOUND_AFTER_STOP_RETRY_BUDGET_SECONDS:.1f}s. "
                    f"The terminal outcome (COMPLETED or FAILED) is not "
                    f"observable from the client."
                )
            # Sleep the retry interval, clamped by the remaining budget.
            await asyncio.sleep(min(retry_interval, deadline - now))
            try:
                return await self._poll_with_retry()
            except SandboxNotFoundError:
                retry_interval = min(
                    retry_interval * DEFAULT_POLL_BACKOFF_FACTOR,
                    DEFAULT_MAX_POLL_INTERVAL_SECONDS,
                )
                continue

    async def _do_poll_complete(self) -> SandboxStatus:
        """Poll until sandbox reaches terminal state and return the status.

        Used as the body of the shared _complete_task so multiple concurrent
        waiters share a single polling loop instead of each hitting the API.
        Returns SandboxStatus instead of raising, so each waiter can apply
        its own raise_on_termination policy. Since all waiters share a single
        shielded task, raising here would force all to see the same exception,
        preventing per-waiter raise_on_termination control.

        Polls indefinitely, relying on external cancellation via stop() for
        termination. Per-waiter timeouts allow individual waiters to give up
        without killing the shared poll (asyncio.shield protects the task).
        """
        assert self._sandbox_id is not None
        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS
        while True:
            try:
                response = await self._poll_with_retry()
            except SandboxNotFoundError:
                # Post-stop NOT_FOUND is a narrow race: the backend persists
                # terminal state for stopped sandboxes, but its DB write may
                # not have committed yet when we poll. Retry briefly so the
                # backend can report its authoritative state.
                if not (self._stop_owned or (self._is_stopping and self._missing_ok_observe)):
                    raise
                # Defensive: if we somehow already hold a terminal state,
                # preserve it rather than let a transient NOT_FOUND replace
                # it. In the normal flow this cannot happen because the
                # outer loop returns on terminal states, but Stop-path code
                # may mutate self._state concurrently.
                if isinstance(self._state, _Terminal) and self._state.status in _TERMINAL_STATUSES:
                    return self._state.status
                response = await self._retry_post_stop_not_found()

            response = await self._grace_repoll_for_exit_code(response)
            self._state = self._apply_sandbox_info(response, source="poll")
            self._status_updated_at = datetime.now(UTC)

            if isinstance(self._state, _Terminal):
                status = self._state.status
                if status == SandboxStatus.COMPLETED:
                    logger.info("Sandbox %s completed", self._sandbox_id)
                elif status == SandboxStatus.TERMINATED:
                    logger.info("Sandbox %s was terminated", self._sandbox_id)
                return status

            # Still running - keep polling
            await asyncio.sleep(poll_interval)
            poll_interval = min(
                poll_interval * DEFAULT_POLL_BACKOFF_FACTOR,
                DEFAULT_MAX_POLL_INTERVAL_SECONDS,
            )

    def _on_complete_task_done(self, task: asyncio.Task[SandboxStatus]) -> None:
        """Callback when _complete_task completes.

        Retrieves and logs exceptions to prevent 'Task exception was never
        retrieved' warnings. Always clears the task reference so future
        waiters start a fresh poll instead of seeing a stale completed task.
        """
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.debug(
                "Complete-polling task for sandbox %s failed: %s",
                self._sandbox_id,
                exc,
            )
        if self._complete_task is task:
            self._complete_task = None

    async def _wait_until_complete_async(
        self,
        timeout: float | None = None,
        raise_on_termination: bool = True,
    ) -> None:
        """Internal async: Poll until sandbox reaches terminal state.

        Multiple concurrent callers share a single polling task to avoid
        redundant GetSandbox API calls. asyncio.shield() prevents one
        caller's cancellation/timeout from killing the poll for others.
        """
        await self._ensure_started_async()
        if self._is_cancelled:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        # Already terminal - apply raise policy and return
        if isinstance(self._state, _Terminal):
            self._raise_or_return_for_terminal(
                self._state, raise_on_termination=raise_on_termination
            )
            return

        effective_timeout = timeout if timeout is not None else self._request_timeout_seconds

        async with self._complete_lock:
            if self._is_cancelled:
                raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
            # Re-check after lock: another coroutine may have reached terminal.
            if isinstance(self._state, _Terminal):
                self._raise_or_return_for_terminal(
                    self._state, raise_on_termination=raise_on_termination
                )
                return
            if self._complete_task is None:
                self._complete_task = asyncio.create_task(self._do_poll_complete())
                self._complete_task.add_done_callback(self._on_complete_task_done)
            task = self._complete_task

        sandbox_id = self._sandbox_id
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=effective_timeout)
        except TimeoutError:
            # See the TimeoutError handler in _wait_until_running_async: a
            # deadline expiring inside the exit-code grace window latches the
            # in-hand terminal response instead of raising a spurious timeout.
            pending = self._grace_pending_response
            if pending is not None:
                self._state = self._apply_sandbox_info(pending, source="poll")
                self._status_updated_at = datetime.now(UTC)
            if isinstance(self._state, _Terminal):
                return self._raise_or_return_for_terminal(
                    self._state, raise_on_termination=raise_on_termination
                )
            raise SandboxTimeoutError(f"Timed out waiting for sandbox {sandbox_id}") from None
        except asyncio.CancelledError:
            if self._stop_owned:
                raise SandboxNotRunningError(f"Sandbox {sandbox_id} has been stopped") from None
            if (
                isinstance(self._state, _Terminal)
                and self._state.status == SandboxStatus.TERMINATED
            ):
                # missing_ok stop can cancel an in-flight waiter before
                # _stop_owned is set.  Route through the normal terminal
                # policy gate so raise_on_termination=False suppresses the
                # error as callers expect.
                return self._raise_or_return_for_terminal(
                    self._state, raise_on_termination=raise_on_termination
                )
            raise

        assert isinstance(self._state, _Terminal)
        self._raise_or_return_for_terminal(self._state, raise_on_termination=raise_on_termination)

    # Lifecycle methods

    def start(self) -> OperationRef[None]:
        """Send StartSandbox to backend, return OperationRef immediately.

        Does NOT wait for RUNNING status. Use wait() to block until ready.
        Call .result() to block until the start request is accepted.

        Returns:
            OperationRef[None]: Use .result() to block until backend accepts.

        Examples:
            ```python
            sandbox = Sandbox(command="sleep", args=["infinity"])
            sandbox.start().result()
            print(f"Started sandbox: {sandbox.sandbox_id}")
            sandbox.wait()  # Block until RUNNING
            ```
        """

        async def _start_and_discard() -> None:
            await self._start_async()

        future = self._loop_manager.run_async(_start_and_discard())
        return OperationRef(future)

    def wait(self, timeout: float | None = None) -> Sandbox:
        """Block until sandbox reaches RUNNING or a terminal state.

        Returns when sandbox is RUNNING or has already completed (COMPLETED/UNSPECIFIED).

        Args:
            timeout: Maximum seconds to wait. None means use default timeout.

        Returns:
            Self for method chaining. Check .status to determine final state.

        Raises:
            SandboxFailedError: If sandbox fails to start
            SandboxTerminatedError: If sandbox was terminated externally
            SandboxTimeoutError: If timeout expires

        Examples:
            ```python
            sb = Sandbox.run("sleep", "infinity").wait()
            result = sb.exec(["echo", "ready"]).result()
            ```
        """
        self._loop_manager.run_sync(self._wait_until_running_async(timeout))
        return self

    def wait_until_complete(
        self,
        timeout: float | None = None,
        *,
        raise_on_termination: bool = True,
    ) -> OperationRef[Sandbox]:
        """Wait until sandbox reaches terminal state (COMPLETED/FAILED/TERMINATED).

        Returns an OperationRef that resolves when the sandbox reaches a terminal state.
        After resolving, returncode will be available when the backend
        recorded one (see the ``returncode`` property for the cases where it
        stays None).

        Args:
            timeout: Maximum seconds to wait. None means use default timeout.
            raise_on_termination: If True (default), raises SandboxTerminatedError
                when this client called stop() or the backend reports legacy
                TERMINATED status. External kills (infrastructure, lifetime limits,
                other clients) that result in COMPLETED are not detectable until
                the backend provides termination_reason metadata.
                Set to False to suppress SandboxTerminatedError entirely.

        Returns:
            OperationRef[Sandbox]: Use .result() to block or await in async contexts.

        Raises:
            SandboxTimeoutError: If timeout expires
            SandboxTerminatedError: If sandbox was stopped by this client or
                reported as TERMINATED by backend (and raise_on_termination=True)
            SandboxFailedError: If sandbox failed

        Note:
            ``poll_retry_budget_seconds`` is a hard sub-timeout inside the
            user's ``timeout`` parameter. A 30s retry budget with a 300s user
            timeout can surface budget-exhaustion errors around 30s. Callers
            that want longer retry should configure
            ``poll_retry_budget_seconds`` accordingly.

        Examples:
            ```python
            sb = Sandbox.run("python", "-c", "print('done')")
            sb.wait_until_complete().result()
            print(f"Exit code: {sb.returncode}")
            ```
        """

        async def _wait() -> Sandbox:
            await self._wait_until_complete_async(timeout, raise_on_termination)
            return self

        future = self._loop_manager.run_async(_wait())
        return OperationRef(future)

    def __await__(self) -> Generator[Any, None, Sandbox]:
        """Make sandbox awaitable - await sandbox waits until RUNNING.

        Routes through _loop_manager to avoid cross-event-loop issues.
        Auto-starts if not already started.

        Examples:
            ```python
            sb = Sandbox.run("sleep", "infinity")
            await sb  # Wait until RUNNING
            result = await sb.exec(["echo", "hello"])
            ```
        """

        async def _await_running() -> Sandbox:
            await self._ensure_started_async()
            await self._wait_until_running_async()
            return self

        future = self._loop_manager.run_async(_await_running())
        return asyncio.wrap_future(future).__await__()

    async def _await_terminal_after_stop(self) -> None:
        """Ensure a complete-polling task is running and wait for terminal state.

        Shared by both the Stop-RPC path and the already-stopping path so that
        stop().result() always resolves only after the sandbox reaches a
        terminal state (COMPLETED or FAILED).
        """
        async with self._complete_lock:
            if isinstance(self._state, _Terminal):
                return
            if self._complete_task is None:
                self._complete_task = asyncio.create_task(self._do_poll_complete())
                self._complete_task.add_done_callback(self._on_complete_task_done)
            task = self._complete_task

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass

    async def _do_stop(
        self,
        *,
        snapshot_on_stop: bool = False,
        graceful_shutdown_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
        missing_ok: bool = False,
        wait_for_ready: bool = True,
        request_id: str | None = None,
    ) -> None:
        """Body of the shared _stop_task: send Stop RPC, poll to terminal, cleanup.

        Only the first caller's parameters (snapshot_on_stop,
        graceful_shutdown_seconds, wait_for_ready, request_id) are used.
        Later stop() calls join the existing task.
        """
        sent_rpc = False

        # Acquire _start_lock to serialize with startup
        async with self._start_lock:
            if self._is_done:
                return
            if self._is_stopping:
                # Already draining (e.g. background poll saw TERMINATING).
                # Skip the Stop RPC but fall through to await terminal.
                logger.debug(
                    "Sandbox %s already stopping, waiting for terminal",
                    self._sandbox_id,
                )
                if missing_ok:
                    # Widen the NOT_FOUND retry gate in _do_poll_complete so
                    # the observe-only waiter treats NOT_FOUND as a race and
                    # retries briefly for an authoritative terminal state,
                    # rather than propagating SandboxNotFoundError.
                    self._missing_ok_observe = True
            elif self._sandbox_id is None:
                self._state = _NotStarted(cancelled=True)
                return
            else:
                sandbox_id = self._sandbox_id
                prev = self._state

                await self._ensure_client()
                assert self._stub is not None

                # The backend runs a snapshot-on-stop in two sequential phases:
                # it archives the mount (bounded by max_timeout_seconds, the FSS
                # default) and THEN deletes the pod (bounded by
                # graceful_shutdown_seconds). max_timeout_seconds bounds only the
                # archive; the client deadline must cover BOTH phases, so it is
                # the sum (see client_deadline below) — not max_timeout alone.
                # A plain stop has no archive phase and stays bounded by graceful
                # shutdown.
                grace_seconds = (
                    math.ceil(graceful_shutdown_seconds)
                    if graceful_shutdown_seconds > 0
                    else int(graceful_shutdown_seconds)
                )
                if snapshot_on_stop:
                    max_timeout = int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS)
                    client_deadline = (
                        max_timeout + grace_seconds + int(DEFAULT_FSS_STOP_CLIENT_SLACK_SECONDS)
                    )
                else:
                    max_timeout = grace_seconds + int(DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS)
                    client_deadline = max_timeout + int(DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS)
                # The renamed proto field is file_system_snapshot_on_stop;
                # wait_for_ready/request_id are only valid alongside it,
                # so only send them when a snapshot is requested.
                snapshot_volumes: list[str] = []
                if snapshot_on_stop:
                    if not self._scratch_volume_names:
                        raise SandboxSnapshotError(
                            "Cannot snapshot on stop: sandbox has no known scratch volumes",
                            file_system_snapshot_id=None,
                        )
                    snapshot_volumes = list(self._scratch_volume_names)
                request = sandbox_pb2.DeleteSandboxRequest(
                    sandbox_id=sandbox_id,
                    grace_period_seconds=grace_seconds,
                    snapshot_volumes=snapshot_volumes,
                    allow_missing=bool(missing_ok),
                )
                if snapshot_on_stop and request_id:
                    request.request_id = request_id

                # Send Stop RPC first, then update state on success
                try:
                    response = await self._stub.DeleteSandbox(
                        request,
                        timeout=client_deadline,
                        metadata=self._auth_metadata,
                    )
                except grpc.RpcError as e:
                    parsed = parse_error_info(e)
                    if missing_ok and is_not_found(e, parsed, CWSANDBOX_SANDBOX_NOT_FOUND):
                        if snapshot_on_stop:
                            raise SnapshotOnStopConflictError(
                                "Cannot snapshot on stop: sandbox was not found.",
                                file_system_snapshot_id=None,
                            ) from e
                        logger.debug(
                            "Sandbox %s not found during stop (missing_ok=True)",
                            sandbox_id,
                        )
                        self._state = _Terminal(
                            sandbox_id=sandbox_id,
                            status=SandboxStatus.TERMINATED,
                            runner_id=(prev.runner_id if isinstance(prev, (_Running,)) else None),
                            runner_group_id=(
                                prev.runner_group_id if isinstance(prev, (_Running,)) else None
                            ),
                            started_at=(prev.started_at if isinstance(prev, (_Running,)) else None),
                        )
                        if self._complete_task is not None and not self._complete_task.done():
                            self._complete_task.cancel()
                            self._complete_task = None
                        return
                    if (
                        snapshot_on_stop
                        and parsed is not None
                        and parsed.domain == CWSANDBOX_ERROR_DOMAIN
                        and parsed.reason == CWSANDBOX_INVALID_REQUEST
                    ):
                        raise SnapshotOnStopConflictError(
                            "Cannot snapshot on stop: the server rejected "
                            "allow_missing together with snapshot_volumes "
                            "(sandbox missing, or the combination is invalid).",
                            file_system_snapshot_id=None,
                            reason=parsed.reason,
                        ) from e
                    raise _translate_rpc_error(
                        e, sandbox_id=sandbox_id, operation="Stop sandbox"
                    ) from e

                # Capture snapshot IDs produced by snapshot-on-delete, if any.
                snapshot_ids = list(getattr(response, "file_system_snapshot_ids", ()) or ())
                if snapshot_ids:
                    self._file_system_snapshot_id = snapshot_ids[0]
                    self._file_system_snapshot_ids = tuple(snapshot_ids)

                response_has_sandbox = hasattr(response, "HasField") and response.HasField(
                    "sandbox"
                )
                if missing_ok and not response_has_sandbox and not snapshot_ids:
                    if snapshot_on_stop:
                        raise SnapshotOnStopConflictError(
                            "Cannot snapshot on stop: sandbox was not found.",
                            file_system_snapshot_id=None,
                        )
                    self._state = _Terminal(
                        sandbox_id=sandbox_id,
                        status=SandboxStatus.TERMINATED,
                        runner_id=(prev.runner_id if isinstance(prev, (_Running,)) else None),
                        runner_group_id=(
                            prev.runner_group_id if isinstance(prev, (_Running,)) else None
                        ),
                        started_at=(prev.started_at if isinstance(prev, (_Running,)) else None),
                    )
                    self._stop_owned = True
                    if self._complete_task is not None and not self._complete_task.done():
                        self._complete_task.cancel()
                        self._complete_task = None
                    logger.debug(
                        "Sandbox %s already absent during stop (missing_ok=True)",
                        sandbox_id,
                    )
                    return

                # RPC succeeded: transition to _Stopping
                self._state = _Stopping(
                    sandbox_id=sandbox_id,
                    runner_id=(prev.runner_id if isinstance(prev, (_Running,)) else None),
                    runner_group_id=(
                        prev.runner_group_id if isinstance(prev, (_Running,)) else None
                    ),
                    started_at=(prev.started_at if isinstance(prev, (_Running,)) else None),
                )
                self._stop_owned = True
                sent_rpc = True
                logger.info("Sandbox %s stop accepted, draining", sandbox_id)
                if snapshot_ids and wait_for_ready:
                    for snapshot_id in snapshot_ids:
                        await _wait_for_snapshot_via_stub(
                            self._stub,
                            snapshot_id,
                            auth_metadata=self._auth_metadata,
                            timeout=DEFAULT_FSS_STOP_TIMEOUT_SECONDS,
                        )

        # Cancel the running poll only when we sent the Stop RPC.
        # In the observe-only path (_is_stopping), the poll already
        # completed naturally when the background poller saw TERMINATING.
        if sent_rpc and self._running_task is not None and not self._running_task.done():
            self._running_task.cancel()
            self._running_task = None

        await self._await_terminal_after_stop()

    def _on_stop_task_done(self, task: asyncio.Task[None]) -> None:
        """Clear _stop_task reference when task completes."""
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.debug(
                "Stop task for sandbox %s failed: %s",
                self._sandbox_id,
                exc,
            )
        if self._stop_task is task:
            self._stop_task = None

    def _reject_unsatisfiable_snapshot_on_stop(self) -> None:
        """Raise if a ``snapshot_on_stop=True`` request cannot be honored.

        Must be called while holding ``_stop_lock``. ``stop()`` coalesces
        callers onto one shared stop task, so a snapshot-on-stop is honorable
        only when this caller will own a fresh stop (creating the task) or when
        it joins a stop that is itself a snapshot-on-stop. Every other state
        means the sandbox is, or is about to be, torn down without archiving
        the mount: joining would silently drop the snapshot. A sandbox that
        was never started has no mount to archive, so it is left to the normal
        no-op path rather than raising here.
        """
        # Already terminal: a snapshot captured by a prior snapshot-on-stop
        # makes this idempotently satisfiable; otherwise the sandbox is gone
        # with no archive.
        if self._is_done:
            if self._file_system_snapshot_id is not None:
                return
            if self._is_cancelled:
                return
            raise SnapshotOnStopConflictError(
                "Cannot snapshot on stop: sandbox has already stopped."
            )
        # A stop this caller would join is already in flight. Joining is safe
        # only when that stop is itself capturing a snapshot.
        if self._stop_task is not None and not self._stop_task.done():
            if self._stop_snapshot_requested:
                return
            raise SnapshotOnStopConflictError(
                "Cannot snapshot on stop: a plain stop is already in progress for this sandbox."
            )
        # Draining (TERMINATING) with no stop task we own: the backend is
        # already tearing the sandbox down (external stop, TTL, or a
        # poller-observed termination), so no snapshot RPC will be sent.
        if self._is_stopping:
            raise SnapshotOnStopConflictError(
                "Cannot snapshot on stop: sandbox is already terminating."
            )

    async def _stop_async(
        self,
        *,
        snapshot_on_stop: bool = False,
        graceful_shutdown_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
        missing_ok: bool = False,
        wait_for_ready: bool = True,
        request_id: str | None = None,
    ) -> None:
        """Internal async: Stop the sandbox using shared _stop_task pattern.

        First caller creates the task; later callers join it.
        """
        async with self._stop_lock:
            # A snapshot-on-stop request cannot be honored by joining (or
            # observing) a stop that will not archive the mount. Detect that
            # here and raise rather than silently coalescing into a no-snapshot
            # stop, which would destroy the sandbox with no archive and still
            # return success. Plain stops keep coalescing in every case.
            if snapshot_on_stop:
                self._reject_unsatisfiable_snapshot_on_stop()
            if self._is_done:
                logger.debug("stop() called on already-stopped sandbox %s", self._sandbox_id)
                return
            if self._sandbox_id is None and not self._is_stopping:
                logger.debug("stop() called on sandbox that was never started")
                self._state = _NotStarted(cancelled=True)
                return
            if self._stop_task is None:
                self._stop_snapshot_requested = snapshot_on_stop
                self._stop_task = asyncio.create_task(
                    self._do_stop(
                        snapshot_on_stop=snapshot_on_stop,
                        graceful_shutdown_seconds=graceful_shutdown_seconds,
                        missing_ok=missing_ok,
                        wait_for_ready=wait_for_ready,
                        request_id=request_id,
                    )
                )
                self._stop_task.add_done_callback(self._on_stop_task_done)
            task = self._stop_task

        # Join the shared stop task. A joiner that passed missing_ok=True
        # can swallow NOT_FOUND from a shared task the first caller did not
        # mark missing_ok. The reverse (first caller swallowed, joiner wanted
        # the error) is not recoverable: the task already succeeded.
        try:
            await asyncio.shield(task)
        except SandboxNotFoundError:
            if missing_ok:
                return
            raise
        except asyncio.CancelledError:
            pass
        finally:
            # Deregister from session if we own the stop
            if self._stop_owned and self._session is not None:
                self._session._deregister_sandbox(self)
            # Close channels to release resources
            await self._direct_data_plane.close()
            if self._streaming_channel is not None:
                await self._streaming_channel.close(grace=None)
                self._streaming_channel = None
            if self._channel is not None:
                await self._channel.close(grace=None)
                self._channel = None
                self._stub = None

    def stop(
        self,
        *,
        snapshot_on_stop: bool = False,
        graceful_shutdown_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
        missing_ok: bool = False,
        wait_for_ready: bool = True,
        request_id: str | None = None,
    ) -> OperationRef[None]:
        """Stop sandbox, return OperationRef immediately.

        The sandbox transitions through TERMINATING (grace period draining)
        before reaching a terminal state (COMPLETED or FAILED). The returned
        OperationRef resolves when the backend confirms a terminal state, not
        just when the stop RPC succeeds.

        Multiple callers share the same underlying stop task: the first caller
        creates it, subsequent callers join it. A ``snapshot_on_stop=True``
        request that would join (or observe) a stop that is not capturing a
        snapshot, because the sandbox is already stopping, already stopped, or
        a plain ``stop()`` is already in flight, raises
        ``SnapshotOnStopConflictError`` instead of silently completing without
        an archive. Plain stops always coalesce.

        The sandbox is deregistered from its session regardless of whether
        the stop was successful, since the sandbox is no longer usable.

        Args:
            snapshot_on_stop: If True, capture a file-system snapshot (FSS) of
                the configured mount before shutdown. The resulting snapshot ID
                is available via the ``file_system_snapshot_id`` property after
                the returned OperationRef resolves. Requires the sandbox to have
                been started with a ``file_system_snapshot`` mount and the org to
                be enabled for FSS. Raises ``SnapshotOnStopConflictError`` if a
                stop is already in progress that will not capture a snapshot.
            graceful_shutdown_seconds: Time to wait for graceful shutdown. With
                ``snapshot_on_stop=True`` this is the post-archive pod-delete
                grace, applied *after* the snapshot completes, so the client
                deadline covers the archive budget plus this grace. In v1,
                ``grace_period_seconds=0`` means immediate termination (no
                backend grace substitute). The backend caps this at 300s for
                snapshot-on-stop.
            missing_ok: If True, suppress SandboxNotFoundError when the
                sandbox does not exist. With ``snapshot_on_stop=True`` the
                flag is still sent as ``allow_missing``; a missing sandbox
                or a server reject of that combination raises
                ``SnapshotOnStopConflictError`` rather than succeeding
                with no archive.
            wait_for_ready: When ``snapshot_on_stop`` is True, block until the
                snapshot reaches READY (or FAILED) before the stop completes.
                Ignored when ``snapshot_on_stop`` is False.
            request_id: Optional client-supplied key to deduplicate the
                snapshot-on-stop request on retries. Ignored when
                ``snapshot_on_stop`` is False.

        Returns:
            OperationRef[None]: Use .result() to block until terminal.
            Raises SandboxError on failure, SandboxNotFoundError if not found
            (unless missing_ok=True).

        Examples:
            ```python
            sb.stop().result()  # Block until terminal (COMPLETED/FAILED)

            # Ignore if already deleted
            sb.stop(missing_ok=True).result()

            # Snapshot the configured mount on stop, then read the ID
            sb.stop(snapshot_on_stop=True).result()
            file_system_snapshot_id = sb.file_system_snapshot_id

            # wait_until_complete() after stop() resolves when terminal
            sb.stop()
            sb.wait_until_complete().result()  # Polls through TERMINATING
            ```
        """
        future = self._loop_manager.run_async(
            self._stop_async(
                snapshot_on_stop=snapshot_on_stop,
                graceful_shutdown_seconds=graceful_shutdown_seconds,
                missing_ok=missing_ok,
                wait_for_ready=wait_for_ready,
                request_id=request_id,
            )
        )
        return OperationRef(future)

    async def _snapshot_async(
        self,
        *,
        wait_for_ready: bool,
        request_id: str | None,
    ) -> str:
        """Internal async: create a mid-life snapshot and return its ID."""
        await self._ensure_started_async()
        # Snapshotting requires a running sandbox (the backend archives the live
        # mount), so wait for RUNNING like exec/read_file/write_file do; calling
        # on a just-started sandbox would otherwise race startup.
        await self._wait_until_running_async()
        await self._ensure_client()
        assert self._stub is not None
        assert self._sandbox_id is not None
        stub = self._stub
        sandbox_id = self._sandbox_id

        # wait_for_ready blocks on the runner archive, so the client deadline
        # must be generous; otherwise the create RPC returns promptly.
        create_timeout = (
            DEFAULT_FSS_STOP_TIMEOUT_SECONDS + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS
            if wait_for_ready
            else self._request_timeout_seconds
        )
        # When blocking on the archive, also bound the server-side wait (mirror
        # snapshot-on-stop) so the backend's own default request ceiling cannot
        # cut a large snapshot short before the client's FSS deadline.
        # Generate an idempotency key when the caller didn't supply one so a
        # retried create (after a transient failure that may have already
        # committed server-side) dedups instead of creating a second snapshot.
        names = self._scratch_volume_names
        if len(names) > 1:
            raise SandboxSnapshotError(
                "snapshot() cannot choose among multiple scratch volumes "
                f"({', '.join(names)}); start the sandbox with a single volume",
                file_system_snapshot_id=None,
            )
        effective_request_id = request_id or uuid.uuid4().hex
        return await _retry_transient_rpc(
            lambda: _create_snapshot_via_stub(
                stub,
                sandbox_id,
                request_id=effective_request_id,
                wait_for_ready=wait_for_ready,
                auth_metadata=self._auth_metadata,
                timeout=create_timeout,
                scratch_volume_name=names[0] if names else None,
            ),
            budget_seconds=DEFAULT_FSS_RETRY_BUDGET_SECONDS,
            operation="Create file-system snapshot",
            # The create timeout is the snapshot's ceiling (it blocks on the
            # archive), so a client DEADLINE_EXCEEDED is the ceiling being hit,
            # not a transient blip. Treat it as terminal — retrying would run a
            # second full-length attempt and overrun the ceiling (~2x). Genuine
            # transients (unavailability/resource-exhaustion/throttle) still
            # retry. The snapshot may still finish server-side; the caller can
            # poll get_snapshot().
            non_retryable=(SandboxRequestTimeoutError,),
        )

    def snapshot(
        self,
        *,
        wait_for_ready: bool = True,
        request_id: str | None = None,
    ) -> OperationRef[str]:
        """Capture a file-system snapshot (FSS) of the configured mount.

        Snapshots the directory configured via ``file_system_snapshot`` on the
        running sandbox, without stopping it. Starts the sandbox first if it has
        not been started. Restore the snapshot into a new sandbox via
        ``Sandbox.run(file_system_snapshot=FileSystemSnapshotOptions(...,
        file_system_snapshot_id=<id>))``.

        Requires the sandbox to have been started with a ``file_system_snapshot``
        mount and the organization to be enabled for FSS.

        Args:
            wait_for_ready: Block until the snapshot reaches READY (or FAILED)
                before returning. When False, returns once the snapshot is
                created (likely still CREATING).
            request_id: Optional client-supplied key to deduplicate the
                request on retries.

        Returns:
            OperationRef[str]: Use .result() to block (or await) for the new
            snapshot's ID. Call ``Sandbox.get_snapshot(id)`` for the full record
            (status, size, timestamps).

        Raises:
            SandboxSnapshotError: If the snapshot fails (see subclasses for
                ``NOT_SUPPORTED`` when the org is not enabled, quota/size, etc.).

        Examples:
            ```python
            with Sandbox.run(
                file_system_snapshot=FileSystemSnapshotOptions(mount_path="/workspace"),
            ) as sb:
                sb.exec(["sh", "-c", "echo hi > /workspace/note.txt"]).result()
                snapshot_id = sb.snapshot().result()
                # Inspect the full record if needed:
                snap = Sandbox.get_snapshot(snapshot_id).result()
            ```
        """
        future = self._loop_manager.run_async(
            self._snapshot_async(
                wait_for_ready=wait_for_ready,
                request_id=request_id,
            )
        )
        return OperationRef(future)

    async def _stream_logs_async(
        self,
        output_queue: asyncio.Queue[str | Exception | None],
        *,
        follow: bool = False,
        tail_lines: int | None = None,
        since_time: datetime | None = None,
        timestamps: bool = False,
        timeout_seconds: float | None = None,
        container: str | None = None,
    ) -> None:
        """Stream PID-1 logs via v1 unary StreamLogs → LogEntry server stream."""
        inner_exit_clean = False
        try:
            await self._ensure_started_async()
            if self._sandbox_id is None:
                raise SandboxNotRunningError("No sandbox is running")
            if (self._is_stopping or self._is_done) and follow:
                raise SandboxNotRunningError(
                    f"Sandbox {self._sandbox_id} is terminating"
                    " (follow=True requires a running sandbox)"
                )
            if not self._is_done and not self._is_stopping:
                await self._wait_until_running_async()

            sandbox_id = self._sandbox_id
            assert sandbox_id is not None

            if since_time is not None and since_time.tzinfo is None:
                raise ValueError("since_time must be timezone-aware; naive datetimes are rejected")

            session_id = ""
            last_offset = 0
            line_parts: list[str] = []
            line_parts_bytes = 0
            attempt = 0
            done = False
            last_transport_error: grpc.aio.AioRpcError | None = None
            delivered_any = False

            while not done and attempt < STREAMING_RESUME_MAX_ATTEMPTS:
                is_resume = bool(session_id) and attempt > 0
                prepared = await self._prepare_data_plane_call(
                    sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_LOGS,
                    streaming=True,
                    allow_terminal=True,
                )
                stub = prepared.stub

                request = sandbox_pb2.StreamLogsRequest(
                    sandbox_id=sandbox_id,
                    follow=follow,
                )
                _set_request_container(request, container)
                if is_resume:
                    request.resume_log_session_id = session_id
                    request.resume_log_offset = last_offset
                else:
                    # Fresh init (first attempt or after SESSION_NOT_FOUND /
                    # REPLAY_GAP / runner loss). Re-send timestamps so
                    # formatting survives a session reset. Re-send the
                    # caller window until any data arrives; after that,
                    # follow mode advances since_time to now to avoid
                    # replaying hours of logs (a small gap is accepted).
                    request.timestamps = timestamps
                    if tail_lines is not None:
                        request.tail_lines = tail_lines
                    if since_time is not None:
                        ts = timestamp_pb2.Timestamp()
                        ts.FromDatetime(since_time)
                        request.since_time.CopyFrom(ts)

                grpc_timeout = timeout_seconds if not follow else None
                try:
                    call = stub.StreamLogs(
                        request,
                        metadata=prepared.metadata,
                        **({"timeout": grpc_timeout} if grpc_timeout is not None else {}),
                    )
                except BaseException:
                    await prepared.release(discard=True)
                    raise
                prepared.release_when_done(call)

                try:
                    try:
                        async for entry in call:
                            if entry.HasField("error") and entry.error.code:
                                code = entry.error.code
                                msg = entry.error.message or code
                                if code in _STREAMING_FRESH_REINIT_CODES:
                                    if not follow:
                                        raise SandboxError(f"Log stream error: {msg}")
                                    session_id = ""
                                    last_offset = 0
                                    line_parts = []
                                    line_parts_bytes = 0
                                    break
                                if code == _STREAMING_INTERRUPTED:
                                    if not follow:
                                        raise SandboxError(f"Log stream error: {msg}")
                                    break
                                if code == STREAM_TRUNCATED:
                                    raise SandboxStreamTruncatedError(msg)
                                raise SandboxError(f"Log stream error: {msg}")
                            if entry.log_session_id:
                                session_id = entry.log_session_id
                            if entry.next_log_offset:
                                last_offset = int(entry.next_log_offset)
                            chunk = entry.data.decode("utf-8", errors="replace")
                            if not chunk:
                                continue
                            if follow:
                                attempt = 0
                                last_transport_error = None
                            delivered_any = True
                            line_parts.append(chunk)
                            buf = "".join(line_parts)
                            if "\n" in buf:
                                *complete, remainder = buf.split("\n")
                                for line in complete:
                                    encoded_line = (line + "\n").encode("utf-8")
                                    if len(encoded_line) > MAX_LINE_BUFFER_BYTES:
                                        raise SandboxStreamTruncatedError(
                                            f"Log line exceeded {MAX_LINE_BUFFER_BYTES} bytes"
                                        )
                                    await output_queue.put(line + "\n")
                                remainder_bytes = len(remainder.encode("utf-8")) if remainder else 0
                                if remainder_bytes > MAX_LINE_BUFFER_BYTES:
                                    raise SandboxStreamTruncatedError(
                                        f"Log line exceeded {MAX_LINE_BUFFER_BYTES} bytes"
                                    )
                                line_parts = [remainder] if remainder else []
                                line_parts_bytes = remainder_bytes
                            else:
                                line_parts_bytes = len(buf.encode("utf-8"))
                                if line_parts_bytes > MAX_LINE_BUFFER_BYTES:
                                    raise SandboxStreamTruncatedError(
                                        f"Log line exceeded {MAX_LINE_BUFFER_BYTES} bytes"
                                    )
                        else:
                            if line_parts:
                                leftover = "".join(line_parts)
                                if len(leftover.encode("utf-8")) > MAX_LINE_BUFFER_BYTES:
                                    raise SandboxStreamTruncatedError(
                                        f"Log line exceeded {MAX_LINE_BUFFER_BYTES} bytes"
                                    )
                                await output_queue.put(leftover)
                                line_parts = []
                                line_parts_bytes = 0
                            done = True
                    except grpc.aio.AioRpcError as exc:
                        last_transport_error = exc
                        direct_unavailable = prepared.is_direct and _is_unavailable_rpc_error(exc)
                        if direct_unavailable:
                            await prepared.discard()
                            if is_resume and _is_log_session_wrong_shard_error(exc):
                                session_id = ""
                                last_offset = 0
                                line_parts = []
                                line_parts_bytes = 0
                        if prepared.is_direct and _is_runner_shard_retiring_error(exc):
                            pass
                        elif not (_is_resumable_transport_error(exc) and follow):
                            raise _translate_rpc_error(
                                exc, sandbox_id=sandbox_id, operation="Stream logs"
                            ) from exc
                finally:
                    with contextlib.suppress(Exception):
                        call.cancel()
                    await prepared.release()

                if done:
                    break
                if follow and delivered_any and not session_id:
                    since_time = datetime.now(UTC)
                    tail_lines = None
                attempt += 1
                await asyncio.sleep(
                    min(
                        STREAMING_RESUME_BACKOFF_SECONDS * (2 ** max(attempt - 1, 0)),
                        STREAMING_RESUME_MAX_BACKOFF_SECONDS,
                    )
                )

            if not done:
                if last_transport_error is not None:
                    raise SandboxUnavailableError(
                        f"Log stream for sandbox {sandbox_id} could not be resumed"
                    ) from last_transport_error
                raise SandboxUnavailableError(
                    f"Log stream for sandbox {sandbox_id} could not be resumed"
                )
            inner_exit_clean = True
        except Exception as e:
            # Same guaranteed-delivery pattern as read_file streaming: a
            # bounded output_queue is often full exactly when a terminal
            # stream error arrives. Dropping on QueueFull leaves the
            # consumer hung on the next get().
            try:
                output_queue.put_nowait(e)
            except asyncio.QueueFull:
                asyncio.create_task(output_queue.put(e))
        finally:
            if inner_exit_clean:
                await output_queue.put(None)

    async def _prepare_data_plane_call(
        self,
        permission: int,
        *,
        streaming: bool,
        allow_terminal: bool = False,
    ) -> _PreparedDataPlaneCall:
        """Select direct mTLS or the gateway for one sandbox data operation."""
        await self._ensure_started_async()
        if (self._is_done or self._is_stopping) and not allow_terminal:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")
        if not self._is_done and not self._is_stopping:
            await self._wait_until_running_async()
        await self._ensure_client()
        assert self._stub is not None
        assert self._sandbox_id is not None

        if self._data_plane_mode != DataPlaneMode.GATEWAY:
            try:
                lease = await self._direct_data_plane.acquire(
                    control_stub=self._stub,
                    sandbox_id=self._sandbox_id,
                    auth_metadata=self._auth_metadata,
                    permission=permission,
                    request_timeout=self._request_timeout_seconds,
                    strict=self._data_plane_mode == DataPlaneMode.DIRECT,
                )
                return _PreparedDataPlaneCall(stub=lease.stub, metadata=(), direct_lease=lease)
            except (DirectDataPlaneUnavailable, DirectDataPlanePermissionUnavailable) as exc:
                if self._data_plane_mode == DataPlaneMode.DIRECT:
                    raise SandboxUnavailableError(
                        f"Direct data-plane access is unavailable for sandbox {self._sandbox_id}: "
                        f"{exc}"
                    ) from exc
                logger.debug(
                    "Direct data-plane access unavailable for sandbox %s; using gateway: %s",
                    self._sandbox_id,
                    exc,
                )
            except grpc.RpcError as exc:
                raise _translate_rpc_error(
                    exc,
                    sandbox_id=self._sandbox_id,
                    operation="Connect to sandbox data plane",
                ) from exc

        if streaming:
            channel = await self._get_or_create_streaming_channel()
            stub: Any = sandbox_pb2_grpc.SandboxServiceStub(channel)  # type: ignore[no-untyped-call]
        else:
            stub = self._stub
        return _PreparedDataPlaneCall(stub=stub, metadata=self._auth_metadata)

    async def _prepare_streaming_call(self) -> _PreparedDataPlaneCall:
        """Shared StreamExec preamble and transport selection."""
        return await self._prepare_data_plane_call(
            sandbox_pb2.SANDBOX_DATA_PERMISSION_STREAM_EXEC,
            streaming=True,
        )

    async def _exec_streaming_tty_async(
        self,
        command: Sequence[str],
        output_queue: asyncio.Queue[bytes | Exception | None],
        *,
        stdin_queue: asyncio.Queue[bytes | None],
        stdin_writer: StreamWriter,
        resize_queue: asyncio.Queue[tuple[int, int] | None],
        tty_width: int | None = None,
        tty_height: int | None = None,
        container: str | None = None,
    ) -> TerminalResult:
        """Internal async: Execute TTY command, push raw bytes to output queue.

        Unlike _exec_streaming_async, this method:
        - Does not accumulate stdout/stderr into a final in-memory buffer
          (output is streamed via queues and must be consumed by the caller)
        - Pushes raw bytes to the output queue (no UTF-8 decode)
        - Returns TerminalResult (exit code only)
        - Always operates in TTY mode
        - No client-side timeout (interactive sessions are open-ended)
        """
        if not command:
            raise ValueError("Command cannot be empty")

        prepared: _PreparedDataPlaneCall | None = None
        try:
            prepared = await self._prepare_streaming_call()
            stub = prepared.stub
            auth_metadata = prepared.metadata
            # Narrow for closure capture: _prepare_streaming_call() raised if
            # _sandbox_id was None, but mypy cannot propagate that narrowing
            # into the request_generator closure below.
            sandbox_id = self._sandbox_id
            assert sandbox_id is not None

            logger.debug(
                "Opening TTY session in sandbox %s: %s",
                sandbox_id,
                shlex.join(command),
            )

            exit_code: int | None = None
            shutdown_event = asyncio.Event()
            ready_event = asyncio.Event()
            request_error: Exception | None = None
            exec_start_time = time.monotonic()

            async def request_generator() -> AsyncIterator[sandbox_pb2.ExecStreamRequest]:
                """Generate request messages for the TTY bidirectional stream."""
                init_msg = sandbox_pb2.ExecStreamInit(
                    sandbox_id=sandbox_id,
                    command=list(command),
                    tty=True,
                )
                _set_request_container(init_msg, container)
                if tty_width is not None:
                    init_msg.tty_width = tty_width
                if tty_height is not None:
                    init_msg.tty_height = tty_height
                yield sandbox_pb2.ExecStreamRequest(init=init_msg)

                nonlocal request_error
                ready_timeout = 5.0
                try:
                    await asyncio.wait_for(ready_event.wait(), timeout=ready_timeout)
                except TimeoutError:
                    request_error = SandboxTimeoutError(
                        "stdin ready signal not received within timeout"
                    )
                    shutdown_event.set()
                    raise request_error from None

                # Multiplex stdin + resize.  Reuse tasks across iterations;
                # only recreate a task after its result has been consumed.
                shutdown_task = asyncio.create_task(shutdown_event.wait())
                get_task: asyncio.Task[bytes | None] = asyncio.create_task(stdin_queue.get())
                resize_task: asyncio.Task[tuple[int, int] | None] = asyncio.create_task(
                    resize_queue.get()
                )

                while not shutdown_event.is_set():
                    try:
                        done, _pending = await asyncio.wait(
                            [get_task, shutdown_task, resize_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        if shutdown_task in done:
                            remaining: list[asyncio.Task[Any]] = [get_task, resize_task]
                            for t in remaining:
                                if not t.done():
                                    t.cancel()
                                    with contextlib.suppress(asyncio.CancelledError):
                                        await t
                            return

                        if resize_task in done:
                            dims = resize_task.result()
                            resize_task = asyncio.create_task(resize_queue.get())
                            if dims is not None:
                                w, h = dims
                                yield sandbox_pb2.ExecStreamRequest(
                                    resize=sandbox_pb2.ExecStreamResize(width=w, height=h)
                                )
                            if get_task not in done:
                                continue

                        if get_task in done:
                            data = get_task.result()
                            if data is None:
                                yield sandbox_pb2.ExecStreamRequest(
                                    close=sandbox_pb2.ExecStreamClose()
                                )
                                to_cancel: list[asyncio.Task[Any]] = [resize_task, shutdown_task]
                                for t in to_cancel:
                                    if not t.done():
                                        t.cancel()
                                        with contextlib.suppress(asyncio.CancelledError):
                                            await t
                                return

                            get_task = asyncio.create_task(stdin_queue.get())
                            for i in range(0, len(data), STDIN_CHUNK_SIZE):
                                chunk = data[i : i + STDIN_CHUNK_SIZE]
                                yield sandbox_pb2.ExecStreamRequest(stdin=chunk)
                    except asyncio.CancelledError:
                        cleanup: list[asyncio.Task[Any]] = [get_task, resize_task, shutdown_task]
                        for t in cleanup:
                            if not t.done():
                                t.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await t
                        return

            # No client-side timeout for interactive sessions
            call: grpc.aio.StreamStreamCall[
                sandbox_pb2.ExecStreamRequest, sandbox_pb2.ExecStreamResponse
            ] = stub.StreamExec(
                request_iterator=request_generator(),
                timeout=None,
                metadata=auth_metadata,
            )
            prepared.release_when_done(call)

            # Bounded queue propagates backpressure to gRPC reads — when the
            # consumer is slow, collect_responses() blocks on put(), stopping
            # reads from gRPC.  Without this bound, long-lived TTY sessions
            # can accumulate unlimited protobuf messages in memory.
            response_queue: asyncio.Queue[sandbox_pb2.ExecStreamResponse | Exception | None] = (
                asyncio.Queue(maxsize=STREAMING_RESPONSE_QUEUE_SIZE)
            )

            async def collect_responses() -> None:
                """Collect responses from gRPC streaming call into queue."""
                try:
                    async for response in call:
                        await response_queue.put(response)
                        if response.HasField("exit") or response.HasField("error"):
                            return
                except grpc.aio.AioRpcError as e:
                    await response_queue.put(e)
                except Exception as e:
                    await response_queue.put(e)
                finally:
                    await response_queue.put(None)

            collect_task = asyncio.create_task(collect_responses())

            try:
                while True:
                    item = await response_queue.get()
                    if item is None:
                        break
                    if isinstance(item, grpc.aio.AioRpcError):
                        raise _translate_rpc_error(
                            item, sandbox_id=self._sandbox_id, operation="TTY exec"
                        )
                    if isinstance(item, Exception):
                        raise item

                    response = item
                    if response.HasField("ready"):
                        ready_latency = time.monotonic() - exec_start_time
                        logger.debug(
                            "TTY stdin ready signal received",
                            extra={
                                "sandbox_id": self._sandbox_id,
                                "ready_latency_ms": ready_latency * 1000,
                                "ready_at": response.ready.ready_time.ToDatetime().isoformat(),
                            },
                        )
                        ready_event.set()

                    elif response.HasField("output"):
                        # Raw bytes — no decode, no buffer
                        await output_queue.put(response.output.data)

                    elif response.HasField("exit"):
                        ready_event.set()
                        exit_code = response.exit.exit_code
                        break

                    elif response.HasField("error"):
                        ready_event.set()
                        raise _exec_stream_error(response.error.message, response.error.code)
            except asyncio.CancelledError:
                raise
            finally:
                ready_event.set()
                shutdown_event.set()
                collect_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await collect_task
                stdin_writer.set_exception(SandboxExecutionError("Terminal session has ended"))
                await output_queue.put(None)

            if request_error is not None:
                raise request_error

            return TerminalResult(
                returncode=exit_code if exit_code is not None else -1,
                command=list(command),
            )
        except Exception as exc:
            # Early failures (before the inner try/finally) must propagate to
            # the consumer so it doesn't hang waiting on a sentinel that never
            # arrives.  StreamReader stops iteration on Exception, so no
            # trailing None sentinel is needed.
            try:
                output_queue.put_nowait(exc)
            except asyncio.QueueFull:
                asyncio.create_task(output_queue.put(exc))
            raise
        finally:
            if prepared is not None:
                await prepared.release()

    async def _exec_streaming_async(
        self,
        command: Sequence[str],
        stdout_queue: asyncio.Queue[str | Exception | None],
        stderr_queue: asyncio.Queue[str | Exception | None],
        *,
        cwd: str | None = None,
        check: bool = False,
        timeout_seconds: float | None = None,
        stdin_queue: asyncio.Queue[bytes | None] | None = None,
        stdin_writer: StreamWriter | None = None,
        container: str | None = None,
    ) -> ProcessResult:
        """Internal async: Execute command using StreamExec RPC, push output to queues.

        Uses gRPC bidirectional streaming to receive stdout/stderr as they arrive.
        Buffers output while also pushing to queues for real-time streaming.
        Signals end-of-stream with None sentinel when command completes.

        When stdin_queue is provided, data from it is sent to the process's stdin.
        None in stdin_queue signals EOF. Uses done_writing() for proper half-close.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds

        if not command:
            raise ValueError("Command cannot be empty")

        prepared = await self._prepare_streaming_call()
        stub = prepared.stub
        auth_metadata = prepared.metadata
        # Narrow for closure capture: _prepare_streaming_call() raised if
        # _sandbox_id was None, but mypy cannot propagate that narrowing
        # into the request_generator closure below.
        sandbox_id = self._sandbox_id
        assert sandbox_id is not None

        # Wrap command with cwd if provided
        rpc_command = _wrap_command_with_cwd(command, cwd) if cwd else list(command)

        logger.debug(
            "Executing command (streaming) in sandbox %s: %s",
            sandbox_id,
            shlex.join(command),
        )

        stdout_buffer: list[bytes] = []
        stderr_buffer: list[bytes] = []
        exit_code: int | None = None

        # Shutdown event signals request generator to stop when process exits/times out
        shutdown_event = asyncio.Event()
        # Ready event signals that server is ready to receive stdin data
        ready_event = asyncio.Event()
        # Capture exceptions from request_generator (gRPC swallows them otherwise)
        request_error: Exception | None = None
        # Track exec start time for ready latency measurement (only used when stdin enabled)
        exec_start_time = time.monotonic()

        async def request_generator() -> AsyncIterator[sandbox_pb2.ExecStreamRequest]:
            """Generate request messages for the bidirectional stream.

            Yields init message, then stdin data if enabled, then returns.
            The generator naturally completes when all messages are sent,
            which signals gRPC to half-close the send direction.
            """
            # Yield init message first
            init_msg = sandbox_pb2.ExecStreamInit(
                sandbox_id=sandbox_id,
                command=rpc_command,
            )
            _set_request_container(init_msg, container)
            yield sandbox_pb2.ExecStreamRequest(init=init_msg)

            # If stdin is enabled, wait for ready signal before sending data
            if stdin_queue is not None:
                nonlocal request_error
                # Wait for ready signal with timeout
                ready_timeout = min(5.0, timeout) if timeout is not None else 5.0
                try:
                    await asyncio.wait_for(ready_event.wait(), timeout=ready_timeout)
                except TimeoutError:
                    # Capture error for propagation (gRPC swallows generator exceptions)
                    request_error = SandboxTimeoutError(
                        "stdin ready signal not received within timeout"
                    )
                    shutdown_event.set()
                    raise request_error from None

                # Now safe to send stdin data.  Cache the shutdown task across
                # iterations since it only completes once.
                shutdown_task = asyncio.create_task(shutdown_event.wait())
                while not shutdown_event.is_set():
                    # Wait for either queue data or shutdown signal
                    get_task = asyncio.create_task(stdin_queue.get())
                    try:
                        done, pending = await asyncio.wait(
                            [get_task, shutdown_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Cancel pending tasks
                        for task in pending:
                            if task is shutdown_task:
                                continue  # Reuse across iterations
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task

                        # Check if shutdown was triggered
                        if shutdown_task in done:
                            return

                        # Process queue data
                        data = get_task.result()
                        if data is None:  # EOF sentinel - close stdin
                            yield sandbox_pb2.ExecStreamRequest(close=sandbox_pb2.ExecStreamClose())
                            return

                        # Chunk large data into 64KB pieces
                        for i in range(0, len(data), STDIN_CHUNK_SIZE):
                            chunk = data[i : i + STDIN_CHUNK_SIZE]
                            yield sandbox_pb2.ExecStreamRequest(stdin=chunk)
                    except asyncio.CancelledError:
                        return

        # Create the bidirectional streaming call with request iterator
        call_timeout = (
            timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS if timeout is not None else None
        )
        try:
            call: grpc.aio.StreamStreamCall[
                sandbox_pb2.ExecStreamRequest, sandbox_pb2.ExecStreamResponse
            ] = stub.StreamExec(
                request_iterator=request_generator(),
                timeout=call_timeout,
                metadata=auth_metadata,
            )
        except BaseException:
            await prepared.release(discard=True)
            raise
        prepared.release_when_done(call)

        # Queue decouples stream iteration from our processing.
        # Without this, processing suspends the stream and can cause issues.
        response_queue: asyncio.Queue[sandbox_pb2.ExecStreamResponse | Exception | None] = (
            asyncio.Queue()
        )

        async def collect_responses() -> None:
            """Collect responses from gRPC streaming call into queue."""
            try:
                async for response in call:
                    await response_queue.put(response)
                    if response.HasField("exit") or response.HasField("error"):
                        return
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    parsed = parse_error_info(e)
                    await response_queue.put(
                        SandboxTimeoutError(
                            f"Command {shlex.join(command)} timed out after {timeout}s",
                            reason=parsed.reason if parsed is not None else None,
                            metadata=parsed.metadata if parsed is not None else None,
                            retry_delay=parsed.retry_delay if parsed is not None else None,
                        )
                    )
                else:
                    await response_queue.put(
                        _translate_rpc_error(
                            e, sandbox_id=self._sandbox_id, operation="Execute command"
                        )
                    )
            except Exception as e:
                await response_queue.put(e)
            finally:
                await response_queue.put(None)  # Sentinel

        # Start collector task (sender is handled by gRPC via the request_iterator)
        collect_task = asyncio.create_task(collect_responses())

        try:
            while True:
                item = await response_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                response = item
                if response.HasField("ready"):
                    # Server ready to receive stdin data
                    # Log latency only when stdin is enabled (no overhead when stdin=False)
                    if stdin_queue is not None:
                        ready_latency = time.monotonic() - exec_start_time
                        logger.debug(
                            "stdin ready signal received",
                            extra={
                                "sandbox_id": self._sandbox_id,
                                "ready_latency_ms": ready_latency * 1000,
                                "ready_at": response.ready.ready_time.ToDatetime().isoformat(),
                            },
                        )
                    ready_event.set()

                elif response.HasField("output"):
                    data = response.output.data
                    # Decode as UTF-8 for queue (replacing invalid chars)
                    text = data.decode("utf-8", errors="replace")
                    stream_type = response.output.stream

                    if stream_type == sandbox_pb2.ExecStreamOutput.STREAM_STDOUT:
                        stdout_buffer.append(data)
                        await stdout_queue.put(text)
                    elif stream_type == sandbox_pb2.ExecStreamOutput.STREAM_STDERR:
                        stderr_buffer.append(data)
                        await stderr_queue.put(text)
                    else:
                        logger.warning(
                            "Received output with unexpected stream %s, treating as stdout",
                            stream_type,
                        )
                        stdout_buffer.append(data)
                        await stdout_queue.put(text)

                elif response.HasField("exit"):
                    ready_event.set()  # Unblock stdin sender on terminal message
                    exit_code = response.exit.exit_code
                    break

                elif response.HasField("error"):
                    ready_event.set()  # Unblock stdin sender on terminal message
                    raise _exec_stream_error(response.error.message, response.error.code)
        finally:
            # Unblock stdin sender if still waiting for ready signal
            ready_event.set()
            # Signal request generator to stop consuming stdin queue
            shutdown_event.set()
            # Cancel collector task
            collect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collect_task
            # Signal stdin writer that process has exited (prevents writes to exited process)
            if stdin_writer is not None:
                stdin_writer.set_exception(SandboxExecutionError("Process has exited"))
            # Signal end-of-stream
            await stdout_queue.put(None)
            await stderr_queue.put(None)

        # Propagate any error from request_generator (gRPC swallows generator exceptions)
        if request_error is not None:
            raise request_error

        # Combine buffers into final output
        stdout_bytes = b"".join(stdout_buffer)
        stderr_bytes = b"".join(stderr_buffer)
        final_exit_code = exit_code if exit_code is not None else 0

        logger.debug("Command completed with exit code %d", final_exit_code)

        result = ProcessResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            returncode=final_exit_code,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            command=list(command),
        )

        if check and result.returncode != 0:
            raise SandboxExecutionError(
                f"Command {shlex.join(command)} failed with exit code {result.returncode}",
                exec_result=result,
            )

        return result

    def exec(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        check: bool = False,
        timeout_seconds: float | None = None,
        stdin: bool = False,
        container: str | None = None,
    ) -> Process:
        """Execute command, return Process immediately.

        Note: If sandbox is not yet RUNNING, this method waits for it first.
        The timeout_seconds parameter only applies to command execution, not to
        the initial wait for RUNNING status.

        Args:
            command: Command and arguments to execute
            cwd: Working directory for command execution. Must be an absolute path.
                When specified, the command is wrapped with a shell cd.
            check: If True, raise SandboxExecutionError on non-zero returncode
            timeout_seconds: Timeout for command execution (after sandbox is RUNNING).
                Does not include time waiting for sandbox to reach RUNNING status.
            stdin: If True, enable stdin streaming. Process.stdin will be a
                StreamWriter that can send input to the command. If False (default),
                stdin is closed immediately and Process.stdin is None.
            container: Container name to exec into. Empty/None targets the primary.

        Returns:
            Process handle with streaming stdout/stderr. Call .result() to block
            for the final ProcessResult, or iterate over .stdout/.stderr for
            real-time output. When stdin=True, Process.stdin is a StreamWriter.

        Raises:
            ValueError: If command is empty or cwd is invalid (empty or relative path)

        Examples:
            ```python
            # Get result directly
            process = sb.exec(["echo", "hello"])
            result = process.result()
            print(result.stdout)

            # With working directory
            result = sb.exec(["ls", "-la"], cwd="/app").result()

            # Stream output in real-time
            process = sb.exec(["python", "script.py"])
            for line in process.stdout:
                print(line)
            result = process.result()

            # With stdin streaming
            process = sb.exec(["cat"], stdin=True)
            process.stdin.write(b"hello world").result()
            process.stdin.close().result()
            result = process.result()

            # Async usage
            result = await sb.exec(["echo", "hello"])
            ```
        """
        if not command:
            raise ValueError("Command cannot be empty")
        _validate_cwd(cwd)

        # Track exec count for metrics
        with self._exec_stats_lock:
            self._exec_count += 1

        # Unbounded queues prevent data loss when producer fills queue before consumer iterates.
        # Bounded queues caused race conditions with HTTP/2 stream buffering.
        stdout_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
        stderr_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

        # Stdin queue is bounded to provide backpressure
        stdin_queue: asyncio.Queue[bytes | None] | None = None
        stdin_writer: StreamWriter | None = None
        if stdin:
            stdin_queue = asyncio.Queue(maxsize=StreamWriter.QUEUE_SIZE)
            stdin_writer = StreamWriter(stdin_queue, self._loop_manager)

        process_future = self._loop_manager.run_async(
            self._exec_streaming_async(
                command,
                stdout_queue,
                stderr_queue,
                cwd=cwd,
                check=check,
                timeout_seconds=timeout_seconds,
                stdin_queue=stdin_queue,
                stdin_writer=stdin_writer,
                container=container,
            )
        )

        return Process(
            future=process_future,
            command=list(command),
            stdout=StreamReader(stdout_queue, self._loop_manager),
            stderr=StreamReader(stderr_queue, self._loop_manager),
            stdin=stdin_writer,
            stats_callback=self._on_exec_complete,
        )

    def shell(
        self,
        command: Sequence[str] | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        container: str | None = None,
    ) -> TerminalSession:
        """Start an interactive TTY session in the sandbox.

        Returns a TerminalSession optimized for interactive terminal use:
        raw byte output (no decode/re-encode), no output buffering, and
        fire-and-forget stdin.

        Args:
            command: Shell command to execute. Defaults to ["/bin/bash"].
                Accepts a sequence like ["/bin/sh"] or ["/usr/bin/python3"].
            width: Initial terminal width in columns.
            height: Initial terminal height in rows.
            container: Container name for the TTY session. Empty/None targets
                the primary.

        Returns:
            TerminalSession handle with .output (StreamReader[bytes]),
            .stdin (StreamWriter), and .resize(w, h).

        Raises:
            ValueError: If command is explicitly empty.

        Example:
            ```python
            session = sandbox.shell(width=80, height=24)
            session.stdin.writeline("echo hello").result()
            for chunk in session.output:
                sys.stdout.buffer.write(chunk)
            exit_code = session.wait()
            ```
        """
        if command is None:
            command = ["/bin/bash"]
        if not command:
            raise ValueError("Command cannot be empty")

        with self._exec_stats_lock:
            self._exec_count += 1

        # Bounded queue provides backpressure for potentially unbounded TTY output
        # (interactive shells can run indefinitely). Contrast with exec stdout/stderr
        # queues which are unbounded because exec output is finite.
        output_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(
            maxsize=STREAMING_OUTPUT_QUEUE_SIZE
        )

        stdin_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=StreamWriter.QUEUE_SIZE)
        stdin_writer = StreamWriter(stdin_queue, self._loop_manager)

        resize_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()

        session_future = self._loop_manager.run_async(
            self._exec_streaming_tty_async(
                command=command,
                output_queue=output_queue,
                stdin_queue=stdin_queue,
                stdin_writer=stdin_writer,
                resize_queue=resize_queue,
                tty_width=width,
                tty_height=height,
                container=container,
            )
        )

        def _on_tty_complete(fut: concurrent.futures.Future[TerminalResult]) -> None:
            try:
                tty_result = fut.result()
                self._on_exec_complete(tty_result, None)
            except BaseException as exc:
                self._on_exec_complete(None, exc)

        session_future.add_done_callback(_on_tty_complete)

        return TerminalSession(
            future=session_future,
            command=list(command),
            output=StreamReader(output_queue, self._loop_manager, cancel=session_future.cancel),
            stdin=stdin_writer,
            resize_queue=resize_queue,
        )

    async def _exec_streaming_binary_async(
        self,
        command: Sequence[str],
        *,
        stdin: bytes | AsyncIterable[bytes] | None = None,
        timeout_seconds: float | None = None,
        operation: str,
        filepath: str | None = None,
        _retry_retiring: bool = True,
        container: str | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run one non-TTY StreamExec and return raw stdout/stderr bytes.

        See also ``_exec_streaming_async``, the queue-driven public variant;
        keep handshake/timeout/error-translation in sync between the two.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds
        deadline = time.monotonic() + timeout if timeout is not None else None

        if not command:
            raise ValueError("Command cannot be empty")

        prepared = await self._prepare_streaming_call()
        stub = prepared.stub
        # Narrow for closure capture: _prepare_streaming_call() raised if
        # _sandbox_id was None, but mypy cannot propagate that narrowing
        # into the request_generator closure below.
        sandbox_id = self._sandbox_id
        assert sandbox_id is not None

        # Cap stderr buffering to defend against runaway error output driving the
        # client to OOM. Stdout is uncapped because the read fallback needs the
        # full file body.
        stderr_cap_bytes = 16384
        stdout_buffer = bytearray()
        stderr_buffer: list[bytes] = []
        stderr_total_bytes = 0
        stderr_truncated = False
        exit_code: int | None = None
        ready_event = asyncio.Event()
        shutdown_event = asyncio.Event()
        request_error: Exception | None = None

        async def request_generator() -> AsyncIterator[sandbox_pb2.ExecStreamRequest]:
            init_msg = sandbox_pb2.ExecStreamInit(
                sandbox_id=sandbox_id,
                command=list(command),
            )
            _set_request_container(init_msg, container)
            yield sandbox_pb2.ExecStreamRequest(init=init_msg)

            if stdin is None:
                return

            nonlocal request_error
            ready_timeout = min(5.0, timeout) if timeout is not None else 5.0
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=ready_timeout)
            except TimeoutError:
                request_error = SandboxTimeoutError(
                    "stdin ready signal not received within timeout"
                )
                shutdown_event.set()
                raise request_error from None

            if shutdown_event.is_set():
                return

            if isinstance(stdin, (bytes, bytearray, memoryview)):
                buf = bytes(stdin)
                for i in range(0, len(buf), STDIN_CHUNK_SIZE):
                    if shutdown_event.is_set():
                        return
                    chunk = buf[i : i + STDIN_CHUNK_SIZE]
                    yield sandbox_pb2.ExecStreamRequest(stdin=chunk)
            else:
                async for chunk in stdin:
                    if shutdown_event.is_set():
                        return
                    if not chunk:
                        continue
                    data = _coerce_bytes_chunk(chunk)
                    for i in range(0, len(data), STDIN_CHUNK_SIZE):
                        if shutdown_event.is_set():
                            return
                        yield sandbox_pb2.ExecStreamRequest(stdin=data[i : i + STDIN_CHUNK_SIZE])

            yield sandbox_pb2.ExecStreamRequest(close=sandbox_pb2.ExecStreamClose())

        call_timeout = (
            timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS if timeout is not None else None
        )
        try:
            call: grpc.aio.StreamStreamCall[
                sandbox_pb2.ExecStreamRequest, sandbox_pb2.ExecStreamResponse
            ] = stub.StreamExec(
                request_iterator=request_generator(),
                timeout=call_timeout,
                metadata=prepared.metadata,
            )
        except BaseException:
            await prepared.release(discard=True)
            raise
        prepared.release_when_done(call)

        retiring_error: grpc.RpcError | None = None
        try:
            async for response in call:
                if response.HasField("ready"):
                    ready_event.set()
                elif response.HasField("output"):
                    data = response.output.data
                    stream_type = response.output.stream
                    if stream_type == sandbox_pb2.ExecStreamOutput.STREAM_STDERR:
                        if not stderr_truncated:
                            remaining = stderr_cap_bytes - stderr_total_bytes
                            if remaining >= len(data):
                                stderr_buffer.append(data)
                                stderr_total_bytes += len(data)
                            else:
                                if remaining > 0:
                                    stderr_buffer.append(bytes(data[:remaining]))
                                    stderr_total_bytes += remaining
                                stderr_buffer.append(b"... [stderr truncated]")
                                stderr_truncated = True
                    else:
                        stdout_buffer.extend(data)
                elif response.HasField("exit"):
                    ready_event.set()
                    exit_code = response.exit.exit_code
                    break
                elif response.HasField("error"):
                    ready_event.set()
                    raise _exec_stream_error(response.error.message, response.error.code)
        except grpc.RpcError as e:
            # Surface the specific stdin-ready timeout message instead of a
            # generic CANCELLED translation when grpcio masks request_error
            # by cancelling the receiver side.
            if request_error is not None:
                raise request_error from e
            if prepared.is_direct and _is_unavailable_rpc_error(e):
                await prepared.discard()
            replayable_stdin = stdin is None or isinstance(stdin, bytes)
            if (
                prepared.is_direct
                and _retry_retiring
                and replayable_stdin
                and _is_runner_shard_retiring_error(e)
            ):
                retiring_error = e
            else:
                raise _translate_rpc_error(
                    e,
                    sandbox_id=self._sandbox_id,
                    operation=operation,
                    filepath=filepath,
                ) from e
        finally:
            ready_event.set()
            shutdown_event.set()
            with contextlib.suppress(Exception):
                call.cancel()

        if retiring_error is not None:
            await prepared.release(discard=True)
            retry_timeout = self._remaining_budget(deadline)
            if retry_timeout is not None and retry_timeout <= 0:
                raise SandboxTimeoutError(f"{operation} timed out") from retiring_error
            return await self._exec_streaming_binary_async(
                command,
                stdin=stdin,
                timeout_seconds=retry_timeout,
                operation=operation,
                filepath=filepath,
                _retry_retiring=False,
                container=container,
            )

        if request_error is not None:
            raise request_error

        if exit_code is None:
            raise SandboxFileError(
                f"{operation} ended without exit status from sandbox",
                filepath=filepath,
            )

        return (
            exit_code,
            bytes(stdout_buffer),
            b"".join(stderr_buffer),
        )

    async def _read_file_unary_async(
        self, filepath: str, timeout: float, *, container: str | None = None
    ) -> bytes:
        request = sandbox_pb2.ReadFileRequest(
            sandbox_id=self._sandbox_id,
            path=filepath,
        )
        _set_request_container(request, container)
        for attempt in range(2):
            prepared = await self._prepare_data_plane_call(
                sandbox_pb2.SANDBOX_DATA_PERMISSION_READ_FILE,
                streaming=False,
            )
            discard = False
            try:
                response = await prepared.stub.ReadFile(
                    request, timeout=timeout, metadata=prepared.metadata
                )
                return bytes(response.content)
            except grpc.RpcError as e:
                discard = prepared.is_direct and _is_unavailable_rpc_error(e)
                if discard:
                    await prepared.discard()
                if prepared.is_direct and attempt == 0 and _is_runner_shard_retiring_error(e):
                    continue
                raise _translate_rpc_error(
                    e,
                    sandbox_id=self._sandbox_id,
                    operation="Read file",
                    filepath=filepath,
                ) from e
            finally:
                await prepared.release(discard=discard)
        raise AssertionError("unreachable")

    async def _read_file_via_exec_streaming(
        self,
        filepath: str,
        timeout: float,
        *,
        expected_size: int | None = None,
        container: str | None = None,
    ) -> bytes:
        script = (
            "path=$1\n"
            'if [ ! -e "$path" ]; then\n'
            '  printf "%s\\n" "File not found: $path" >&2\n'
            "  exit 2\n"
            "fi\n"
            'if [ -d "$path" ]; then\n'
            '  printf "%s\\n" "Path is a directory: $path" >&2\n'
            "  exit 3\n"
            "fi\n"
            'cat < "$path"\n'
        )
        returncode, stdout, stderr = await self._exec_streaming_binary_async(
            ["/bin/sh", "-c", script, "cwsandbox-read-file", filepath],
            timeout_seconds=timeout,
            operation="Read file",
            filepath=filepath,
            container=container,
        )
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            if not detail:
                detail = f"fallback command exited with status {returncode}"
            # Map fallback script exit codes onto the same AIP-193 reasons the
            # unary path returns, so callers can switch on ``reason`` without
            # caring which path produced the error.
            if returncode == 2:
                raise SandboxFileError(
                    f"File operation failed ({CWSANDBOX_FILE_NOT_FOUND}): {detail}",
                    filepath=filepath,
                    reason=CWSANDBOX_FILE_NOT_FOUND,
                )
            if returncode == 3:
                raise SandboxFileError(
                    f"File operation failed ({CWSANDBOX_FILE_IS_DIRECTORY}): {detail}",
                    filepath=filepath,
                    reason=CWSANDBOX_FILE_IS_DIRECTORY,
                )
            raise SandboxFileError(
                f"Failed to read file '{filepath}' via exec-stream fallback: {detail}",
                filepath=filepath,
                reason=CWSANDBOX_FILE_IO_FAILED,
            )
        # Integrity check (shared with read_file_streaming): detect a silently
        # truncated read and surface it as a typed CWSANDBOX_FILE_TRUNCATED
        # rather than returning a partial file as if complete (issue #1172).
        # The expected size is the server-reported pre-read size from the
        # FILE_TOO_LARGE metadata that triggered this fallback — no extra stat
        # round-trip is needed, and because it was captured *before* the read it
        # cannot false-positive on a file that grows during the read.
        self._verify_no_truncation(
            filepath, delivered=len(stdout), expected=expected_size, operation="read_file"
        )
        return stdout

    async def _stat_file_size_async(
        self, filepath: str, timeout: float, *, container: str | None = None
    ) -> int | None:
        """Best-effort: ask the sandbox for the file's size in bytes.

        Returns ``None`` if the size could not be determined (stat unavailable,
        unexpected output, transient transport error). Used by
        ``read_file_streaming`` to capture a pre-read size baseline for the
        truncation check; a ``None`` result means the check is skipped rather
        than raising: a stat failure on its own is not a streaming-read failure.
        """
        try:
            returncode, stdout, _stderr = await self._exec_streaming_binary_async(
                ["/bin/sh", "-c", 'stat -c %s -- "$1" 2>/dev/null', "cwsandbox-stat", filepath],
                timeout_seconds=timeout,
                operation="Stat file size",
                filepath=filepath,
                container=container,
            )
        except Exception:
            return None
        if returncode != 0:
            return None
        text = stdout.decode("utf-8", errors="replace").strip()
        try:
            value = int(text)
        except ValueError:
            return None
        return value if value >= 0 else None

    @staticmethod
    def _remaining_budget(deadline: float | None) -> float | None:
        """Seconds left until ``deadline`` (a ``time.monotonic()`` value).

        Returns ``None`` when no deadline was set (untimed operation) and a
        floor of 0.0 once the deadline has passed, so a downstream RPC sees a
        non-negative timeout rather than a negative one.
        """
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _stat_budget(self, deadline: float | None) -> float:
        """Timeout for the pre-read ``stat``: the remaining budget, capped short.

        ``stat`` is an O(1) metadata lookup, so it is capped at
        ``STAT_INTEGRITY_TIMEOUT_SECONDS`` and never allowed to exceed the
        operation's remaining wall-clock budget: the stat and the read it
        precedes share one deadline and together stay within the caller's
        timeout.
        """
        remaining = self._remaining_budget(deadline)
        if remaining is None:
            return STAT_INTEGRITY_TIMEOUT_SECONDS
        return min(STAT_INTEGRITY_TIMEOUT_SECONDS, remaining)

    def _verify_no_truncation(
        self, filepath: str, *, delivered: int, expected: int | None, operation: str
    ) -> None:
        """Raise CWSANDBOX_FILE_TRUNCATED if a streamed read came back short.

        Pure comparison shared by read_file (exec-stream fallback) and
        read_file_streaming. ``expected`` is the file's size captured *before*
        the read: the server-reported size for the read_file fallback, or a
        pre-read ``stat`` for read_file_streaming. On a backend where the
        streaming channel silently truncates (e.g. the lossless gate is off),
        the read command exits 0 having produced the whole file while the client
        received only a prefix; without this the caller would consume a partial
        read as if complete (issue #1172).

        Only a SHORT read (``delivered < expected``) is flagged. Specifically:
        - ``expected is None`` (size unknown: server omitted it, or ``stat`` is
          unavailable on a distroless/scratch image): skip. The check is a
          best-effort backstop, not a guarantee; the public docstrings scope
          this.
        - ``expected == 0`` (pseudo-files such as ``/proc/*`` and ``/sys/*``
          report size 0 while ``cat`` legitimately yields content): skip, to
          avoid a false-positive on a fully-delivered read.
        - ``delivered >= expected``: not short, so no raise. Because ``expected``
          is the *pre-read* size, a file appended to during the read grows the
          delivered byte count above the baseline rather than below it: a
          benign concurrent append never trips the check (the false-positive
          that an after-the-fact stat would produce).
        """
        if expected is None or expected == 0 or delivered >= expected:
            return
        raise SandboxFileError(
            f"{operation} of '{filepath}' was truncated: got {delivered} of "
            f"{expected} bytes. Use read_file_streaming and drain it promptly, "
            f"or read the file in smaller parts.",
            filepath=filepath,
            reason=CWSANDBOX_FILE_TRUNCATED,
            metadata={
                "filepath": filepath,
                "operation": operation,
                "size_bytes": str(expected),
                "bytes_delivered": str(delivered),
            },
        )

    async def _read_file_async(
        self,
        filepath: str,
        timeout: float,
        *,
        container: str | None = None,
    ) -> bytes:
        """Internal async: Read a file from the sandbox filesystem."""
        await self._ensure_started_async()
        if self._is_done or self._is_stopping:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        # Wait for sandbox to be running before file operations
        await self._wait_until_running_async()

        await self._ensure_client()
        assert self._stub is not None

        logger.debug("Reading file from sandbox %s: %s", self._sandbox_id, filepath)

        try:
            return await self._read_file_unary_async(filepath, timeout, container=container)
        except SandboxFileError as e:
            if e.reason != CWSANDBOX_FILE_TOO_LARGE:
                raise
            self._record_observed_cap(e)
            size = self._parse_size_from_metadata(e)
            if size is None or size > MAX_AUTO_FALLBACK_BYTES:
                # Refuse to auto-fall back when the file is over the ceiling
                # or when the server did not report its size. The latter is
                # unverifiable from the client; the safe default is to surface
                # the typed error and let the caller opt into streaming.
                raise
            self._notify_streaming_fallback_once(
                "Read file", filepath, size, suggest_method="read_file_streaming"
            )
            # ``size`` is the server-reported pre-read size: feed it to the
            # truncation check directly, so the fallback needs no extra stat
            # round-trip and cannot false-positive on a growing file.
            return await self._read_file_via_exec_streaming(
                filepath, timeout, expected_size=size, container=container
            )
        except SandboxResourceExhaustedError:
            # Backend resource pressure is indistinguishable from message-size
            # rejects on this code path without inspecting error text; remote
            # file size is unknown to the client until first attempt fails, so
            # fall back broadly. Writes are conservative because the client
            # knows the local payload size. The remote size is unknown here, so
            # the truncation check is skipped (no reliable pre-read baseline);
            # see read_file's docstring for the resulting caveat.
            logger.debug(
                "Falling back to exec-streaming read for sandbox %s: %s",
                self._sandbox_id,
                filepath,
            )
            return await self._read_file_via_exec_streaming(filepath, timeout, container=container)

    def read_file(
        self,
        filepath: str,
        *,
        timeout_seconds: float | None = None,
        container: str | None = None,
    ) -> OperationRef[bytes]:
        """Read file from sandbox, return OperationRef immediately.

        Args:
            filepath: Path to file in sandbox
            timeout_seconds: Timeout for the operation
            container: Container to read from. Empty/None targets the primary.

        Returns:
            OperationRef[bytes]: Use .result() to block and retrieve contents.

        Behavior:
            Files up to ~32 MiB are read in a single unary call. Larger files
            (up to ~256 MiB) transparently fall back to a streaming read: the
            first such fallback per Sandbox logs once at INFO. When the server
            reports the file's size, files above ~256 MiB are refused with
            ``CWSANDBOX_FILE_TOO_LARGE``; use ``read_file_streaming`` for those.

            The whole result is held in memory regardless of path. The client
            cannot always know the remote size in advance (e.g. when the backend
            signals the oversized read via resource exhaustion rather than a
            sized ``CWSANDBOX_FILE_TOO_LARGE``), so a very large file can still
            be buffered in full rather than refused: prefer
            ``read_file_streaming`` for anything large to consume it
            incrementally and bound memory.

        Raises:
            SandboxFileError: with ``reason == CWSANDBOX_FILE_TOO_LARGE`` when
                the file exceeds the server cap and the server reported its
                size; or with ``reason == CWSANDBOX_FILE_TRUNCATED`` when a
                streamed read comes back short of the file's size (truncation
                detected against the pre-read size).
            SandboxStreamBackpressureError: when a large read falls back to
                streaming and the output is produced faster than the client
                reads it (a subclass of SandboxExecutionError).

        Examples:
            ```python
            data = sb.read_file("/output/result.txt").result()
            ```
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds
        future = self._loop_manager.run_async(
            self._read_file_async(filepath, timeout, container=container)
        )
        return OperationRef(future)

    async def _write_file_unary_async(
        self,
        filepath: str,
        contents: bytes,
        timeout: float,
        *,
        container: str | None = None,
    ) -> None:
        request = sandbox_pb2.WriteFileRequest(
            sandbox_id=self._sandbox_id,
            path=filepath,
            content=contents,
        )
        _set_request_container(request, container)
        for attempt in range(2):
            prepared = await self._prepare_data_plane_call(
                sandbox_pb2.SANDBOX_DATA_PERMISSION_WRITE_FILE,
                streaming=False,
            )
            discard = False
            try:
                await prepared.stub.WriteFile(
                    request,
                    timeout=timeout,
                    metadata=prepared.metadata,
                )
                return
            except grpc.RpcError as e:
                discard = prepared.is_direct and _is_unavailable_rpc_error(e)
                if discard:
                    await prepared.discard()
                if prepared.is_direct and attempt == 0 and _is_runner_shard_retiring_error(e):
                    continue
                raise _translate_rpc_error(
                    e,
                    sandbox_id=self._sandbox_id,
                    operation="Write file",
                    filepath=filepath,
                ) from e
            finally:
                await prepared.release(discard=discard)

    async def _write_file_via_exec_streaming(
        self,
        filepath: str,
        contents: bytes,
        timeout: float,
        *,
        container: str | None = None,
    ) -> None:
        script = (
            "path=$1\n"
            "expected=$2\n"
            'if ! cat > "$path"; then\n'
            '  printf "%s\\n" "Failed to write input stream to $path" >&2\n'
            "  exit 1\n"
            "fi\n"
            'actual=$(wc -c < "$path") || exit 1\n'
            "set -- $actual\n"
            "actual=$1\n"
            'if [ "$actual" != "$expected" ]; then\n'
            '  printf "%s\\n" "Expected $expected bytes but wrote $actual bytes; '
            'target may be partial or truncated" >&2\n'
            "  exit 1\n"
            "fi\n"
        )
        try:
            returncode, _, stderr = await self._exec_streaming_binary_async(
                ["/bin/sh", "-c", script, "cwsandbox-write-file", filepath, str(len(contents))],
                stdin=contents,
                timeout_seconds=timeout,
                operation="Write file",
                filepath=filepath,
                container=container,
            )
        except SandboxStreamBackpressureError:
            # A too-slow producer is its own actionable, typed failure. Let it
            # propagate so write_file surfaces the SAME error as write_file_streaming
            # and read_file for this condition — remasking it as a generic
            # "may be truncated" SandboxFileError would hide the real cause and
            # diverge the public error model across the three entry points.
            raise
        except (TypeError, ValueError):
            # A caller programming error (e.g. a non-bytes-like chunk) is not a
            # transport/truncation failure — let it propagate unchanged rather
            # than disguising it as a "may be truncated" SandboxFileError, which
            # would send the caller debugging the network instead of their code.
            raise
        except Exception as e:
            # The exec-stream write does direct-cat-to-target (no temp file +
            # rename), so any interruption — gRPC timeout, transport error,
            # mid-stream cancel — may leave a partially written file. Surface
            # that to callers so they can decide whether to retry vs delete.
            raise SandboxFileError(
                f"Failed to write file '{filepath}' via exec-stream fallback. "
                f"The target may be partial or truncated. Upstream error: {e!r}",
                filepath=filepath,
            ) from e
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            if not detail:
                detail = f"fallback command exited with status {returncode}"
            raise SandboxFileError(
                "Failed to write file "
                f"'{filepath}' via exec-stream fallback: {detail}. "
                "The target may be partial or truncated.",
                filepath=filepath,
            )

    def _file_op_cap(self) -> int:
        """Per-call cap to apply before dispatching a unary file op.

        Uses the server-reported cap when one has been observed; otherwise
        falls back to ``DEFAULT_FILE_OPERATION_CAP_BYTES``. The result is clamped
        to ``MAX_FILE_UNARY_BYTES`` (a frame-safe ceiling below the channel's max
        message length): even if a cluster reports a cap at or above the channel
        limit, a payload at the reported cap could not survive protobuf framing
        on the unary path, so anything above the clamp is routed to streaming
        instead of being sent unary and rejected for frame size.
        """
        observed = self._observed_file_op_cap_bytes
        if observed is not None and observed > 0:
            return min(observed, MAX_FILE_UNARY_BYTES)
        return min(DEFAULT_FILE_OPERATION_CAP_BYTES, MAX_FILE_UNARY_BYTES)

    def _record_observed_cap(self, exc: SandboxFileError) -> None:
        """Cache the server's max_size_bytes when present on a FILE_TOO_LARGE.

        The raw server value is stored as observed; the frame-safe clamp is
        applied at the point of use in ``_file_op_cap`` rather than here, so the
        cached value stays a faithful record of what the server reported.
        """
        meta = exc.metadata or {}
        raw = meta.get("max_size_bytes")
        if not raw:
            return
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return
        if value > 0:
            self._observed_file_op_cap_bytes = value

    @staticmethod
    def _parse_size_from_metadata(exc: SandboxFileError) -> int | None:
        """Return ``size_bytes`` from ErrorInfo metadata, or None if absent."""
        meta = exc.metadata or {}
        raw = meta.get("size_bytes")
        if not raw:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def _notify_streaming_fallback_once(
        self,
        operation: str,
        filepath: str,
        size: int,
        suggest_method: str,
    ) -> None:
        """One-shot notification when an auto-fallback to streaming fires.

        Emits INFO on the first occurrence per Sandbox instance and DEBUG
        thereafter. INFO is deliberate: the fallback is a recovered path,
        not a failure, and elevating to WARNING would couple SDK-internal
        recovery to caller incident pipelines.
        """
        if self._streaming_fallback_warned:
            logger.debug(
                "Streaming fallback for %s on %s (%d bytes)",
                operation,
                filepath,
                size,
            )
            return
        logger.info(
            "%s for '%s' (%d bytes) is being streamed; prefer %s() for large files.",
            operation,
            filepath,
            size,
            suggest_method,
        )
        self._streaming_fallback_warned = True

    async def _write_file_async(
        self,
        filepath: str,
        contents: bytes,
        timeout: float,
        *,
        container: str | None = None,
    ) -> None:
        """Internal async: Write a file to the sandbox filesystem."""
        await self._ensure_started_async()
        if self._is_done or self._is_stopping:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")

        await self._wait_until_running_async()

        await self._ensure_client()
        assert self._stub is not None

        size = len(contents)
        logger.debug(
            "Writing file to sandbox %s: %s (%d bytes)",
            self._sandbox_id,
            filepath,
            size,
        )

        if size > MAX_AUTO_FALLBACK_BYTES:
            raise SandboxFileError(
                f"Refusing to write '{filepath}': {size} bytes exceeds the "
                f"auto-fallback ceiling of {MAX_AUTO_FALLBACK_BYTES} bytes. "
                "Use write_file_streaming() to stream large payloads.",
                filepath=filepath,
                reason=CWSANDBOX_FILE_TOO_LARGE,
                metadata={
                    "size_bytes": str(size),
                    "max_size_bytes": str(MAX_AUTO_FALLBACK_BYTES),
                    # Public method name, not the internal RPC ("AddFile"), so
                    # the metadata["operation"] value is a consistent contract
                    # across read_file / write_file / *_streaming.
                    "operation": "write_file",
                    "filepath": filepath,
                },
            )

        cap = self._file_op_cap()
        if size > cap:
            self._notify_streaming_fallback_once(
                "Write file", filepath, size, suggest_method="write_file_streaming"
            )
            await self._write_file_via_exec_streaming(
                filepath, contents, timeout, container=container
            )
            return

        try:
            await self._write_file_unary_async(filepath, contents, timeout, container=container)
        except SandboxFileError as e:
            if e.reason != CWSANDBOX_FILE_TOO_LARGE:
                raise
            self._record_observed_cap(e)
            if size > MAX_AUTO_FALLBACK_BYTES:
                raise
            self._notify_streaming_fallback_once(
                "Write file", filepath, size, suggest_method="write_file_streaming"
            )
            await self._write_file_via_exec_streaming(
                filepath, contents, timeout, container=container
            )
        except SandboxResourceExhaustedError as e:
            # Legacy gRPC frame-size signal. Distinguishable from real backend
            # pressure only by message text, so the fallback fires only on the
            # frame-size shape; everything else re-raises.
            text = str(e).lower()
            if "message" not in text or "larger than max" not in text:
                raise
            logger.debug(
                "Falling back to exec-streaming write for sandbox %s: %s",
                self._sandbox_id,
                filepath,
            )
            await self._write_file_via_exec_streaming(
                filepath, contents, timeout, container=container
            )

    def write_file(
        self,
        filepath: str,
        contents: bytes,
        *,
        timeout_seconds: float | None = None,
        container: str | None = None,
    ) -> OperationRef[None]:
        """Write file to sandbox, return OperationRef immediately.

        Args:
            filepath: Path to file in sandbox
            contents: File contents as bytes
            timeout_seconds: Timeout for the operation
            container: Container to write into. Empty/None targets the primary.

        Returns:
            OperationRef[None]: Use .result() to block until complete.

        Behavior:
            Payloads up to ~32 MiB are written in a single unary call. Larger
            payloads (up to ~256 MiB) transparently fall back to a streaming
            write: the first such fallback per Sandbox logs once at INFO.
            Payloads above ~256 MiB are refused; use ``write_file_streaming``
            for those.

        Raises:
            SandboxFileError: with ``reason == CWSANDBOX_FILE_TOO_LARGE`` when
                the payload exceeds the server cap, or (without that reason) if
                a streamed write fails mid-stream and may have left a partial
                file.
            SandboxStreamBackpressureError: when a large write falls back to
                streaming and the source produces data faster than it can be
                sent (a subclass of SandboxExecutionError).

        Examples:
            ```python
            sb.write_file("/input/data.txt", b"content").result()
            ```
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds
        future = self._loop_manager.run_async(
            self._write_file_async(filepath, contents, timeout, container=container)
        )
        return OperationRef(future)

    async def _write_file_streaming_async(
        self,
        filepath: str,
        source: bytes | Iterable[bytes] | AsyncIterable[bytes],
        timeout: float,
        *,
        container: str | None = None,
    ) -> None:
        await self._ensure_started_async()
        if self._is_done or self._is_stopping:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")
        await self._wait_until_running_async()
        await self._ensure_client()
        assert self._stub is not None

        async def to_async_iter() -> AsyncIterator[bytes]:
            if isinstance(source, (bytes, bytearray, memoryview)):
                buf = source if isinstance(source, bytes) else bytes(source)
                for i in range(0, len(buf), STREAMING_WRITE_CHUNK_SIZE):
                    yield buf[i : i + STREAMING_WRITE_CHUNK_SIZE]
                return
            if isinstance(source, AsyncIterable):
                async for chunk in source:
                    yield _coerce_bytes_chunk(chunk)
                return
            # Synchronous iterable: pull each chunk off the event loop so a
            # blocking source (file handle, NFS/FUSE read, network generator)
            # parks an executor thread instead of stalling the shared loop and
            # every other operation on it.
            async for chunk in _iter_sync_source_in_executor(source):
                yield chunk

        script = (
            "path=$1\n"
            'if ! cat > "$path"; then\n'
            '  printf "%s\\n" "Failed to write input stream to $path" >&2\n'
            "  exit 1\n"
            "fi\n"
        )
        try:
            returncode, _, stderr = await self._exec_streaming_binary_async(
                ["/bin/sh", "-c", script, "cwsandbox-write-file-streaming", filepath],
                stdin=to_async_iter(),
                timeout_seconds=timeout,
                operation="Stream write file",
                filepath=filepath,
                container=container,
            )
        except SandboxStreamBackpressureError:
            # A too-slow producer is its own actionable failure — surface the
            # typed backpressure error with its guidance, don't remask it as a
            # generic "may be truncated" file error.
            raise
        except (TypeError, ValueError):
            # A non-bytes-like chunk from the caller's source raises TypeError
            # (see _coerce_bytes_chunk); that is a caller programming error, not
            # a transport/truncation failure. Let it propagate unchanged so the
            # documented "raises TypeError" contract holds, rather than
            # disguising it as a SandboxFileError.
            raise
        except Exception as e:
            raise SandboxFileError(
                f"Failed to stream-write file '{filepath}'. "
                f"The target may be partial or truncated. Upstream error: {e!r}",
                filepath=filepath,
            ) from e
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            if not detail:
                detail = f"stream-write command exited with status {returncode}"
            raise SandboxFileError(
                f"Failed to stream-write file '{filepath}': {detail}. "
                "The target may be partial or truncated.",
                filepath=filepath,
            )

    def write_file_streaming(
        self,
        filepath: str,
        source: bytes | Iterable[bytes] | AsyncIterable[bytes],
        *,
        timeout_seconds: float | None = None,
        container: str | None = None,
    ) -> OperationRef[None]:
        """Stream a file to the sandbox without materializing the full payload.

        Prefer this over ``write_file`` for payloads larger than roughly
        32 MiB, or any time the data is already an iterator (file handle,
        generator, async producer).

        Args:
            filepath: Absolute path inside the sandbox.
            source: Payload as ``bytes``, a sync ``Iterable[bytes]``, or an
                ``AsyncIterable[bytes]``. Input is split into frame-safe chunks
                before transmission.
                Yielded items must be ``bytes``, ``bytearray``, or
                ``memoryview``; anything else raises ``TypeError``.
            timeout_seconds: Wall-clock timeout for the streaming write.

        Returns:
            ``OperationRef[None]``: call ``.result()`` to block until complete.

        Raises:
            SandboxStreamBackpressureError: if the source produces data faster
                than it can be sent and the stream is ended early. Yield from a
                source you can pace, or pre-chunk large uploads; see that
                exception's docstring for guidance.

        Caveats:
            The destination is written directly (no temp-and-rename). A
            mid-stream cancel or transport error may leave a partial file.
            The streaming transfer also does not survive a sandbox restart.

            A synchronous source (e.g. a file handle from ``open(...)``) is
            pulled on a worker thread, so a blocking ``read`` does not stall the
            SDK's event loop. An async source is awaited directly. Either is
            fine; pick whichever is more natural for your data.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds
        future = self._loop_manager.run_async(
            self._write_file_streaming_async(filepath, source, timeout, container=container)
        )
        return OperationRef(future)

    async def _read_file_streaming_async(
        self,
        filepath: str,
        output_queue: asyncio.Queue[bytes | Exception | None],
        timeout: float,
        _retry_retiring: bool = True,
        *,
        container: str | None = None,
    ) -> None:
        try:
            # Absolute wall-clock deadline for the whole operation (stat + read),
            # so the pre-read stat consumes from the same budget as the read and
            # the two together never exceed the caller's timeout.
            deadline = time.monotonic() + timeout if timeout is not None else None
            await self._ensure_started_async()
            if self._is_done or self._is_stopping:
                raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
            if self._sandbox_id is None:
                raise SandboxNotRunningError("No sandbox is running")
            await self._wait_until_running_async()
            # Capture into a local so the inner closure has a non-Optional binding.
            # Subsequent awaits invalidate mypy's narrowing of self._sandbox_id.
            assert self._sandbox_id is not None
            sandbox_id = self._sandbox_id

            # Capture the file's size BEFORE the read so the post-read truncation
            # check has a stable baseline: a file appended to during the read
            # only grows the delivered count, so a benign concurrent append can
            # never look like a short read (issue #1172 false-positive fix). The
            # stat draws from the operation's remaining budget, capped short
            # because it is an O(1) metadata lookup.
            expected_size = await self._stat_file_size_async(
                filepath, self._stat_budget(deadline), container=container
            )

            prepared = await self._prepare_streaming_call()
            stub = prepared.stub

            stderr_buf = bytearray()
            stderr_cap = STREAMING_READ_STDERR_CAP_BYTES
            exit_code: int | None = None
            total_bytes = 0

            async def request_generator() -> AsyncIterator[sandbox_pb2.ExecStreamRequest]:
                # Init only — no stdin frames and no explicit close. The command
                # reads the file from its argument, never stdin, so an early
                # stdin close is unnecessary and can race server-side stream
                # setup. The request stream half-closes when this generator
                # returns, matching the stdin-less exec path.
                init_msg = sandbox_pb2.ExecStreamInit(
                    sandbox_id=sandbox_id,
                    command=["/bin/cat", "--", filepath],
                )
                _set_request_container(init_msg, container)
                yield sandbox_pb2.ExecStreamRequest(init=init_msg)

            # The read gets the budget remaining after the pre-read stat, so the
            # two phases together honor the caller's overall timeout.
            read_budget = self._remaining_budget(deadline)
            call_timeout = (
                read_budget + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS
                if read_budget is not None
                else None
            )
            try:
                call: grpc.aio.StreamStreamCall[
                    sandbox_pb2.ExecStreamRequest, sandbox_pb2.ExecStreamResponse
                ] = stub.StreamExec(
                    request_iterator=request_generator(),
                    timeout=call_timeout,
                    metadata=prepared.metadata,
                )
            except BaseException:
                await prepared.release(discard=True)
                raise
            prepared.release_when_done(call)
            retry_retiring = False
            try:
                async for response in call:
                    if response.HasField("output"):
                        stream_type = response.output.stream
                        data = response.output.data
                        if stream_type == sandbox_pb2.ExecStreamOutput.STREAM_STDERR:
                            remaining = stderr_cap - len(stderr_buf)
                            if remaining > 0:
                                stderr_buf.extend(data[:remaining])
                        else:
                            total_bytes += len(data)
                            await output_queue.put(bytes(data))
                    elif response.HasField("exit"):
                        exit_code = response.exit.exit_code
                        break
                    elif response.HasField("error"):
                        raise _exec_stream_error(response.error.message, response.error.code)
            except grpc.RpcError as exc:
                if prepared.is_direct and _is_unavailable_rpc_error(exc):
                    await prepared.discard()
                if prepared.is_direct and _retry_retiring and _is_runner_shard_retiring_error(exc):
                    retry_retiring = True
                else:
                    raise _translate_rpc_error(
                        exc,
                        sandbox_id=self._sandbox_id,
                        operation="read_file_streaming",
                        filepath=filepath,
                    ) from exc
            finally:
                with contextlib.suppress(Exception):
                    call.cancel()

            if retry_retiring:
                await prepared.release(discard=True)
                retry_budget = self._remaining_budget(deadline)
                if retry_budget is None or retry_budget > 0:
                    await self._read_file_streaming_async(
                        filepath,
                        output_queue,
                        timeout if retry_budget is None else retry_budget,
                        _retry_retiring=False,
                        container=container,
                    )
                    return
                raise SandboxTimeoutError(f"Timed out reading file '{filepath}'")

            if exit_code is None:
                raise SandboxFileError(
                    f"Stream-read of '{filepath}' ended without exit status",
                    filepath=filepath,
                )
            if exit_code != 0:
                detail = bytes(stderr_buf).decode("utf-8", errors="replace").strip()
                if not detail:
                    detail = f"stream-read command exited with status {exit_code}"
                raise SandboxFileError(
                    f"Failed to stream-read file '{filepath}': {detail}",
                    filepath=filepath,
                    reason=CWSANDBOX_FILE_IO_FAILED,
                )
            # Integrity check (shared with the read_file fallback): a
            # short-stream-with-exit-0 means the channel silently dropped output;
            # surface as a typed CWSANDBOX_FILE_TRUNCATED (issue #1172). Gated by
            # size band: silent truncation only manifests on large payloads, so
            # below TRUNCATION_CHECK_MIN_BYTES the check cannot catch anything and
            # is skipped (passing expected=None). ``expected_size`` was captured
            # before the read, so a concurrent append cannot false-positive.
            check_expected = (
                expected_size
                if expected_size is not None and expected_size >= TRUNCATION_CHECK_MIN_BYTES
                else None
            )
            self._verify_no_truncation(
                filepath,
                delivered=total_bytes,
                expected=check_expected,
                operation="read_file_streaming",
            )
            await output_queue.put(None)
        except Exception as exc:
            # Deliver the terminal exception with GUARANTEED delivery, not
            # best-effort. On the slow-reader path the bounded output_queue is
            # full exactly when the terminal STREAM_BACKPRESSURE / error frame
            # arrives; a non-blocking put that drops on QueueFull would silently
            # lose the error and leave the consumer blocked forever on the next
            # get() — turning the loud failure this feature exists to surface
            # back into a silent hang. This runs on the long-lived background
            # loop, so create_task on QueueFull is valid.
            try:
                output_queue.put_nowait(exc)
            except asyncio.QueueFull:
                asyncio.create_task(output_queue.put(exc))

    def read_file_streaming(
        self,
        filepath: str,
        *,
        timeout_seconds: float | None = None,
        container: str | None = None,
    ) -> StreamReader[bytes]:
        """Stream a file from the sandbox in chunks without buffering the whole payload.

        Prefer this over ``read_file`` for files larger than roughly 32 MiB,
        or any time you want to consume the file incrementally (write to
        disk, hash on the fly, parse line by line).

        For large files, the SDK captures the file's size *before* reading and,
        once the stream finishes, verifies that at least that many bytes were
        delivered. If fewer arrived, the iterator raises ``SandboxFileError``
        with reason ``CWSANDBOX_FILE_TRUNCATED`` so callers can detect a silent
        short read rather than consuming a partial file. (Using the pre-read
        size means a file appended to during the read is never mistaken for a
        truncation.) The check is skipped for small files, where silent
        truncation does not occur, and is best-effort when the size cannot be
        determined.

        If your loop reads chunks slower than the file streams (e.g. you do
        slow work between iterations), the read may be ended early with
        ``SandboxStreamBackpressureError``. Iterate promptly and move slow work
        off the read loop; see that exception's docstring for guidance.

        Args:
            filepath: Absolute path inside the sandbox.
            timeout_seconds: Wall-clock timeout for the streaming read.

        Returns:
            ``StreamReader[bytes]`` yielding chunks in order. End-of-file is
            signaled by normal iterator exhaustion. Errors (missing file,
            permission denied, truncation, a too-slow reader) are re-raised
            when the consumer iterates past them.

        Example:
            ```python
            with contextlib.closing(sb.read_file_streaming("/data/big.bin")) as reader:
                with open("local.bin", "wb") as f:
                    for chunk in reader:
                        f.write(chunk)
            ```

        Caveats:
            The streaming transfer does not survive a sandbox restart; a
            long transfer that coincides with a restart will fail mid-stream.

            Callers should iterate the reader to completion or call
            ``close()`` on it. The SDK installs a finalizer to cancel the
            background task on garbage collection, but explicit close
            releases resources sooner.

            A bounded amount of output is buffered ahead of your loop to smooth
            out bursts and apply backpressure, but it is not a hard memory
            ceiling: resident memory still grows with how far behind your loop
            falls. Keep the read loop tight and move slow per-chunk work off it
            (see ``examples/large_file_streaming.py``).
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds
        output_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(
            maxsize=STREAMING_OUTPUT_QUEUE_SIZE
        )
        future = self._loop_manager.run_async(
            self._read_file_streaming_async(filepath, output_queue, timeout, container=container)
        )
        reader = StreamReader(
            output_queue,
            self._loop_manager,
            cancel=future.cancel,
        )
        # Cancel the producer if the consumer abandons the reader without
        # iterating to completion or calling close(). Otherwise the producer
        # parks on a full queue and holds the gRPC call open.
        weakref.finalize(reader, future.cancel)
        return reader

    def stream_logs(
        self,
        *,
        follow: bool = False,
        tail_lines: int | None = None,
        since_time: datetime | None = None,
        timestamps: bool = False,
        timeout_seconds: float | None = None,
        container: str | None = None,
    ) -> StreamReader[str]:
        """Stream logs from the sandbox's main process.

        Streams stdout/stderr from the sandbox's **main command**: the
        entrypoint passed to ``Sandbox.run()`` (or the default shell-trapped
        keep-alive). Output from commands started via ``exec()`` is **not**
        included; use ``Process.stdout``/``Process.stderr`` for those.

        .. note::

            Sandboxes created with the default keep-alive command do not
            produce any log output. To see logs here, pass a command that
            writes to stdout/stderr when calling ``Sandbox.run()``.

        Returns a StreamReader that yields log lines as strings. The method
        returns immediately; iteration on the StreamReader blocks until
        data arrives.

        Args:
            follow: If True, continuously stream new logs (like ``tail -f``).
                If False, stream existing logs from the running sandbox and
                stop. Stopped sandboxes reject ``StreamLogs``.
            tail_lines: Number of most recent lines to retrieve. If None,
                returns all available lines.
            since_time: Only return logs after this timestamp. Must be
                timezone-aware; naive datetimes raise ``ValueError``.
            timestamps: If True, prefix each line with an ISO 8601 timestamp
                from the server.
            timeout_seconds: Client-side deadline for the gRPC call. Defaults
                to ``request_timeout_seconds`` when ``follow=False``, and
                ``None`` (no timeout) when ``follow=True``.
            container: Container whose logs to stream. Empty/None targets the
                primary.

        Returns:
            StreamReader yielding log lines as strings. Iterate synchronously
            with ``for line in reader`` or asynchronously with
            ``async for line in reader``.

        Raises:
            SandboxNotRunningError: If ``follow=True`` and the sandbox has
                been stopped.
            SandboxError: If the log stream encounters an error.

        Example:
            ```python
            # One-shot: get recent logs
            for line in sandbox.stream_logs(tail_lines=100):
                print(line, end="")

            # Follow mode: stream continuously
            for line in sandbox.stream_logs(follow=True):
                print(line, end="")

            # Async usage
            async for line in sandbox.stream_logs(follow=True):
                print(line, end="")
            ```
        """
        # Default timeout: request_timeout for finite streams, None for follow
        if timeout_seconds is None and not follow:
            timeout_seconds = self._request_timeout_seconds

        # Bounded queue provides backpressure for potentially unbounded log output
        # (follow=True streams indefinitely). Contrast with exec stdout/stderr
        # queues which are unbounded because exec output is finite.
        output_queue: asyncio.Queue[str | Exception | None] = asyncio.Queue(
            maxsize=STREAMING_OUTPUT_QUEUE_SIZE
        )

        future = self._loop_manager.run_async(
            self._stream_logs_async(
                output_queue,
                follow=follow,
                tail_lines=tail_lines,
                since_time=since_time,
                timestamps=timestamps,
                timeout_seconds=timeout_seconds,
                container=container,
            )
        )

        return StreamReader(output_queue, self._loop_manager, cancel=future.cancel)
