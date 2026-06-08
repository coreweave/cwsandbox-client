# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

from __future__ import annotations

import asyncio
import builtins
import contextlib
import logging
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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import grpc
import grpc.aio
from google.protobuf import timestamp_pb2

from cwsandbox._auth import resolve_auth_metadata
from cwsandbox._defaults import (
    DEFAULT_BASE_URL,
    DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS,
    DEFAULT_FILE_OPERATION_CAP_BYTES,
    DEFAULT_FSS_RETRY_BUDGET_SECONDS,
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
from cwsandbox._error_info import (
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
    CWSANDBOX_SANDBOX_NOT_FOUND,
    FILE_ERROR_REASONS,
    SNAPSHOT_INTERNAL_REASONS,
    SNAPSHOT_TRANSIENT_REASONS,
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
    gateway_pb2,
    gateway_pb2_grpc,
    streaming_pb2,
    streaming_pb2_grpc,
)
from cwsandbox._resources import normalize_resources
from cwsandbox._types import (
    ExecOutcome,
    FileSystemSnapshot,
    FileSystemSnapshotBucketConfig,
    FileSystemSnapshotBucketMode,
    FileSystemSnapshotOptions,
    FileSystemSnapshotStatus,
    FileSystemSnapshotTrigger,
    NetworkOptions,
    OperationRef,
    Process,
    ProcessResult,
    ResourceOptions,
    Secret,
    StreamReader,
    StreamWriter,
    TerminalResult,
    TerminalSession,
)
from cwsandbox.exceptions import (
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
    SnapshotQuotaExceededError,
    SnapshotSizeExceededError,
    SnapshotWaitTimeoutError,
)

if TYPE_CHECKING:
    import concurrent.futures

    from cwsandbox._session import Session

logger = logging.getLogger(__name__)


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
        """Convert protobuf status enum to SandboxStatus."""
        try:
            proto_name = gateway_pb2.SandboxStatus.Name(proto_status)
            enum_name = proto_name.replace("SANDBOX_STATUS_", "")
            return cls[enum_name]
        except (ValueError, KeyError):
            logger.warning("Unknown sandbox status %s, treating as UNSPECIFIED", proto_status)
            return cls.UNSPECIFIED

    def to_proto(self) -> int:
        """Convert SandboxStatus to protobuf enum"""
        proto_name = f"SANDBOX_STATUS_{self.name}"
        return gateway_pb2.SandboxStatus.Value(proto_name)


def _fss_status_from_proto(value: int) -> FileSystemSnapshotStatus:
    """Convert a proto FileSystemSnapshotStatus enum to the SDK enum."""
    try:
        name = gateway_pb2.FileSystemSnapshotStatus.Name(value).replace(
            "FILE_SYSTEM_SNAPSHOT_STATUS_", ""
        )
        return FileSystemSnapshotStatus[name]
    except (ValueError, KeyError):
        logger.warning("Unknown snapshot status %s, treating as UNSPECIFIED", value)
        return FileSystemSnapshotStatus.UNSPECIFIED


def _fss_trigger_from_proto(value: int) -> FileSystemSnapshotTrigger:
    """Convert a proto FileSystemSnapshotTrigger enum to the SDK enum."""
    try:
        name = gateway_pb2.FileSystemSnapshotTrigger.Name(value).replace(
            "FILE_SYSTEM_SNAPSHOT_TRIGGER_", ""
        )
        return FileSystemSnapshotTrigger[name]
    except (ValueError, KeyError):
        logger.warning("Unknown snapshot trigger %s, treating as UNSPECIFIED", value)
        return FileSystemSnapshotTrigger.UNSPECIFIED


def _fss_bucket_mode_from_proto(value: int) -> FileSystemSnapshotBucketMode:
    """Convert a proto FileSystemSnapshotBucketMode enum to the SDK enum."""
    try:
        name = gateway_pb2.FileSystemSnapshotBucketMode.Name(value).replace(
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


def _snapshot_from_proto(proto: gateway_pb2.FileSystemSnapshot) -> FileSystemSnapshot:
    """Convert a proto FileSystemSnapshot to the SDK dataclass."""
    return FileSystemSnapshot(
        file_system_snapshot_id=proto.file_system_snapshot_id,
        status=_fss_status_from_proto(proto.status),
        status_reason=proto.status_reason,
        size_bytes=proto.size_bytes,
        source_sandbox_id=proto.source_sandbox_id,
        trigger=_fss_trigger_from_proto(proto.trigger),
        idempotency_key=proto.idempotency_key,
        object_bucket=proto.object_bucket,
        created_at=_proto_timestamp_to_datetime(proto, "created_at"),
        updated_at=_proto_timestamp_to_datetime(proto, "updated_at"),
        completed_at=_proto_timestamp_to_datetime(proto, "completed_at"),
    )


def _bucket_config_from_proto(
    proto: gateway_pb2.FileSystemSnapshotBucketConfig,
) -> FileSystemSnapshotBucketConfig:
    """Convert a proto FileSystemSnapshotBucketConfig to the SDK dataclass."""
    return FileSystemSnapshotBucketConfig(
        mode=_fss_bucket_mode_from_proto(proto.mode),
        bucket_name=proto.bucket_name,
        region=proto.region,
        effective_bucket_name=proto.effective_bucket_name,
    )


def _coerce_file_system_snapshot(
    value: FileSystemSnapshotOptions | Mapping[str, Any] | None,
) -> FileSystemSnapshotOptions | None:
    """Coerce the ``file_system_snapshot`` argument to FileSystemSnapshotOptions.

    Accepts a FileSystemSnapshotOptions, a plain mapping with matching keys, or
    None. Raises TypeError for anything else.
    """
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


def _file_system_mount_kwargs(opts: FileSystemSnapshotOptions) -> dict[str, Any]:
    """Build StartSandboxRequest.file_system (SandboxFileSystemMount) kwargs."""
    mount: dict[str, Any] = {"mount_path": opts.mount_path}
    if opts.size is not None:
        mount["size"] = opts.size
    if opts.file_system_snapshot_id is not None:
        mount["file_system_snapshot"] = {"file_system_snapshot_id": opts.file_system_snapshot_id}
    return mount


async def _create_snapshot_via_stub(
    stub: gateway_pb2_grpc.GatewayServiceStub,
    sandbox_id: str,
    *,
    idempotency_key: str | None,
    wait_for_ready: bool,
    auth_metadata: tuple[tuple[str, str], ...],
    timeout: float,
    max_timeout_seconds: int | None = None,
) -> str:
    """Call CreateFileSystemSnapshot on ``stub``; return the new snapshot ID."""
    request = gateway_pb2.CreateFileSystemSnapshotRequest(
        sandbox_id=sandbox_id,
        wait_for_ready=wait_for_ready,
    )
    if idempotency_key:
        request.idempotency_key = idempotency_key
    if max_timeout_seconds is not None:
        request.max_timeout_seconds = max_timeout_seconds
    try:
        response = await stub.CreateFileSystemSnapshot(
            request, timeout=timeout, metadata=auth_metadata
        )
    except grpc.RpcError as e:
        raise _translate_rpc_error(
            e, sandbox_id=sandbox_id, operation="Create file-system snapshot"
        ) from e
    if not response.success:
        raise SandboxSnapshotError(
            f"Failed to create file-system snapshot: {response.error_message or 'unknown error'}"
        )
    return str(response.file_system_snapshot_id)


async def _get_snapshot_via_stub(
    stub: gateway_pb2_grpc.GatewayServiceStub,
    file_system_snapshot_id: str,
    *,
    auth_metadata: tuple[tuple[str, str], ...],
    timeout: float,
) -> FileSystemSnapshot:
    """Call GetFileSystemSnapshot on ``stub``; return the snapshot record."""
    request = gateway_pb2.GetFileSystemSnapshotRequest(
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
    background loop would stall every other operation — heartbeats, other
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


def _translate_rpc_error(
    e: grpc.RpcError,
    *,
    sandbox_id: str | None = None,
    operation: str = "operation",
    filepath: str | None = None,
    file_system_snapshot_id: str | None = None,
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


_RETRYABLE_POLL_EXCEPTIONS: tuple[type[CWSandboxError], ...] = (
    SandboxUnavailableError,
    SandboxRequestTimeoutError,
    SandboxResourceExhaustedError,
)


# In-band error codes the server may emit on a streaming response.  These
# are application-level (the gRPC call itself succeeds); the codes describe
# server-side outcomes for the streaming session.  Mirrors the documented
# error contract in streaming_pb2.pyi (LogStreamError.code).
#
# Per the wire contract every LogStreamError is terminal — the server will
# not send further frames on the same call.  The client's recovery action
# is dictated by the code:
#
#   SESSION_NOT_FOUND / REPLAY_GAP / RUNNER_UNAVAILABLE / RUNNER_DRAINING
#       reconnect with a FRESH init (no resume_session_id / resume_offset)
#       to pick up the live tail from the current head.
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
    restart or a transient network blip qualify.  Anything else — including
    DEADLINE_EXCEEDED (caller-requested timeout), NOT_FOUND, PERMISSION_DENIED,
    INVALID_ARGUMENT — propagates as-is.

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
    exception — matching the pattern in ``_classify_poll_error``.
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
) -> _T:
    """Run ``attempt`` with bounded retry on transient CWSandbox errors.

    ``attempt`` performs exactly one RPC try and must raise a *translated*
    ``CWSandboxError`` on failure (i.e. wrap the stub call and
    ``_translate_rpc_error``). Only classes in ``_RETRYABLE_POLL_EXCEPTIONS``
    are retried - transient unavailability, request-deadline, resource
    exhaustion, and FSS backend-throttling (which subclasses
    ``SandboxUnavailableError``). Every other error, including ``NOT_FOUND``
    and ``FAILED_PRECONDITION``, is fatal and re-raised on the first attempt.

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
            if _classify_poll_error(exc) != "retryable" or budget_seconds <= 0:
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
    profile_id: str | None = None
    runner_group_id: str | None = None
    started_at: datetime | None = None


@dataclass(frozen=True)
class _Stopping:
    sandbox_id: str
    status: SandboxStatus = SandboxStatus.TERMINATING
    runner_id: str | None = None
    profile_id: str | None = None
    runner_group_id: str | None = None
    started_at: datetime | None = None


@dataclass(frozen=True)
class _Terminal:
    sandbox_id: str
    status: SandboxStatus
    returncode: int | None = None
    runner_id: str | None = None
    profile_id: str | None = None
    runner_group_id: str | None = None
    started_at: datetime | None = None


_LifecycleState = _NotStarted | _Starting | _Running | _Stopping | _Terminal


class _SandboxInfoLike(Protocol):
    """Structural type for protobuf sandbox info responses.

    Required fields are always present. Optional fields are guarded by
    hasattr/getattr in _apply_sandbox_info.
    """

    @property
    def sandbox_id(self) -> Any: ...
    @property
    def sandbox_status(self) -> Any: ...


_RUNNING_STATUSES = frozenset({SandboxStatus.RUNNING, SandboxStatus.PAUSED})
_TERMINAL_STATUSES = frozenset(
    {SandboxStatus.COMPLETED, SandboxStatus.FAILED, SandboxStatus.TERMINATED}
)


def _lifecycle_state_from_info(
    *,
    sandbox_id: str,
    status: SandboxStatus,
    runner_id: str | None = None,
    profile_id: str | None = None,
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
            profile_id=profile_id,
            runner_group_id=runner_group_id,
            started_at=started_at,
        )
    if status == SandboxStatus.TERMINATING:
        return _Stopping(
            sandbox_id=sandbox_id,
            status=status,
            runner_id=runner_id,
            profile_id=profile_id,
            runner_group_id=runner_group_id,
            started_at=started_at,
        )
    if status in _TERMINAL_STATUSES:
        return _Terminal(
            sandbox_id=sandbox_id,
            status=status,
            returncode=returncode,
            runner_id=runner_id,
            profile_id=profile_id,
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
        profile_id: Profile ID for this sandbox.
        returncode: Exit code if sandbox completed.
        started_at: When sandbox started running.
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        defaults: SandboxDefaults | None = None,
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
        environment_variables: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        secrets: Sequence[Secret | dict[str, Any]] | None = None,
        _session: Session | None = None,
    ) -> None:
        """Initialize a sandbox (does not start it).

        Args:
            command: Optional command to run in the sandbox
            args: Optional arguments for the command
            defaults: Optional SandboxDefaults to apply
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
            profile_ids: Optional list of profile IDs for infrastructure selection.
                See SandboxDefaults.profile_ids for semantics. Prefer
                ``profile_names`` when selecting by name.
            profile_names: Optional list of profile names for infrastructure
                selection (preferred over profile_ids). See
                SandboxDefaults.profile_names for semantics.
            runner_ids: Optional list of runner IDs
            resources: Resource configuration. Accepts ResourceOptions for separate
                requests/limits, or a flat dict for backward-compatible Guaranteed QoS.
            mounted_files: Files to mount into the sandbox at startup. Each dict
                should have ``mount_path`` (str) and ``file_content`` (bytes).
                Note: Mounted files are read-only at runtime. To modify a file,
                use ``sandbox.write_file()`` after the sandbox is running.
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            network: Network configuration (NetworkOptions dataclass)
            file_system_snapshot: File-system snapshot (FSS) mount configuration.
                Accepts a FileSystemSnapshotOptions or a dict with ``mount_path``,
                optional ``size``, and optional ``file_system_snapshot_id``. When
                ``file_system_snapshot_id`` is set, the snapshot is restored into
                ``mount_path`` at start (fork); otherwise the mount starts empty.
                Requires the organization to be enabled for FSS.
            max_timeout_seconds: Maximum timeout for sandbox operations
            environment_variables: Environment variables to inject into the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive config only.
            annotations: Kubernetes pod annotations for the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive metadata only.
            secrets: Secrets to inject as environment variables.
                Merged with defaults (defaults first, then this list).
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

        # Note: These can be None for sandboxes discovered via list()/from_id()
        self._command: str | None = command or self._defaults.command
        self._args: list[str] | None = args if args is not None else list(self._defaults.args)

        # Apply defaults with explicit values taking precedence
        self._container_image: str | None = container_image or self._defaults.container_image
        self._base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or self._defaults.base_url
        ).rstrip("/")
        self._request_timeout_seconds = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else self._defaults.request_timeout_seconds
        )
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
        self._max_lifetime_seconds = (
            max_lifetime_seconds
            if max_lifetime_seconds is not None
            else self._defaults.max_lifetime_seconds
        )

        self._tags: list[str] | None = self._defaults.merge_tags(tags)
        self._environment_variables = self._defaults.merge_environment_variables(
            environment_variables
        )
        self._annotations = self._defaults.merge_annotations(annotations)

        self._profile_ids = _resolve_selector(profile_ids, self._defaults.profile_ids)
        self._profile_names = _resolve_selector(profile_names, self._defaults.profile_names)
        self._runner_ids = _resolve_selector(runner_ids, self._defaults.runner_ids)

        self._start_kwargs: dict[str, Any] = {}
        # Use explicit resources or fall back to defaults, then normalize
        effective_resources = resources if resources is not None else self._defaults.resources
        normalized = normalize_resources(effective_resources)
        if normalized is not None:
            self._start_kwargs["resources"] = normalized
        if mounted_files is not None:
            self._start_kwargs["mounted_files"] = mounted_files
        if s3_mount is not None:
            self._start_kwargs["s3_mount"] = s3_mount
        if ports is not None:
            self._start_kwargs["ports"] = ports
        # Use explicit network or fall back to defaults
        effective_network = network if network is not None else self._defaults.network
        if effective_network is not None:
            self._start_kwargs["network"] = effective_network
        # Use explicit file-system snapshot mount or fall back to defaults.
        effective_fss = (
            file_system_snapshot
            if file_system_snapshot is not None
            else self._defaults.file_system_snapshot
        )
        effective_fss = _coerce_file_system_snapshot(effective_fss)
        if effective_fss is not None:
            self._start_kwargs["file_system_snapshot"] = effective_fss
        if max_timeout_seconds is not None:
            self._start_kwargs["max_timeout_seconds"] = max_timeout_seconds
        merged_secrets = list(self._defaults.secrets or ()) + [
            Secret(**s) if isinstance(s, dict) else s for s in (secrets or ())
        ]
        if merged_secrets:
            seen: dict[str, Secret] = {}
            for secret in merged_secrets:
                env_var = secret.env_var
                assert env_var is not None  # guaranteed by Secret.__post_init__
                if env_var in seen and secret != seen[env_var]:
                    raise ValueError(
                        f"Conflicting secrets for env_var {env_var!r}: "
                        f"Secret(store={seen[env_var].store!r}, name={seen[env_var].name!r}, "
                        f"field={seen[env_var].field!r}) vs "
                        f"Secret(store={secret.store!r}, name={secret.name!r}, "
                        f"field={secret.field!r})"
                    )
                seen[env_var] = secret
            self._start_kwargs["secrets"] = list(seen.values())

        self._channel: grpc.aio.Channel | None = None
        self._stub: gateway_pb2_grpc.GatewayServiceStub | None = None
        self._auth_metadata: tuple[tuple[str, str], ...] = ()
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

        # Shared stop task so repeated stop() calls join the same operation
        self._stop_task: asyncio.Task[None] | None = None
        self._stop_lock = asyncio.Lock()
        self._stop_owned: bool = False
        # Set when a caller invokes stop(missing_ok=True) on a sandbox that
        # is already draining (observe-only path). Widens the NOT_FOUND
        # retry gate in _do_poll_complete so the observe-only waiter treats
        # NOT_FOUND as a backend race (retry briefly for authoritative
        # terminal state) rather than propagating SandboxNotFoundError.
        self._missing_ok_observe: bool = False

        self._status_updated_at: datetime | None = None
        self._service_address: str | None = None
        self._exposed_ports: tuple[tuple[int, str], ...] | None = None
        self._applied_ingress_mode: str | None = None
        self._applied_egress_mode: str | None = None
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
        environment_variables: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        secrets: Sequence[Secret | dict[str, Any]] | None = None,
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
            request_timeout_seconds: Timeout for API requests (client-side)
            poll_retry_budget_seconds: Wall-clock budget for retrying transient
                errors on the sandbox-status poll loop (default: 30s). Set to
                0 to disable retry.
            poll_rpc_timeout_seconds: Per-call timeout for poll Get RPCs
                (default: 15s). Separate from request_timeout_seconds.
            max_lifetime_seconds: Max sandbox lifetime (server-side)
            tags: Optional tags for the sandbox
            profile_ids: Optional list of profile IDs for infrastructure selection.
                See SandboxDefaults.profile_ids for semantics. Prefer
                ``profile_names`` when selecting by name.
            profile_names: Optional list of profile names for infrastructure
                selection (preferred over profile_ids). See
                SandboxDefaults.profile_names for semantics.
            runner_ids: Optional list of runner IDs
            resources: Resource configuration. Accepts ResourceOptions for separate
                requests/limits, or a flat dict for backward-compatible Guaranteed QoS.
            mounted_files: Files to mount into the sandbox at startup. Each dict
                should have ``mount_path`` (str) and ``file_content`` (bytes).
                Note: Mounted files are read-only at runtime. To modify a file,
                use ``sandbox.write_file()`` after the sandbox is running.
            s3_mount: S3 bucket mount configuration
            ports: Port mappings for the sandbox
            network: Network configuration (NetworkOptions dataclass)
            file_system_snapshot: File-system snapshot (FSS) mount configuration.
                Accepts a FileSystemSnapshotOptions or a dict with ``mount_path``,
                optional ``size``, and optional ``file_system_snapshot_id``. When
                ``file_system_snapshot_id`` is set, the snapshot is restored into
                ``mount_path`` at start (fork); otherwise the mount starts empty.
            max_timeout_seconds: Maximum timeout for sandbox operations
            environment_variables: Environment variables to inject into the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive config only.
            annotations: Kubernetes pod annotations for the sandbox.
                Merges with and overrides matching keys from the session defaults.
                Use for non-sensitive metadata only.
            secrets: Secrets to inject as environment variables.
                Merged with defaults (defaults first, then this list).
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
            environment_variables=environment_variables,
            annotations=annotations,
            secrets=secrets,
        )
        logger.debug("Creating sandbox with command: %s", command)
        sandbox.start().result()
        return sandbox

    @classmethod
    def session(
        cls,
        defaults: SandboxDefaults | Mapping[str, Any] | None = None,
    ) -> Session:
        """Create a session for managing multiple sandboxes.

        Sessions provide:
        - Shared configuration via defaults
        - Automatic cleanup of orphaned sandboxes
        - Function execution via @session.function() decorator

        Args:
            defaults: Optional defaults to apply to sandboxes created via session

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

        return Session(defaults)

    @classmethod
    def _from_sandbox_info(
        cls,
        info: _SandboxInfoLike,
        *,
        base_url: str,
        timeout_seconds: float,
        poll_retry_budget_seconds: float = DEFAULT_POLL_RETRY_BUDGET_SECONDS,
        poll_rpc_timeout_seconds: float = DEFAULT_POLL_RPC_TIMEOUT_SECONDS,
    ) -> Sandbox:
        """Create a Sandbox instance from a protobuf sandbox info response."""
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
        sandbox._profile_ids = None
        sandbox._profile_names = None
        sandbox._runner_ids = None
        sandbox._environment_variables = {}
        sandbox._annotations = {}
        sandbox._channel = None
        sandbox._stub = None
        sandbox._auth_metadata = ()
        sandbox._streaming_channel = None
        sandbox._streaming_channel_lock = asyncio.Lock()
        sandbox._session = None
        sandbox._defaults = SandboxDefaults()
        sandbox._start_kwargs = {}
        sandbox._start_lock = asyncio.Lock()
        sandbox._running_task = None
        sandbox._running_lock = asyncio.Lock()
        sandbox._complete_task = None
        sandbox._complete_lock = asyncio.Lock()
        sandbox._stop_task = None
        sandbox._stop_lock = asyncio.Lock()
        sandbox._stop_owned = False
        sandbox._missing_ok_observe = False
        sandbox._loop_manager = _LoopManager.get()
        sandbox._service_address = None
        sandbox._exposed_ports = None
        sandbox._applied_ingress_mode = None
        sandbox._applied_egress_mode = None
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
            profile_id=getattr(info, "profile_id", None) or None,
            runner_group_id=getattr(info, "runner_group_id", None) or None,
            started_at=started_at,
        )
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
        include_stopped: bool = False,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
    ) -> OperationRef[builtins.list[Sandbox]]:
        """List existing sandboxes with optional filters.

        Returns OperationRef that resolves to Sandbox instances usable for
        operations like exec(), stop(), get_status(), read_file(), write_file().

        By default, only active (non-terminal) sandboxes are returned.
        Set ``include_stopped=True`` to widen the search to include terminal
        sandboxes (completed, failed, terminated).
        A terminal status filter (e.g. ``status="completed"``) also widens
        the search automatically.

        Args:
            tags: Filter by tags (sandboxes must have ALL specified tags)
            status: Filter by status ("running", "completed", "failed", etc.)
            profile_ids: Optional list of profile IDs for infrastructure selection.
                See SandboxDefaults.profile_ids for semantics. Prefer
                ``profile_names`` when selecting by name.
            profile_names: Optional list of profile names for infrastructure
                selection (preferred over profile_ids). See
                SandboxDefaults.profile_names for semantics.
            runner_ids: Filter by runner IDs
            include_stopped: If True, include terminal sandboxes (completed,
                failed, terminated). Defaults to False.
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)
            poll_retry_budget_seconds: Wall-clock budget for retrying transient
                errors on the sandbox-status poll loop (default: 30s). Set to 0
                to disable retry. Applied to returned Sandbox instances.
            poll_rpc_timeout_seconds: Per-call timeout for poll Get RPCs
                (default: 15s). Separate from ``timeout_seconds``. Applied to
                returned Sandbox instances.

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
                tags=["my-batch-job"], include_stopped=True
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
                include_stopped=include_stopped,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                poll_retry_budget_seconds=poll_retry_budget_seconds,
                poll_rpc_timeout_seconds=poll_rpc_timeout_seconds,
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
        include_stopped: bool = False,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
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

        auth_metadata = resolve_auth_metadata()

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request_kwargs: dict[str, Any] = {}
            if normalized_tags:
                request_kwargs["tags"] = list(normalized_tags)
            if status_enum:
                request_kwargs["status"] = status_enum.to_proto()
            if profile_ids is not None:
                request_kwargs["profile_ids"] = profile_ids
            if profile_names is not None:
                request_kwargs["profile_names"] = profile_names
            if runner_ids is not None:
                request_kwargs["runner_ids"] = runner_ids

            if include_stopped:
                request_kwargs["include_stopped"] = True
            request = gateway_pb2.ListSandboxesRequest(**request_kwargs)
            try:
                sandbox_infos = await paginate_async(
                    stub.List,
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
                    sb,
                    base_url=effective_base_url,
                    timeout_seconds=timeout,
                    poll_retry_budget_seconds=effective_poll_retry_budget,
                    poll_rpc_timeout_seconds=effective_poll_rpc_timeout,
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
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
    ) -> OperationRef[Sandbox]:
        """Attach to an existing sandbox by ID.

        Creates a Sandbox instance connected to an existing sandbox,
        allowing operations like exec(), stop(), get_status(), etc.

        Args:
            sandbox_id: The ID of the existing sandbox
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default)
            timeout_seconds: Request timeout (default: 300s)
            poll_retry_budget_seconds: Wall-clock budget for retrying transient
                errors on the sandbox-status poll loop (default: 30s). Set to 0
                to disable retry. Applied to the returned Sandbox instance.
            poll_rpc_timeout_seconds: Per-call timeout for poll Get RPCs
                (default: 15s). Separate from ``timeout_seconds``. Applied to
                the returned Sandbox instance.

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
                timeout_seconds=timeout_seconds,
                poll_retry_budget_seconds=poll_retry_budget_seconds,
                poll_rpc_timeout_seconds=poll_rpc_timeout_seconds,
            )
        )
        return OperationRef(future)

    @classmethod
    async def _from_id_async(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        poll_retry_budget_seconds: float | None = None,
        poll_rpc_timeout_seconds: float | None = None,
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

        auth_metadata = resolve_auth_metadata()

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request = gateway_pb2.GetSandboxRequest(sandbox_id=sandbox_id)
            try:
                response = await stub.Get(request, timeout=timeout, metadata=auth_metadata)
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, sandbox_id=sandbox_id, operation="Get sandbox") from e

            return cls._from_sandbox_info(
                response,
                base_url=effective_base_url,
                timeout_seconds=timeout,
                poll_retry_budget_seconds=effective_poll_retry_budget,
                poll_rpc_timeout_seconds=effective_poll_rpc_timeout,
            )
        finally:
            await channel.close(grace=None)

    @classmethod
    def delete(
        cls,
        sandbox_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> OperationRef[None]:
        """Delete a sandbox by ID without creating a Sandbox instance.

        This is a convenience method for cleanup scenarios where you
        don't need to perform other operations on the sandbox.

        Args:
            sandbox_id: The sandbox ID to delete
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default)
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

        auth_metadata = resolve_auth_metadata()

        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]

        try:
            request = gateway_pb2.DeleteSandboxRequest(sandbox_id=sandbox_id)
            try:
                response = await stub.Delete(request, timeout=timeout, metadata=auth_metadata)
            except grpc.RpcError as e:
                parsed = parse_error_info(e)
                if missing_ok and is_not_found(e, parsed, CWSANDBOX_SANDBOX_NOT_FOUND):
                    return
                raise _translate_rpc_error(
                    e, sandbox_id=sandbox_id, operation="Delete sandbox"
                ) from e

            if not response.success:
                raise SandboxError(f"Failed to delete sandbox: {response.error_message}")
        finally:
            await channel.close(grace=None)

    @classmethod
    def get_snapshot(
        cls,
        file_system_snapshot_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> OperationRef[FileSystemSnapshot]:
        """Fetch a file-system snapshot (FSS) record by ID.

        Snapshots are org-scoped: any snapshot owned by your organization is
        visible, regardless of which sandbox created it.

        Args:
            file_system_snapshot_id: The snapshot ID to fetch.
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
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
                file_system_snapshot_id, base_url=base_url, timeout_seconds=timeout_seconds
            )
        )
        return OperationRef(future)

    @classmethod
    async def _get_snapshot_async(
        cls,
        file_system_snapshot_id: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> FileSystemSnapshot:
        """Internal async: fetch a snapshot record by ID."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        auth_metadata = resolve_auth_metadata()
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]
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

        auth_metadata = resolve_auth_metadata()
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]
        try:

            async def _attempt() -> builtins.list[Any]:
                # Build the request inside the attempt: paginate_async mutates
                # page_token in place, so a retry must start from a fresh
                # request (page 1) rather than resuming from the last token.
                request = gateway_pb2.ListFileSystemSnapshotsRequest()
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
        timeout_seconds: float | None = None,
        missing_ok: bool = False,
    ) -> OperationRef[None]:
        """Delete a file-system snapshot (FSS) by ID.

        Deleting a snapshot does not affect sandboxes already restored from it.

        Args:
            file_system_snapshot_id: The snapshot ID to delete.
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
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
        auth_metadata = resolve_auth_metadata()
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]
        try:
            request = gateway_pb2.DeleteFileSystemSnapshotRequest(
                file_system_snapshot_id=file_system_snapshot_id
            )
            attempts = {"n": 0}

            async def _attempt() -> None:
                attempts["n"] += 1
                try:
                    response = await stub.DeleteFileSystemSnapshot(
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
                if not response.success:
                    raise SandboxSnapshotError(
                        f"Failed to delete file-system snapshot: "
                        f"{response.error_message or 'unknown error'}",
                        file_system_snapshot_id=file_system_snapshot_id,
                    )

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
        timeout_seconds: float | None = None,
    ) -> OperationRef[FileSystemSnapshotBucketConfig]:
        """Fetch the organization's FSS object-storage bucket configuration.

        Args:
            base_url: Override API URL (default: CWSANDBOX_BASE_URL env or default).
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
                base_url=base_url, timeout_seconds=timeout_seconds
            )
        )
        return OperationRef(future)

    @classmethod
    async def _get_snapshot_bucket_config_async(
        cls,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> FileSystemSnapshotBucketConfig:
        """Internal async: fetch the org's FSS bucket configuration."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        auth_metadata = resolve_auth_metadata()
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]
        try:
            request = gateway_pb2.GetFileSystemSnapshotBucketConfigRequest()

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
        timeout_seconds: float | None = None,
    ) -> FileSystemSnapshotBucketConfig:
        """Internal async: set the org's FSS bucket configuration."""
        effective_base_url = (
            base_url or os.environ.get("CWSANDBOX_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        auth_metadata = resolve_auth_metadata()
        target, is_secure = parse_grpc_target(effective_base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]
        try:
            request = gateway_pb2.SetFileSystemSnapshotBucketConfigRequest(
                bucket_name=bucket_name,
                region=region,
            )

            async def _attempt() -> FileSystemSnapshotBucketConfig:
                try:
                    proto = await stub.SetFileSystemSnapshotBucketConfig(
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
    def profile_id(self) -> str | None:
        """Profile where sandbox is running, or None if not started."""
        if isinstance(self._state, (_Running, _Stopping, _Terminal)):
            return self._state.profile_id
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
    def service_address(self) -> str | None:
        """External address for accessing sandbox services.

        Returns an address like '166.19.9.70:8080' for network-accessible sandboxes
        (SSH, web services). Availability depends on runner configuration.

        Returns None if:
        - Sandbox hasn't been started yet
        - Sandbox was obtained via from_id() or list()
        - Runner uses ClusterIP instead of LoadBalancer
        """
        return self._service_address

    @property
    def exposed_ports(self) -> tuple[tuple[int, str], ...] | None:
        """Exposed ports for the sandbox.

        Returns a tuple of (container_port, name) tuples for ports exposed by
        the sandbox. Useful for network-accessible sandboxes.

        Returns None if:
        - Sandbox hasn't been started yet
        - Sandbox was obtained via from_id() or list()
        - No ports were exposed
        """
        return self._exposed_ports

    @property
    def applied_ingress_mode(self) -> str | None:
        """The ingress mode applied by the backend (set after start)."""
        return self._applied_ingress_mode

    @property
    def applied_egress_mode(self) -> str | None:
        """The egress mode applied by the backend (set after start)."""
        return self._applied_egress_mode

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
            info: Protobuf response with sandbox_status, runner_id, profile_id,
                runner_group_id, started_at_time, and optionally returncode fields.
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
        # returncode is only meaningful for completed sandboxes observed via polling
        if source == "poll" and status == SandboxStatus.COMPLETED and hasattr(info, "returncode"):
            returncode = info.returncode
        else:
            returncode = None

        new_state = _lifecycle_state_from_info(
            sandbox_id=sandbox_id,
            status=status,
            runner_id=getattr(info, "runner_id", None) or None,
            profile_id=getattr(info, "profile_id", None) or None,
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

        request = gateway_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
        try:
            response = await self._stub.Get(
                request,
                timeout=self._poll_rpc_timeout_seconds,
                metadata=self._auth_metadata,
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

        auth_metadata = resolve_auth_metadata()
        target, is_secure = parse_grpc_target(self._base_url)
        channel = create_channel(target, is_secure)
        stub = gateway_pb2_grpc.GatewayServiceStub(channel)  # type: ignore[no-untyped-call]
        self._channel = channel
        self._stub = stub
        self._auth_metadata = auth_metadata
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
    ) -> gateway_pb2.GetSandboxResponse:
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

            request = gateway_pb2.GetSandboxRequest(sandbox_id=self._sandbox_id)
            try:
                response: gateway_pb2.GetSandboxResponse = await self._stub.Get(
                    request,
                    timeout=effective_rpc_timeout,
                    metadata=self._auth_metadata,
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
                gateway_pb2.SANDBOX_STATUS_RUNNING,
                gateway_pb2.SANDBOX_STATUS_PAUSED,
                gateway_pb2.SANDBOX_STATUS_TERMINATING,
                gateway_pb2.SANDBOX_STATUS_COMPLETED,
                gateway_pb2.SANDBOX_STATUS_FAILED,
                gateway_pb2.SANDBOX_STATUS_TERMINATED,
                gateway_pb2.SANDBOX_STATUS_UNSPECIFIED,
            ):
                return response

            # Transient states - keep polling
            await asyncio.sleep(poll_interval)
            poll_interval = min(
                poll_interval * DEFAULT_POLL_BACKOFF_FACTOR,
                DEFAULT_MAX_POLL_INTERVAL_SECONDS,
            )

    async def _poll_with_retry(self) -> gateway_pb2.GetSandboxResponse:
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
        """Internal async: Send StartSandbox to backend, return sandbox_id.

        Does NOT wait for RUNNING status. Idempotent - safe to call multiple times.
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

            request_kwargs: dict[str, Any] = {
                "command": self._command,
                "args": self._args or [],
                "container_image": self._container_image,
                "tags": self._tags or [],
            }

            if self._max_lifetime_seconds is not None:
                request_kwargs["max_lifetime_seconds"] = int(self._max_lifetime_seconds)
            if self._profile_ids is not None:
                request_kwargs["profile_ids"] = self._profile_ids
            if self._profile_names is not None:
                request_kwargs["profile_names"] = self._profile_names
            if self._runner_ids is not None:
                request_kwargs["runner_ids"] = self._runner_ids
            if self._environment_variables:
                request_kwargs["environment_variables"] = self._environment_variables
            if self._annotations:
                request_kwargs["pod_annotations"] = self._annotations

            request_kwargs.update(self._start_kwargs)

            # Convert ResourceOptions to new proto fields (resource_limits / resource_requests)
            resources_opt = request_kwargs.pop("resources", None)
            if resources_opt is not None:
                if isinstance(resources_opt, ResourceOptions):
                    limits_kwargs: dict[str, Any] = {}
                    if resources_opt.limits:
                        if "cpu" in resources_opt.limits:
                            limits_kwargs["cpu"] = resources_opt.limits["cpu"]
                        if "memory" in resources_opt.limits:
                            limits_kwargs["memory"] = resources_opt.limits["memory"]
                    if resources_opt.gpu:
                        gpu_kwargs: dict[str, Any] = {}
                        if "count" in resources_opt.gpu:
                            gpu_kwargs["gpu_count"] = resources_opt.gpu["count"]
                        if "type" in resources_opt.gpu:
                            gpu_kwargs["gpu_type"] = resources_opt.gpu["type"]
                        if "memory_gb" in resources_opt.gpu:
                            gpu_kwargs["gpu_memory_gb"] = resources_opt.gpu["memory_gb"]
                        if gpu_kwargs:
                            limits_kwargs["gpu"] = gpu_kwargs
                    if limits_kwargs:
                        request_kwargs["resource_limits"] = limits_kwargs

                    requests_kwargs: dict[str, Any] = {}
                    if resources_opt.requests:
                        if "cpu" in resources_opt.requests:
                            requests_kwargs["cpu"] = resources_opt.requests["cpu"]
                        if "memory" in resources_opt.requests:
                            requests_kwargs["memory"] = resources_opt.requests["memory"]
                    if resources_opt.gpu:
                        gpu_ref = limits_kwargs.get("gpu")
                        requests_kwargs["gpu"] = dict(gpu_ref) if gpu_ref else gpu_ref
                    if requests_kwargs:
                        request_kwargs["resource_requests"] = requests_kwargs

            # Convert NetworkOptions to dict for proto
            network = request_kwargs.get("network")
            if network is not None and isinstance(network, NetworkOptions):
                net_opts = network
                network_dict: dict[str, Any] = {}
                if net_opts.ingress_mode is not None:
                    network_dict["ingress_mode"] = net_opts.ingress_mode
                if net_opts.exposed_ports is not None:
                    network_dict["exposed_ports"] = list(net_opts.exposed_ports)
                if net_opts.egress_mode is not None:
                    network_dict["egress_mode"] = net_opts.egress_mode
                request_kwargs["network"] = network_dict

            # Convert flat Secret list to grouped proto SecretStoreReference dicts
            secrets = request_kwargs.pop("secrets", None)
            if secrets is not None and secrets and isinstance(secrets[0], Secret):
                grouped: dict[str, list[dict[str, str]]] = {}
                for s in secrets:
                    grouped.setdefault(s.store, []).append(
                        {"path": s.name, "field": s.field, "env_var": s.env_var}
                    )
                request_kwargs["secret_stores"] = [
                    {"store_name": store, "secrets": mappings}
                    for store, mappings in grouped.items()
                ]

            # Convert FileSystemSnapshotOptions to the proto file_system mount.
            fss_opts = request_kwargs.pop("file_system_snapshot", None)
            if fss_opts is not None:
                request_kwargs["file_system"] = _file_system_mount_kwargs(fss_opts)

            logger.debug("Starting sandbox with image %s", self._container_image)

            request = gateway_pb2.StartSandboxRequest(**request_kwargs)
            self._start_accepted_at = time.monotonic()
            try:
                response = await self._stub.Start(
                    request, timeout=self._request_timeout_seconds, metadata=self._auth_metadata
                )
            except grpc.RpcError as e:
                raise _translate_rpc_error(e, operation="Start sandbox") from e

            sandbox_id = str(response.sandbox_id)
            self._sandbox_id = sandbox_id
            self._status_updated_at = datetime.now(UTC)
            self._state = _Starting(sandbox_id=sandbox_id)
            self._service_address = response.service_address or None
            self._exposed_ports = (
                tuple((p.container_port, p.name) for p in response.exposed_ports)
                if response.exposed_ports
                else None
            )
            self._applied_ingress_mode = response.applied_ingress_mode or None
            self._applied_egress_mode = response.applied_egress_mode or None

            # Extract resource limits/requests echoed back from the start response
            if response.HasField("requested_resource_limits"):
                rl = response.requested_resource_limits
                d = {}
                if rl.cpu:
                    d["cpu"] = rl.cpu
                if rl.memory:
                    d["memory"] = rl.memory
                self._resource_limits = d if d else None
                if rl.HasField("gpu"):
                    gpu = rl.gpu
                    gd: dict[str, Any] = {}
                    if gpu.gpu_count:
                        gd["count"] = gpu.gpu_count
                    if gpu.gpu_type:
                        gd["type"] = gpu.gpu_type
                    if gpu.gpu_memory_gb:
                        gd["memory_gb"] = gpu.gpu_memory_gb
                    self._resource_gpu = gd if gd else None
            if response.HasField("requested_resource_requests"):
                rr = response.requested_resource_requests
                d = {}
                if rr.cpu:
                    d["cpu"] = rr.cpu
                if rr.memory:
                    d["memory"] = rr.memory
                self._resource_requests = d if d else None

            logger.debug("Sandbox %s created (pending)", sandbox_id)
            return sandbox_id

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

    async def _retry_post_stop_not_found(self) -> gateway_pb2.GetSandboxResponse:
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
        After resolving, returncode will be available.

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
        idempotency_key: str | None = None,
    ) -> None:
        """Body of the shared _stop_task: send Stop RPC, poll to terminal, cleanup.

        Only the first caller's parameters (snapshot_on_stop,
        graceful_shutdown_seconds, wait_for_ready, idempotency_key) are used.
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

                # Snapshot-on-stop blocks on the runner archive, so the ceiling
                # must be generous; use the FSS default. A plain stop stays
                # bounded by graceful shutdown.
                if snapshot_on_stop:
                    max_timeout = int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS)
                else:
                    max_timeout = int(graceful_shutdown_seconds) + int(
                        DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS
                    )
                # The renamed proto field is file_system_snapshot_on_stop;
                # wait_for_ready/idempotency_key are only valid alongside it,
                # so only send them when a snapshot is requested.
                request = gateway_pb2.StopSandboxRequest(
                    sandbox_id=sandbox_id,
                    graceful_shutdown_seconds=int(graceful_shutdown_seconds),
                    file_system_snapshot_on_stop=snapshot_on_stop,
                    max_timeout_seconds=max_timeout,
                )
                if snapshot_on_stop:
                    request.wait_for_ready = wait_for_ready
                    if idempotency_key:
                        request.idempotency_key = idempotency_key

                # Send Stop RPC first, then update state on success
                try:
                    response = await self._stub.Stop(
                        request,
                        timeout=max_timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS,
                        metadata=self._auth_metadata,
                    )
                except grpc.RpcError as e:
                    parsed = parse_error_info(e)
                    if missing_ok and is_not_found(e, parsed, CWSANDBOX_SANDBOX_NOT_FOUND):
                        logger.debug(
                            "Sandbox %s not found during stop (missing_ok=True)",
                            sandbox_id,
                        )
                        self._state = _Terminal(
                            sandbox_id=sandbox_id,
                            status=SandboxStatus.TERMINATED,
                            runner_id=(prev.runner_id if isinstance(prev, (_Running,)) else None),
                            profile_id=(prev.profile_id if isinstance(prev, (_Running,)) else None),
                            runner_group_id=(
                                prev.runner_group_id if isinstance(prev, (_Running,)) else None
                            ),
                            started_at=(prev.started_at if isinstance(prev, (_Running,)) else None),
                        )
                        if self._complete_task is not None and not self._complete_task.done():
                            self._complete_task.cancel()
                            self._complete_task = None
                        return
                    raise _translate_rpc_error(
                        e, sandbox_id=sandbox_id, operation="Stop sandbox"
                    ) from e

                if not response.success:
                    error_msg = response.error_message or "unknown error"
                    raise SandboxError(f"Failed to stop sandbox: {error_msg}")

                # Capture the snapshot ID produced by snapshot-on-stop, if any.
                # Only populated when file_system_snapshot_on_stop was accepted.
                if response.file_system_snapshot_id:
                    self._file_system_snapshot_id = response.file_system_snapshot_id

                # RPC succeeded: transition to _Stopping
                self._state = _Stopping(
                    sandbox_id=sandbox_id,
                    runner_id=(prev.runner_id if isinstance(prev, (_Running,)) else None),
                    profile_id=(prev.profile_id if isinstance(prev, (_Running,)) else None),
                    runner_group_id=(
                        prev.runner_group_id if isinstance(prev, (_Running,)) else None
                    ),
                    started_at=(prev.started_at if isinstance(prev, (_Running,)) else None),
                )
                self._stop_owned = True
                sent_rpc = True
                logger.info("Sandbox %s stop accepted, draining", sandbox_id)

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

    async def _stop_async(
        self,
        *,
        snapshot_on_stop: bool = False,
        graceful_shutdown_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
        missing_ok: bool = False,
        wait_for_ready: bool = True,
        idempotency_key: str | None = None,
    ) -> None:
        """Internal async: Stop the sandbox using shared _stop_task pattern.

        First caller creates the task; later callers join it.
        """
        async with self._stop_lock:
            if self._is_done:
                logger.debug("stop() called on already-stopped sandbox %s", self._sandbox_id)
                return
            if self._sandbox_id is None and not self._is_stopping:
                logger.debug("stop() called on sandbox that was never started")
                self._state = _NotStarted(cancelled=True)
                return
            if self._stop_task is None:
                self._stop_task = asyncio.create_task(
                    self._do_stop(
                        snapshot_on_stop=snapshot_on_stop,
                        graceful_shutdown_seconds=graceful_shutdown_seconds,
                        missing_ok=missing_ok,
                        wait_for_ready=wait_for_ready,
                        idempotency_key=idempotency_key,
                    )
                )
                self._stop_task.add_done_callback(self._on_stop_task_done)
            task = self._stop_task

        # Join the shared stop task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
        finally:
            # Deregister from session if we own the stop
            if self._stop_owned and self._session is not None:
                self._session._deregister_sandbox(self)
            # Close channels to release resources
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
        idempotency_key: str | None = None,
    ) -> OperationRef[None]:
        """Stop sandbox, return OperationRef immediately.

        The sandbox transitions through TERMINATING (grace period draining)
        before reaching a terminal state (COMPLETED or FAILED). The returned
        OperationRef resolves when the backend confirms a terminal state, not
        just when the stop RPC succeeds.

        Multiple callers share the same underlying stop task: the first caller
        creates it, subsequent callers join it.

        The sandbox is deregistered from its session regardless of whether
        the stop was successful, since the sandbox is no longer usable.

        Args:
            snapshot_on_stop: If True, capture a file-system snapshot (FSS) of
                the configured mount before shutdown. The resulting snapshot ID
                is available via the ``file_system_snapshot_id`` property after
                the returned OperationRef resolves. Requires the sandbox to have
                been started with a ``file_system_snapshot`` mount and the org to
                be enabled for FSS.
            graceful_shutdown_seconds: Time to wait for graceful shutdown.
            missing_ok: If True, suppress SandboxNotFoundError when sandbox
                doesn't exist.
            wait_for_ready: When ``snapshot_on_stop`` is True, block until the
                snapshot reaches READY (or FAILED) before the stop completes.
                Ignored when ``snapshot_on_stop`` is False.
            idempotency_key: Optional client-supplied key to deduplicate the
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
                idempotency_key=idempotency_key,
            )
        )
        return OperationRef(future)

    async def _snapshot_async(
        self,
        *,
        wait_for_ready: bool,
        idempotency_key: str | None,
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
        create_max_timeout = int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS) if wait_for_ready else None
        # Generate an idempotency key when the caller didn't supply one so a
        # retried create (after a transient failure that may have already
        # committed server-side) dedups instead of creating a second snapshot.
        effective_idempotency_key = idempotency_key or uuid.uuid4().hex
        return await _retry_transient_rpc(
            lambda: _create_snapshot_via_stub(
                stub,
                sandbox_id,
                idempotency_key=effective_idempotency_key,
                wait_for_ready=wait_for_ready,
                auth_metadata=self._auth_metadata,
                timeout=create_timeout,
                max_timeout_seconds=create_max_timeout,
            ),
            budget_seconds=DEFAULT_FSS_RETRY_BUDGET_SECONDS,
            operation="Create file-system snapshot",
        )

    def snapshot(
        self,
        *,
        wait_for_ready: bool = True,
        idempotency_key: str | None = None,
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
            idempotency_key: Optional client-supplied key to deduplicate the
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
                idempotency_key=idempotency_key,
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
    ) -> None:
        """Internal async: Stream logs from sandbox, push lines to queue.

        Uses gRPC bidirectional streaming to receive log data as it arrives.
        Buffers partial lines and pushes complete lines to output_queue.
        Signals end-of-stream with None sentinel when log stream completes.

        Resume handling: when the server includes a session_id and cumulative
        byte offset on each log frame, we remember the last-received values
        and, on a transient transport failure, re-init with resume_session_id
        and resume_offset so the stream picks up where it left off.  Servers
        that do not advertise a session_id (or reply with an in-band
        SESSION_NOT_FOUND on the resume init) fall back to a fresh init —
        the live tail restarts from the current head with no replay of
        older bytes.  ``tail_lines`` and ``since_time`` are only applied to
        the very first attempt; fresh-fallback re-inits send no replay
        filters so the user does not see the original window re-emitted on
        every transient disconnect.
        """
        # Track whether the inner block exited normally.  Only on a clean
        # exit do we emit the EOF sentinel.  On the failure path the
        # outer ``except`` below emits the exception, and StreamReader
        # stops iteration on Exception, so the sentinel would be
        # redundant — and emitting it first (via an unconditional
        # ``finally``) would race the exception to the queue and cause
        # the consumer to see a clean EOS instead of the actual failure.
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

            await self._ensure_client()
            auth_metadata = self._auth_metadata

            logger.debug("Streaming logs from sandbox %s (follow=%s)", self._sandbox_id, follow)

            # Resume state.  ``session_id`` is the opaque token the server
            # echoes on every data frame; ``last_offset`` is the cumulative
            # byte count after the most recently *delivered* chunk.  Bytes
            # that were in flight at disconnect (sent by the server but not
            # consumed by us) are NOT in last_offset, so the server replays
            # them on resume — that is the at-least-once delivery contract.
            session_id = ""
            last_offset = 0
            # Line-buffer state spans attempts so a partial line received
            # before a disconnect is concatenated with the bytes replayed on
            # resume.  The server's at-least-once replay may duplicate a
            # final partial line, which is acceptable — the alternative
            # (dropping the buffer on disconnect) would drop user output.
            line_parts: list[str] = []
            line_parts_bytes = 0

            # Narrow for closure capture: the None-check above already
            # raised, but mypy cannot propagate that narrowing into the
            # request_generator closure below.
            sandbox_id = self._sandbox_id
            assert sandbox_id is not None

            attempt = 0
            done = False
            # Retain the most recent transport error so it can chain into
            # the synthesized SandboxUnavailableError if the retry budget
            # is exhausted.  Without this, an SRE debugging a resume-
            # failure run sees only the generic "could not be resumed"
            # message with no underlying gRPC status code.
            last_transport_error: grpc.aio.AioRpcError | None = None
            try:
                while not done and attempt < STREAMING_RESUME_MAX_ATTEMPTS:
                    is_first_attempt = attempt == 0
                    is_resume = bool(session_id) and attempt > 0
                    if is_resume:
                        logger.debug(
                            "Resuming log stream for sandbox %s (attempt %d, offset %d)",
                            self._sandbox_id,
                            attempt,
                            last_offset,
                        )

                    # Reacquire the streaming channel and stub on every
                    # attempt.  ``_get_or_create_streaming_channel`` returns
                    # the cached channel when it is still healthy, so this
                    # is free in the common case.  After an external
                    # teardown (the integration test forcibly closes the
                    # cached channel; ``stop()`` invalidates it during
                    # shutdown), this is what gives the retry loop a live
                    # stub to run the next ``StreamLogs`` against.
                    channel = await self._get_or_create_streaming_channel()
                    stub = streaming_pb2_grpc.GatewayStreamingServiceStub(channel)  # type: ignore[no-untyped-call]

                    shutdown_event = asyncio.Event()

                    def make_request_generator(
                        resume_id: str,
                        resume_off: int,
                        first_attempt: bool,
                        shutdown: asyncio.Event,
                    ) -> Callable[[], AsyncIterator[streaming_pb2.LogStreamRequest]]:
                        async def request_generator() -> AsyncIterator[
                            streaming_pb2.LogStreamRequest
                        ]:
                            init = streaming_pb2.LogStreamInit(
                                sandbox_id=sandbox_id,
                                follow=follow,
                            )
                            if resume_id:
                                # Per the wire contract: when resume_session_id
                                # is set, all fields except sandbox_id are
                                # ignored.  Keep the others off the wire to
                                # make that explicit and avoid log noise
                                # from servers that validate the rule.
                                init.resume_session_id = resume_id
                                init.resume_offset = resume_off
                            elif first_attempt:
                                # tail_lines / since_time describe the
                                # *original* replay window the caller asked
                                # for.  Re-emitting them on every fresh-
                                # fallback re-init would replay the same
                                # window after each transient disconnect —
                                # for ``stream_logs(follow=True,
                                # tail_lines=100)`` the user would see the
                                # last 100 lines re-emitted whenever
                                # SESSION_NOT_FOUND / REPLAY_GAP / RUNNER_*
                                # forced a fresh init.  ``timestamps`` is
                                # a formatting flag, not a replay window,
                                # so it stays across attempts.
                                if tail_lines is not None:
                                    init.tail_lines = tail_lines
                                if since_time is not None:
                                    ts = timestamp_pb2.Timestamp()
                                    ts.FromDatetime(since_time)
                                    init.since_time.CopyFrom(ts)
                            if timestamps:
                                init.timestamps = True
                            yield streaming_pb2.LogStreamRequest(init=init)
                            if follow:
                                await shutdown.wait()
                                yield streaming_pb2.LogStreamRequest(
                                    close=streaming_pb2.LogStreamClose()
                                )

                        return request_generator

                    # For follow=False, apply a client-side deadline so a
                    # stalled server doesn't block the client indefinitely.
                    # follow=True streams are intentionally unbounded.
                    grpc_timeout = timeout_seconds if not follow else None

                    call: grpc.aio.StreamStreamCall[
                        streaming_pb2.LogStreamRequest, streaming_pb2.LogStreamResponse
                    ] = stub.StreamLogs(
                        request_iterator=make_request_generator(
                            session_id, last_offset, is_first_attempt, shutdown_event
                        )(),
                        metadata=auth_metadata,
                        **({"timeout": grpc_timeout} if grpc_timeout is not None else {}),
                    )

                    response_queue: asyncio.Queue[
                        streaming_pb2.LogStreamResponse | Exception | None
                    ] = asyncio.Queue(maxsize=STREAMING_RESPONSE_QUEUE_SIZE)

                    async def collect_responses(
                        active_call: grpc.aio.StreamStreamCall[
                            streaming_pb2.LogStreamRequest, streaming_pb2.LogStreamResponse
                        ],
                        queue: asyncio.Queue[streaming_pb2.LogStreamResponse | Exception | None],
                    ) -> None:
                        try:
                            async for response in active_call:
                                await queue.put(response)
                                if response.HasField("error") or response.HasField("complete"):
                                    return
                        except grpc.aio.AioRpcError as exc_inner:
                            await queue.put(exc_inner)
                        except Exception as exc_inner:
                            await queue.put(exc_inner)
                        finally:
                            await queue.put(None)

                    collect_task = asyncio.create_task(collect_responses(call, response_queue))

                    # Outcome of this attempt: "done" (terminal), "resume"
                    # (transient — try again), or "fresh" (server told us
                    # the session is gone, drop resume state and re-init).
                    attempt_outcome: str = "done"
                    pending_error: Exception | None = None

                    try:
                        while True:
                            item = await response_queue.get()
                            if item is None:
                                # Stream ended without a terminal frame and
                                # without a transport error — treat as done.
                                break
                            if isinstance(item, grpc.aio.AioRpcError):
                                if follow and _is_resumable_transport_error(item):
                                    # The outer `while attempt < MAX_ATTEMPTS`
                                    # is the only budget — let it decide
                                    # whether another retry is possible.
                                    # Marking the outcome "resume" here keeps
                                    # the exhaustion path (the synthesized
                                    # SandboxUnavailableError below) the
                                    # single place that translates "budget
                                    # spent" to a user-visible error.
                                    #
                                    # We do NOT gate on session_id being
                                    # populated: if the first transport
                                    # error happens before any frame
                                    # arrives, session_id is still "", and
                                    # the next request_generator will fall
                                    # into the empty-resume_id branch — a
                                    # bounded fresh-init retry, which is
                                    # exactly the right behavior for a
                                    # flaky gateway connection at the
                                    # opening edge of the tail.
                                    logger.debug(
                                        "Log stream transport error %s; will attempt resume",
                                        item.code(),
                                    )
                                    last_transport_error = item
                                    attempt_outcome = "resume"
                                else:
                                    translated = _translate_rpc_error(
                                        item,
                                        sandbox_id=self._sandbox_id,
                                        operation="Stream logs",
                                    )
                                    # Chain the gRPC error so callers can
                                    # inspect the underlying status code via
                                    # __cause__ — matches the convention
                                    # used at other _translate_rpc_error
                                    # call sites in this module.
                                    translated.__cause__ = item
                                    pending_error = translated
                                break
                            if isinstance(item, Exception):
                                pending_error = item
                                break

                            response = item
                            if response.HasField("data"):
                                data_msg = response.data
                                # Capture resume metadata.  These fields are
                                # only populated by servers that speak the
                                # resume protocol; on older builds they are
                                # the proto defaults ("" and 0), which means
                                # we never attempt resume — exactly right.
                                incoming_session_id = data_msg.session_id
                                if incoming_session_id:
                                    session_id = incoming_session_id
                                if data_msg.offset:
                                    last_offset = data_msg.offset

                                text = data_msg.data.decode("utf-8", errors="replace")
                                line_parts.append(text)
                                line_parts_bytes += len(text)

                                if "\n" not in text and line_parts_bytes < MAX_LINE_BUFFER_BYTES:
                                    continue

                                combined = "".join(line_parts)
                                line_parts.clear()
                                line_parts_bytes = 0
                                parts = combined.split("\n")
                                for part in parts[:-1]:
                                    await output_queue.put(part + "\n")
                                if parts[-1]:
                                    line_parts.append(parts[-1])
                                    line_parts_bytes = len(parts[-1])

                            elif response.HasField("error"):
                                code = response.error.code or None
                                # The wire contract documents every
                                # LogStreamError as terminal — the server
                                # will not send further frames on this call.
                                # REPLAY_GAP also warrants an operator-
                                # visible warning, because data below the
                                # server's replay window was permanently
                                # missed.
                                if code == _STREAMING_REPLAY_GAP:
                                    logger.warning(
                                        "Log stream replay gap for sandbox %s"
                                        " (offset %d below server replay window);"
                                        " reconnecting with fresh init",
                                        self._sandbox_id,
                                        last_offset,
                                    )
                                if follow and code in _STREAMING_FRESH_REINIT_CODES:
                                    # The server told us to reconnect from
                                    # scratch.  Drop resume state and clear
                                    # the cross-attempt line buffer — on a
                                    # fresh init we are NOT going to receive
                                    # the bytes that preceded any partial
                                    # line, so preserving it would splice
                                    # unrelated future bytes into a stale
                                    # fragment.
                                    logger.debug(
                                        "Log stream %s; falling back to fresh init",
                                        code,
                                    )
                                    session_id = ""
                                    last_offset = 0
                                    line_parts.clear()
                                    line_parts_bytes = 0
                                    attempt_outcome = "fresh"
                                    break
                                # Every other code — including
                                # INVALID_RESUME_OFFSET (echoed offset is
                                # corrupt; retrying with the same state
                                # would loop), SANDBOX_NOT_FOUND, and
                                # PERMISSION_DENIED — is terminal, no
                                # retry.  Surface as a SandboxError.
                                pending_error = SandboxError(
                                    f"Log stream error: {response.error.message}",
                                    reason=code,
                                )
                                break

                            elif response.HasField("complete"):
                                # Server signaled clean completion.  Exit
                                # the inner loop with attempt_outcome
                                # still "done"; the outer loop will set
                                # done=True and the `finally` below cancels
                                # the call and collector.  Any frames the
                                # server pushed after `complete` would be
                                # a protocol violation and are discarded.
                                break

                        # Determine terminal-vs-retry outcome.
                        if attempt_outcome == "done":
                            done = True
                            if pending_error is not None:
                                await output_queue.put(pending_error)
                                return
                    except asyncio.CancelledError:
                        raise
                    finally:
                        shutdown_event.set()
                        with contextlib.suppress(Exception):
                            call.cancel()
                        collect_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await collect_task

                    if not done:
                        # Bounded exponential backoff before resume/fresh
                        # init.  Capped so the orphan window on the server
                        # (30s) doesn't expire while the client sleeps.
                        backoff = min(
                            STREAMING_RESUME_BACKOFF_SECONDS * (2**attempt),
                            STREAMING_RESUME_MAX_BACKOFF_SECONDS,
                        )
                        await asyncio.sleep(backoff)

                    attempt += 1

                if not done and attempt >= STREAMING_RESUME_MAX_ATTEMPTS:
                    # Exhausted the retry budget after the final transport
                    # error.  pending_error is set only on paths that
                    # chose not to resume; the resume-chosen path leaves
                    # pending_error=None, so synthesize a stable user-
                    # facing error but chain the underlying gRPC error
                    # so SREs can see the real status code post-mortem.
                    synthesized = SandboxUnavailableError(
                        "Log stream disconnected and could not be resumed",
                    )
                    if last_transport_error is not None:
                        synthesized.__cause__ = last_transport_error
                    await output_queue.put(synthesized)
                    return

                # Flush any remaining partial line on clean completion.
                if line_parts:
                    await output_queue.put("".join(line_parts))
                inner_exit_clean = True
            finally:
                # See ``inner_exit_clean`` declaration above for rationale.
                if inner_exit_clean:
                    await output_queue.put(None)
        except Exception as exc:
            # A failure escaped the retry loop (early-setup failure, an
            # asyncio cancellation propagating mid-loop, a per-attempt
            # setup error that is not an AioRpcError, etc.).  Surface it
            # to the consumer.  StreamReader stops iteration on Exception,
            # so no trailing None sentinel is needed — and the inner
            # ``finally`` deliberately did not emit one for this path.
            # ``put_nowait`` avoids blocking if the consumer stopped
            # draining.
            try:
                output_queue.put_nowait(exc)
            except asyncio.QueueFull:
                pass

    async def _prepare_streaming_call(
        self,
    ) -> streaming_pb2_grpc.GatewayStreamingServiceStub:
        """Shared StreamExec preamble: ensure running, return a stub."""
        await self._ensure_started_async()
        if self._is_done or self._is_stopping:
            raise SandboxNotRunningError(f"Sandbox {self._sandbox_id} has been stopped")
        if self._sandbox_id is None:
            raise SandboxNotRunningError("No sandbox is running")
        await self._wait_until_running_async()
        await self._ensure_client()
        channel = await self._get_or_create_streaming_channel()
        return streaming_pb2_grpc.GatewayStreamingServiceStub(channel)  # type: ignore[no-untyped-call]

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

        try:
            stub = await self._prepare_streaming_call()
            auth_metadata = self._auth_metadata
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

            async def request_generator() -> AsyncIterator[streaming_pb2.ExecStreamRequest]:
                """Generate request messages for the TTY bidirectional stream."""
                init_msg = streaming_pb2.ExecStreamInit(
                    sandbox_id=sandbox_id,
                    command=list(command),
                    tty=True,
                )
                if tty_width is not None:
                    init_msg.tty_width = tty_width
                if tty_height is not None:
                    init_msg.tty_height = tty_height
                yield streaming_pb2.ExecStreamRequest(init=init_msg)

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
                                yield streaming_pb2.ExecStreamRequest(
                                    resize=streaming_pb2.ExecStreamResize(width=w, height=h)
                                )
                            if get_task not in done:
                                continue

                        if get_task in done:
                            data = get_task.result()
                            if data is None:
                                yield streaming_pb2.ExecStreamRequest(
                                    close=streaming_pb2.ExecStreamClose()
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
                                yield streaming_pb2.ExecStreamRequest(
                                    stdin=streaming_pb2.ExecStreamData(data=chunk)
                                )
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
                streaming_pb2.ExecStreamRequest, streaming_pb2.ExecStreamResponse
            ] = stub.StreamExec(
                request_iterator=request_generator(),
                timeout=None,
                metadata=auth_metadata,
            )

            # Bounded queue propagates backpressure to gRPC reads — when the
            # consumer is slow, collect_responses() blocks on put(), stopping
            # reads from gRPC.  Without this bound, long-lived TTY sessions
            # can accumulate unlimited protobuf messages in memory.
            response_queue: asyncio.Queue[streaming_pb2.ExecStreamResponse | Exception | None] = (
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
                                "ready_at": response.ready.ready_at.ToDatetime().isoformat(),
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

        stub = await self._prepare_streaming_call()
        auth_metadata = self._auth_metadata
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

        async def request_generator() -> AsyncIterator[streaming_pb2.ExecStreamRequest]:
            """Generate request messages for the bidirectional stream.

            Yields init message, then stdin data if enabled, then returns.
            The generator naturally completes when all messages are sent,
            which signals gRPC to half-close the send direction.
            """
            # Yield init message first
            yield streaming_pb2.ExecStreamRequest(
                init=streaming_pb2.ExecStreamInit(
                    sandbox_id=sandbox_id,
                    command=rpc_command,
                )
            )

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
                            yield streaming_pb2.ExecStreamRequest(
                                close=streaming_pb2.ExecStreamClose()
                            )
                            return

                        # Chunk large data into 64KB pieces
                        for i in range(0, len(data), STDIN_CHUNK_SIZE):
                            chunk = data[i : i + STDIN_CHUNK_SIZE]
                            yield streaming_pb2.ExecStreamRequest(
                                stdin=streaming_pb2.ExecStreamData(data=chunk)
                            )
                    except asyncio.CancelledError:
                        return

        # Create the bidirectional streaming call with request iterator
        call_timeout = (
            timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS if timeout is not None else None
        )
        call: grpc.aio.StreamStreamCall[
            streaming_pb2.ExecStreamRequest, streaming_pb2.ExecStreamResponse
        ] = stub.StreamExec(
            request_iterator=request_generator(),
            timeout=call_timeout,
            metadata=auth_metadata,
        )

        # Queue decouples stream iteration from our processing.
        # Without this, processing suspends the stream and can cause issues.
        response_queue: asyncio.Queue[streaming_pb2.ExecStreamResponse | Exception | None] = (
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
                    await response_queue.put(e)
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
                                "ready_at": response.ready.ready_at.ToDatetime().isoformat(),
                            },
                        )
                    ready_event.set()

                elif response.HasField("output"):
                    data = response.output.data
                    # Decode as UTF-8 for queue (replacing invalid chars)
                    text = data.decode("utf-8", errors="replace")
                    stream_type = response.output.stream_type

                    if stream_type == streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDOUT:
                        stdout_buffer.append(data)
                        await stdout_queue.put(text)
                    elif stream_type == streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR:
                        stderr_buffer.append(data)
                        await stderr_queue.put(text)
                    else:
                        logger.warning(
                            "Received output with unexpected stream_type %s, treating as stdout",
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
    ) -> tuple[int, bytes, bytes]:
        """Run one non-TTY StreamExec and return raw stdout/stderr bytes.

        See also ``_exec_streaming_async`` — the queue-driven public variant;
        keep handshake/timeout/error-translation in sync between the two.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._request_timeout_seconds

        if not command:
            raise ValueError("Command cannot be empty")

        stub = await self._prepare_streaming_call()
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

        async def request_generator() -> AsyncIterator[streaming_pb2.ExecStreamRequest]:
            yield streaming_pb2.ExecStreamRequest(
                init=streaming_pb2.ExecStreamInit(
                    sandbox_id=sandbox_id,
                    command=list(command),
                )
            )

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
                    yield streaming_pb2.ExecStreamRequest(
                        stdin=streaming_pb2.ExecStreamData(data=chunk)
                    )
            else:
                # AsyncIterable[bytes] — caller controls chunk size; each
                # yielded chunk is passed through unmodified so the caller
                # never materializes the full payload in memory.
                async for chunk in stdin:
                    if shutdown_event.is_set():
                        return
                    if not chunk:
                        continue
                    yield streaming_pb2.ExecStreamRequest(
                        stdin=streaming_pb2.ExecStreamData(data=_coerce_bytes_chunk(chunk))
                    )

            yield streaming_pb2.ExecStreamRequest(close=streaming_pb2.ExecStreamClose())

        call_timeout = (
            timeout + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS if timeout is not None else None
        )
        call: grpc.aio.StreamStreamCall[
            streaming_pb2.ExecStreamRequest, streaming_pb2.ExecStreamResponse
        ] = stub.StreamExec(
            request_iterator=request_generator(),
            timeout=call_timeout,
            metadata=self._auth_metadata,
        )

        try:
            async for response in call:
                if response.HasField("ready"):
                    ready_event.set()
                elif response.HasField("output"):
                    data = response.output.data
                    stream_type = response.output.stream_type
                    if stream_type == streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR:
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

    async def _read_file_unary_async(self, filepath: str, timeout: float) -> bytes:
        assert self._stub is not None
        request = gateway_pb2.RetrieveFileSandboxRequest(
            sandbox_id=self._sandbox_id,
            filepath=filepath,
            max_timeout_seconds=int(timeout),
        )

        try:
            response = await self._stub.RetrieveFile(
                request, timeout=timeout, metadata=self._auth_metadata
            )
        except grpc.RpcError as e:
            raise _translate_rpc_error(
                e,
                sandbox_id=self._sandbox_id,
                operation="Read file",
                filepath=filepath,
            ) from e

        if not response.success:
            logger.warning("Failed to read file %s from sandbox %s", filepath, self._sandbox_id)
            raise SandboxFileError(
                f"Failed to read file '{filepath}': {response.error_message}",
                filepath=filepath,
            )

        return bytes(response.file_contents)

    async def _read_file_via_exec_streaming(
        self, filepath: str, timeout: float, *, expected_size: int | None = None
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

    async def _stat_file_size_async(self, filepath: str, timeout: float) -> int | None:
        """Best-effort: ask the sandbox for the file's size in bytes.

        Returns ``None`` if the size could not be determined (stat unavailable,
        unexpected output, transient transport error). Used by
        ``read_file_streaming`` to capture a pre-read size baseline for the
        truncation check; a ``None`` result means the check is skipped rather
        than raising — a stat failure on its own is not a streaming-read failure.
        """
        try:
            returncode, stdout, _stderr = await self._exec_streaming_binary_async(
                ["/bin/sh", "-c", 'stat -c %s -- "$1" 2>/dev/null', "cwsandbox-stat", filepath],
                timeout_seconds=timeout,
                operation="Stat file size",
                filepath=filepath,
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
        operation's remaining wall-clock budget — the stat and the read it
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
        the read — the server-reported size for the read_file fallback, or a
        pre-read ``stat`` for read_file_streaming. On a backend where the
        streaming channel silently truncates (e.g. the lossless gate is off),
        the read command exits 0 having produced the whole file while the client
        received only a prefix; without this the caller would consume a partial
        read as if complete (issue #1172).

        Only a SHORT read (``delivered < expected``) is flagged. Specifically:
        - ``expected is None`` (size unknown — server omitted it, or ``stat`` is
          unavailable on a distroless/scratch image): skip. The check is a
          best-effort backstop, not a guarantee; the public docstrings scope
          this.
        - ``expected == 0`` (pseudo-files such as ``/proc/*`` and ``/sys/*``
          report size 0 while ``cat`` legitimately yields content): skip, to
          avoid a false-positive on a fully-delivered read.
        - ``delivered >= expected``: not short, so no raise. Because ``expected``
          is the *pre-read* size, a file appended to during the read grows the
          delivered byte count above the baseline rather than below it — a
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
            return await self._read_file_unary_async(filepath, timeout)
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
            return await self._read_file_via_exec_streaming(filepath, timeout, expected_size=size)
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
            return await self._read_file_via_exec_streaming(filepath, timeout)

    def read_file(
        self,
        filepath: str,
        *,
        timeout_seconds: float | None = None,
    ) -> OperationRef[bytes]:
        """Read file from sandbox, return OperationRef immediately.

        Args:
            filepath: Path to file in sandbox
            timeout_seconds: Timeout for the operation

        Returns:
            OperationRef[bytes]: Use .result() to block and retrieve contents.

        Behavior:
            Files up to ~32 MiB are read in a single unary call. Larger files
            (up to ~256 MiB) transparently fall back to a streaming read — the
            first such fallback per Sandbox logs once at INFO. When the server
            reports the file's size, files above ~256 MiB are refused with
            ``CWSANDBOX_FILE_TOO_LARGE``; use ``read_file_streaming`` for those.

            The whole result is held in memory regardless of path. The client
            cannot always know the remote size in advance (e.g. when the backend
            signals the oversized read via resource exhaustion rather than a
            sized ``CWSANDBOX_FILE_TOO_LARGE``), so a very large file can still
            be buffered in full rather than refused — prefer
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
        future = self._loop_manager.run_async(self._read_file_async(filepath, timeout))
        return OperationRef(future)

    async def _write_file_unary_async(
        self,
        filepath: str,
        contents: bytes,
        timeout: float,
    ) -> None:
        assert self._stub is not None
        request = gateway_pb2.AddFileSandboxRequest(
            sandbox_id=self._sandbox_id,
            filepath=filepath,
            file_contents=contents,
            max_timeout_seconds=int(timeout),
        )

        try:
            response = await self._stub.AddFile(
                request, timeout=timeout, metadata=self._auth_metadata
            )
        except grpc.RpcError as e:
            raise _translate_rpc_error(
                e,
                sandbox_id=self._sandbox_id,
                operation="Write file",
                filepath=filepath,
            ) from e

        if not response.success:
            logger.warning("Failed to write file %s to sandbox %s", filepath, self._sandbox_id)
            raise SandboxFileError(
                f"Failed to write file '{filepath}'",
                filepath=filepath,
            )

    async def _write_file_via_exec_streaming(
        self,
        filepath: str,
        contents: bytes,
        timeout: float,
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
            await self._write_file_via_exec_streaming(filepath, contents, timeout)
            return

        try:
            await self._write_file_unary_async(filepath, contents, timeout)
        except SandboxFileError as e:
            if e.reason != CWSANDBOX_FILE_TOO_LARGE:
                raise
            self._record_observed_cap(e)
            if size > MAX_AUTO_FALLBACK_BYTES:
                raise
            self._notify_streaming_fallback_once(
                "Write file", filepath, size, suggest_method="write_file_streaming"
            )
            await self._write_file_via_exec_streaming(filepath, contents, timeout)
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
            await self._write_file_via_exec_streaming(filepath, contents, timeout)

    def write_file(
        self,
        filepath: str,
        contents: bytes,
        *,
        timeout_seconds: float | None = None,
    ) -> OperationRef[None]:
        """Write file to sandbox, return OperationRef immediately.

        Args:
            filepath: Path to file in sandbox
            contents: File contents as bytes
            timeout_seconds: Timeout for the operation

        Returns:
            OperationRef[None]: Use .result() to block until complete.

        Behavior:
            Payloads up to ~32 MiB are written in a single unary call. Larger
            payloads (up to ~256 MiB) transparently fall back to a streaming
            write — the first such fallback per Sandbox logs once at INFO.
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
        future = self._loop_manager.run_async(self._write_file_async(filepath, contents, timeout))
        return OperationRef(future)

    async def _write_file_streaming_async(
        self,
        filepath: str,
        source: bytes | Iterable[bytes] | AsyncIterable[bytes],
        timeout: float,
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
    ) -> OperationRef[None]:
        """Stream a file to the sandbox without materializing the full payload.

        Prefer this over ``write_file`` for payloads larger than roughly
        32 MiB, or any time the data is already an iterator (file handle,
        generator, async producer).

        Args:
            filepath: Absolute path inside the sandbox.
            source: Payload as ``bytes``, a sync ``Iterable[bytes]``, or an
                ``AsyncIterable[bytes]``. Each yielded chunk is sent as-is;
                ``bytes`` input is sliced into 1 MiB chunks internally.
                Yielded items must be ``bytes``, ``bytearray``, or
                ``memoryview`` — anything else raises ``TypeError``.
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
            self._write_file_streaming_async(filepath, source, timeout)
        )
        return OperationRef(future)

    async def _read_file_streaming_async(
        self,
        filepath: str,
        output_queue: asyncio.Queue[bytes | Exception | None],
        timeout: float,
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
            await self._ensure_client()
            stub = await self._prepare_streaming_call()
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
            expected_size = await self._stat_file_size_async(filepath, self._stat_budget(deadline))

            stderr_buf = bytearray()
            stderr_cap = STREAMING_READ_STDERR_CAP_BYTES
            exit_code: int | None = None
            total_bytes = 0

            async def request_generator() -> AsyncIterator[streaming_pb2.ExecStreamRequest]:
                yield streaming_pb2.ExecStreamRequest(
                    init=streaming_pb2.ExecStreamInit(
                        sandbox_id=sandbox_id,
                        command=["/bin/cat", "--", filepath],
                    )
                )
                # Close stdin immediately so cat reads the file and exits.
                yield streaming_pb2.ExecStreamRequest(close=streaming_pb2.ExecStreamClose())

            # The read gets the budget remaining after the pre-read stat, so the
            # two phases together honor the caller's overall timeout.
            read_budget = self._remaining_budget(deadline)
            call_timeout = (
                read_budget + DEFAULT_CLIENT_TIMEOUT_BUFFER_SECONDS
                if read_budget is not None
                else None
            )
            call: grpc.aio.StreamStreamCall[
                streaming_pb2.ExecStreamRequest, streaming_pb2.ExecStreamResponse
            ] = stub.StreamExec(
                request_iterator=request_generator(),
                timeout=call_timeout,
                metadata=self._auth_metadata,
            )
            try:
                async for response in call:
                    if response.HasField("output"):
                        stream_type = response.output.stream_type
                        data = response.output.data
                        if stream_type == streaming_pb2.ExecStreamOutput.STREAM_TYPE_STDERR:
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
            finally:
                with contextlib.suppress(Exception):
                    call.cancel()

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
            self._read_file_streaming_async(filepath, output_queue, timeout)
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
    ) -> StreamReader[str]:
        """Stream logs from the sandbox's main process.

        Streams stdout/stderr from the sandbox's **main command** — the
        entrypoint passed to ``Sandbox.run()`` (or the default shell-trapped
        keep-alive). Output from commands started via ``exec()`` is **not**
        included; use ``Process.stdout``/``Process.stderr`` for those.

        .. note::

            Sandboxes created with the default keep-alive command do not
            produce any log output. To see logs here, pass a command that
            writes to stdout/stderr when calling ``Sandbox.run()``.

        Returns a StreamReader that yields log lines as strings. The method
        returns immediately — iteration on the StreamReader blocks until
        data arrives.

        Can also retrieve historical logs from stopped sandboxes when
        ``follow=False``.

        Args:
            follow: If True, continuously stream new logs (like ``tail -f``).
                If False, stream existing logs and stop. Only running
                sandboxes support ``follow=True``.
            tail_lines: Number of most recent lines to retrieve. If None,
                returns all available lines.
            since_time: Only return logs after this timestamp.
            timestamps: If True, prefix each line with an ISO 8601 timestamp
                from the server.
            timeout_seconds: Client-side deadline for the gRPC call. Defaults
                to ``request_timeout_seconds`` when ``follow=False``, and
                ``None`` (no timeout) when ``follow=True``.

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

            # Retrieve logs from a stopped sandbox
            sb = Sandbox.from_id("sbx-abc123").result()
            for line in sb.stream_logs(tail_lines=50):
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
            )
        )

        return StreamReader(output_queue, self._loop_manager, cancel=future.cancel)
