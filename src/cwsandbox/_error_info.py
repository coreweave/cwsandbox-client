# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""AIP-193 structured-error parser for gRPC trailing metadata.

Parses ``google.rpc.Status`` structured error details out of the
``grpc-status-details-bin`` trailing metadata entry produced by
servers that follow Google's AIP-193 error model, and returns the
subset of fields the SDK cares about (``ErrorInfo.reason``,
``ErrorInfo.domain``, ``ErrorInfo.metadata``, and
``RetryInfo.retry_delay`` when present, and ``BadRequest`` field
violations).

The parser is defensive: any failure to decode or extract details
returns ``None`` rather than raising, so callers can safely fall
back to status-code-only error handling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import grpc
from google.protobuf import message as _message
from google.rpc import error_details_pb2, status_pb2

from cwsandbox.exceptions import FieldViolation

_STATUS_DETAILS_KEY = "grpc-status-details-bin"

# Domain value used by the CoreWeave backend when emitting ErrorInfo.
# Reason-to-exception mapping is only applied when the ErrorInfo carries
# this domain, so third-party gRPC intermediaries or future non-CoreWeave
# services don't accidentally collide with CWSANDBOX_* reason strings.
CWSANDBOX_ERROR_DOMAIN = "cwsandbox.com"

# CWSANDBOX_* reason strings emitted by the CoreWeave backend when
# ``ErrorInfo.domain == CWSANDBOX_ERROR_DOMAIN``. Kept in sync with the
# backend AIP-193 error catalog.

# File operation reasons
CWSANDBOX_FILE_NOT_FOUND = "CWSANDBOX_FILE_NOT_FOUND"
CWSANDBOX_FILE_IS_DIRECTORY = "CWSANDBOX_FILE_IS_DIRECTORY"
CWSANDBOX_FILE_IO_FAILED = "CWSANDBOX_FILE_IO_FAILED"
CWSANDBOX_FILE_PERMISSION_DENIED = "CWSANDBOX_FILE_PERMISSION_DENIED"
# A size-policy refusal: the payload exceeds the server/client size cap and no
# data was lost. The fix is to switch to the streaming APIs. Distinct from
# CWSANDBOX_FILE_TRUNCATED, which signals data loss on a read that already used
# streaming.
CWSANDBOX_FILE_TOO_LARGE = "CWSANDBOX_FILE_TOO_LARGE"
# A post-hoc short read: a streamed read delivered fewer bytes than the file
# held, so data WAS lost. Surfaced by the SDK's own integrity check (not the
# backend), and switching to streaming is a no-op because the read already
# streamed. Kept separate from CWSANDBOX_FILE_TOO_LARGE so callers can tell a
# "too big, use streaming" refusal apart from a "your streamed read was
# truncated" failure without sniffing metadata.
CWSANDBOX_FILE_TRUNCATED = "CWSANDBOX_FILE_TRUNCATED"

FILE_ERROR_REASONS: frozenset[str] = frozenset(
    {
        CWSANDBOX_FILE_NOT_FOUND,
        CWSANDBOX_FILE_IS_DIRECTORY,
        CWSANDBOX_FILE_IO_FAILED,
        CWSANDBOX_FILE_PERMISSION_DENIED,
        CWSANDBOX_FILE_TOO_LARGE,
        CWSANDBOX_FILE_TRUNCATED,
    }
)

# Not-found reasons per context
CWSANDBOX_SANDBOX_NOT_FOUND = "CWSANDBOX_SANDBOX_NOT_FOUND"
CWSANDBOX_RUNNER_NOT_FOUND = "CWSANDBOX_RUNNER_NOT_FOUND"
CWSANDBOX_PROFILE_NOT_FOUND = "CWSANDBOX_PROFILE_NOT_FOUND"

# Timeout reasons
CWSANDBOX_COMMAND_TIMEOUT = "CWSANDBOX_COMMAND_TIMEOUT"

# Streaming-exec terminal error code. NOT an AIP-193 ErrorInfo reason: it is a
# free-form string carried in ``ExecStreamError.code`` on the exec/file stream
# (not in gRPC trailing metadata), so it does not flow through the ErrorInfo
# parser. The server emits it when an output stream is ended early because it
# was not being read fast enough to keep up with the command's output — an
# explicit failure instead of silently dropping output. The SDK maps it to
# ``SandboxStreamBackpressureError`` and exposes it on that exception's
# ``.stream_code`` attribute, keeping it out of the AIP-193 ``.reason``
# namespace.
STREAM_BACKPRESSURE = "STREAM_BACKPRESSURE"

# Terminal ExecStreamError code emitted when a command's output was truncated in
# transit even though the command exited normally — the transport dropped a tail
# of the output. The SDK maps it to ``SandboxStreamTruncatedError`` and exposes
# it on that exception's ``.stream_code`` attribute, keeping it out of the
# AIP-193 ``.reason`` namespace.
STREAM_TRUNCATED = "STREAM_TRUNCATED"

# Unavailable reasons
CWSANDBOX_RUNNER_UNAVAILABLE = "CWSANDBOX_RUNNER_UNAVAILABLE"
CWSANDBOX_BACKEND_UNAVAILABLE = "CWSANDBOX_BACKEND_UNAVAILABLE"

UNAVAILABLE_REASONS: frozenset[str] = frozenset(
    {
        CWSANDBOX_RUNNER_UNAVAILABLE,
        CWSANDBOX_BACKEND_UNAVAILABLE,
    }
)

# File-system snapshot (FSS) reasons. Emitted by the snapshot create/restore,
# snapshot-on-stop, and snapshot management RPCs.
CWSANDBOX_FSS_NOT_FOUND = "CWSANDBOX_FSS_NOT_FOUND"
CWSANDBOX_FSS_NOT_READY = "CWSANDBOX_FSS_NOT_READY"
CWSANDBOX_FSS_NOT_SUPPORTED = "CWSANDBOX_FSS_NOT_SUPPORTED"
CWSANDBOX_FSS_SIZE_EXCEEDED = "CWSANDBOX_FSS_SIZE_EXCEEDED"
CWSANDBOX_FSS_QUOTA_EXCEEDED = "CWSANDBOX_FSS_QUOTA_EXCEEDED"
CWSANDBOX_FSS_BUCKET_MISMATCH = "CWSANDBOX_FSS_BUCKET_MISMATCH"
CWSANDBOX_FSS_BUCKET_UNAVAILABLE = "CWSANDBOX_FSS_BUCKET_UNAVAILABLE"
CWSANDBOX_FSS_RESTORE_FAILED = "CWSANDBOX_FSS_RESTORE_FAILED"
CWSANDBOX_FSS_CREATE_FAILED = "CWSANDBOX_FSS_CREATE_FAILED"
CWSANDBOX_FSS_CREATE_TIMED_OUT = "CWSANDBOX_FSS_CREATE_TIMED_OUT"
CWSANDBOX_FSS_AUTH_FAILED = "CWSANDBOX_FSS_AUTH_FAILED"
CWSANDBOX_FSS_TRANSPORT_FAILED = "CWSANDBOX_FSS_TRANSPORT_FAILED"
CWSANDBOX_FSS_WAIT_TIMEOUT = "CWSANDBOX_FSS_WAIT_TIMEOUT"
CWSANDBOX_FSS_CANCELED = "CWSANDBOX_FSS_CANCELED"
CWSANDBOX_FSS_BACKEND_THROTTLED = "CWSANDBOX_FSS_BACKEND_THROTTLED"
CWSANDBOX_FSS_INFLIGHT_LIMIT = "CWSANDBOX_FSS_INFLIGHT_LIMIT"

# Internal/terminal FSS failures that map to the generic SandboxSnapshotError.
SNAPSHOT_INTERNAL_REASONS: frozenset[str] = frozenset(
    {
        CWSANDBOX_FSS_BUCKET_UNAVAILABLE,
        CWSANDBOX_FSS_RESTORE_FAILED,
        CWSANDBOX_FSS_CREATE_FAILED,
        CWSANDBOX_FSS_CREATE_TIMED_OUT,
        CWSANDBOX_FSS_AUTH_FAILED,
        CWSANDBOX_FSS_TRANSPORT_FAILED,
        CWSANDBOX_FSS_CANCELED,
    }
)

# Transient FSS failures (gRPC UNAVAILABLE): safe to retry with backoff.
SNAPSHOT_TRANSIENT_REASONS: frozenset[str] = frozenset(
    {
        CWSANDBOX_FSS_BACKEND_THROTTLED,
        CWSANDBOX_FSS_INFLIGHT_LIMIT,
    }
)


@dataclass(frozen=True)
class ParsedError:
    """Structured error fields parsed from gRPC trailing metadata details.

    Attributes:
        reason: The ``ErrorInfo.reason`` string (e.g. ``CWSANDBOX_FILE_NOT_FOUND``),
            or ``None`` if no ``ErrorInfo`` detail was present.
        domain: The ``ErrorInfo.domain`` namespace (e.g. ``cwsandbox.com``), or
            empty string when ``ErrorInfo`` is absent.
        metadata: The ``ErrorInfo.metadata`` map as a read-only ``Mapping``.
            Empty mapping (never ``None``) when ``ErrorInfo`` is present
            without metadata. Callers that need to mutate or own the data
            should shallow-copy via ``dict(parsed.metadata)``.
        retry_delay: The ``RetryInfo.retry_delay`` as a ``timedelta``, or ``None``
            if no ``RetryInfo`` detail was present or the value was out of range.
        field_violations: ``BadRequest.field_violations`` entries emitted by the
            server. Empty tuple when no field violations were present.
    """

    reason: str | None = None
    domain: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    retry_delay: timedelta | None = None
    field_violations: tuple[FieldViolation, ...] = ()


def parse_error_info(err: grpc.RpcError) -> ParsedError | None:
    """Parse ``google.rpc.Status`` details from gRPC trailing metadata.

    Walks ``err.trailing_metadata()`` for every ``grpc-status-details-bin``
    entry (the key may repeat in gRPC metadata). The first entry that decodes
    to a ``google.rpc.Status`` containing at least one ``ErrorInfo``,
    ``RetryInfo``, or ``BadRequest`` detail wins; a malformed leading entry does
    not suppress a later valid one.

    Returns ``None`` when no entry yields usable structured details. Never
    raises - bad protobuf payloads, out-of-range RetryInfo durations, or any
    other decode failure (including exceptions from iterating a pathological
    trailing-metadata container) falls through to ``None``.
    """
    getter = getattr(err, "trailing_metadata", None)
    if getter is None:
        return None
    try:
        trailing = getter()
    except Exception:
        return None
    if trailing is None or not isinstance(trailing, Iterable):
        return None

    try:
        for entry in trailing:
            # Accept both tuple-style and .key/.value-style metadata entries.
            key = getattr(entry, "key", None)
            value = getattr(entry, "value", None)
            if not isinstance(key, str):
                try:
                    key, value = entry
                except (TypeError, ValueError):
                    continue
                if not isinstance(key, str):
                    continue
            if key.lower() != _STATUS_DETAILS_KEY:
                continue
            if not isinstance(value, (bytes, bytearray)):
                continue

            status = status_pb2.Status()
            try:
                status.ParseFromString(bytes(value))
            except _message.DecodeError:
                continue

            parsed = _extract_parsed_error(status)
            if parsed is not None:
                return parsed
    except Exception:
        # Defensive: never let metadata-iteration oddities mask the original
        # RPC failure. Fall through to status-code-only handling.
        return None

    return None


def _extract_parsed_error(status: status_pb2.Status) -> ParsedError | None:
    reason: str | None = None
    domain = ""
    metadata: Mapping[str, str] = {}
    retry_delay: timedelta | None = None
    field_violations: list[FieldViolation] = []

    for detail in status.details:
        # Treat an empty proto3-default `reason` as "not present" so a later
        # ErrorInfo with a real reason can still win.
        if reason is None and detail.Is(error_details_pb2.ErrorInfo.DESCRIPTOR):
            info = error_details_pb2.ErrorInfo()
            try:
                detail.Unpack(info)
            except _message.DecodeError:
                continue
            if not info.reason:
                continue
            reason = info.reason
            domain = info.domain
            metadata = info.metadata
        elif retry_delay is None and detail.Is(error_details_pb2.RetryInfo.DESCRIPTOR):
            retry = error_details_pb2.RetryInfo()
            try:
                detail.Unpack(retry)
            except _message.DecodeError:
                continue
            # An absent retry_delay field looks like Duration(0, 0) after
            # Unpack; skip so a later RetryInfo with a real value can win.
            if not retry.HasField("retry_delay"):
                continue
            retry_delay = _retry_delay_from_duration(retry.retry_delay)
        elif detail.Is(error_details_pb2.BadRequest.DESCRIPTOR):
            bad_request = error_details_pb2.BadRequest()
            try:
                detail.Unpack(bad_request)
            except _message.DecodeError:
                continue
            for violation in bad_request.field_violations:
                parsed_violation = _field_violation_from_proto(violation)
                if _has_field_violation_data(parsed_violation):
                    field_violations.append(parsed_violation)

    if reason is None and retry_delay is None and not field_violations:
        return None
    return ParsedError(
        reason=reason,
        domain=domain,
        metadata=metadata,
        retry_delay=retry_delay,
        field_violations=tuple(field_violations),
    )


def _retry_delay_from_duration(duration: Any) -> timedelta | None:
    try:
        result = duration.ToTimedelta()
    except (OverflowError, ValueError, TypeError, AttributeError):
        return None
    return result if isinstance(result, timedelta) else None


def _field_violation_from_proto(
    violation: error_details_pb2.BadRequest.FieldViolation,
) -> FieldViolation:
    return FieldViolation(
        field=violation.field,
        description=_field_violation_description_from_proto(violation),
    )


def _has_field_violation_data(violation: FieldViolation) -> bool:
    return bool(violation.field or violation.description)


def _field_violation_description_from_proto(
    violation: error_details_pb2.BadRequest.FieldViolation,
) -> str:
    description = violation.description
    if isinstance(description, str) and description:
        return description
    if violation.HasField("localized_message"):
        message = violation.localized_message.message
        if isinstance(message, str) and message:
            return message
    reason = violation.reason
    return reason if isinstance(reason, str) else ""


def format_field_violations(
    field_violations: Iterable[FieldViolation],
) -> str:
    """Return a compact user-facing string for field violations."""
    messages: list[str] = []
    for violation in field_violations:
        if violation.field and violation.description:
            messages.append(f"{violation.field}: {violation.description}")
        elif violation.field:
            messages.append(violation.field)
        elif violation.description:
            messages.append(violation.description)
    return "; ".join(messages)


def rpc_error_details(err: grpc.RpcError, parsed: ParsedError | None) -> str:
    """Return gRPC details plus any parsed field violations."""
    details = err.details()
    violation_details = (
        format_field_violations(parsed.field_violations) if parsed is not None else ""
    )
    if details and violation_details:
        return f"{details}; field violations: {violation_details}"
    if details:
        return details
    if violation_details:
        return f"field violations: {violation_details}"
    return str(err)


def is_not_found(
    err: grpc.RpcError,
    parsed: ParsedError | None,
    reason: str,
) -> bool:
    """Return True if ``err`` represents a not-found condition for ``reason``.

    Considers both the transport-level status code and the AIP-193 reason
    (when domain is trusted). Lets ``missing_ok``-style call sites honor
    both signals consistently.
    """
    if err.code() == grpc.StatusCode.NOT_FOUND:
        return True
    if parsed is None or parsed.domain != CWSANDBOX_ERROR_DOMAIN:
        return False
    return parsed.reason == reason
