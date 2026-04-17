# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""AIP-193 ErrorInfo parser for gRPC trailing metadata.

Parses ``google.rpc.Status`` structured error details out of the
``grpc-status-details-bin`` trailing metadata entry produced by
servers that follow Google's AIP-193 error model, and returns the
subset of fields the SDK cares about (``ErrorInfo.reason``,
``ErrorInfo.domain``, ``ErrorInfo.metadata``, and
``RetryInfo.retry_delay`` when present).

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

FILE_ERROR_REASONS: frozenset[str] = frozenset(
    {
        CWSANDBOX_FILE_NOT_FOUND,
        CWSANDBOX_FILE_IS_DIRECTORY,
        CWSANDBOX_FILE_IO_FAILED,
        CWSANDBOX_FILE_PERMISSION_DENIED,
    }
)

# Not-found reasons per context
CWSANDBOX_SANDBOX_NOT_FOUND = "CWSANDBOX_SANDBOX_NOT_FOUND"
CWSANDBOX_RUNNER_NOT_FOUND = "CWSANDBOX_RUNNER_NOT_FOUND"
CWSANDBOX_PROFILE_NOT_FOUND = "CWSANDBOX_PROFILE_NOT_FOUND"

# Timeout reasons
CWSANDBOX_COMMAND_TIMEOUT = "CWSANDBOX_COMMAND_TIMEOUT"

# Unavailable reasons
CWSANDBOX_RUNNER_UNAVAILABLE = "CWSANDBOX_RUNNER_UNAVAILABLE"
CWSANDBOX_BACKEND_UNAVAILABLE = "CWSANDBOX_BACKEND_UNAVAILABLE"

UNAVAILABLE_REASONS: frozenset[str] = frozenset(
    {
        CWSANDBOX_RUNNER_UNAVAILABLE,
        CWSANDBOX_BACKEND_UNAVAILABLE,
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
    """

    reason: str | None = None
    domain: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    retry_delay: timedelta | None = None


def parse_error_info(err: grpc.RpcError) -> ParsedError | None:
    """Parse ``google.rpc.Status`` details from gRPC trailing metadata.

    Walks ``err.trailing_metadata()`` for every ``grpc-status-details-bin``
    entry (the key may repeat in gRPC metadata). The first entry that decodes
    to a ``google.rpc.Status`` containing at least one ``ErrorInfo`` or
    ``RetryInfo`` detail wins; a malformed leading entry does not suppress a
    later valid one.

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
        if reason is not None and retry_delay is not None:
            break

    if reason is None and retry_delay is None:
        return None
    return ParsedError(
        reason=reason,
        domain=domain,
        metadata=metadata,
        retry_delay=retry_delay,
    )


def _retry_delay_from_duration(duration: Any) -> timedelta | None:
    try:
        result = duration.ToTimedelta()
    except (OverflowError, ValueError, TypeError, AttributeError):
        return None
    return result if isinstance(result, timedelta) else None


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
