# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""gRPC transport utilities for CWSandbox SDK.

Provides channel management and shared error translation for gRPC
communication with the CWSandbox backend.

Generated SDK imports:
- from cwsandbox._proto import gateway_pb2_grpc (GatewayServiceStub)
- from cwsandbox._proto import streaming_pb2_grpc (GatewayStreamingServiceStub)
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

import grpc
import grpc.aio

from cwsandbox._error_info import ParsedError, parse_error_info
from cwsandbox.exceptions import CWSandboxAuthenticationError, CWSandboxError


def parse_grpc_target(base_url: str) -> tuple[str, bool]:
    """Parse a URL into a gRPC target and security flag.

    Args:
        base_url: HTTP(S) URL to parse (e.g., "https://api.cwsandbox.com")

    Returns:
        Tuple of (target, is_secure) where:
        - target: "host:port" string for gRPC channel
        - is_secure: True for HTTPS, False for HTTP

    Raises:
        ValueError: If URL scheme is not http/https or if URL has a path
    """
    parsed = urlparse(base_url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https scheme, got: {parsed.scheme!r}")

    if parsed.path and parsed.path != "/":
        raise ValueError(f"gRPC does not support URL paths, got: {parsed.path!r}")

    is_secure = parsed.scheme == "https"

    host = parsed.hostname
    if not host:
        raise ValueError(f"URL must have a hostname: {base_url!r}")

    if parsed.port:
        port = parsed.port
    else:
        port = 443 if is_secure else 80

    target = f"{host}:{port}"
    return target, is_secure


def create_channel(
    target: str,
    is_secure: bool,
) -> grpc.aio.Channel:
    """Create a gRPC async channel.

    Args:
        target: gRPC target in "host:port" format
        is_secure: If True, use TLS; if False, use insecure channel

    Returns:
        An async gRPC channel
    """
    if is_secure:
        credentials = grpc.ssl_channel_credentials()
        return grpc.aio.secure_channel(target, credentials)
    else:
        return grpc.aio.insecure_channel(target)


def translate_grpc_error(
    e: grpc.RpcError,
    *,
    operation: str = "operation",
    fallback_cls: type[CWSandboxError] = CWSandboxError,
    parsed: ParsedError | None = None,
) -> CWSandboxError:
    """Translate a gRPC error into a transport-level CWSandbox exception.

    Handles cross-cutting gRPC status codes (auth, timeout, unavailable).
    Domain modules should check domain-specific codes (e.g. NOT_FOUND)
    before calling this function.

    Any AIP-193 ``ErrorInfo`` / ``RetryInfo`` details present in the gRPC
    trailing metadata are parsed and attached to the returned exception
    (``reason``, ``metadata``, ``retry_delay``) regardless of which branch
    picks the exception class.

    Args:
        e: The gRPC error to translate.
        operation: Description of the operation for error messages.
        fallback_cls: Exception class for non-auth errors. Callers should
            pass their domain base class (e.g. ``SandboxError``,
            ``DiscoveryError``) so that ``except DomainError`` catches
            transport failures too. Defaults to ``CWSandboxError``.
        parsed: Pre-parsed AIP-193 details to avoid re-walking
            ``trailing_metadata()``. When ``None``, this function parses
            internally; callers that already parsed may pass the value.

    Returns:
        An appropriate CWSandbox exception. Caller should
        ``raise result from e`` to preserve the chain.
    """
    if parsed is None:
        parsed = parse_error_info(e)

    code = e.code()
    details = e.details() or str(e)
    reason = parsed.reason if parsed is not None else None
    metadata = parsed.metadata if parsed is not None else None
    retry_delay = parsed.retry_delay if parsed is not None else None

    if code == grpc.StatusCode.UNAUTHENTICATED:
        return CWSandboxAuthenticationError(
            f"Authentication failed: {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if code == grpc.StatusCode.PERMISSION_DENIED:
        return CWSandboxAuthenticationError(
            f"Permission denied: {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return fallback_cls(
            f"{operation} timed out: {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    if code == grpc.StatusCode.UNAVAILABLE:
        return fallback_cls(
            f"Service unavailable: {details}",
            reason=reason,
            metadata=metadata,
            retry_delay=retry_delay,
        )
    return fallback_cls(
        f"{operation} failed: {details}",
        reason=reason,
        metadata=metadata,
        retry_delay=retry_delay,
    )


async def paginate_async(
    rpc_method: Any,
    request: Any,
    items_field: str,
    metadata: tuple[tuple[str, str], ...],
    timeout: float,
    *,
    operation: str = "Request",
) -> list[Any]:
    """Auto-paginate a list RPC.

    Follows ``next_page_token`` until the server returns an empty token or
    the overall deadline is reached.

    Args:
        rpc_method: Bound stub method (e.g. ``stub.ListAvailableRunners``).
        request: The protobuf request message. Its ``page_token`` field is
            mutated in-place between pages.
        items_field: Name of the repeated field on the response that holds
            the result items (e.g. ``"runners"``).
        metadata: gRPC call metadata (auth headers).
        timeout: Total wall-clock seconds allowed for all pages.
        operation: Human-readable label used in timeout/loop error messages.

    Returns:
        Flat list of proto items collected across all pages.

    Raises:
        CWSandboxError: On timeout, pagination loop, or exceeding page limit.
    """
    all_items: list[Any] = []
    deadline = time.monotonic() + timeout
    max_pages = 100
    seen_tokens: set[str] = set()

    for _ in range(max_pages):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CWSandboxError(f"{operation} timed out during pagination")

        response = await rpc_method(request, metadata=metadata, timeout=remaining)
        items = getattr(response, items_field)
        all_items.extend(items)

        next_token = response.next_page_token
        if not next_token:
            break
        if next_token in seen_tokens:
            raise CWSandboxError(f"{operation} pagination loop detected: repeated page token")
        seen_tokens.add(next_token)
        request.page_token = next_token
    else:
        raise CWSandboxError(f"{operation} pagination exceeded {max_pages} pages")

    return all_items
