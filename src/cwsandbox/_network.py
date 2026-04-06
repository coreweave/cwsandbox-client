# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""gRPC transport utilities for CWSandbox SDK.

Provides channel management and shared error translation for gRPC
communication with the CWSandbox backend.

Generated SDK imports:
- from cwsandbox._proto import atc_pb2_grpc (ATCServiceStub)
- from cwsandbox._proto import streaming_pb2_grpc (ATCStreamingServiceStub)
"""

from __future__ import annotations

from urllib.parse import urlparse

import grpc
import grpc.aio

from cwsandbox.exceptions import CWSandboxAuthenticationError, CWSandboxError


def parse_grpc_target(base_url: str) -> tuple[str, bool]:
    """Parse a URL into a gRPC target and security flag.

    Args:
        base_url: HTTP(S) URL to parse (e.g., "https://atc.cw-sandbox.com")

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
) -> CWSandboxError:
    """Translate a gRPC error into a transport-level CWSandbox exception.

    Handles cross-cutting gRPC status codes (auth, timeout, unavailable).
    Domain modules should check domain-specific codes (e.g. NOT_FOUND)
    before calling this function.

    Args:
        e: The gRPC error to translate.
        operation: Description of the operation for error messages.
        fallback_cls: Exception class for non-auth errors. Callers should
            pass their domain base class (e.g. ``SandboxError``,
            ``DiscoveryError``) so that ``except DomainError`` catches
            transport failures too. Defaults to ``CWSandboxError``.

    Returns:
        An appropriate CWSandbox exception. Caller should
        ``raise result from e`` to preserve the chain.
    """
    code = e.code()
    details = e.details() or str(e)

    if code == grpc.StatusCode.UNAUTHENTICATED:
        return CWSandboxAuthenticationError(f"Authentication failed: {details}")
    if code == grpc.StatusCode.PERMISSION_DENIED:
        return CWSandboxAuthenticationError(f"Permission denied: {details}")
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return fallback_cls(f"{operation} timed out: {details}")
    if code == grpc.StatusCode.UNAVAILABLE:
        return fallback_cls(f"Service unavailable: {details}")
    return fallback_cls(f"{operation} failed: {details}")
