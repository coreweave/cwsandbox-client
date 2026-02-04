"""gRPC channel management for Aviato SDK.

This module provides utilities for creating and managing gRPC channels
to communicate with the Aviato backend.

Generated SDK imports:
- from coreweave.aviato.v1beta1 import atc_pb2_grpc (ATCServiceStub)
- from coreweave.aviato.v1beta1 import streaming_pb2_grpc (ATCStreamingServiceStub)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

import grpc
import grpc.aio

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_grpc_target(base_url: str) -> tuple[str, bool]:
    """Parse a URL into a gRPC target and security flag.

    Args:
        base_url: HTTP(S) URL to parse (e.g., "https://atc.cwaviato.com")

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
    interceptors: Sequence[grpc.aio.ClientInterceptor] | None = None,
) -> grpc.aio.Channel:
    """Create a gRPC async channel.

    Args:
        target: gRPC target in "host:port" format
        is_secure: If True, use TLS; if False, use insecure channel
        interceptors: Optional sequence of client interceptors

    Returns:
        An async gRPC channel
    """
    interceptor_list = list(interceptors) if interceptors else None

    if is_secure:
        credentials = grpc.ssl_channel_credentials()
        return grpc.aio.secure_channel(
            target,
            credentials,
            interceptors=interceptor_list,
        )
    else:
        return grpc.aio.insecure_channel(
            target,
            interceptors=interceptor_list,
        )
