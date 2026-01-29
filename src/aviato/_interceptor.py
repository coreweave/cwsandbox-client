"""ConnectRPC interceptors for Aviato client.

Provides interceptors for adding authentication headers to connectrpc requests.
This adapts to connect-python 0.8.1+ API which uses interceptors instead of
accepting httpx clients directly with headers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from connectrpc.request import RequestContext


class AuthHeaderInterceptor:
    """Interceptor that adds auth headers to every connectrpc request.

    Implements the MetadataInterceptor protocol from connectrpc, which is
    called for all RPC method types (unary, client streaming, server streaming,
    and bidirectional streaming).

    Example:
        ```python
        from aviato._interceptor import create_auth_interceptors

        interceptors = create_auth_interceptors()
        client = ATCServiceClient(
            address=base_url,
            interceptors=interceptors,
        )
        ```
    """

    def __init__(self, headers: dict[str, str]) -> None:
        """Initialize with headers to add to each request.

        Args:
            headers: Dictionary of header key-value pairs to add.
        """
        self._headers = headers

    async def on_start(self, ctx: RequestContext) -> None:  # type: ignore[type-arg]
        """Add auth headers when an RPC starts.

        This method is called by connectrpc's MetadataInterceptorInvoker
        before each RPC call, regardless of method type.

        Args:
            ctx: The request context containing headers to modify.
        """
        request_headers = ctx.request_headers()
        for key, value in self._headers.items():
            request_headers[key] = value

    async def on_end(self, token: None, ctx: RequestContext) -> None:  # type: ignore[type-arg]
        """Called when the RPC ends. No-op for auth headers."""
        pass


def create_auth_interceptors() -> list[AuthHeaderInterceptor]:
    """Create interceptors for auth headers based on resolved credentials.

    Resolves authentication credentials using the standard auth resolution
    order (AVIATO_API_KEY, WANDB_* env vars, ~/.netrc) and creates an
    interceptor to add those headers to each request.

    Returns:
        List containing a single AuthHeaderInterceptor configured with
        resolved auth headers. Returns list with interceptor even if no
        auth headers are resolved (empty headers dict).

    Example:
        ```python
        interceptors = create_auth_interceptors()
        client = ATCServiceClient(
            address="https://atc.cwaviato.com",
            interceptors=interceptors,
        )
        ```
    """
    from aviato._auth import resolve_auth

    auth = resolve_auth()
    return [AuthHeaderInterceptor(auth.headers)]
