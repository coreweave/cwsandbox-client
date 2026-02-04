"""Authentication resolution for Aviato client.

Supports two auth strategies:
1. Aviato: Uses AVIATO_API_KEY env var -> Authorization: Bearer header
2. W&B: Uses WANDB_* env vars or ~/.netrc -> x-api-key, x-entity-id, x-project-name headers

Resolution order: Aviato credentials take priority if present.
"""

from __future__ import annotations

import logging
import netrc
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import grpc
import grpc.aio

from aviato._defaults import DEFAULT_PROJECT_NAME, WANDB_NETRC_HOST
from aviato.exceptions import WandbAuthError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthHeaders:
    """Resolved authentication headers and strategy used."""

    headers: dict[str, str]
    strategy: Literal["aviato", "wandb", "none"]

    def __bool__(self) -> bool:
        """Return True if any auth headers are present."""
        return bool(self.headers)


class _AuthMode:
    """Configuration for an authentication mode."""

    def __init__(self, try_auth: Callable[[], AuthHeaders | None]) -> None:
        self.try_auth = try_auth


def resolve_auth() -> AuthHeaders:
    """Resolve authentication headers from available credentials.

    Tries each auth mode in priority order (defined in _AUTH_MODES) and
    returns the first one that succeeds.

    Resolution order:
    1. AVIATO_API_KEY env var (Aviato auth)
    2. WANDB_API_KEY + WANDB_ENTITY_NAME env vars (W&B auth)
    3. ~/.netrc api.wandb.ai + WANDB_ENTITY_NAME env var (W&B auth)
    4. No auth (empty headers)

    Returns:
        AuthHeaders with resolved headers and strategy name
    """
    for mode in _AUTH_MODES:
        auth = mode.try_auth()
        if auth is not None:
            logger.debug("Using %s authentication", auth.strategy)
            return auth

    logger.debug("No authentication credentials found")
    return AuthHeaders(headers={}, strategy="none")


def _try_aviato_auth() -> AuthHeaders | None:
    """Try to resolve Aviato authentication from env var.

    Returns:
        AuthHeaders if AVIATO_API_KEY is set, None otherwise
    """
    api_key = os.environ.get("AVIATO_API_KEY")
    if not api_key:
        return None

    return AuthHeaders(
        headers={"Authorization": f"Bearer {api_key}"},
        strategy="aviato",
    )


def _try_wandb_auth() -> AuthHeaders | None:
    """Try to resolve W&B authentication from env vars or netrc.

    API key can come from WANDB_API_KEY env var or ~/.netrc.
    Entity must be set via WANDB_ENTITY_NAME env var.

    Returns:
        AuthHeaders if valid W&B credentials found, None otherwise

    Raises:
        WandbAuthError: If API key is found but entity is missing
    """
    # Check for API key first (env var, then netrc)
    api_key = os.environ.get("WANDB_API_KEY") or _read_api_key_from_netrc()

    if not api_key:
        # No W&B credentials configured
        return None

    # API key found - entity is now required
    entity = os.environ.get("WANDB_ENTITY_NAME")
    if not entity:
        raise WandbAuthError(
            "WANDB_API_KEY or ~/.netrc credentials found, but WANDB_ENTITY_NAME is not set. "
            "Set WANDB_ENTITY_NAME to your W&B entity/team name."
        )

    project = os.environ.get("WANDB_PROJECT_NAME", DEFAULT_PROJECT_NAME)

    return AuthHeaders(
        headers={
            "x-api-key": api_key,
            "x-entity-id": entity,
            "x-project-name": project,
        },
        strategy="wandb",
    )


def _read_api_key_from_netrc() -> str | None:
    """Read W&B API key from ~/.netrc file.

    Looks for machine 'api.wandb.ai' and extracts the password field.

    Returns:
        API key string if found, None otherwise
    """
    netrc_path = Path.home() / ".netrc"

    try:
        nrc = netrc.netrc(str(netrc_path))
    except FileNotFoundError:
        logger.debug("No .netrc file found at %s", netrc_path)
        return None
    except netrc.NetrcParseError as e:
        logger.warning("Failed to parse .netrc: %s", e)
        return None

    auth = nrc.authenticators(WANDB_NETRC_HOST)
    if auth is None:
        logger.debug("No entry for %s in .netrc", WANDB_NETRC_HOST)
        return None

    # auth is (login, account, password)
    _login, _account, password = auth
    return password


# Auth modes in priority order - first successful returns
_AUTH_MODES = [
    _AuthMode(try_auth=_try_aviato_auth),
    _AuthMode(try_auth=_try_wandb_auth),
]


class AuthInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor,  # type: ignore[type-arg]
    grpc.aio.StreamStreamClientInterceptor,  # type: ignore[type-arg]
):
    """gRPC interceptor that adds auth headers to every RPC call.

    Implements both UnaryUnaryClientInterceptor and StreamStreamClientInterceptor
    to handle both unary and bidirectional streaming calls.

    Note: gRPC requires metadata keys to be lowercase. This interceptor
    normalizes all header keys to lowercase on initialization.
    """

    def __init__(self, headers: dict[str, str]) -> None:
        """Initialize the interceptor with auth headers.

        Args:
            headers: Dict of header name to value. Keys are normalized to lowercase.
        """
        # gRPC requires lowercase metadata keys
        self._metadata: tuple[tuple[str, str], ...] = tuple(
            (k.lower(), v) for k, v in headers.items()
        )

    def _add_metadata(
        self, client_call_details: grpc.aio.ClientCallDetails
    ) -> grpc.aio.ClientCallDetails:
        """Create new ClientCallDetails with auth metadata added.

        Args:
            client_call_details: Original call details

        Returns:
            New ClientCallDetails with auth metadata merged
        """
        existing = client_call_details.metadata or ()
        merged = tuple(existing) + self._metadata
        # ClientCallDetails is a NamedTuple at runtime with _replace method
        return client_call_details._replace(metadata=merged)  # type: ignore[attr-defined, no-any-return]

    async def intercept_unary_unary(
        self,
        continuation: Callable[
            [grpc.aio.ClientCallDetails, Any],
            grpc.aio.UnaryUnaryCall[Any, Any],
        ],
        client_call_details: grpc.aio.ClientCallDetails,
        request: Any,
    ) -> Any:
        """Intercept unary-unary calls to add auth metadata."""
        new_details = self._add_metadata(client_call_details)
        return await continuation(new_details, request)

    async def intercept_stream_stream(
        self,
        continuation: Callable[
            [grpc.aio.ClientCallDetails, Any],
            grpc.aio.StreamStreamCall[Any, Any],
        ],
        client_call_details: grpc.aio.ClientCallDetails,
        request_iterator: Any,
    ) -> grpc.aio.StreamStreamCall[Any, Any]:
        """Intercept stream-stream calls to add auth metadata."""
        new_details = self._add_metadata(client_call_details)
        # StreamStreamCall is not awaitable - return it directly
        return continuation(new_details, request_iterator)


def create_auth_interceptors() -> list[AuthInterceptor]:
    """Create gRPC interceptors for auth headers based on resolved credentials.

    Resolves authentication credentials using the standard auth resolution
    order (AVIATO_API_KEY, WANDB_* env vars, ~/.netrc) and creates an
    interceptor to add those headers to each request.

    Returns:
        List containing a single AuthInterceptor configured with resolved auth
        headers. Returns list with interceptor even if no auth headers are
        resolved (empty headers dict).
    """
    auth = resolve_auth()
    return [AuthInterceptor(auth.headers)]
