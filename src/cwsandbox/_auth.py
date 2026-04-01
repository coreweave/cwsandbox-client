# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Authentication resolution for CWSandbox client.

Auth is resolved in the following order:
1. API key: Uses CWSANDBOX_API_KEY env var -> Authorization: Bearer header
2. Registered auth mode in current process
3. No auth
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthHeaders:
    """Resolved authentication headers and strategy used."""

    headers: dict[str, str]
    strategy: str

    def __bool__(self) -> bool:
        """Return True if any auth headers are present."""
        return bool(self.headers)


class _AuthMode:
    """Configuration for an authentication mode."""

    def __init__(self, name: str, try_auth: Callable[[], AuthHeaders | None]) -> None:
        self.name = name
        self.try_auth = try_auth


_REGISTERED_AUTH_MODES: list[_AuthMode] = []
_AUTH_MODES_LOCK = threading.Lock()


def register_auth_mode(
    name: str,
    try_auth: Callable[[], AuthHeaders | None],
) -> None:
    """Register an additional auth mode.

    Registered auth modes are consulted after explicit CoreWeave API-key auth.
    Registration is idempotent by name.
    """
    with _AUTH_MODES_LOCK:
        if any(auth_mode.name == name for auth_mode in _REGISTERED_AUTH_MODES):
            return
        _REGISTERED_AUTH_MODES.append(_AuthMode(name=name, try_auth=try_auth))


def unregister_auth_mode(name: str) -> None:
    """Unregister a previously registered auth mode by name."""
    with _AUTH_MODES_LOCK:
        _REGISTERED_AUTH_MODES[:] = [
            auth_mode for auth_mode in _REGISTERED_AUTH_MODES if auth_mode.name != name
        ]


def _iter_auth_modes() -> list[_AuthMode]:
    if not _REGISTERED_AUTH_MODES:
        return _BUILTIN_AUTH_MODES

    with _AUTH_MODES_LOCK:
        registered_auth_modes = list(_REGISTERED_AUTH_MODES)

    return [*_BUILTIN_AUTH_MODES, *registered_auth_modes]


def resolve_auth() -> AuthHeaders:
    """Resolve authentication headers from available credentials.

    Tries auth modes in priority order and returns the first one that succeeds.

    Resolution order:
    1. CWSANDBOX_API_KEY env var (API key auth)
    2. Registered auth modes
    3. No auth (empty headers)

    Returns:
        AuthHeaders with resolved headers and strategy name
    """
    for mode in _iter_auth_modes():
        auth = mode.try_auth()
        if auth is not None:
            logger.debug("Using %s authentication", auth.strategy)
            return auth

    logger.debug("No authentication credentials found")
    return AuthHeaders(headers={}, strategy="none")


def resolve_auth_metadata() -> tuple[tuple[str, str], ...]:
    """Resolve authentication credentials and return as gRPC metadata tuples.

    Convenience wrapper around resolve_auth() that returns metadata in the
    format expected by gRPC calls (lowercase key-value tuples).

    Returns:
        Tuple of (key, value) pairs suitable for gRPC metadata parameter
    """
    auth = resolve_auth()
    return tuple((k.lower(), v) for k, v in auth.headers.items())


def _try_api_key_auth() -> AuthHeaders | None:
    """Try to resolve API key authentication from env var.

    Returns:
        AuthHeaders if CWSANDBOX_API_KEY is set, None otherwise
    """
    api_key = os.environ.get("CWSANDBOX_API_KEY")
    if not api_key:
        return None

    return AuthHeaders(
        headers={"Authorization": f"Bearer {api_key}"},
        strategy="api_key",
    )


_BUILTIN_AUTH_MODES = [
    _AuthMode(name="api_key", try_auth=_try_api_key_auth),
]
