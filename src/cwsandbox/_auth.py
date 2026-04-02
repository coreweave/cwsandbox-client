# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Authentication resolution for CWSandbox client.

Auth is resolved from a single active mode.

By default, the built-in CoreWeave auth mode is active and resolves:
1. `CWSANDBOX_API_KEY` bearer auth, if present
2. otherwise no auth

Provider integrations can replace that active mode for the current process.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from cwsandbox.exceptions import CWSandboxAuthenticationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthHeaders:
    """Resolved authentication headers and strategy used."""

    headers: dict[str, str]
    strategy: str

    def __bool__(self) -> bool:
        """Return True if any auth headers are present."""
        return bool(self.headers)


@dataclass(frozen=True)
class _AuthMode:
    """Configuration for an authentication mode."""

    name: str
    get_auth: Callable[[], AuthHeaders]


def _resolve_builtin_auth() -> AuthHeaders:
    """Resolve the built-in CoreWeave auth mode."""
    api_key = os.environ.get("CWSANDBOX_API_KEY")
    if api_key:
        return AuthHeaders(
            headers={"Authorization": f"Bearer {api_key}"},
            strategy="api_key",
        )

    return AuthHeaders(headers={}, strategy="none")


_BUILTIN_AUTH_MODE = _AuthMode(name="builtin", get_auth=_resolve_builtin_auth)
_ACTIVE_AUTH_MODE = _BUILTIN_AUTH_MODE


def set_auth_mode(
    name: str,
    get_auth: Callable[[], AuthHeaders],
) -> None:
    """Set the active auth mode for this process.

    The active mode replaces the built-in auth mode until it is reset.
    Configuration is process-global and last-writer-wins.
    The callback must return AuthHeaders or raise CWSandboxAuthenticationError.
    """
    global _ACTIVE_AUTH_MODE
    _ACTIVE_AUTH_MODE = _AuthMode(name=name, get_auth=get_auth)


def _reset_auth_mode_for_testing() -> None:
    """Reset the active auth mode to the built-in default.

    This exists for test isolation and should not be used by integrations.
    """
    global _ACTIVE_AUTH_MODE
    _ACTIVE_AUTH_MODE = _BUILTIN_AUTH_MODE


def resolve_auth() -> AuthHeaders:
    """Resolve authentication headers from available credentials.

    Uses the current active auth mode and returns the resolved headers.

    Returns:
        AuthHeaders with resolved headers and strategy name
    """
    mode = _ACTIVE_AUTH_MODE
    auth = mode.get_auth()
    # In case None is still returned at runtime, raise auth error instead of
    # AttributeError from logging auth.strategy.
    if auth is None:
        raise CWSandboxAuthenticationError(
            f"Configured auth mode {mode.name} returned no credentials"
        )
    logger.debug("Using auth mode %s with strategy %s", mode.name, auth.strategy)
    return auth


def resolve_auth_metadata() -> tuple[tuple[str, str], ...]:
    """Resolve authentication credentials and return as gRPC metadata tuples.

    Convenience wrapper around resolve_auth() that returns metadata in the
    format expected by gRPC calls (lowercase key-value tuples).

    Returns:
        Tuple of (key, value) pairs suitable for gRPC metadata parameter
    """
    auth = resolve_auth()
    return tuple((k.lower(), v) for k, v in auth.headers.items())
