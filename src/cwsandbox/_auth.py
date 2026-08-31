# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Authentication resolution for the CWSandbox client.

Authentication is selected per operation or per client instance. Omitting the
selection preserves the built-in CoreWeave API-key behavior. W&B credential
resolution is explicit and requires the optional ``wandb`` integration.

The legacy process-global ``set_auth_mode`` hook remains for compatibility,
but explicit per-instance authentication always takes precedence.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable
from urllib.parse import urlsplit

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


@runtime_checkable
class AuthProvider(Protocol):
    """Provider that resolves authentication for a Sandbox API endpoint."""

    def resolve_auth(self, *, base_url: str) -> AuthHeaders:
        """Resolve headers for ``base_url`` or raise an authentication error."""


class AuthStrategy(StrEnum):
    """Built-in authentication strategies."""

    COREWEAVE_API_KEY = "coreweave_api_key"
    WANDB = "wandb"


AuthConfig: TypeAlias = AuthStrategy | AuthProvider | AuthHeaders


@dataclass(frozen=True)
class _AuthMode:
    """Legacy process-global authentication mode."""

    name: str
    get_auth: Callable[[], AuthHeaders]


def _resolve_coreweave_auth(*, required: bool) -> AuthHeaders:
    api_key = os.environ.get("CWSANDBOX_API_KEY")
    if api_key:
        return AuthHeaders(
            headers={"Authorization": f"Bearer {api_key}"},
            strategy="api_key",
        )
    if required:
        raise CWSandboxAuthenticationError("CoreWeave authentication requires CWSANDBOX_API_KEY.")
    return AuthHeaders(headers={}, strategy="none")


def _wandb_missing_error() -> CWSandboxAuthenticationError:
    return CWSandboxAuthenticationError(
        "W&B authentication requires the optional wandb dependency. "
        'Install it with: pip install "cwsandbox[wandb]"'
    )


def _resolve_wandb_auth() -> AuthHeaders:
    """Resolve W&B API-key auth without importing W&B until it is needed."""
    try:
        import wandb
        from wandb.sdk import wandb_setup
        from wandb.sdk.lib import wbauth
        from wandb.sdk.lib.wbauth import saas
    except ModuleNotFoundError as exc:
        if exc.name == "wandb" or (exc.name and exc.name.startswith("wandb.")):
            raise _wandb_missing_error() from None
        raise

    settings = wandb_setup.singleton().settings
    host = wbauth.HostUrl(settings.base_url, app_url=settings.app_url)

    try:
        auth = wbauth.session_credentials(host=host)
        if auth is None:
            auth = wbauth.authenticate_session(
                host=host,
                source="cwsandbox",
                no_offline=True,
            )
    except Exception as exc:
        raise CWSandboxAuthenticationError(f"Failed to resolve W&B credentials: {exc}") from exc

    if auth is None:
        raise CWSandboxAuthenticationError(
            "No W&B credentials found. Set WANDB_API_KEY or run `wandb login`."
        )

    if not isinstance(auth, wbauth.AuthApiKey):
        raise CWSandboxAuthenticationError(
            "CWSandbox currently supports only W&B user API-key authentication."
        )

    headers = {
        "x-wandb-api-key": auth.api_key,
        "x-wandb-sdk-version": wandb.__version__,
    }
    parsed_host = urlsplit(auth.host.url)
    hostname = parsed_host.hostname
    if hostname is None:
        raise CWSandboxAuthenticationError("W&B base URL must include a hostname.")
    hostname = hostname.lower()
    try:
        port = parsed_host.port
    except ValueError as exc:
        raise CWSandboxAuthenticationError(f"W&B base URL has an invalid port: {exc}") from exc

    scheme = parsed_host.scheme.lower()
    if not saas.is_wandb_domain(f"{scheme}://{hostname}"):
        default_port = {"http": 80, "https": 443}.get(scheme)
        headers["x-wandb-host"] = (
            f"{hostname}:{port}" if port is not None and port != default_port else hostname
        )
    if settings.entity:
        headers["x-entity-id"] = settings.entity
    if settings.project:
        headers["x-project-name"] = settings.project

    return AuthHeaders(headers=headers, strategy="wandb_api_key")


_BUILTIN_AUTH_MODE = _AuthMode(
    name="builtin",
    get_auth=lambda: _resolve_coreweave_auth(required=False),
)
_ACTIVE_AUTH_MODE = _BUILTIN_AUTH_MODE


def set_auth_mode(
    name: str,
    get_auth: Callable[[], AuthHeaders],
) -> None:
    """Set the legacy process-global auth mode.

    New integrations should pass ``auth=`` to ``Sandbox``, ``Session``, or
    class-level operations instead. This hook remains for compatibility with
    integrations that install an auth mode at import time.
    """
    global _ACTIVE_AUTH_MODE
    _ACTIVE_AUTH_MODE = _AuthMode(name=name, get_auth=get_auth)


def _reset_auth_mode_for_testing() -> None:
    """Reset the legacy active auth mode to the built-in default."""
    global _ACTIVE_AUTH_MODE
    _ACTIVE_AUTH_MODE = _BUILTIN_AUTH_MODE


def _validate_resolved_auth(auth: AuthHeaders | None, *, provider_name: str) -> AuthHeaders:
    if auth is None:
        raise CWSandboxAuthenticationError(
            f"Configured auth provider {provider_name} returned no credentials"
        )
    if not isinstance(auth, AuthHeaders):
        raise CWSandboxAuthenticationError(
            f"Configured auth provider {provider_name} returned "
            f"{type(auth).__name__}, expected AuthHeaders"
        )
    return auth


def resolve_auth(
    auth: AuthConfig | None = None,
    *,
    base_url: str = "",
) -> AuthHeaders:
    """Resolve one authentication strategy.

    Args:
        auth: An ``AuthStrategy``, resolved ``AuthHeaders``, or an
            ``AuthProvider``. ``None`` preserves a legacy global override when
            installed and otherwise uses ``AuthStrategy.COREWEAVE_API_KEY``.
        base_url: Sandbox API endpoint passed to custom providers.
    """
    if auth is None:
        mode = _ACTIVE_AUTH_MODE
        resolved = mode.get_auth()
        if resolved is None:
            raise CWSandboxAuthenticationError(
                f"Configured auth mode {mode.name} returned no credentials"
            )
        resolved = _validate_resolved_auth(resolved, provider_name=mode.name)
        logger.debug("Using auth mode %s with strategy %s", mode.name, resolved.strategy)
        return resolved

    if isinstance(auth, AuthHeaders):
        return auth

    if isinstance(auth, AuthStrategy):
        if auth is AuthStrategy.COREWEAVE_API_KEY:
            return _resolve_coreweave_auth(required=True)
        if auth is AuthStrategy.WANDB:
            return _resolve_wandb_auth()

    if isinstance(auth, AuthProvider):
        resolved = auth.resolve_auth(base_url=base_url)
        return _validate_resolved_auth(resolved, provider_name=type(auth).__name__)

    raise TypeError(
        "auth must be an AuthStrategy, AuthHeaders, "
        f"an AuthProvider, or None; got {type(auth).__name__}"
    )


def resolve_auth_metadata(
    auth: AuthConfig | None = None,
    *,
    base_url: str = "",
) -> tuple[tuple[str, str], ...]:
    """Resolve credentials as lowercase gRPC metadata tuples."""
    resolved = resolve_auth(auth, base_url=base_url)
    return tuple((key.lower(), value) for key, value in resolved.headers.items())
