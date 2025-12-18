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
from typing import Literal

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
