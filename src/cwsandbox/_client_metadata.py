# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Request-scoped client metadata sent as gRPC metadata."""

from __future__ import annotations

from importlib import metadata as importlib_metadata

HEADER_CWSANDBOX_CLIENT_VERSION = "x-cwsandbox-client-version"
HEADER_SANDBOX_INTEGRATION = "x-sandbox-integration"

_INTEGRATION_METADATA_VALUE = ""


def _cwsandbox_version() -> str:
    try:
        return importlib_metadata.version("cwsandbox")
    except importlib_metadata.PackageNotFoundError:
        from cwsandbox import __version__

        return __version__


def set_integration_metadata(integration: str) -> None:
    """Set the integration name attached to future sandbox API requests.

    The value is process-global and optional. Passing an empty string clears the
    integration metadata.
    """
    global _INTEGRATION_METADATA_VALUE
    _INTEGRATION_METADATA_VALUE = integration


def _reset_client_metadata_for_testing() -> None:
    """Reset process-global client metadata for unit-test isolation."""
    set_integration_metadata("")


def client_metadata_headers() -> tuple[tuple[str, str], ...]:
    """Return cwsandbox-managed client metadata headers."""
    headers = [(HEADER_CWSANDBOX_CLIENT_VERSION, _cwsandbox_version())]
    if _INTEGRATION_METADATA_VALUE:
        headers.append((HEADER_SANDBOX_INTEGRATION, _INTEGRATION_METADATA_VALUE))
    return tuple(headers)
