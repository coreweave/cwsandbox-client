# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""HTTPS share-token product endpoints.

Demonstrates:
- Creating a PUBLIC HTTPS endpoint with auth=SHARE_TOKEN
- Reading the create-only EndpointShareToken
- Using EndpointShareToken.as_headers() to attach X-Sandbox-Share-Token
- Redacted string/repr behavior for accidental-log protection
- Logging the URL + "share token: received" only — never the raw token
- from_id omits the token; the live handle keeps it

Usage:
    python examples/https_share_token.py
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

from cwsandbox import (
    Endpoint,
    EndpointAuth,
    EndpointKind,
    EndpointShareToken,
    Sandbox,
    SandboxDefaults,
    Service,
    ServiceVisibility,
)
from cwsandbox.exceptions import SandboxError


def _wait_for_url(sandbox: Sandbox, *, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sandbox.get_status()
        if sandbox.service_urls:
            return sandbox.service_urls[0][2]
        time.sleep(0.5)
    raise SystemExit("service_urls stayed empty after wait")


def _header_get(url: str, token: EndpointShareToken) -> int:
    request = urllib.request.Request(url, headers=token.as_headers())
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status
        except urllib.error.HTTPError:
            pass
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise SystemExit("Authenticated GET did not return 200")


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.11",
        tags=("example", "example-https-share-token"),
    )
    service = Service(
        port=8000,
        name="http",
        visibility=ServiceVisibility.PUBLIC,
        endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.SHARE_TOKEN),
    )
    try:
        with Sandbox.run(
            "python",
            "-m",
            "http.server",
            "8000",
            defaults=defaults,
            services=[service],
        ) as sandbox:
            token = sandbox.endpoint_share_token
            if token is None:
                sandbox.start().result()
                token = sandbox.endpoint_share_token
            if token is None:
                raise SystemExit(
                    "Share-token recovery exhausted on this create handle; "
                    "Get/from_id cannot recover it."
                )

            print(f"Sandbox: {sandbox.sandbox_id}")
            sandbox.wait()
            url = _wait_for_url(sandbox)
            print(f"URL: {url}; share token: received")
            print("from_id / get_status omit the token; this handle keeps the create-time value.")

            ok_status = _header_get(url, token)
            print(f"Authenticated GET: {ok_status}")
            if ok_status != 200:
                raise SystemExit("Authenticated GET expected 200")

            try:
                urllib.request.urlopen(url, timeout=10)
                denied_status = 200
            except urllib.error.HTTPError as exc:
                denied_status = exc.code
            print(f"Unauthenticated GET: {denied_status}")
            if denied_status != 401:
                raise SystemExit("Unauthenticated GET expected 401")

            reattached = Sandbox.from_id(sandbox.sandbox_id).result()
            if reattached.endpoint_share_token is not None:
                raise SystemExit("from_id must omit the create-only share token")
            if sandbox.endpoint_share_token is None:
                raise SystemExit("live handle must keep the create-time share token")
    except SandboxError as exc:
        if exc.reason == "CWSANDBOX_HTTPS_SHARE_TOKEN_NOT_SUPPORTED":
            raise SystemExit(f"No runner advertises share-token HTTPS. {exc}") from exc
        raise


if __name__ == "__main__":
    main()
