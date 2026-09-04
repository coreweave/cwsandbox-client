# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""TLS passthrough product endpoints.

Demonstrates:
- Creating a PUBLIC TLS_PASSTHROUGH endpoint
- Reading host:port from Sandbox.service_addresses[].address
- Sandbox.from_id and Sandbox.list echoing the same address
- GETting the edge with SNI equal to that host (workload owns the cert)

Usage:
    python examples/tls_passthrough.py
"""

from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request

from cwsandbox import (
    Endpoint,
    EndpointKind,
    Sandbox,
    SandboxDefaults,
    Service,
    ServiceVisibility,
    TlsPassthroughEndpointStatus,
)
from cwsandbox.exceptions import SandboxError

_TLS_BODY = "product-tls-ok"
_TLS_SCRIPT = """
from http.server import BaseHTTPRequestHandler, HTTPServer
import ssl, subprocess
subprocess.check_call([
    "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
    "-keyout", "/tmp/tls.key", "-out", "/tmp/tls.crt",
    "-days", "1", "-subj", "/CN=tls-probe",
])
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"product-tls-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = HTTPServer(("0.0.0.0", 8443), H)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain("/tmp/tls.crt", "/tmp/tls.key")
server.socket = ctx.wrap_socket(server.socket, server_side=True)
server.serve_forever()
"""


def _tls_get(address: str) -> str:
    host, port = address.rsplit(":", 1)
    url = f"https://{host}/" if port == "443" else f"https://{host}:{port}/"
    ctx = ssl._create_unverified_context()
    deadline = time.monotonic() + 60.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=10) as response:
                return response.read().decode()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2)
    raise SystemExit(f"TLS GET {url} failed: {last_error}")


def _require_same_addresses(
    label: str,
    got: tuple[TlsPassthroughEndpointStatus, ...],
    want: tuple[TlsPassthroughEndpointStatus, ...],
) -> None:
    if not got:
        raise SystemExit(f"{label} did not fill service_addresses")
    if got != want:
        raise SystemExit(f"{label} service_addresses {got} != Create {want}")


def main() -> None:
    defaults = SandboxDefaults(
        container_image="python:3.9",
        tags=("example", "tls-passthrough"),
    )
    service = Service(
        port=8443,
        name="tls",
        visibility=ServiceVisibility.PUBLIC,
        endpoint=Endpoint(kind=EndpointKind.TLS_PASSTHROUGH),
    )
    try:
        with Sandbox.run(
            "python3",
            "-c",
            _TLS_SCRIPT,
            defaults=defaults,
            services=[service],
        ) as sandbox:
            print(f"Sandbox: {sandbox.sandbox_id}")
            print(f"Create addresses: {sandbox.service_addresses}")
            if not sandbox.service_addresses:
                raise SystemExit("Create did not fill service_addresses")
            created = sandbox.service_addresses
            address = created[0].address
            sandbox.wait()

            fetched = Sandbox.from_id(sandbox.sandbox_id).result()
            print(f"from_id addresses: {fetched.service_addresses}")
            _require_same_addresses("from_id", fetched.service_addresses, created)

            listed = Sandbox.list(tags=["tls-passthrough"]).result()
            match = next(
                (item for item in listed if item.sandbox_id == sandbox.sandbox_id),
                None,
            )
            if match is None:
                raise SystemExit(f"List did not return {sandbox.sandbox_id}")
            print(f"list addresses: {match.service_addresses}")
            _require_same_addresses("list", match.service_addresses, created)

            body = _tls_get(address)
            print(f"TLS GET body: {body}")
            if body != _TLS_BODY:
                raise SystemExit(f"expected {_TLS_BODY!r}, got {body!r}")
    except SandboxError as exc:
        if exc.reason == "CWSANDBOX_TLS_PASSTHROUGH_ENDPOINTS_NOT_SUPPORTED":
            raise SystemExit(f"No runner advertises TLS passthrough. {exc}") from exc
        raise


if __name__ == "__main__":
    main()
