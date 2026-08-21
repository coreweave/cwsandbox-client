# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for DNS-name HTTPS egress.

These tests require a running CWSandbox backend whose selected runner
advertises DNS-name egress and whose policy admits the declared names.
When the fleet cannot place a name grant, they skip rather than fail.

Set CWSANDBOX_BASE_URL and CWSANDBOX_API_KEY before running.
"""

from __future__ import annotations

import pytest

from cwsandbox import EgressRule, NetworkOptions, Sandbox, SandboxDefaults
from cwsandbox._error_info import (
    CWSANDBOX_NO_SUITABLE_RUNNER,
    CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
)
from cwsandbox.exceptions import SandboxError, SandboxValidationError

_SKIP_PLACEMENT_REASONS = frozenset(
    {
        CWSANDBOX_NO_SUITABLE_RUNNER,
        CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
    }
)

GRANTED_EXACT = "pypi.org"
GRANTED_WILD = "*.pypi.org"
UNGRANTED = "example.com"


def _skip_if_dns_egress_unavailable(exc: BaseException) -> None:
    """Skip when the fleet cannot admit a dns_name create."""
    if isinstance(exc, SandboxValidationError):
        fields = " ".join(v.field for v in exc.field_violations)
        if "dns_name" in f"{fields} {exc}".lower():
            pytest.skip(f"runner policy does not admit DNS-name egress: {exc}")
    if isinstance(exc, SandboxError) and exc.reason in _SKIP_PLACEMENT_REASONS:
        pytest.skip(f"no runner can host DNS-name egress: {exc}")


def _https_get(sandbox: Sandbox, url: str, *, timeout: float) -> int:
    probe = (
        "import sys, urllib.request; "
        "urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2]))"
    )
    result = sandbox.exec(
        ["python", "-c", probe, url, str(timeout)],
        timeout_seconds=timeout + 15,
    ).result()
    return result.returncode


@pytest.fixture
def dns_egress_defaults(sandbox_defaults: SandboxDefaults) -> SandboxDefaults:
    """Longer lifetime so create can wait out the Envoy init probe."""
    return sandbox_defaults.with_overrides(max_lifetime_seconds=300)


def test_dns_name_https_grant_and_miss(dns_egress_defaults: SandboxDefaults) -> None:
    """Declared names reach HTTPS; a name outside the grant does not."""
    network = NetworkOptions(
        egress=[
            EgressRule(dns_name=GRANTED_EXACT),
            EgressRule(dns_name=GRANTED_WILD),
        ]
    )
    try:
        with Sandbox.run(defaults=dns_egress_defaults, network=network) as sandbox:
            sandbox.wait()
            assert GRANTED_EXACT in sandbox.dns_egress_names
            assert GRANTED_WILD in sandbox.dns_egress_names
            assert _https_get(sandbox, f"https://{GRANTED_EXACT}", timeout=20.0) == 0
            assert _https_get(sandbox, f"https://{UNGRANTED}", timeout=8.0) != 0
    except SandboxError as exc:
        _skip_if_dns_egress_unavailable(exc)
        raise
