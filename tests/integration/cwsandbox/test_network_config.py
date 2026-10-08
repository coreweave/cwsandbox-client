# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Integration tests for NetworkConfig and live network updates.

These tests require a CWSandbox backend that serves NetworkConfigService
with bound updates. Against an older backend the RPCs are unimplemented
and the tests skip. Live hostname grants also need a runner that can add
names live; the tests skip when no runner can.

Set CWSANDBOX_BASE_URL and CWSANDBOX_API_KEY before running.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from cwsandbox import EgressRule, NetworkConfig, NetworkOptions, Sandbox, SandboxDefaults
from cwsandbox._error_info import (
    CWSANDBOX_NO_SUITABLE_RUNNER,
    CWSANDBOX_NOT_IMPLEMENTED,
    CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED,
)
from cwsandbox.exceptions import (
    CWSandboxError,
    DNSEgressUnavailableError,
    NetworkConfigEtagMismatchError,
    NetworkConfigInUseError,
    SandboxError,
)

GRANTED = "pypi.org"
UNGRANTED = "example.com"


def _skip_if_unsupported(exc: BaseException) -> None:
    """Skip when the backend predates NetworkConfig."""
    if isinstance(exc, CWSandboxError) and (
        exc.reason == CWSANDBOX_NOT_IMPLEMENTED or "UNIMPLEMENTED" in str(exc).upper()
    ):
        pytest.skip(f"backend does not serve NetworkConfig: {exc}")


def _skip_if_live_names_unavailable(exc: BaseException) -> None:
    """Skip when no runner can add a hostname to a running sandbox."""
    if isinstance(exc, DNSEgressUnavailableError):
        pytest.skip(f"DNS-name egress unavailable: {exc}")
    if isinstance(exc, SandboxError) and (
        "dns_name" in str(exc).lower()
        or "hostname" in str(exc).lower()
        or exc.reason in {CWSANDBOX_NO_SUITABLE_RUNNER, CWSANDBOX_PLACEMENT_CONSTRAINT_UNSATISFIED}
    ):
        pytest.skip(f"no runner can grant hostnames live: {exc}")


def _https_get(sandbox: Sandbox, url: str, *, timeout: float) -> int:
    # urlopen's timeout applies per resolved address, so a blocked host with
    # several addresses can outlast the exec deadline. The alarm caps the
    # whole probe, DNS lookup included.
    probe = (
        "import signal, sys, urllib.request; t = float(sys.argv[2]); "
        "signal.alarm(int(t) + 2); urllib.request.urlopen(sys.argv[1], timeout=t)"
    )
    result = sandbox.exec(
        ["python", "-c", probe, url, str(timeout)],
        timeout_seconds=timeout + 15,
    ).result()
    return result.returncode


@pytest.fixture
def base_url(sandbox_defaults: SandboxDefaults) -> str | None:
    return sandbox_defaults.base_url


@pytest.fixture
def unbound_config(base_url: str | None) -> Iterator[NetworkConfig]:
    """A caller-created config, deleted after the test."""
    config_id = f"it-{uuid.uuid4().hex[:12]}"
    try:
        config = NetworkConfig.create(
            config_id,
            network=NetworkOptions(deny_ingress=True),
            base_url=base_url,
        ).result()
    except CWSandboxError as exc:
        _skip_if_unsupported(exc)
        raise
    try:
        yield config
    finally:
        config.delete(allow_missing=True).result()


def test_unbound_crud_round_trip(unbound_config: NetworkConfig, base_url: str | None) -> None:
    """Create, get, guarded update, list, and delete an unbound config."""
    fetched = NetworkConfig.get(unbound_config.network_config_id, base_url=base_url).result()
    assert fetched.network.deny_ingress is True
    assert fetched.sandbox_id is None
    assert fetched.platform_managed is False

    # update() refreshes the object in place, so keep the pre-update etag.
    stale_etag = fetched.etag
    updated = fetched.update(NetworkOptions(deny_egress=True), etag=stale_etag).result()
    assert updated.network.deny_egress is True
    assert updated.network.deny_ingress is True  # outside the default mask
    assert updated.etag != stale_etag

    with pytest.raises(NetworkConfigEtagMismatchError):
        updated.update(NetworkOptions(deny_egress=False), etag=stale_etag).result()

    listed = NetworkConfig.list(platform_managed=False, base_url=base_url).result()
    assert unbound_config.network_config_id in {c.network_config_id for c in listed}


def test_bind_precreated_config(
    unbound_config: NetworkConfig, sandbox_defaults: SandboxDefaults
) -> None:
    """A precreated config binds 1:1 and refuses a second live sandbox."""
    config_id = unbound_config.network_config_id
    with Sandbox.run(defaults=sandbox_defaults, network_config_id=config_id) as sandbox:
        sandbox.wait()
        assert sandbox.network_config_id == config_id
        with pytest.raises(NetworkConfigInUseError):
            Sandbox.run(defaults=sandbox_defaults, network_config_id=config_id).wait()


def test_update_network_adds_and_removes_hostname(sandbox_defaults: SandboxDefaults) -> None:
    """A minted config can grant a hostname live and clear it again."""
    defaults = sandbox_defaults.with_overrides(max_lifetime_seconds=300)
    with Sandbox.run(defaults=defaults) as sandbox:
        sandbox.wait()
        if sandbox.network_config_id is None:
            pytest.skip("backend did not report a NetworkConfig for the sandbox")
        assert sandbox.network_config_id.startswith("nc-")
        if _https_get(sandbox, f"https://{GRANTED}", timeout=8.0) == 0:
            pytest.skip(f"policy default already allows {GRANTED}; grant is not observable")
        try:
            sandbox.update_network(NetworkOptions(egress=[EgressRule(dns_name=GRANTED)])).result()
        except CWSandboxError as exc:
            _skip_if_unsupported(exc)
            _skip_if_live_names_unavailable(exc)
            raise
        assert GRANTED in sandbox.dns_egress_names
        # The status echo is not enforcement; check traffic from inside the sandbox.
        assert _https_get(sandbox, f"https://{GRANTED}", timeout=20.0) == 0
        assert _https_get(sandbox, f"https://{UNGRANTED}", timeout=8.0) != 0

        sandbox.update_network(NetworkOptions(egress=[])).result()
        assert GRANTED not in sandbox.dns_egress_names
        assert _https_get(sandbox, f"https://{GRANTED}", timeout=8.0) != 0
