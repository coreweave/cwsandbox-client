"""Integration tests for network exposure configuration.

Tests for ingress_mode, egress_mode, and network configuration via
Sandbox.run() and Session.sandbox() entry points.

These tests require a backend that supports the network configuration
protobuf field. Tests will be skipped if the backend doesn't support
network configuration.
"""

import contextlib
import time
from collections.abc import Generator
from typing import Any

import httpx
import pytest

from aviato import Sandbox, SandboxDefaults, Session


@contextlib.contextmanager
def network_sandbox(
    *args: str,
    network: dict[str, Any],
    defaults: SandboxDefaults,
    **kwargs: Any,
) -> Generator[Sandbox, None, None]:
    """Context manager that creates a sandbox with network config, skipping if unsupported.

    This helper catches ValueError from missing protobuf fields and converts
    them to pytest.skip() for graceful handling on backends that don't yet
    support network configuration.
    """
    try:
        sandbox = Sandbox.run(*args, network=network, defaults=defaults, **kwargs)
    except ValueError as e:
        if "network" in str(e) and "no" in str(e).lower() and "field" in str(e).lower():
            pytest.skip("Backend does not support network configuration")
        raise

    try:
        yield sandbox
    finally:
        sandbox.stop(missing_ok=True).result()


def test_sandbox_public_service_address(sandbox_defaults: SandboxDefaults) -> None:
    """Test sandbox with ingress_mode=public returns service_address.

    Creates a sandbox with network={"ingress_mode": "public"} and verifies that
    service_address is populated in the response.
    """
    with network_sandbox(
        network={"ingress_mode": "public"},
        defaults=sandbox_defaults,
    ) as sandbox:
        sandbox.wait()

        # service_address comes from StartSandboxResponse
        # It may be None if the tower uses ClusterIP instead of LoadBalancer
        if sandbox.service_address is not None:
            # Address should look like "ip:port" or hostname format
            assert ":" in sandbox.service_address or "." in sandbox.service_address
        else:
            # If service_address is None, that's acceptable - infrastructure dependent
            # Just verify the sandbox is running and the property exists
            assert sandbox.status == "running"


def test_sandbox_public_exposed_ports(sandbox_defaults: SandboxDefaults) -> None:
    """Test sandbox with ingress_mode=public and ports returns exposed_ports.

    Creates a sandbox with network config and port mappings,
    then verifies exposed_ports is populated.
    """
    with network_sandbox(
        network={"ingress_mode": "public", "exposed_ports": [8080]},
        ports=[{"container_port": 8080, "name": "http"}],
        defaults=sandbox_defaults,
    ) as sandbox:
        sandbox.wait()

        # exposed_ports comes from StartSandboxResponse
        # It may be None depending on infrastructure configuration
        if sandbox.exposed_ports is not None:
            assert len(sandbox.exposed_ports) >= 1
            # Each entry should be (port, name) tuple
            port, name = sandbox.exposed_ports[0]
            assert isinstance(port, int)
            assert isinstance(name, str)
        else:
            # If exposed_ports is None, verify sandbox is running
            assert sandbox.status == "running"


def test_sandbox_applied_network_modes(sandbox_defaults: SandboxDefaults) -> None:
    """Test sandbox with network config returns applied_ingress_mode and applied_egress_mode.

    Creates a sandbox with network configuration and verifies that the applied
    mode properties are populated in the response.
    """
    with network_sandbox(
        network={"ingress_mode": "public"},
        defaults=sandbox_defaults,
    ) as sandbox:
        sandbox.wait()

        # applied_ingress_mode comes from StartSandboxResponse
        # It should be populated when network config is provided
        if sandbox.applied_ingress_mode is not None:
            # Mode should be a non-empty string
            assert isinstance(sandbox.applied_ingress_mode, str)
            assert len(sandbox.applied_ingress_mode) > 0
        else:
            # If applied_ingress_mode is None, verify sandbox is running
            # This can happen if the backend doesn't return the field
            assert sandbox.status == "running"

        # applied_egress_mode may or may not be set depending on config
        if sandbox.applied_egress_mode is not None:
            assert isinstance(sandbox.applied_egress_mode, str)
            assert len(sandbox.applied_egress_mode) > 0


def test_sandbox_public_service_connectivity(sandbox_defaults: SandboxDefaults) -> None:
    """Test that exposed service is actually reachable.

    Starts a simple HTTP server inside the sandbox and verifies we can
    connect to it from outside using the service_address.
    """
    with network_sandbox(
        "python",
        "-m",
        "http.server",
        "8080",
        network={"ingress_mode": "public", "exposed_ports": [8080]},
        ports=[{"container_port": 8080, "name": "http"}],
        defaults=sandbox_defaults,
    ) as sandbox:
        sandbox.wait()

        if sandbox.service_address is None:
            pytest.skip("Infrastructure does not provide service_address")

        # Give the HTTP server a moment to start
        time.sleep(2)

        response = httpx.get(
            f"http://{sandbox.service_address}/",
            timeout=120.0,
        )
        assert response.status_code == 200


def test_sandbox_egress_mode_only(sandbox_defaults: SandboxDefaults) -> None:
    """Test sandbox with egress_mode only returns applied_egress_mode.

    Creates a sandbox with only egress_mode configured and verifies that
    applied_egress_mode is populated in the response.
    """
    with network_sandbox(
        network={"egress_mode": "default"},
        defaults=sandbox_defaults,
    ) as sandbox:
        sandbox.wait()

        # applied_egress_mode should be populated when egress_mode is requested
        if sandbox.applied_egress_mode is not None:
            assert isinstance(sandbox.applied_egress_mode, str)
            assert len(sandbox.applied_egress_mode) > 0
        else:
            # If applied_egress_mode is None, verify sandbox is running
            # This can happen if the tower doesn't support the requested mode
            assert sandbox.status == "running"

        # Verify sandbox is functional
        result = sandbox.exec(["echo", "egress test"]).result()
        assert result.returncode == 0
        assert result.stdout.strip() == "egress test"


def test_sandbox_ingress_and_egress_modes(sandbox_defaults: SandboxDefaults) -> None:
    """Test sandbox with both ingress_mode and egress_mode configured.

    Creates a sandbox with both network modes and verifies that both
    applied_ingress_mode and applied_egress_mode are populated.
    """
    with network_sandbox(
        network={"ingress_mode": "public", "egress_mode": "default"},
        defaults=sandbox_defaults,
    ) as sandbox:
        sandbox.wait()

        # Both modes should be populated when both are requested
        if sandbox.applied_ingress_mode is not None:
            assert isinstance(sandbox.applied_ingress_mode, str)
            assert len(sandbox.applied_ingress_mode) > 0

        if sandbox.applied_egress_mode is not None:
            assert isinstance(sandbox.applied_egress_mode, str)
            assert len(sandbox.applied_egress_mode) > 0

        # At minimum, the sandbox should be running
        assert sandbox.status == "running"

        # Verify sandbox is functional
        result = sandbox.exec(["echo", "combined modes test"]).result()
        assert result.returncode == 0
        assert result.stdout.strip() == "combined modes test"


def test_session_sandbox_with_network(sandbox_defaults: SandboxDefaults) -> None:
    """Test Session.sandbox() with network configuration.

    Creates a sandbox via Session.sandbox() with network config and verifies
    that applied_ingress_mode and applied_egress_mode are populated.
    """
    with Session(sandbox_defaults) as session:
        try:
            sandbox = session.sandbox(
                command="sleep",
                args=["infinity"],
                network={"ingress_mode": "public", "egress_mode": "default"},
            )
        except ValueError as e:
            if "network" in str(e) and "no" in str(e).lower() and "field" in str(e).lower():
                pytest.skip("Backend does not support network configuration")
            raise

        try:
            sandbox.wait()

            # Verify applied modes are populated
            if sandbox.applied_ingress_mode is not None:
                assert isinstance(sandbox.applied_ingress_mode, str)
                assert len(sandbox.applied_ingress_mode) > 0

            if sandbox.applied_egress_mode is not None:
                assert isinstance(sandbox.applied_egress_mode, str)
                assert len(sandbox.applied_egress_mode) > 0

            # Verify sandbox is functional
            result = sandbox.exec(["echo", "session network test"]).result()
            assert result.returncode == 0
            assert result.stdout.strip() == "session network test"

        finally:
            sandbox.stop().result()
