# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for Sandbox network_config_id binding and update_network()."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from cwsandbox import (
    EgressRule,
    NetworkConfig,
    NetworkOptions,
    Sandbox,
    SandboxDefaults,
    SandboxError,
    SandboxStatus,
    Session,
)
from cwsandbox._proto import sandbox_pb2
from cwsandbox._sandbox import _Running, _Terminal


def _response(sandbox_id: str, *, network_config_id: str = "") -> sandbox_pb2.Sandbox:
    return sandbox_pb2.Sandbox(
        sandbox_id=sandbox_id,
        spec=sandbox_pb2.SandboxSpec(network_config_id=network_config_id),
        status=sandbox_pb2.SandboxStatus(state=sandbox_pb2.STATE_PENDING),
    )


def _run_with_mock_stub(
    *args: str, response: sandbox_pb2.Sandbox | None = None, **kwargs: Any
) -> tuple[Sandbox, MagicMock]:
    stub = MagicMock()
    stub.CreateSandbox = AsyncMock(return_value=response or _response("sb-1"))
    stub.CreateSandboxFromTemplate = AsyncMock(return_value=response or _response("sb-1"))

    async def ensure_client(sandbox: Sandbox) -> None:
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = stub

    with patch.object(Sandbox, "_ensure_client", ensure_client):
        sandbox = Sandbox.run(*args, **kwargs)
    return sandbox, stub


def _finish(sandbox: Sandbox) -> None:
    sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)


class TestNetworkConfigIdOnCreate:
    def test_create_sends_network_config_id_without_network(self) -> None:
        sandbox, stub = _run_with_mock_stub(
            network_config_id="team-net",
            defaults=SandboxDefaults(network=NetworkOptions(deny_egress=True)),
        )
        spec = stub.CreateSandbox.call_args.args[0].sandbox.spec
        assert spec.network_config_id == "team-net"
        assert not spec.HasField("network")
        assert sandbox.network_config_id == "team-net"
        _finish(sandbox)

    def test_create_echo_sets_minted_id(self) -> None:
        sandbox, stub = _run_with_mock_stub(
            network=NetworkOptions(egress=[EgressRule(dns_name="pypi.org")]),
            response=_response("sb-1", network_config_id="nc-abc"),
        )
        spec = stub.CreateSandbox.call_args.args[0].sandbox.spec
        assert spec.network_config_id == ""
        assert sandbox.network_config_id == "nc-abc"
        _finish(sandbox)

    def test_without_echo_network_config_id_is_none(self) -> None:
        sandbox, _ = _run_with_mock_stub()
        assert sandbox.network_config_id is None
        _finish(sandbox)

    def test_network_and_network_config_id_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            Sandbox(network=NetworkOptions(deny_egress=True), network_config_id="team-net")

    def test_empty_network_config_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Sandbox(network_config_id="")

    def test_template_override_carries_network_config_id(self) -> None:
        stub = MagicMock()
        stub.CreateSandboxFromTemplate = AsyncMock(return_value=_response("sb-1"))

        async def ensure_client(sandbox: Sandbox) -> None:
            sandbox._channel = MagicMock()
            sandbox._channel.close = AsyncMock()
            sandbox._stub = stub

        with patch.object(Sandbox, "_ensure_client", ensure_client):
            sandbox = Sandbox.run_from_template("template-123", network_config_id="team-net")

        overrides = stub.CreateSandboxFromTemplate.call_args.args[0].overrides
        assert overrides.network_config_id == "team-net"
        assert not overrides.HasField("network")
        _finish(sandbox)

    def test_run_from_file_rejects_network_config_id(self) -> None:
        with pytest.raises(TypeError):
            Sandbox.run_from_file(
                b"services: {}\n",
                primary_service="app",
                network_config_id="team-net",  # type: ignore[call-arg]
            )

    def test_session_sandbox_forwards_network_config_id(self) -> None:
        session = Session()
        sandbox = session.sandbox(network_config_id="team-net")
        assert sandbox._requested_network_config_id == "team-net"
        assert sandbox.network_config_id == "team-net"

    def test_from_view_reads_network_config_id(self) -> None:
        sandbox, _ = _run_with_mock_stub()
        view_response = sandbox_pb2.Sandbox(
            sandbox_id="sb-1",
            spec=sandbox_pb2.SandboxSpec(network_config_id="nc-xyz"),
            status=sandbox_pb2.SandboxStatus(state=sandbox_pb2.STATE_RUNNING),
        )
        sandbox._stub = MagicMock()
        sandbox._stub.GetSandbox = AsyncMock(return_value=view_response)
        with patch.object(Sandbox, "_ensure_client", AsyncMock()):
            sandbox.get_status()
        assert sandbox.network_config_id == "nc-xyz"
        _finish(sandbox)


def _running_sandbox(network_config_id: str | None = "nc-1") -> Sandbox:
    sandbox = Sandbox(
        command="sleep",
        args=["infinity"],
        base_url="https://api.example.com",
        request_timeout_seconds=42.0,
    )
    sandbox._sandbox_id = "sb-1"
    sandbox._state = _Running(sandbox_id="sb-1")
    sandbox._network_config_id = network_config_id
    sandbox._channel = MagicMock()
    sandbox._stub = MagicMock()
    return sandbox


def _running_view(*, network_config_id: str = "nc-1", names: tuple[str, ...] = ()) -> Any:
    return sandbox_pb2.Sandbox(
        sandbox_id="sb-1",
        spec=sandbox_pb2.SandboxSpec(network_config_id=network_config_id),
        status=sandbox_pb2.SandboxStatus(
            state=sandbox_pb2.STATE_RUNNING,
            effective_egress=[sandbox_pb2.EgressRule(https_hostname=n) for n in names],
        ),
    )


class TestUpdateNetwork:
    def test_updates_bound_config_and_refreshes_echo(self) -> None:
        sandbox = _running_sandbox()
        sandbox._stub.GetSandbox = AsyncMock(return_value=_running_view(names=("pypi.org",)))
        updated = NetworkConfig(network_config_id="nc-1", etag="v2")
        update = AsyncMock(return_value=updated)
        with (
            patch.object(Sandbox, "_ensure_client", AsyncMock()),
            patch.object(Sandbox, "_wait_until_running_async", AsyncMock()),
            patch("cwsandbox._sandbox._update_network_config_async", update),
        ):
            result = sandbox.update_network(
                {"egress": [{"dns_name": "pypi.org"}]}, etag="v1"
            ).result()

        assert result is updated
        args, kwargs = update.call_args
        assert args[0] == "nc-1"
        assert args[1] == {"egress": [{"dns_name": "pypi.org"}]}
        assert kwargs["etag"] == "v1"
        assert kwargs["update_mask"] is None
        assert kwargs["base_url"] == "https://api.example.com"
        assert kwargs["timeout_seconds"] == 42.0
        assert sandbox.dns_egress_names == ("pypi.org",)
        _finish(sandbox)

    def test_fetches_id_when_unknown(self) -> None:
        sandbox = _running_sandbox(network_config_id=None)
        sandbox._stub.GetSandbox = AsyncMock(return_value=_running_view(network_config_id="nc-9"))
        update = AsyncMock(return_value=NetworkConfig(network_config_id="nc-9"))
        with (
            patch.object(Sandbox, "_ensure_client", AsyncMock()),
            patch.object(Sandbox, "_wait_until_running_async", AsyncMock()),
            patch("cwsandbox._sandbox._update_network_config_async", update),
        ):
            sandbox.update_network(NetworkOptions(deny_egress=True)).result()
        assert update.call_args.args[0] == "nc-9"
        _finish(sandbox)

    def test_raises_when_backend_reports_no_config(self) -> None:
        sandbox = _running_sandbox(network_config_id=None)
        sandbox._stub.GetSandbox = AsyncMock(return_value=_running_view(network_config_id=""))
        update = AsyncMock()
        with (
            patch.object(Sandbox, "_ensure_client", AsyncMock()),
            patch.object(Sandbox, "_wait_until_running_async", AsyncMock()),
            patch("cwsandbox._sandbox._update_network_config_async", update),
        ):
            with pytest.raises(SandboxError, match="has no NetworkConfig"):
                sandbox.update_network(NetworkOptions(deny_egress=True)).result()
        update.assert_not_called()
        _finish(sandbox)

    def test_refresh_failure_does_not_hide_success(self) -> None:
        class _Unavailable(grpc.RpcError):
            def code(self) -> grpc.StatusCode:
                return grpc.StatusCode.UNAVAILABLE

            def details(self) -> str:
                return "blip"

            def trailing_metadata(self) -> tuple[()]:
                return ()

        sandbox = _running_sandbox()
        sandbox._stub.GetSandbox = AsyncMock(side_effect=_Unavailable())
        updated = NetworkConfig(network_config_id="nc-1")
        with (
            patch.object(Sandbox, "_ensure_client", AsyncMock()),
            patch.object(Sandbox, "_wait_until_running_async", AsyncMock()),
            patch(
                "cwsandbox._sandbox._update_network_config_async",
                AsyncMock(return_value=updated),
            ),
        ):
            assert sandbox.update_network(NetworkOptions(egress=[])).result() is updated
        _finish(sandbox)
