# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for NetworkConfig CRUD."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.protobuf import duration_pb2
from google.rpc import error_details_pb2, status_pb2

from cwsandbox import (
    DNSEgressMode,
    EgressRule,
    IngressRule,
    NetworkConfig,
    NetworkOptions,
    SandboxUnavailableError,
)
from cwsandbox._error_info import (
    CWSANDBOX_DNS_EGRESS_UNAVAILABLE,
    CWSANDBOX_ETAG_MISMATCH,
    CWSANDBOX_NETWORK_CONFIG_IN_USE,
    CWSANDBOX_NETWORK_CONFIG_NOT_FOUND,
)
from cwsandbox._network_config import _resolve_update_mask
from cwsandbox._proto import network_config_pb2, sandbox_pb2
from cwsandbox.exceptions import (
    DNSEgressUnavailableError,
    NetworkConfigEtagMismatchError,
    NetworkConfigInUseError,
    NetworkConfigNotFoundError,
)


def _rpc_error(
    reason: str | None,
    code: grpc.StatusCode = grpc.StatusCode.FAILED_PRECONDITION,
    *,
    metadata: dict[str, str] | None = None,
    retry_seconds: int | None = None,
) -> grpc.RpcError:
    status = status_pb2.Status(message="boom")
    if reason is not None:
        info = error_details_pb2.ErrorInfo(
            reason=reason, domain="cwsandbox.com", metadata=metadata or {}
        )
        status.details.add().Pack(info)
    if retry_seconds is not None:
        retry = error_details_pb2.RetryInfo(
            retry_delay=duration_pb2.Duration(seconds=retry_seconds)
        )
        status.details.add().Pack(retry)
    status_bytes = status.SerializeToString()

    class _Err(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return code

        def details(self) -> str:
            return "boom"

        def trailing_metadata(self) -> tuple[tuple[str, bytes], ...]:
            return (("grpc-status-details-bin", status_bytes),)

    return _Err()


def _patch_channel(stub: MagicMock) -> tuple[object, object, object, object]:
    mock_channel = MagicMock()
    mock_channel.close = AsyncMock()
    return (
        patch(
            "cwsandbox._network_config.resolve_auth_metadata",
            return_value=(("authorization", "t"),),
        ),
        patch("cwsandbox._network_config.parse_grpc_target", return_value=("test:443", True)),
        patch("cwsandbox._network_config.create_channel", return_value=mock_channel),
        patch(
            "cwsandbox._network_config.network_config_pb2_grpc.NetworkConfigServiceStub",
            return_value=stub,
        ),
    )


def _config_proto(
    network_config_id: str = "team-net",
    *,
    sandbox_id: str | None = None,
    platform_managed: bool = False,
    etag: str = "v1",
) -> network_config_pb2.NetworkConfig:
    status = network_config_pb2.NetworkConfigStatus(platform_managed=platform_managed)
    if sandbox_id is not None:
        status.sandbox_id = sandbox_id
    status.create_time.FromSeconds(1_700_000_000)
    status.update_time.FromSeconds(1_700_000_100)
    return network_config_pb2.NetworkConfig(
        network_config_id=network_config_id,
        spec=sandbox_pb2.NetworkOptions(
            egress=[sandbox_pb2.EgressRule(https_hostname="pypi.org")],
            ingress=[sandbox_pb2.IngressRule(any=True)],
            dns_egress=sandbox_pb2.DNS_EGRESS_MODE_DENY,
        ),
        status=status,
        etag=etag,
    )


class TestNetworkConfigCrud:
    def test_create_maps_request_and_response(self) -> None:
        stub = MagicMock()
        stub.CreateNetworkConfig = AsyncMock(return_value=_config_proto())
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            config = NetworkConfig.create(
                "team-net",
                network={"egress": [{"dns_name": "pypi.org"}], "dns_egress": "deny"},
            ).result()

        req = stub.CreateNetworkConfig.call_args[0][0]
        assert req.network_config.network_config_id == "team-net"
        assert [r.https_hostname for r in req.network_config.spec.egress] == ["pypi.org"]
        assert req.network_config.spec.dns_egress == sandbox_pb2.DNS_EGRESS_MODE_DENY
        assert req.request_id
        assert config.network_config_id == "team-net"
        assert config.network.egress == (EgressRule(dns_name="pypi.org"),)
        assert config.network.ingress == (IngressRule(any=True),)
        assert config.network.dns_egress is DNSEgressMode.DENY
        assert config.network.deny_egress is False
        assert config.sandbox_id is None
        assert config.platform_managed is False
        assert config.etag == "v1"
        assert config.created_at is not None
        assert config.updated_at is not None

    def test_create_without_network_sends_empty_spec(self) -> None:
        stub = MagicMock()
        stub.CreateNetworkConfig = AsyncMock(return_value=_config_proto())
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            NetworkConfig.create("team-net", request_id="req-1").result()

        req = stub.CreateNetworkConfig.call_args[0][0]
        assert req.HasField("network_config")
        assert req.network_config.HasField("spec")
        assert len(req.network_config.spec.egress) == 0
        assert req.request_id == "req-1"

    def test_create_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="network_config_id"):
            NetworkConfig.create("").result()

    def test_get_maps_binding(self) -> None:
        stub = MagicMock()
        stub.GetNetworkConfig = AsyncMock(
            return_value=_config_proto("nc-abc", sandbox_id="sb-1", platform_managed=True)
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            config = NetworkConfig.get("nc-abc").result()

        assert stub.GetNetworkConfig.call_args[0][0].network_config_id == "nc-abc"
        assert config.sandbox_id == "sb-1"
        assert config.platform_managed is True
        assert "sb-1" in repr(config)

    def test_get_not_found(self) -> None:
        stub = MagicMock()
        stub.GetNetworkConfig = AsyncMock(
            side_effect=_rpc_error(
                CWSANDBOX_NETWORK_CONFIG_NOT_FOUND,
                code=grpc.StatusCode.NOT_FOUND,
                metadata={"network_config_id": "missing"},
            )
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(NetworkConfigNotFoundError) as exc_info:
                NetworkConfig.get("missing").result()
        assert exc_info.value.network_config_id == "missing"

    def test_bare_not_found_maps_to_network_config_not_found(self) -> None:
        stub = MagicMock()
        stub.GetNetworkConfig = AsyncMock(
            side_effect=_rpc_error(None, code=grpc.StatusCode.NOT_FOUND)
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(NetworkConfigNotFoundError) as exc_info:
                NetworkConfig.get("gone").result()
        assert exc_info.value.network_config_id == "gone"

    def test_list_filters_and_paginates(self) -> None:
        stub = MagicMock()
        stub.ListNetworkConfigs = AsyncMock(
            side_effect=[
                network_config_pb2.ListNetworkConfigsResponse(
                    network_configs=[_config_proto("a-net")], next_page_token="p2"
                ),
                network_config_pb2.ListNetworkConfigsResponse(
                    network_configs=[_config_proto("b-net")]
                ),
            ]
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            configs = NetworkConfig.list(platform_managed=False).result()

        assert [c.network_config_id for c in configs] == ["a-net", "b-net"]
        first = stub.ListNetworkConfigs.call_args_list[0][0][0]
        assert first.HasField("platform_managed")
        assert first.platform_managed is False

    def test_list_without_filter_leaves_field_unset(self) -> None:
        stub = MagicMock()
        stub.ListNetworkConfigs = AsyncMock(
            return_value=network_config_pb2.ListNetworkConfigsResponse()
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            assert NetworkConfig.list().result() == []
        assert not stub.ListNetworkConfigs.call_args[0][0].HasField("platform_managed")

    def test_delete_in_use(self) -> None:
        stub = MagicMock()
        stub.DeleteNetworkConfig = AsyncMock(
            side_effect=_rpc_error(CWSANDBOX_NETWORK_CONFIG_IN_USE)
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(NetworkConfigInUseError):
                NetworkConfig(network_config_id="team-net").delete().result()

    def test_delete_allow_missing(self) -> None:
        stub = MagicMock()
        stub.DeleteNetworkConfig = AsyncMock(
            side_effect=_rpc_error(
                CWSANDBOX_NETWORK_CONFIG_NOT_FOUND, code=grpc.StatusCode.NOT_FOUND
            )
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            assert (
                NetworkConfig(network_config_id="team-net").delete(allow_missing=True).result()
                is None
            )
            with pytest.raises(NetworkConfigNotFoundError):
                NetworkConfig(network_config_id="team-net").delete().result()
        assert stub.DeleteNetworkConfig.call_args_list[0][0][0].allow_missing is True


class TestNetworkConfigUpdate:
    def test_update_derives_mask_from_set_fields(self) -> None:
        stub = MagicMock()
        stub.UpdateNetworkConfig = AsyncMock(return_value=_config_proto(etag="v2"))
        patches = _patch_channel(stub)
        config = NetworkConfig(network_config_id="team-net")
        with patches[0], patches[1], patches[2], patches[3]:
            result = config.update(NetworkOptions(egress=[])).result()

        req = stub.UpdateNetworkConfig.call_args[0][0]
        assert req.network_config_id == "team-net"
        assert list(req.update_mask.paths) == ["spec.egress"]
        assert len(req.network_config.spec.egress) == 0
        assert not req.network_config.HasField("etag")
        assert result is config
        assert config.etag == "v2"

    def test_update_sends_etag_and_explicit_mask(self) -> None:
        stub = MagicMock()
        stub.UpdateNetworkConfig = AsyncMock(return_value=_config_proto())
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            NetworkConfig(network_config_id="team-net").update(
                NetworkOptions(egress=[EgressRule(dns_name="pypi.org")], deny_ingress=True),
                update_mask=["egress", "spec.egress", "*"],
                etag="v1",
            ).result()

        req = stub.UpdateNetworkConfig.call_args[0][0]
        assert list(req.update_mask.paths) == ["spec.egress", "*"]
        assert req.network_config.etag == "v1"

    def test_update_read_modify_write_replaces_every_field(self) -> None:
        read = NetworkConfig(network_config_id="team-net")
        from cwsandbox._spec import network_from_proto

        read.network = network_from_proto(_config_proto().spec)
        edited = dataclasses.replace(read.network, ingress=())
        assert _resolve_update_mask(edited, None) == [
            "spec.egress",
            "spec.ingress",
            "spec.deny_egress",
            "spec.deny_ingress",
            "spec.dns_egress",
        ]

    def test_update_requires_a_field_or_mask(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            NetworkConfig(network_config_id="team-net").update(NetworkOptions()).result()

    def test_update_empty_mask_is_sent_as_no_op(self) -> None:
        assert _resolve_update_mask(NetworkOptions(), []) == []

    def test_update_rejects_unknown_mask_path(self) -> None:
        with pytest.raises(ValueError, match="unknown update_mask path"):
            _resolve_update_mask(NetworkOptions(), ["spec.services"])
        with pytest.raises(TypeError):
            _resolve_update_mask(NetworkOptions(), "spec.egress")

    def test_update_etag_mismatch(self) -> None:
        stub = MagicMock()
        stub.UpdateNetworkConfig = AsyncMock(
            side_effect=_rpc_error(CWSANDBOX_ETAG_MISMATCH, code=grpc.StatusCode.ABORTED)
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(NetworkConfigEtagMismatchError) as exc_info:
                NetworkConfig(network_config_id="team-net").update(
                    NetworkOptions(deny_egress=True), etag="stale"
                ).result()
        assert exc_info.value.network_config_id == "team-net"
        assert stub.UpdateNetworkConfig.call_count == 1

    def test_update_retries_dns_egress_unavailable(self) -> None:
        stub = MagicMock()
        stub.UpdateNetworkConfig = AsyncMock(
            side_effect=[
                _rpc_error(
                    CWSANDBOX_DNS_EGRESS_UNAVAILABLE,
                    code=grpc.StatusCode.UNAVAILABLE,
                    retry_seconds=0,
                ),
                _config_proto(),
            ]
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3], patch("asyncio.sleep", AsyncMock()):
            NetworkConfig(network_config_id="team-net").update(
                NetworkOptions(egress=[EgressRule(dns_name="pypi.org")])
            ).result()
        assert stub.UpdateNetworkConfig.call_count == 2

    def test_dns_egress_unavailable_is_retryable_type(self) -> None:
        from cwsandbox._sandbox import _translate_rpc_error

        exc = _translate_rpc_error(
            _rpc_error(
                CWSANDBOX_DNS_EGRESS_UNAVAILABLE,
                code=grpc.StatusCode.UNAVAILABLE,
                retry_seconds=5,
            ),
            operation="Update network config",
            network_config_id="team-net",
        )
        assert isinstance(exc, DNSEgressUnavailableError)
        assert isinstance(exc, SandboxUnavailableError)
        assert exc.retry_delay == timedelta(seconds=5)

    def test_update_deadline_is_not_retried(self) -> None:
        from cwsandbox.exceptions import SandboxRequestTimeoutError

        stub = MagicMock()
        stub.UpdateNetworkConfig = AsyncMock(
            side_effect=_rpc_error(None, code=grpc.StatusCode.DEADLINE_EXCEEDED)
        )
        patches = _patch_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(SandboxRequestTimeoutError):
                NetworkConfig(network_config_id="team-net").update(
                    NetworkOptions(deny_egress=True)
                ).result()
        assert stub.UpdateNetworkConfig.call_count == 1
