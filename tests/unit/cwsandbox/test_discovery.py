# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._discovery module."""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest

from cwsandbox._discovery import (
    EgressMode,
    IngressMode,
    Runway,
    Tower,
    TowerResources,
    _paginate_async,
    _runway_from_proto,
    _tower_from_proto,
    format_bytes,
    format_cpu,
    get_runway,
    get_tower,
    list_runways,
    list_towers,
)
from cwsandbox._proto import discovery_pb2
from cwsandbox.exceptions import (
    CWSandboxAuthenticationError,
    CWSandboxError,
    DiscoveryError,
    RunwayNotFoundError,
    TowerNotFoundError,
)

# ---------------------------------------------------------------------------
# Type tests
# ---------------------------------------------------------------------------


class TestTowerResources:
    """Tests for TowerResources dataclass."""

    def test_creation(self) -> None:
        res = TowerResources(
            available_cpu_millicores=2000,
            available_memory_bytes=8589934592,
            available_gpu_count=1,
            running_sandboxes=5,
        )
        assert res.available_cpu_millicores == 2000
        assert res.available_memory_bytes == 8589934592
        assert res.available_gpu_count == 1
        assert res.running_sandboxes == 5

    def test_frozen(self) -> None:
        res = TowerResources(
            available_cpu_millicores=2000,
            available_memory_bytes=0,
            available_gpu_count=0,
            running_sandboxes=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.available_cpu_millicores = 9999  # type: ignore[misc]


class TestIngressMode:
    """Tests for IngressMode dataclass."""

    def test_creation(self) -> None:
        mode = IngressMode(name="public")
        assert mode.name == "public"

    def test_frozen(self) -> None:
        mode = IngressMode(name="public")
        with pytest.raises(dataclasses.FrozenInstanceError):
            mode.name = "private"  # type: ignore[misc]


class TestEgressMode:
    """Tests for EgressMode dataclass."""

    def test_creation(self) -> None:
        mode = EgressMode(name="internet")
        assert mode.name == "internet"

    def test_frozen(self) -> None:
        mode = EgressMode(name="internet")
        with pytest.raises(dataclasses.FrozenInstanceError):
            mode.name = "blocked"  # type: ignore[misc]


class TestTower:
    """Tests for Tower dataclass."""

    def _make_tower(self, **overrides: object) -> Tower:
        defaults: dict[str, object] = {
            "tower_id": "tower-1",
            "tower_group_id": "group-1",
            "tags": ("tag1", "tag2"),
            "healthy": True,
            "runway_names": ("default",),
            "connected_at": datetime(2025, 1, 1, tzinfo=UTC),
            "max_cpu_millicores": 4000,
            "max_memory_bytes": 17179869184,
            "max_gpu_count": 2,
            "supported_gpu_types": ("A100",),
            "supported_architectures": ("amd64",),
            "supports_privileged": True,
            "available_storage_classes": ("ssd",),
            "resources": None,
        }
        defaults.update(overrides)
        return Tower(**defaults)  # type: ignore[arg-type]

    def test_creation_all_fields(self) -> None:
        tower = self._make_tower()
        assert tower.tower_id == "tower-1"
        assert tower.tower_group_id == "group-1"
        assert tower.tags == ("tag1", "tag2")
        assert tower.healthy is True
        assert tower.runway_names == ("default",)
        assert tower.connected_at == datetime(2025, 1, 1, tzinfo=UTC)
        assert tower.max_cpu_millicores == 4000
        assert tower.max_memory_bytes == 17179869184
        assert tower.max_gpu_count == 2
        assert tower.supported_gpu_types == ("A100",)
        assert tower.supported_architectures == ("amd64",)
        assert tower.supports_privileged is True
        assert tower.available_storage_classes == ("ssd",)
        assert tower.resources is None

    def test_frozen(self) -> None:
        tower = self._make_tower()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tower.tower_id = "other"  # type: ignore[misc]

    def test_resources_populated(self) -> None:
        res = TowerResources(
            available_cpu_millicores=2000,
            available_memory_bytes=8589934592,
            available_gpu_count=1,
            running_sandboxes=3,
        )
        tower = self._make_tower(resources=res)
        assert tower.resources is res
        assert tower.resources.running_sandboxes == 3

    def test_repr_human_readable(self) -> None:
        tower = self._make_tower(
            max_cpu_millicores=4000,
            max_memory_bytes=17179869184,
            max_gpu_count=2,
            runway_names=("default", "gpu"),
        )
        r = repr(tower)
        assert "tower_id='tower-1'" in r
        assert "healthy=True" in r
        assert "cpu=4.0 vCPU" in r
        assert "memory=16.0 GiB" in r
        assert "gpus=2" in r
        assert "['default', 'gpu']" in r

    def test_post_init_normalizes_lists_to_tuples(self) -> None:
        tower = Tower(
            tower_id="t",
            tower_group_id="g",
            tags=["a", "b"],  # type: ignore[arg-type]
            healthy=True,
            runway_names=["r1"],  # type: ignore[arg-type]
            connected_at=datetime(2025, 1, 1, tzinfo=UTC),
            max_cpu_millicores=0,
            max_memory_bytes=0,
            max_gpu_count=0,
            supported_gpu_types=["H100"],  # type: ignore[arg-type]
            supported_architectures=["arm64"],  # type: ignore[arg-type]
            supports_privileged=False,
            available_storage_classes=["nfs"],  # type: ignore[arg-type]
        )
        assert isinstance(tower.tags, tuple)
        assert isinstance(tower.runway_names, tuple)
        assert isinstance(tower.supported_gpu_types, tuple)
        assert isinstance(tower.supported_architectures, tuple)
        assert isinstance(tower.available_storage_classes, tuple)

    def test_post_init_preserves_tuples(self) -> None:
        tags = ("x",)
        tower = self._make_tower(tags=tags)
        assert tower.tags is tags


class TestRunway:
    """Tests for Runway dataclass."""

    def test_creation(self) -> None:
        runway = Runway(
            runway_name="default",
            tower_id="tower-1",
            supported_gpu_types=("A100",),
            supported_architectures=("amd64",),
            ingress_modes=(IngressMode(name="public"),),
            egress_modes=(EgressMode(name="internet"),),
        )
        assert runway.runway_name == "default"
        assert runway.tower_id == "tower-1"
        assert runway.ingress_modes == (IngressMode(name="public"),)
        assert runway.egress_modes == (EgressMode(name="internet"),)

    def test_post_init_normalizes_lists(self) -> None:
        runway = Runway(
            runway_name="r",
            tower_id="t",
            supported_gpu_types=["A100"],  # type: ignore[arg-type]
            supported_architectures=["amd64"],  # type: ignore[arg-type]
            ingress_modes=[IngressMode(name="public")],  # type: ignore[arg-type]
            egress_modes=[EgressMode(name="internet")],  # type: ignore[arg-type]
        )
        assert isinstance(runway.supported_gpu_types, tuple)
        assert isinstance(runway.supported_architectures, tuple)
        assert isinstance(runway.ingress_modes, tuple)
        assert isinstance(runway.egress_modes, tuple)


# ---------------------------------------------------------------------------
# Format utility tests
# ---------------------------------------------------------------------------


class TestFormatBytes:
    """Tests for format_bytes utility."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0 B"),
            (1023, "1023 B"),
            (1024, "1.0 KiB"),
            (1536, "1.5 KiB"),
            (1048576, "1.0 MiB"),
            (1073741824, "1.0 GiB"),
            (17179869184, "16.0 GiB"),
            (1099511627776, "1.0 TiB"),
        ],
    )
    def test_format_bytes(self, value: int, expected: str) -> None:
        assert format_bytes(value) == expected


class TestFormatCpu:
    """Tests for format_cpu utility."""

    @pytest.mark.parametrize(
        ("millicores", "expected"),
        [
            (0, "0.0 vCPU"),
            (100, "0.1 vCPU"),
            (500, "0.5 vCPU"),
            (1000, "1.0 vCPU"),
            (4000, "4.0 vCPU"),
        ],
    )
    def test_format_cpu(self, millicores: int, expected: str) -> None:
        assert format_cpu(millicores) == expected


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestDiscoveryExceptions:
    """Tests for discovery-specific exceptions."""

    def test_tower_not_found_error(self) -> None:
        exc = TowerNotFoundError("Tower not found: 'abc'", tower_id="abc")
        assert exc.tower_id == "abc"
        assert "Tower not found" in str(exc)
        assert isinstance(exc, CWSandboxError)

    def test_runway_not_found_error_with_tower_id(self) -> None:
        exc = RunwayNotFoundError(
            "Runway not found: 'default'",
            runway_name="default",
            tower_id="tower-1",
        )
        assert exc.runway_name == "default"
        assert exc.tower_id == "tower-1"
        assert isinstance(exc, CWSandboxError)

    def test_runway_not_found_error_without_tower_id(self) -> None:
        exc = RunwayNotFoundError(
            "Runway not found: 'default'",
            runway_name="default",
        )
        assert exc.runway_name == "default"
        assert exc.tower_id is None


class TestDiscoveryErrorHierarchy:
    """Tests for discovery exception inheritance."""

    def test_tower_not_found_is_discovery_error(self) -> None:
        exc = TowerNotFoundError("not found", tower_id="abc")
        assert isinstance(exc, DiscoveryError)
        assert isinstance(exc, CWSandboxError)

    def test_runway_not_found_is_discovery_error(self) -> None:
        exc = RunwayNotFoundError("not found", runway_name="default")
        assert isinstance(exc, DiscoveryError)
        assert isinstance(exc, CWSandboxError)


# ---------------------------------------------------------------------------
# Proto-to-dataclass conversion tests
# ---------------------------------------------------------------------------


class TestTowerFromProto:
    """Tests for _tower_from_proto conversion."""

    def _make_proto(
        self,
        *,
        with_capabilities: bool = True,
        with_resources: bool = False,
    ) -> discovery_pb2.AvailableTower:
        proto = discovery_pb2.AvailableTower(
            tower_id="tower-1",
            tower_group_id="group-1",
            tags=["tag1"],
            healthy=True,
            runway_names=["default"],
        )
        proto.connected_at.FromDatetime(datetime(2025, 6, 15, 12, 0, 0))
        if with_capabilities:
            proto.capabilities.CopyFrom(
                discovery_pb2.TowerCapabilitySummary(
                    max_cpu_millicores=4000,
                    max_memory_bytes=17179869184,
                    max_gpu_count=2,
                    supported_gpu_types=["A100"],
                    supported_architectures=["amd64"],
                    supports_privileged=True,
                    available_storage_classes=["ssd"],
                )
            )
        if with_resources:
            proto.resources.CopyFrom(
                discovery_pb2.TowerResourceSummary(
                    available_cpu_millicores=2000,
                    available_memory_bytes=8589934592,
                    available_gpu_count=1,
                    running_sandboxes=5,
                )
            )
        return proto

    def test_basic_conversion(self) -> None:
        proto = self._make_proto()
        tower = _tower_from_proto(proto)

        assert tower.tower_id == "tower-1"
        assert tower.tower_group_id == "group-1"
        assert tower.tags == ("tag1",)
        assert tower.healthy is True
        assert tower.runway_names == ("default",)
        assert tower.connected_at.tzinfo is UTC
        assert tower.max_cpu_millicores == 4000
        assert tower.max_memory_bytes == 17179869184
        assert tower.max_gpu_count == 2
        assert tower.supported_gpu_types == ("A100",)
        assert tower.supported_architectures == ("amd64",)
        assert tower.supports_privileged is True
        assert tower.available_storage_classes == ("ssd",)
        assert tower.resources is None

    def test_without_capabilities(self) -> None:
        proto = self._make_proto(with_capabilities=False)
        tower = _tower_from_proto(proto)

        assert tower.max_cpu_millicores == 0
        assert tower.max_memory_bytes == 0
        assert tower.max_gpu_count == 0
        assert tower.supported_gpu_types == ()
        assert tower.supported_architectures == ()
        assert tower.supports_privileged is False
        assert tower.available_storage_classes == ()

    def test_with_resources(self) -> None:
        proto = self._make_proto(with_resources=True)
        tower = _tower_from_proto(proto)

        assert tower.resources is not None
        assert tower.resources.available_cpu_millicores == 2000
        assert tower.resources.available_memory_bytes == 8589934592
        assert tower.resources.available_gpu_count == 1
        assert tower.resources.running_sandboxes == 5


class TestRunwayFromProto:
    """Tests for _runway_from_proto conversion."""

    def test_basic_conversion(self) -> None:
        proto = discovery_pb2.RunwaySummary(
            runway_name="default",
            tower_id="tower-1",
            supported_gpu_types=["A100", "H100"],
            supported_architectures=["amd64"],
            service_exposure_modes=[
                discovery_pb2.ServiceExposureMode(name="public"),
            ],
            egress_modes=[
                discovery_pb2.EgressMode(name="internet"),
            ],
        )
        runway = _runway_from_proto(proto)

        assert runway.runway_name == "default"
        assert runway.tower_id == "tower-1"
        assert runway.supported_gpu_types == ("A100", "H100")
        assert runway.supported_architectures == ("amd64",)
        assert runway.ingress_modes == (IngressMode(name="public"),)
        assert runway.egress_modes == (EgressMode(name="internet"),)

    def test_empty_modes(self) -> None:
        proto = discovery_pb2.RunwaySummary(
            runway_name="bare",
            tower_id="tower-2",
        )
        runway = _runway_from_proto(proto)
        assert runway.ingress_modes == ()
        assert runway.egress_modes == ()


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------


class TestPaginateAsync:
    """Tests for _paginate_async."""

    def _run(self, coro: object) -> object:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)  # type: ignore[arg-type]
        finally:
            loop.close()

    def test_single_page(self) -> None:
        response = MagicMock()
        response.items = ["a", "b"]
        response.next_page_token = ""

        rpc = AsyncMock(return_value=response)
        request = MagicMock()
        request.page_token = ""

        result = self._run(_paginate_async(rpc, request, "items", (), timeout=10.0))
        assert result == ["a", "b"]
        rpc.assert_awaited_once()

    def test_multi_page(self) -> None:
        page1 = MagicMock()
        page1.items = ["a"]
        page1.next_page_token = "tok1"

        page2 = MagicMock()
        page2.items = ["b"]
        page2.next_page_token = "tok2"

        page3 = MagicMock()
        page3.items = ["c"]
        page3.next_page_token = ""

        rpc = AsyncMock(side_effect=[page1, page2, page3])
        request = MagicMock()
        request.page_token = ""

        result = self._run(_paginate_async(rpc, request, "items", (), timeout=30.0))
        assert result == ["a", "b", "c"]
        assert rpc.await_count == 3

    def test_empty_response(self) -> None:
        response = MagicMock()
        response.items = []
        response.next_page_token = ""

        rpc = AsyncMock(return_value=response)
        request = MagicMock()
        request.page_token = ""

        result = self._run(_paginate_async(rpc, request, "items", (), timeout=10.0))
        assert result == []

    def test_repeated_token_raises(self) -> None:
        page = MagicMock()
        page.items = ["a"]
        page.next_page_token = "same-token"

        rpc = AsyncMock(return_value=page)
        request = MagicMock()
        request.page_token = ""

        with pytest.raises(CWSandboxError, match="pagination loop"):
            self._run(_paginate_async(rpc, request, "items", (), timeout=10.0))

    def test_max_pages_exceeded(self) -> None:
        """Exceeding the 100-page limit raises CWSandboxError."""

        call_count = 0

        async def _fake_rpc(req: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.items = [call_count]
            resp.next_page_token = f"tok-{call_count}"
            return resp

        rpc = _fake_rpc
        request = MagicMock()
        request.page_token = ""

        with pytest.raises(CWSandboxError, match="exceeded 100 pages"):
            self._run(_paginate_async(rpc, request, "items", (), timeout=600.0))

    def test_timeout_during_pagination(self) -> None:
        """A deadline that expires mid-pagination raises CWSandboxError."""

        async def _slow_rpc(req: object, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.items = ["a"]
            resp.next_page_token = "next"
            return resp

        request = MagicMock()
        request.page_token = ""

        with pytest.raises(CWSandboxError, match="timed out"):
            self._run(_paginate_async(_slow_rpc, request, "items", (), timeout=0.0))


# ---------------------------------------------------------------------------
# Base URL resolution tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation on public functions."""

    def test_get_tower_empty_string(self) -> None:
        with pytest.raises(ValueError, match="tower_id must not be empty"):
            get_tower("")

    def test_get_tower_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="tower_id must not be empty"):
            get_tower("   ")

    def test_get_runway_empty_string(self) -> None:
        with pytest.raises(ValueError, match="runway_name must not be empty"):
            get_runway("")

    def test_get_runway_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="runway_name must not be empty"):
            get_runway("  ")


# ---------------------------------------------------------------------------
# gRPC client tests (mocked)
# ---------------------------------------------------------------------------

# Shared patch targets
_PATCH_LM = "cwsandbox._discovery._LoopManager"
_PATCH_CHANNEL = "cwsandbox._discovery.create_channel"
_PATCH_AUTH = "cwsandbox._discovery.resolve_auth_metadata"
_PATCH_PARSE = "cwsandbox._discovery.parse_grpc_target"
_PATCH_STUB = "cwsandbox._discovery.discovery_pb2_grpc.DiscoveryServiceStub"


def _setup_grpc_mocks(
    mock_lm: MagicMock,
    mock_channel: MagicMock,
    mock_auth: MagicMock,
    mock_parse: MagicMock,
) -> MagicMock:
    """Wire up the standard gRPC mock plumbing.

    Returns the channel mock (an AsyncMock) so callers can set up stub
    expectations via the stub constructor mock.
    """
    loop = asyncio.new_event_loop()

    def _run_async_side_effect(coro: Any) -> concurrent.futures.Future[Any]:
        """Simulate run_async by returning a Future wrapping the coroutine."""
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            result = loop.run_until_complete(coro)
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)
        return future

    mock_lm.get.return_value.run_async.side_effect = _run_async_side_effect
    mock_parse.return_value = ("test-target:443", True)
    channel = AsyncMock()
    mock_channel.return_value = channel
    mock_auth.return_value = (("authorization", "Bearer test-key"),)
    return channel


def _make_tower_proto() -> discovery_pb2.AvailableTower:
    """Build a minimal AvailableTower proto for stub responses."""
    proto = discovery_pb2.AvailableTower(
        tower_id="tower-abc",
        tower_group_id="group-1",
        tags=["test"],
        healthy=True,
        runway_names=["default"],
    )
    proto.connected_at.FromDatetime(datetime(2025, 1, 1))
    proto.capabilities.CopyFrom(
        discovery_pb2.TowerCapabilitySummary(
            max_cpu_millicores=8000,
            max_memory_bytes=34359738368,
            max_gpu_count=4,
            supported_gpu_types=["A100"],
            supported_architectures=["amd64"],
            supports_privileged=True,
            available_storage_classes=["ssd"],
        )
    )
    return proto


def _make_runway_proto() -> discovery_pb2.RunwaySummary:
    """Build a minimal RunwaySummary proto for stub responses."""
    return discovery_pb2.RunwaySummary(
        runway_name="default",
        tower_id="tower-abc",
        supported_gpu_types=["A100"],
        supported_architectures=["amd64"],
        service_exposure_modes=[
            discovery_pb2.ServiceExposureMode(name="public"),
        ],
        egress_modes=[
            discovery_pb2.EgressMode(name="internet"),
        ],
    )


class TestListTowers:
    """Tests for list_towers public function."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_basic_view(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [_make_tower_proto()]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        channel = mock_channel.return_value
        result = list_towers()

        assert len(result) == 1
        assert result[0].tower_id == "tower-abc"
        # Verify basic view is the default
        call_args = stub.ListAvailableTowers.call_args
        request = call_args[0][0]
        assert request.view == discovery_pb2.TOWER_VIEW_BASIC
        channel.close.assert_awaited_once_with(grace=None)

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_full_view_with_resources(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [_make_tower_proto()]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_towers(include_resources=True)

        call_args = stub.ListAvailableTowers.call_args
        request = call_args[0][0]
        assert request.view == discovery_pb2.TOWER_VIEW_FULL

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_filters_propagated(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = []
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_towers(
            tower_group_id="grp",
            runway_name="rw",
            gpu_type="H100",
            architecture="arm64",
        )

        call_args = stub.ListAvailableTowers.call_args
        request = call_args[0][0]
        assert request.tower_group_id == "grp"
        assert request.runway_name == "rw"
        assert request.gpu_type == "H100"
        assert request.architecture == "arm64"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_auth_metadata_sent(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = []
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_towers()

        call_kwargs = stub.ListAvailableTowers.call_args[1]
        assert call_kwargs["metadata"] == (("authorization", "Bearer test-key"),)


class TestGetTower:
    """Tests for get_tower public function."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_returns_tower(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        stub = MagicMock()
        stub.GetAvailableTower = AsyncMock(return_value=_make_tower_proto())
        mock_stub_cls.return_value = stub

        tower = get_tower("tower-abc")

        assert tower.tower_id == "tower-abc"
        call_args = stub.GetAvailableTower.call_args
        request = call_args[0][0]
        assert request.tower_id == "tower-abc"
        assert request.view == discovery_pb2.TOWER_VIEW_FULL

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_not_found_raises_tower_not_found(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.NOT_FOUND,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="not found",
        )
        stub = MagicMock()
        stub.GetAvailableTower = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(TowerNotFoundError) as exc_info:
            get_tower("missing")
        assert exc_info.value.tower_id == "missing"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_unauthenticated_raises_auth_error(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAUTHENTICATED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="invalid token",
        )
        stub = MagicMock()
        stub.GetAvailableTower = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(CWSandboxAuthenticationError, match="invalid token"):
            get_tower("tower-1")


class TestListRunways:
    """Tests for list_runways public function."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_returns_runways(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runways = [_make_runway_proto()]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListRunways = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runways()

        assert len(result) == 1
        assert result[0].runway_name == "default"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_filters_propagated(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runways = []
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListRunways = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_runways(
            gpu_type="H100",
            architecture="arm64",
            tower_id="tower-1",
        )

        call_args = stub.ListRunways.call_args
        request = call_args[0][0]
        assert request.gpu_type == "H100"
        assert request.architecture == "arm64"
        assert request.tower_id == "tower-1"


class TestGetRunway:
    """Tests for get_runway public function."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_returns_runway(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        stub = MagicMock()
        stub.GetRunway = AsyncMock(return_value=_make_runway_proto())
        mock_stub_cls.return_value = stub

        runway = get_runway("default")

        assert runway.runway_name == "default"
        call_args = stub.GetRunway.call_args
        request = call_args[0][0]
        assert request.runway_name == "default"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_with_tower_id(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        stub = MagicMock()
        stub.GetRunway = AsyncMock(return_value=_make_runway_proto())
        mock_stub_cls.return_value = stub

        get_runway("default", tower_id="tower-abc")

        call_args = stub.GetRunway.call_args
        request = call_args[0][0]
        assert request.tower_id == "tower-abc"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_not_found_raises_runway_not_found(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.NOT_FOUND,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="not found",
        )
        stub = MagicMock()
        stub.GetRunway = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(RunwayNotFoundError) as exc_info:
            get_runway("missing", tower_id="tower-1")
        assert exc_info.value.runway_name == "missing"
        assert exc_info.value.tower_id == "tower-1"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_unavailable_raises_discovery_error(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="connection refused",
        )
        stub = MagicMock()
        stub.GetRunway = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        channel = mock_channel.return_value
        with pytest.raises(DiscoveryError, match="unavailable"):
            get_runway("default")
        channel.close.assert_awaited_once_with(grace=None)

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_deadline_exceeded_raises_discovery_error(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.DEADLINE_EXCEEDED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="timeout",
        )
        stub = MagicMock()
        stub.GetRunway = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(DiscoveryError, match="timed out"):
            get_runway("default")


# ---------------------------------------------------------------------------
# Client-side filtering tests - list_runways
# ---------------------------------------------------------------------------

_PATCH_LIST_RUNWAYS_ASYNC = "cwsandbox._discovery._list_runways_async"


def _make_runway_proto_with_modes(
    name: str,
    tower_id: str,
    ingress_names: list[str],
    egress_names: list[str],
) -> discovery_pb2.RunwaySummary:
    """Build a RunwaySummary proto with specific ingress/egress modes."""
    return discovery_pb2.RunwaySummary(
        runway_name=name,
        tower_id=tower_id,
        supported_gpu_types=["A100"],
        supported_architectures=["amd64"],
        service_exposure_modes=[discovery_pb2.ServiceExposureMode(name=n) for n in ingress_names],
        egress_modes=[discovery_pb2.EgressMode(name=n) for n in egress_names],
    )


class TestListRunwaysClientSideFilters:
    """Tests for client-side ingress/egress mode filtering on list_runways."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_ingress_mode_filter(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runways = [
            _make_runway_proto_with_modes("rw-1", "t-1", ["public", "private"], ["internet"]),
            _make_runway_proto_with_modes("rw-2", "t-1", ["public"], ["internet"]),
            _make_runway_proto_with_modes("rw-3", "t-2", ["private"], ["internet"]),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListRunways = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runways(ingress_mode="public")

        assert len(result) == 2
        assert {r.runway_name for r in result} == {"rw-1", "rw-2"}

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_egress_mode_filter(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runways = [
            _make_runway_proto_with_modes("rw-1", "t-1", ["public"], ["internet"]),
            _make_runway_proto_with_modes("rw-2", "t-1", ["public"], ["blocked"]),
            _make_runway_proto_with_modes("rw-3", "t-2", ["public"], ["internet", "blocked"]),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListRunways = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runways(egress_mode="blocked")

        assert len(result) == 2
        assert {r.runway_name for r in result} == {"rw-2", "rw-3"}

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_both_mode_filters(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runways = [
            _make_runway_proto_with_modes("rw-1", "t-1", ["public"], ["internet"]),
            _make_runway_proto_with_modes("rw-2", "t-1", ["public"], ["blocked"]),
            _make_runway_proto_with_modes("rw-3", "t-2", ["private"], ["internet"]),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListRunways = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runways(ingress_mode="public", egress_mode="internet")

        assert len(result) == 1
        assert result[0].runway_name == "rw-1"


# ---------------------------------------------------------------------------
# Client-side filtering tests - list_towers
# ---------------------------------------------------------------------------


def _make_tower_proto_with_resources(
    tower_id: str,
    cpu: int,
    memory: int,
    gpu: int,
    running: int = 0,
) -> discovery_pb2.AvailableTower:
    """Build a tower proto with resource availability."""
    proto = discovery_pb2.AvailableTower(
        tower_id=tower_id,
        tower_group_id="group-1",
        tags=["test"],
        healthy=True,
        runway_names=["default"],
    )
    proto.connected_at.FromDatetime(datetime(2025, 1, 1))
    proto.capabilities.CopyFrom(
        discovery_pb2.TowerCapabilitySummary(
            max_cpu_millicores=16000,
            max_memory_bytes=68719476736,
            max_gpu_count=8,
            supported_gpu_types=["A100"],
            supported_architectures=["amd64"],
            supports_privileged=True,
            available_storage_classes=["ssd"],
        )
    )
    proto.resources.CopyFrom(
        discovery_pb2.TowerResourceSummary(
            available_cpu_millicores=cpu,
            available_memory_bytes=memory,
            available_gpu_count=gpu,
            running_sandboxes=running,
        )
    )
    return proto


def _make_tower_proto_no_resources(tower_id: str) -> discovery_pb2.AvailableTower:
    """Build a tower proto without resource information."""
    proto = discovery_pb2.AvailableTower(
        tower_id=tower_id,
        tower_group_id="group-1",
        tags=["test"],
        healthy=True,
        runway_names=["default"],
    )
    proto.connected_at.FromDatetime(datetime(2025, 1, 1))
    proto.capabilities.CopyFrom(
        discovery_pb2.TowerCapabilitySummary(
            max_cpu_millicores=8000,
            max_memory_bytes=34359738368,
            max_gpu_count=4,
            supported_gpu_types=["A100"],
            supported_architectures=["amd64"],
            supports_privileged=True,
            available_storage_classes=["ssd"],
        )
    )
    return proto


class TestListTowersClientSideFilters:
    """Tests for client-side capacity and networking filtering on list_towers."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_min_cpu_filter(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_tower_proto_with_resources("t-2", cpu=1000, memory=8_000_000_000, gpu=2),
            _make_tower_proto_with_resources("t-3", cpu=8000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_towers(min_available_cpu_millicores=4000)

        assert len(result) == 2
        assert {t.tower_id for t in result} == {"t-1", "t-3"}

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_min_memory_filter(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=16_000_000_000, gpu=0),
            _make_tower_proto_with_resources("t-2", cpu=4000, memory=4_000_000_000, gpu=0),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_towers(min_available_memory_bytes=8_000_000_000)

        assert len(result) == 1
        assert result[0].tower_id == "t-1"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_min_gpu_filter(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=4),
            _make_tower_proto_with_resources("t-2", cpu=4000, memory=8_000_000_000, gpu=1),
            _make_tower_proto_with_resources("t-3", cpu=4000, memory=8_000_000_000, gpu=0),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_towers(min_available_gpu_count=2)

        assert len(result) == 1
        assert result[0].tower_id == "t-1"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_capacity_filter_auto_enables_resources(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_towers(min_available_cpu_millicores=1)

        call_args = stub.ListAvailableTowers.call_args
        request = call_args[0][0]
        assert request.view == discovery_pb2.TOWER_VIEW_FULL

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_capacity_filter_excludes_none_resources(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_tower_proto_no_resources("t-2"),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_towers(min_available_cpu_millicores=1)

        assert len(result) == 1
        assert result[0].tower_id == "t-1"

    @patch(_PATCH_LIST_RUNWAYS_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_ingress_mode_filter_on_towers(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_runways_async: AsyncMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_tower_proto_with_resources("t-2", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_runways_async.return_value = [
            Runway(
                runway_name="rw-1",
                tower_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Runway(
                runway_name="rw-2",
                tower_id="t-2",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="private"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
        ]

        result = list_towers(ingress_mode="public")

        assert len(result) == 1
        assert result[0].tower_id == "t-1"
        mock_list_runways_async.assert_awaited_once()

    @patch(_PATCH_LIST_RUNWAYS_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_egress_mode_filter_on_towers(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_runways_async: AsyncMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_tower_proto_with_resources("t-2", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_tower_proto_with_resources("t-3", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_runways_async.return_value = [
            Runway(
                runway_name="rw-1",
                tower_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Runway(
                runway_name="rw-2",
                tower_id="t-2",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="public"),),
                egress_modes=(EgressMode(name="blocked"),),
            ),
            Runway(
                runway_name="rw-3",
                tower_id="t-3",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="public"),),
                egress_modes=(EgressMode(name="internet"), EgressMode(name="blocked")),
            ),
        ]

        result = list_towers(egress_mode="internet")

        assert len(result) == 2
        assert {t.tower_id for t in result} == {"t-1", "t-3"}
        mock_list_runways_async.assert_awaited_once()

    @patch(_PATCH_LIST_RUNWAYS_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_capacity_plus_ingress_mode_combined(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_runways_async: AsyncMock,
    ) -> None:
        """f-10: Combined capacity and network mode filters."""
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=8000, memory=16_000_000_000, gpu=4),
            _make_tower_proto_with_resources("t-2", cpu=2000, memory=4_000_000_000, gpu=1),
            _make_tower_proto_with_resources("t-3", cpu=8000, memory=16_000_000_000, gpu=4),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_runways_async.return_value = [
            Runway(
                runway_name="rw-1",
                tower_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Runway(
                runway_name="rw-2",
                tower_id="t-2",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Runway(
                runway_name="rw-3",
                tower_id="t-3",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="private"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
        ]

        result = list_towers(min_available_cpu_millicores=4000, ingress_mode="public")

        assert len(result) == 1
        assert result[0].tower_id == "t-1"
        mock_list_runways_async.assert_awaited_once()

    @patch(_PATCH_LIST_RUNWAYS_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_list_runways_async_failure_propagates(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_runways_async: AsyncMock,
    ) -> None:
        """f-11: Error in _list_runways_async propagates through list_towers."""
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_runways_async.side_effect = CWSandboxError("runway fetch failed")

        with pytest.raises(CWSandboxError, match="runway fetch failed"):
            list_towers(ingress_mode="public")

    @patch(_PATCH_LIST_RUNWAYS_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_timeout_budget_shared_with_runway_fetch(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_runways_async: AsyncMock,
    ) -> None:
        """g-14: Verify remaining time (not original timeout) is passed to _list_runways_async."""
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.towers = [
            _make_tower_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_runways_async.return_value = [
            Runway(
                runway_name="rw-1",
                tower_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                ingress_modes=(IngressMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
        ]

        # Simulate 7 seconds elapsing between deadline set and runway fetch.
        # With a 30s timeout, the remaining budget should be ~23s, not 30s.
        call_count = 0

        def advancing_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            # First calls: deadline creation and paginate timeout
            # Later calls: the remaining-time check before _list_runways_async
            return 1000.0 + (call_count - 1) * 3.5

        with patch("cwsandbox._discovery.time") as mock_time:
            mock_time.monotonic = advancing_monotonic
            list_towers(ingress_mode="public")

        # _list_runways_async receives a timeout arg (positional arg index 2)
        runway_call_args = mock_list_runways_async.call_args
        timeout_passed = runway_call_args[0][2]
        assert timeout_passed < 30.0, f"Expected remaining budget < 30s, got {timeout_passed}"
        assert timeout_passed > 0, "Timeout budget must be positive"


# ---------------------------------------------------------------------------
# gRPC error path tests for list functions
# ---------------------------------------------------------------------------


class TestListTowersGrpcErrors:
    """f-9: gRPC error path coverage for list_towers."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_unavailable_raises_discovery_error(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="connection refused",
        )
        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(DiscoveryError, match="unavailable"):
            list_towers()


class TestListRunwaysGrpcErrors:
    """f-9: gRPC error path coverage for list_runways."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_unavailable_raises_discovery_error(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="connection refused",
        )
        stub = MagicMock()
        stub.ListRunways = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        channel = mock_channel.return_value
        with pytest.raises(DiscoveryError, match="unavailable"):
            list_runways()
        channel.close.assert_awaited_once_with(grace=None)


# ---------------------------------------------------------------------------
# Multi-page pagination end-to-end tests
# ---------------------------------------------------------------------------


class TestMultiPagePagination:
    """f-12: Multi-page end-to-end tests through public list functions."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_list_towers_multi_page(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        page1 = MagicMock()
        page1.towers = [_make_tower_proto()]
        page1.next_page_token = "page2"

        page2 = MagicMock()
        page2.towers = [_make_tower_proto()]
        page2.towers[0].tower_id = "tower-def"
        page2.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableTowers = AsyncMock(side_effect=[page1, page2])
        mock_stub_cls.return_value = stub

        result = list_towers()

        assert len(result) == 2
        assert {t.tower_id for t in result} == {"tower-abc", "tower-def"}
        assert stub.ListAvailableTowers.call_count == 2

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_list_runways_multi_page(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        page1 = MagicMock()
        page1.runways = [_make_runway_proto()]
        page1.next_page_token = "page2"

        page2 = MagicMock()
        page2.runways = [_make_runway_proto()]
        page2.runways[0].runway_name = "gpu-runway"
        page2.next_page_token = ""

        stub = MagicMock()
        stub.ListRunways = AsyncMock(side_effect=[page1, page2])
        mock_stub_cls.return_value = stub

        result = list_runways()

        assert len(result) == 2
        assert {r.runway_name for r in result} == {"default", "gpu-runway"}
        assert stub.ListRunways.call_count == 2


# ---------------------------------------------------------------------------
# get_runway NOT_FOUND without tower_id
# ---------------------------------------------------------------------------


class TestGetRunwayNotFoundVariants:
    """f-13: get_runway NOT_FOUND with tower_id=None."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_not_found_without_tower_id(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.NOT_FOUND,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="not found",
        )
        stub = MagicMock()
        stub.GetRunway = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(RunwayNotFoundError) as exc_info:
            get_runway("missing")
        assert exc_info.value.runway_name == "missing"
        assert exc_info.value.tower_id is None
