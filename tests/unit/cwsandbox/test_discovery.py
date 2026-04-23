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
    Profile,
    Runner,
    RunnerResources,
    ServiceExposureMode,
    _profile_from_proto,
    _runner_from_proto,
    format_bytes,
    format_cpu,
    get_profile,
    get_runner,
    list_profiles,
    list_runners,
)
from cwsandbox._proto import discovery_pb2
from cwsandbox.exceptions import (
    CWSandboxAuthenticationError,
    CWSandboxError,
    DiscoveryError,
    ProfileNotFoundError,
    RunnerNotFoundError,
)

# ---------------------------------------------------------------------------
# Type tests
# ---------------------------------------------------------------------------


class TestRunnerResources:
    """Tests for RunnerResources dataclass."""

    def test_creation(self) -> None:
        res = RunnerResources(
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
        res = RunnerResources(
            available_cpu_millicores=2000,
            available_memory_bytes=0,
            available_gpu_count=0,
            running_sandboxes=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.available_cpu_millicores = 9999  # type: ignore[misc]


class TestServiceExposureMode:
    """Tests for ServiceExposureMode dataclass."""

    def test_creation(self) -> None:
        mode = ServiceExposureMode(name="public")
        assert mode.name == "public"

    def test_frozen(self) -> None:
        mode = ServiceExposureMode(name="public")
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


class TestRunner:
    """Tests for Runner dataclass."""

    def _make_runner(self, **overrides: object) -> Runner:
        defaults: dict[str, object] = {
            "runner_id": "runner-1",
            "runner_group_id": "group-1",
            "tags": ("tag1", "tag2"),
            "healthy": True,
            "profile_names": ("default",),
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
        return Runner(**defaults)  # type: ignore[arg-type]

    def test_creation_all_fields(self) -> None:
        runner = self._make_runner()
        assert runner.runner_id == "runner-1"
        assert runner.runner_group_id == "group-1"
        assert runner.tags == ("tag1", "tag2")
        assert runner.healthy is True
        assert runner.profile_names == ("default",)
        assert runner.connected_at == datetime(2025, 1, 1, tzinfo=UTC)
        assert runner.max_cpu_millicores == 4000
        assert runner.max_memory_bytes == 17179869184
        assert runner.max_gpu_count == 2
        assert runner.supported_gpu_types == ("A100",)
        assert runner.supported_architectures == ("amd64",)
        assert runner.supports_privileged is True
        assert runner.available_storage_classes == ("ssd",)
        assert runner.resources is None

    def test_frozen(self) -> None:
        runner = self._make_runner()
        with pytest.raises(dataclasses.FrozenInstanceError):
            runner.runner_id = "other"  # type: ignore[misc]

    def test_resources_populated(self) -> None:
        res = RunnerResources(
            available_cpu_millicores=2000,
            available_memory_bytes=8589934592,
            available_gpu_count=1,
            running_sandboxes=3,
        )
        runner = self._make_runner(resources=res)
        assert runner.resources is res
        assert runner.resources.running_sandboxes == 3

    def test_repr_human_readable(self) -> None:
        runner = self._make_runner(
            max_cpu_millicores=4000,
            max_memory_bytes=17179869184,
            max_gpu_count=2,
            profile_names=("default", "gpu"),
        )
        r = repr(runner)
        assert "runner_id='runner-1'" in r
        assert "healthy=True" in r
        assert "cpu=4.0 vCPU" in r
        assert "memory=16.0 GiB" in r
        assert "gpus=2" in r
        assert "['default', 'gpu']" in r

    def test_post_init_normalizes_lists_to_tuples(self) -> None:
        runner = Runner(
            runner_id="t",
            runner_group_id="g",
            tags=["a", "b"],  # type: ignore[arg-type]
            healthy=True,
            profile_names=["r1"],  # type: ignore[arg-type]
            connected_at=datetime(2025, 1, 1, tzinfo=UTC),
            max_cpu_millicores=0,
            max_memory_bytes=0,
            max_gpu_count=0,
            supported_gpu_types=["H100"],  # type: ignore[arg-type]
            supported_architectures=["arm64"],  # type: ignore[arg-type]
            supports_privileged=False,
            available_storage_classes=["nfs"],  # type: ignore[arg-type]
        )
        assert isinstance(runner.tags, tuple)
        assert isinstance(runner.profile_names, tuple)
        assert isinstance(runner.supported_gpu_types, tuple)
        assert isinstance(runner.supported_architectures, tuple)
        assert isinstance(runner.available_storage_classes, tuple)

    def test_post_init_preserves_tuples(self) -> None:
        tags = ("x",)
        runner = self._make_runner(tags=tags)
        assert runner.tags is tags


class TestProfile:
    """Tests for Profile dataclass."""

    def test_creation(self) -> None:
        profile = Profile(
            profile_name="default",
            runner_id="runner-1",
            supported_gpu_types=("A100",),
            supported_architectures=("amd64",),
            service_exposure_modes=(ServiceExposureMode(name="public"),),
            egress_modes=(EgressMode(name="internet"),),
        )
        assert profile.profile_name == "default"
        assert profile.runner_id == "runner-1"
        assert profile.service_exposure_modes == (ServiceExposureMode(name="public"),)
        assert profile.egress_modes == (EgressMode(name="internet"),)

    def test_post_init_normalizes_lists(self) -> None:
        profile = Profile(
            profile_name="r",
            runner_id="t",
            supported_gpu_types=["A100"],  # type: ignore[arg-type]
            supported_architectures=["amd64"],  # type: ignore[arg-type]
            service_exposure_modes=[ServiceExposureMode(name="public")],  # type: ignore[arg-type]
            egress_modes=[EgressMode(name="internet")],  # type: ignore[arg-type]
        )
        assert isinstance(profile.supported_gpu_types, tuple)
        assert isinstance(profile.supported_architectures, tuple)
        assert isinstance(profile.service_exposure_modes, tuple)
        assert isinstance(profile.egress_modes, tuple)


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

    def test_runner_not_found_error(self) -> None:
        exc = RunnerNotFoundError("Runner not found: 'abc'", runner_id="abc")
        assert exc.runner_id == "abc"
        assert "Runner not found" in str(exc)
        assert isinstance(exc, CWSandboxError)

    def test_profile_not_found_error_with_runner_id(self) -> None:
        exc = ProfileNotFoundError(
            "Profile not found: 'default'",
            profile_name="default",
            runner_id="runner-1",
        )
        assert exc.profile_name == "default"
        assert exc.runner_id == "runner-1"
        assert isinstance(exc, CWSandboxError)

    def test_profile_not_found_error_without_runner_id(self) -> None:
        exc = ProfileNotFoundError(
            "Profile not found: 'default'",
            profile_name="default",
        )
        assert exc.profile_name == "default"
        assert exc.runner_id is None


class TestDiscoveryErrorHierarchy:
    """Tests for discovery exception inheritance."""

    def test_runner_not_found_is_discovery_error(self) -> None:
        exc = RunnerNotFoundError("not found", runner_id="abc")
        assert isinstance(exc, DiscoveryError)
        assert isinstance(exc, CWSandboxError)

    def test_profile_not_found_is_discovery_error(self) -> None:
        exc = ProfileNotFoundError("not found", profile_name="default")
        assert isinstance(exc, DiscoveryError)
        assert isinstance(exc, CWSandboxError)


# ---------------------------------------------------------------------------
# Proto-to-dataclass conversion tests
# ---------------------------------------------------------------------------


class TestRunnerFromProto:
    """Tests for _runner_from_proto conversion."""

    def _make_proto(
        self,
        *,
        with_capabilities: bool = True,
        with_resources: bool = False,
    ) -> discovery_pb2.AvailableRunner:
        proto = discovery_pb2.AvailableRunner(
            runner_id="runner-1",
            runner_group_id="group-1",
            tags=["tag1"],
            healthy=True,
            profile_names=["default"],
        )
        proto.connected_at.FromDatetime(datetime(2025, 6, 15, 12, 0, 0))
        if with_capabilities:
            proto.capabilities.CopyFrom(
                discovery_pb2.RunnerCapabilitySummary(
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
                discovery_pb2.RunnerResourceSummary(
                    available_cpu_millicores=2000,
                    available_memory_bytes=8589934592,
                    available_gpu_count=1,
                    running_sandboxes=5,
                )
            )
        return proto

    def test_basic_conversion(self) -> None:
        proto = self._make_proto()
        runner = _runner_from_proto(proto)

        assert runner.runner_id == "runner-1"
        assert runner.runner_group_id == "group-1"
        assert runner.tags == ("tag1",)
        assert runner.healthy is True
        assert runner.profile_names == ("default",)
        assert runner.connected_at.tzinfo is UTC
        assert runner.max_cpu_millicores == 4000
        assert runner.max_memory_bytes == 17179869184
        assert runner.max_gpu_count == 2
        assert runner.supported_gpu_types == ("A100",)
        assert runner.supported_architectures == ("amd64",)
        assert runner.supports_privileged is True
        assert runner.available_storage_classes == ("ssd",)
        assert runner.resources is None

    def test_without_capabilities(self) -> None:
        proto = self._make_proto(with_capabilities=False)
        runner = _runner_from_proto(proto)

        assert runner.max_cpu_millicores == 0
        assert runner.max_memory_bytes == 0
        assert runner.max_gpu_count == 0
        assert runner.supported_gpu_types == ()
        assert runner.supported_architectures == ()
        assert runner.supports_privileged is False
        assert runner.available_storage_classes == ()

    def test_with_resources(self) -> None:
        proto = self._make_proto(with_resources=True)
        runner = _runner_from_proto(proto)

        assert runner.resources is not None
        assert runner.resources.available_cpu_millicores == 2000
        assert runner.resources.available_memory_bytes == 8589934592
        assert runner.resources.available_gpu_count == 1
        assert runner.resources.running_sandboxes == 5


class TestProfileFromProto:
    """Tests for _profile_from_proto conversion."""

    def test_basic_conversion(self) -> None:
        proto = discovery_pb2.ProfileSummary(
            profile_name="default",
            runner_id="runner-1",
            supported_gpu_types=["A100", "H100"],
            supported_architectures=["amd64"],
            service_exposure_modes=[
                discovery_pb2.ServiceExposureMode(name="public"),
            ],
            egress_modes=[
                discovery_pb2.EgressMode(name="internet"),
            ],
        )
        profile = _profile_from_proto(proto)

        assert profile.profile_name == "default"
        assert profile.runner_id == "runner-1"
        assert profile.supported_gpu_types == ("A100", "H100")
        assert profile.supported_architectures == ("amd64",)
        assert profile.service_exposure_modes == (ServiceExposureMode(name="public"),)
        assert profile.egress_modes == (EgressMode(name="internet"),)

    def test_empty_modes(self) -> None:
        proto = discovery_pb2.ProfileSummary(
            profile_name="bare",
            runner_id="runner-2",
        )
        profile = _profile_from_proto(proto)
        assert profile.service_exposure_modes == ()
        assert profile.egress_modes == ()


# ---------------------------------------------------------------------------
# Base URL resolution tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation on public functions."""

    def test_get_runner_empty_string(self) -> None:
        with pytest.raises(ValueError, match="runner_id must not be empty"):
            get_runner("")

    def test_get_runner_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="runner_id must not be empty"):
            get_runner("   ")

    def test_get_profile_empty_string(self) -> None:
        with pytest.raises(ValueError, match="profile_name must not be empty"):
            get_profile("")

    def test_get_profile_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="profile_name must not be empty"):
            get_profile("  ")


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


def _make_runner_proto() -> discovery_pb2.AvailableRunner:
    """Build a minimal AvailableRunner proto for stub responses."""
    proto = discovery_pb2.AvailableRunner(
        runner_id="runner-abc",
        runner_group_id="group-1",
        tags=["test"],
        healthy=True,
        profile_names=["default"],
    )
    proto.connected_at.FromDatetime(datetime(2025, 1, 1))
    proto.capabilities.CopyFrom(
        discovery_pb2.RunnerCapabilitySummary(
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


def _make_profile_proto() -> discovery_pb2.ProfileSummary:
    """Build a minimal ProfileSummary proto for stub responses."""
    return discovery_pb2.ProfileSummary(
        profile_name="default",
        runner_id="runner-abc",
        supported_gpu_types=["A100"],
        supported_architectures=["amd64"],
        service_exposure_modes=[
            discovery_pb2.ServiceExposureMode(name="public"),
        ],
        egress_modes=[
            discovery_pb2.EgressMode(name="internet"),
        ],
    )


class TestListRunners:
    """Tests for list_runners public function."""

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
        response.runners = [_make_runner_proto()]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        channel = mock_channel.return_value
        result = list_runners()

        assert len(result) == 1
        assert result[0].runner_id == "runner-abc"
        # Verify basic view is the default
        call_args = stub.ListAvailableRunners.call_args
        request = call_args[0][0]
        assert request.view == discovery_pb2.RUNNER_VIEW_BASIC
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
        response.runners = [_make_runner_proto()]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_runners(include_resources=True)

        call_args = stub.ListAvailableRunners.call_args
        request = call_args[0][0]
        assert request.view == discovery_pb2.RUNNER_VIEW_FULL

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
        response.runners = []
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_runners(
            runner_group_id="grp",
            profile_name="rw",
            gpu_type="H100",
            architecture="arm64",
        )

        call_args = stub.ListAvailableRunners.call_args
        request = call_args[0][0]
        assert request.runner_group_id == "grp"
        assert request.profile_name == "rw"
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
        response.runners = []
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_runners()

        call_kwargs = stub.ListAvailableRunners.call_args[1]
        assert call_kwargs["metadata"] == (("authorization", "Bearer test-key"),)


class TestGetRunner:
    """Tests for get_runner public function."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_returns_runner(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        stub = MagicMock()
        stub.GetAvailableRunner = AsyncMock(return_value=_make_runner_proto())
        mock_stub_cls.return_value = stub

        runner = get_runner("runner-abc")

        assert runner.runner_id == "runner-abc"
        call_args = stub.GetAvailableRunner.call_args
        request = call_args[0][0]
        assert request.runner_id == "runner-abc"
        assert request.view == discovery_pb2.RUNNER_VIEW_FULL

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_not_found_raises_runner_not_found(
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
        stub.GetAvailableRunner = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(RunnerNotFoundError) as exc_info:
            get_runner("missing")
        assert exc_info.value.runner_id == "missing"

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
        stub.GetAvailableRunner = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(CWSandboxAuthenticationError, match="invalid token"):
            get_runner("runner-1")


class TestListProfiles:
    """Tests for list_profiles public function."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_returns_profiles(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.profiles = [_make_profile_proto()]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListProfiles = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_profiles()

        assert len(result) == 1
        assert result[0].profile_name == "default"

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
        response.profiles = []
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListProfiles = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_profiles(
            gpu_type="H100",
            architecture="arm64",
            runner_id="runner-1",
        )

        call_args = stub.ListProfiles.call_args
        request = call_args[0][0]
        assert request.gpu_type == "H100"
        assert request.architecture == "arm64"
        assert request.runner_id == "runner-1"


class TestGetProfile:
    """Tests for get_profile public function."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_returns_profile(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        stub = MagicMock()
        stub.GetProfile = AsyncMock(return_value=_make_profile_proto())
        mock_stub_cls.return_value = stub

        profile = get_profile("default")

        assert profile.profile_name == "default"
        call_args = stub.GetProfile.call_args
        request = call_args[0][0]
        assert request.profile_name == "default"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_with_runner_id(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        stub = MagicMock()
        stub.GetProfile = AsyncMock(return_value=_make_profile_proto())
        mock_stub_cls.return_value = stub

        get_profile("default", runner_id="runner-abc")

        call_args = stub.GetProfile.call_args
        request = call_args[0][0]
        assert request.runner_id == "runner-abc"

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_not_found_raises_profile_not_found(
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
        stub.GetProfile = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(ProfileNotFoundError) as exc_info:
            get_profile("missing", runner_id="runner-1")
        assert exc_info.value.profile_name == "missing"
        assert exc_info.value.runner_id == "runner-1"

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
        stub.GetProfile = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        channel = mock_channel.return_value
        with pytest.raises(DiscoveryError, match="unavailable"):
            get_profile("default")
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
        stub.GetProfile = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(DiscoveryError, match="timed out"):
            get_profile("default")


# ---------------------------------------------------------------------------
# Client-side filtering tests - list_profiles
# ---------------------------------------------------------------------------

_PATCH_LIST_PROFILES_ASYNC = "cwsandbox._discovery._list_profiles_async"


def _make_profile_proto_with_modes(
    name: str,
    runner_id: str,
    ingress_names: list[str],
    egress_names: list[str],
) -> discovery_pb2.ProfileSummary:
    """Build a ProfileSummary proto with specific ingress/egress modes."""
    return discovery_pb2.ProfileSummary(
        profile_name=name,
        runner_id=runner_id,
        supported_gpu_types=["A100"],
        supported_architectures=["amd64"],
        service_exposure_modes=[discovery_pb2.ServiceExposureMode(name=n) for n in ingress_names],
        egress_modes=[discovery_pb2.EgressMode(name=n) for n in egress_names],
    )


class TestListProfilesClientSideFilters:
    """Tests for client-side ingress/egress mode filtering on list_profiles."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_service_exposure_mode_filter(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.profiles = [
            _make_profile_proto_with_modes("prof-1", "t-1", ["public", "private"], ["internet"]),
            _make_profile_proto_with_modes("prof-2", "t-1", ["public"], ["internet"]),
            _make_profile_proto_with_modes("prof-3", "t-2", ["private"], ["internet"]),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListProfiles = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_profiles(service_exposure_mode="public")

        assert len(result) == 2
        assert {r.profile_name for r in result} == {"prof-1", "prof-2"}

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
        response.profiles = [
            _make_profile_proto_with_modes("prof-1", "t-1", ["public"], ["internet"]),
            _make_profile_proto_with_modes("prof-2", "t-1", ["public"], ["blocked"]),
            _make_profile_proto_with_modes("prof-3", "t-2", ["public"], ["internet", "blocked"]),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListProfiles = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_profiles(egress_mode="blocked")

        assert len(result) == 2
        assert {r.profile_name for r in result} == {"prof-2", "prof-3"}

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
        response.profiles = [
            _make_profile_proto_with_modes("prof-1", "t-1", ["public"], ["internet"]),
            _make_profile_proto_with_modes("prof-2", "t-1", ["public"], ["blocked"]),
            _make_profile_proto_with_modes("prof-3", "t-2", ["private"], ["internet"]),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListProfiles = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_profiles(service_exposure_mode="public", egress_mode="internet")

        assert len(result) == 1
        assert result[0].profile_name == "prof-1"


# ---------------------------------------------------------------------------
# Client-side filtering tests - list_runners
# ---------------------------------------------------------------------------


def _make_runner_proto_with_resources(
    runner_id: str,
    cpu: int,
    memory: int,
    gpu: int,
    running: int = 0,
) -> discovery_pb2.AvailableRunner:
    """Build a runner proto with resource availability."""
    proto = discovery_pb2.AvailableRunner(
        runner_id=runner_id,
        runner_group_id="group-1",
        tags=["test"],
        healthy=True,
        profile_names=["default"],
    )
    proto.connected_at.FromDatetime(datetime(2025, 1, 1))
    proto.capabilities.CopyFrom(
        discovery_pb2.RunnerCapabilitySummary(
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
        discovery_pb2.RunnerResourceSummary(
            available_cpu_millicores=cpu,
            available_memory_bytes=memory,
            available_gpu_count=gpu,
            running_sandboxes=running,
        )
    )
    return proto


def _make_runner_proto_no_resources(runner_id: str) -> discovery_pb2.AvailableRunner:
    """Build a runner proto without resource information."""
    proto = discovery_pb2.AvailableRunner(
        runner_id=runner_id,
        runner_group_id="group-1",
        tags=["test"],
        healthy=True,
        profile_names=["default"],
    )
    proto.connected_at.FromDatetime(datetime(2025, 1, 1))
    proto.capabilities.CopyFrom(
        discovery_pb2.RunnerCapabilitySummary(
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


class TestListRunnersClientSideFilters:
    """Tests for client-side capacity and networking filtering on list_runners."""

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
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_runner_proto_with_resources("t-2", cpu=1000, memory=8_000_000_000, gpu=2),
            _make_runner_proto_with_resources("t-3", cpu=8000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runners(min_available_cpu_millicores=4000)

        assert len(result) == 2
        assert {t.runner_id for t in result} == {"t-1", "t-3"}

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
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=16_000_000_000, gpu=0),
            _make_runner_proto_with_resources("t-2", cpu=4000, memory=4_000_000_000, gpu=0),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runners(min_available_memory_bytes=8_000_000_000)

        assert len(result) == 1
        assert result[0].runner_id == "t-1"

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
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=4),
            _make_runner_proto_with_resources("t-2", cpu=4000, memory=8_000_000_000, gpu=1),
            _make_runner_proto_with_resources("t-3", cpu=4000, memory=8_000_000_000, gpu=0),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runners(min_available_gpu_count=2)

        assert len(result) == 1
        assert result[0].runner_id == "t-1"

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
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        list_runners(min_available_cpu_millicores=1)

        call_args = stub.ListAvailableRunners.call_args
        request = call_args[0][0]
        assert request.view == discovery_pb2.RUNNER_VIEW_FULL

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
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_runner_proto_no_resources("t-2"),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        result = list_runners(min_available_cpu_millicores=1)

        assert len(result) == 1
        assert result[0].runner_id == "t-1"

    @patch(_PATCH_LIST_PROFILES_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_service_exposure_mode_filter_on_runners(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_profiles_async: AsyncMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_runner_proto_with_resources("t-2", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_profiles_async.return_value = [
            Profile(
                profile_name="prof-1",
                runner_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Profile(
                profile_name="prof-2",
                runner_id="t-2",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="private"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
        ]

        result = list_runners(service_exposure_mode="public")

        assert len(result) == 1
        assert result[0].runner_id == "t-1"
        mock_list_profiles_async.assert_awaited_once()

    @patch(_PATCH_LIST_PROFILES_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_egress_mode_filter_on_runners(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_profiles_async: AsyncMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_runner_proto_with_resources("t-2", cpu=4000, memory=8_000_000_000, gpu=2),
            _make_runner_proto_with_resources("t-3", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_profiles_async.return_value = [
            Profile(
                profile_name="prof-1",
                runner_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Profile(
                profile_name="prof-2",
                runner_id="t-2",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="public"),),
                egress_modes=(EgressMode(name="blocked"),),
            ),
            Profile(
                profile_name="prof-3",
                runner_id="t-3",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="public"),),
                egress_modes=(EgressMode(name="internet"), EgressMode(name="blocked")),
            ),
        ]

        result = list_runners(egress_mode="internet")

        assert len(result) == 2
        assert {t.runner_id for t in result} == {"t-1", "t-3"}
        mock_list_profiles_async.assert_awaited_once()

    @patch(_PATCH_LIST_PROFILES_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_capacity_plus_service_exposure_mode_combined(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_profiles_async: AsyncMock,
    ) -> None:
        """f-10: Combined capacity and network mode filters."""
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=8000, memory=16_000_000_000, gpu=4),
            _make_runner_proto_with_resources("t-2", cpu=2000, memory=4_000_000_000, gpu=1),
            _make_runner_proto_with_resources("t-3", cpu=8000, memory=16_000_000_000, gpu=4),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_profiles_async.return_value = [
            Profile(
                profile_name="prof-1",
                runner_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Profile(
                profile_name="prof-2",
                runner_id="t-2",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
            Profile(
                profile_name="prof-3",
                runner_id="t-3",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="private"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
        ]

        result = list_runners(min_available_cpu_millicores=4000, service_exposure_mode="public")

        assert len(result) == 1
        assert result[0].runner_id == "t-1"
        mock_list_profiles_async.assert_awaited_once()

    @patch(_PATCH_LIST_PROFILES_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_list_profiles_async_failure_propagates(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_profiles_async: AsyncMock,
    ) -> None:
        """f-11: Error in _list_profiles_async propagates through list_runners."""
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_profiles_async.side_effect = CWSandboxError("profile fetch failed")

        with pytest.raises(CWSandboxError, match="profile fetch failed"):
            list_runners(service_exposure_mode="public")

    @patch(_PATCH_LIST_PROFILES_ASYNC, new_callable=AsyncMock)
    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_timeout_budget_shared_with_profile_fetch(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
        mock_list_profiles_async: AsyncMock,
    ) -> None:
        """g-14: Verify remaining time (not original timeout) is passed to _list_profiles_async."""
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        response = MagicMock()
        response.runners = [
            _make_runner_proto_with_resources("t-1", cpu=4000, memory=8_000_000_000, gpu=2),
        ]
        response.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(return_value=response)
        mock_stub_cls.return_value = stub

        mock_list_profiles_async.return_value = [
            Profile(
                profile_name="prof-1",
                runner_id="t-1",
                supported_gpu_types=("A100",),
                supported_architectures=("amd64",),
                service_exposure_modes=(ServiceExposureMode(name="public"),),
                egress_modes=(EgressMode(name="internet"),),
            ),
        ]

        # Simulate 7 seconds elapsing between deadline set and profile fetch.
        # With a 30s timeout, the remaining budget should be ~23s, not 30s.
        call_count = 0

        def advancing_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            # First calls: deadline creation and paginate timeout
            # Later calls: the remaining-time check before _list_profiles_async
            return 1000.0 + (call_count - 1) * 3.5

        with patch("cwsandbox._discovery.time") as mock_time:
            mock_time.monotonic = advancing_monotonic
            list_runners(service_exposure_mode="public")

        # _list_profiles_async receives a timeout arg (positional arg index 2)
        profile_call_args = mock_list_profiles_async.call_args
        timeout_passed = profile_call_args[0][2]
        assert timeout_passed < 30.0, f"Expected remaining budget < 30s, got {timeout_passed}"
        assert timeout_passed > 0, "Timeout budget must be positive"


# ---------------------------------------------------------------------------
# gRPC error path tests for list functions
# ---------------------------------------------------------------------------


class TestListRunnersGrpcErrors:
    """f-9: gRPC error path coverage for list_runners."""

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
        stub.ListAvailableRunners = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(DiscoveryError, match="unavailable"):
            list_runners()


class TestListProfilesGrpcErrors:
    """f-9: gRPC error path coverage for list_profiles."""

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
        stub.ListProfiles = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        channel = mock_channel.return_value
        with pytest.raises(DiscoveryError, match="unavailable"):
            list_profiles()
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
    def test_list_runners_multi_page(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        page1 = MagicMock()
        page1.runners = [_make_runner_proto()]
        page1.next_page_token = "page2"

        page2 = MagicMock()
        page2.runners = [_make_runner_proto()]
        page2.runners[0].runner_id = "runner-def"
        page2.next_page_token = ""

        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(side_effect=[page1, page2])
        mock_stub_cls.return_value = stub

        result = list_runners()

        assert len(result) == 2
        assert {t.runner_id for t in result} == {"runner-abc", "runner-def"}
        assert stub.ListAvailableRunners.call_count == 2

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_list_profiles_multi_page(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)

        page1 = MagicMock()
        page1.profiles = [_make_profile_proto()]
        page1.next_page_token = "page2"

        page2 = MagicMock()
        page2.profiles = [_make_profile_proto()]
        page2.profiles[0].profile_name = "gpu-profile"
        page2.next_page_token = ""

        stub = MagicMock()
        stub.ListProfiles = AsyncMock(side_effect=[page1, page2])
        mock_stub_cls.return_value = stub

        result = list_profiles()

        assert len(result) == 2
        assert {r.profile_name for r in result} == {"default", "gpu-profile"}
        assert stub.ListProfiles.call_count == 2


# ---------------------------------------------------------------------------
# get_profile NOT_FOUND without runner_id
# ---------------------------------------------------------------------------


class TestGetProfileNotFoundVariants:
    """f-13: get_profile NOT_FOUND with runner_id=None."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_not_found_without_runner_id(
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
        stub.GetProfile = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(ProfileNotFoundError) as exc_info:
            get_profile("missing")
        assert exc_info.value.profile_name == "missing"
        assert exc_info.value.runner_id is None


# ---------------------------------------------------------------------------
# NOT_FOUND with AIP-193 ErrorInfo: structured fields propagate
# ---------------------------------------------------------------------------


def _pack_error_info_status(*, reason: str, metadata: dict[str, str] | None = None) -> bytes:
    """Build a serialized google.rpc.Status carrying an ErrorInfo detail."""
    from google.protobuf import any_pb2
    from google.rpc import error_details_pb2, status_pb2

    status = status_pb2.Status(code=5, message="not found")
    info = error_details_pb2.ErrorInfo(
        reason=reason,
        domain="cwsandbox.com",
        metadata=metadata or {},
    )
    packed = any_pb2.Any()
    packed.Pack(info)
    status.details.append(packed)
    return status.SerializeToString()


class TestDiscoveryNotFoundStructuredFields:
    """NOT_FOUND in discovery must thread ErrorInfo into the direct exception."""

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_get_runner_not_found_propagates_error_info(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)
        status_bytes = _pack_error_info_status(
            reason="CWSANDBOX_RUNNER_NOT_FOUND",
            metadata={"runner_id": "missing"},
        )
        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.NOT_FOUND,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(
                ("grpc-status-details-bin", status_bytes),
            ),
            details="not found",
        )
        stub = MagicMock()
        stub.GetAvailableRunner = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(RunnerNotFoundError) as exc_info:
            get_runner("missing")

        assert exc_info.value.runner_id == "missing"
        assert exc_info.value.reason == "CWSANDBOX_RUNNER_NOT_FOUND"
        assert exc_info.value.metadata == {"runner_id": "missing"}
        assert exc_info.value.retry_delay is None

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_get_runner_not_found_without_error_info_preserves_none(
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
        stub.GetAvailableRunner = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(RunnerNotFoundError) as exc_info:
            get_runner("missing")

        assert exc_info.value.reason is None
        assert exc_info.value.metadata == {}
        assert exc_info.value.retry_delay is None

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_get_profile_not_found_propagates_error_info(
        self,
        mock_lm: MagicMock,
        mock_channel: MagicMock,
        mock_auth: MagicMock,
        mock_parse: MagicMock,
        mock_stub_cls: MagicMock,
    ) -> None:
        _setup_grpc_mocks(mock_lm, mock_channel, mock_auth, mock_parse)
        status_bytes = _pack_error_info_status(
            reason="CWSANDBOX_PROFILE_NOT_FOUND",
            metadata={"profile_name": "missing"},
        )
        err = grpc.aio.AioRpcError(
            code=grpc.StatusCode.NOT_FOUND,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(
                ("grpc-status-details-bin", status_bytes),
            ),
            details="not found",
        )
        stub = MagicMock()
        stub.GetProfile = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(ProfileNotFoundError) as exc_info:
            get_profile("missing", runner_id="runner-1")

        assert exc_info.value.profile_name == "missing"
        assert exc_info.value.runner_id == "runner-1"
        assert exc_info.value.reason == "CWSANDBOX_PROFILE_NOT_FOUND"
        assert exc_info.value.metadata == {"profile_name": "missing"}
        assert exc_info.value.retry_delay is None

    @patch(_PATCH_STUB)
    @patch(_PATCH_PARSE)
    @patch(_PATCH_AUTH)
    @patch(_PATCH_CHANNEL)
    @patch(_PATCH_LM)
    def test_get_profile_not_found_without_error_info_preserves_none(
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
        stub.GetProfile = AsyncMock(side_effect=err)
        mock_stub_cls.return_value = stub

        with pytest.raises(ProfileNotFoundError) as exc_info:
            get_profile("missing")

        assert exc_info.value.reason is None
        assert exc_info.value.metadata == {}
        assert exc_info.value.retry_delay is None
