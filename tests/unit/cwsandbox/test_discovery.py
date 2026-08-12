# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._discovery (v1 runners + capabilities)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from cwsandbox._discovery import (
    RunnerResources,
    _runner_from_proto,
    format_bytes,
    format_cpu,
    get_runner,
    list_runners,
)
from cwsandbox._proto import discovery_pb2, sandbox_pb2
from cwsandbox.exceptions import (
    CWSandboxAuthenticationError,
    DiscoveryError,
    RunnerNotFoundError,
)


def _ts(seconds: int = 1_700_000_000) -> Timestamp:
    t = Timestamp()
    t.FromSeconds(seconds)
    return t


def _make_runner_proto(
    *,
    runner_id: str = "runner-1",
    organization_id: str = "org-1",
    with_caps: bool = True,
    with_resources: bool = False,
    visibilities: list[int] | None = None,
) -> discovery_pb2.AvailableRunner:
    proto = discovery_pb2.AvailableRunner(
        runner_id=runner_id,
        organization_id=organization_id,
        runner_group_id="rg-1",
        tags=["gpu"],
        healthy=True,
        is_shared=False,
        connected_at=_ts(),
    )
    if with_caps:
        proto.capabilities.CopyFrom(
            discovery_pb2.RunnerCapabilitySummary(
                max_cpu_millicores=4000,
                max_memory_bytes=16 << 30,
                max_gpu_count=2,
                supported_gpu_types=["A100"],
                supported_architectures=["amd64"],
                supports_privileged=True,
                available_storage_classes=["ssd"],
                supported_service_visibilities=visibilities
                or [sandbox_pb2.VISIBILITY_PUBLIC, sandbox_pb2.VISIBILITY_PRIVATE],
            )
        )
    if with_resources:
        proto.resources.CopyFrom(
            discovery_pb2.RunnerResourceSummary(
                available_cpu_millicores=2000,
                available_memory_bytes=8 << 30,
                available_gpu_count=1,
                running_sandboxes=3,
            )
        )
    return proto


class TestFormatters:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0 B"),
            (1024, "1.0 KiB"),
            (16 << 30, "16.0 GiB"),
        ],
    )
    def test_format_bytes(self, value: int, expected: str) -> None:
        assert format_bytes(value) == expected

    @pytest.mark.parametrize(
        ("millicores", "expected"),
        [(4000, "4.0 vCPU"), (500, "0.5 vCPU")],
    )
    def test_format_cpu(self, millicores: int, expected: str) -> None:
        assert format_cpu(millicores) == expected


class TestRunnerTypes:
    def test_resources_frozen(self) -> None:
        res = RunnerResources(
            available_cpu_millicores=1,
            available_memory_bytes=2,
            available_gpu_count=0,
            running_sandboxes=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.available_cpu_millicores = 9  # type: ignore[misc]

    def test_runner_from_proto_with_caps_and_resources(self) -> None:
        runner = _runner_from_proto(_make_runner_proto(with_resources=True))
        assert runner.runner_id == "runner-1"
        assert runner.organization_id == "org-1"
        assert runner.max_cpu_millicores == 4000
        assert runner.supported_service_visibilities == ("public", "private")
        assert runner.resources is not None
        assert runner.resources.running_sandboxes == 3
        assert isinstance(runner.connected_at, datetime)
        assert runner.connected_at.tzinfo == UTC

    def test_runner_from_proto_without_caps(self) -> None:
        runner = _runner_from_proto(_make_runner_proto(with_caps=False))
        assert runner.max_cpu_millicores == 0
        assert runner.supported_service_visibilities == ()
        assert runner.resources is None

    def test_repr_includes_visibilities(self) -> None:
        runner = _runner_from_proto(_make_runner_proto())
        text = repr(runner)
        assert "runner-1" in text
        assert "public" in text


class TestExceptions:
    def test_runner_not_found(self) -> None:
        err = RunnerNotFoundError("missing", runner_id="r1")
        assert isinstance(err, DiscoveryError)
        assert err.runner_id == "r1"


class TestInputValidation:
    def test_get_runner_requires_ids(self) -> None:
        with pytest.raises(ValueError, match="runner_id"):
            get_runner("", organization_id="org")
        with pytest.raises(ValueError, match="organization_id"):
            get_runner("runner", organization_id=" ")


def _patch_channel_and_stub(stub: MagicMock) -> Any:
    channel = MagicMock()
    channel.close = AsyncMock()
    return patch.multiple(
        "cwsandbox._discovery",
        create_channel=MagicMock(return_value=channel),
        parse_grpc_target=MagicMock(return_value=("host:443", True)),
        resolve_auth_metadata=MagicMock(return_value=(("authorization", "Bearer x"),)),
        discovery_pb2_grpc=MagicMock(DiscoveryServiceStub=MagicMock(return_value=stub)),
    )


class TestListRunners:
    def test_basic_list_and_filters(self) -> None:
        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(
            return_value=discovery_pb2.ListAvailableRunnersResponse(
                runners=[_make_runner_proto(with_resources=True)],
                next_page_token="",
            )
        )
        with _patch_channel_and_stub(stub):
            runners = list_runners(
                runner_group_id="rg-1",
                gpu_type="A100",
                architecture="amd64",
                healthy_only=True,
                include_resources=True,
                service_visibility="public",
                min_available_cpu_millicores=1000,
            )
        assert len(runners) == 1
        assert runners[0].runner_id == "runner-1"
        req = stub.ListAvailableRunners.await_args.args[0]
        assert req.runner_group_id == "rg-1"
        assert req.gpu_type == "A100"
        assert req.architecture == "amd64"
        assert req.healthy_only is True
        assert req.view == discovery_pb2.RUNNER_VIEW_FULL
        assert req.service_visibility == sandbox_pb2.VISIBILITY_PUBLIC

    def test_capacity_filter_excludes(self) -> None:
        stub = MagicMock()
        stub.ListAvailableRunners = AsyncMock(
            return_value=discovery_pb2.ListAvailableRunnersResponse(
                runners=[_make_runner_proto(with_resources=True)],
                next_page_token="",
            )
        )
        with _patch_channel_and_stub(stub):
            runners = list_runners(min_available_cpu_millicores=10_000)
        assert runners == []

    def test_unavailable_maps_to_discovery_error(self) -> None:
        stub = MagicMock()
        err = grpc.aio.AioRpcError(grpc.StatusCode.UNAVAILABLE, None, None, details="down")
        stub.ListAvailableRunners = AsyncMock(side_effect=err)
        with _patch_channel_and_stub(stub):
            with pytest.raises(DiscoveryError):
                list_runners()


class TestGetRunner:
    def test_returns_runner(self) -> None:
        stub = MagicMock()
        stub.GetAvailableRunner = AsyncMock(return_value=_make_runner_proto(with_resources=True))
        with _patch_channel_and_stub(stub):
            runner = get_runner("runner-1", organization_id="org-1")
        assert runner.runner_id == "runner-1"
        req = stub.GetAvailableRunner.await_args.args[0]
        assert req.organization_id == "org-1"
        assert req.view == discovery_pb2.RUNNER_VIEW_FULL

    def test_not_found(self) -> None:
        stub = MagicMock()
        err = grpc.aio.AioRpcError(grpc.StatusCode.NOT_FOUND, None, None, details="gone")
        stub.GetAvailableRunner = AsyncMock(side_effect=err)
        with (
            _patch_channel_and_stub(stub),
            patch("cwsandbox._discovery.is_not_found", return_value=True),
            patch(
                "cwsandbox._discovery.parse_error_info",
                return_value=MagicMock(
                    reason="CWSANDBOX_RUNNER_NOT_FOUND", metadata=None, retry_delay=None
                ),
            ),
        ):
            with pytest.raises(RunnerNotFoundError):
                get_runner("missing", organization_id="org-1")

    def test_unauthenticated(self) -> None:
        stub = MagicMock()
        err = grpc.aio.AioRpcError(grpc.StatusCode.UNAUTHENTICATED, None, None, details="auth")
        stub.GetAvailableRunner = AsyncMock(side_effect=err)
        with _patch_channel_and_stub(stub):
            with pytest.raises(CWSandboxAuthenticationError):
                get_runner("runner-1", organization_id="org-1")
