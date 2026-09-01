# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for registered Volume CRUD."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.rpc import error_details_pb2, status_pb2

from cwsandbox import PvcVolumeSource, Volume, VolumeState
from cwsandbox._error_info import CWSANDBOX_VOLUME_IN_USE, CWSANDBOX_VOLUME_NOT_FOUND
from cwsandbox._proto import volume_pb2
from cwsandbox.exceptions import VolumeInUseError, VolumeNotFoundError, VolumeWaitTimeoutError


def _volume_rpc_error(
    reason: str, code: grpc.StatusCode = grpc.StatusCode.FAILED_PRECONDITION
) -> grpc.RpcError:
    info = error_details_pb2.ErrorInfo(reason=reason, domain="cwsandbox.com")
    status = status_pb2.Status(message="boom")
    status.details.add().Pack(info)
    status_bytes = status.SerializeToString()

    class _Err(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return code

        def details(self) -> str:
            return "boom"

        def trailing_metadata(self) -> tuple[tuple[str, bytes], ...]:
            return (("grpc-status-details-bin", status_bytes),)

    return _Err()


def _patch_volume_channel(stub: MagicMock) -> tuple[object, object, object, object]:
    mock_channel = MagicMock()
    mock_channel.close = AsyncMock()
    return (
        patch("cwsandbox._volume.resolve_auth_metadata", return_value=(("authorization", "t"),)),
        patch("cwsandbox._volume.parse_grpc_target", return_value=("test:443", True)),
        patch("cwsandbox._volume.create_channel", return_value=mock_channel),
        patch("cwsandbox._volume.volume_pb2_grpc.VolumeServiceStub", return_value=stub),
    )


def _ready_proto(volume_id: str = "team-data") -> volume_pb2.Volume:
    return volume_pb2.Volume(
        volume_id=volume_id,
        spec=volume_pb2.VolumeSpec(
            pvc=volume_pb2.PvcVolumeSource(
                runner_id="runner-1",
                namespace="ml",
                claim_name="data",
                sub_path="datasets",
            ),
            description="shared",
        ),
        status=volume_pb2.VolumeStatus(
            state=volume_pb2.VOLUME_STATE_READY,
            locality=volume_pb2.VOLUME_LOCALITY_CLUSTER_LOCAL,
            capacity="1Ti",
            access_modes=["ReadWriteMany"],
        ),
    )


class TestVolume:
    def test_create_maps_request(self) -> None:
        stub = MagicMock()
        stub.CreateVolume = AsyncMock(return_value=_ready_proto())
        patches = _patch_volume_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            volume = Volume.create(
                "team-data",
                pvc=PvcVolumeSource(
                    runner_id="runner-1",
                    namespace="ml",
                    claim_name="data",
                    sub_path="datasets",
                ),
                description="shared",
            ).result()

        req = stub.CreateVolume.call_args[0][0]
        assert req.volume.volume_id == "team-data"
        assert req.volume.spec.pvc.claim_name == "data"
        assert req.volume.spec.pvc.sub_path == "datasets"
        assert volume.state == VolumeState.READY
        assert volume.capacity == "1Ti"
        assert volume.pvc is not None
        assert volume.pvc.runner_id == "runner-1"

    def test_get_not_found(self) -> None:
        stub = MagicMock()
        stub.GetVolume = AsyncMock(
            side_effect=_volume_rpc_error(
                CWSANDBOX_VOLUME_NOT_FOUND, code=grpc.StatusCode.NOT_FOUND
            )
        )
        patches = _patch_volume_channel(stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(VolumeNotFoundError):
                Volume.get("missing").result()

    def test_delete_in_use(self) -> None:
        stub = MagicMock()
        stub.DeleteVolume = AsyncMock(side_effect=_volume_rpc_error(CWSANDBOX_VOLUME_IN_USE))
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data")
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(VolumeInUseError):
                volume.delete().result()

    def test_wait_until_ready_polls(self) -> None:
        validating = _ready_proto()
        validating.status.state = volume_pb2.VOLUME_STATE_VALIDATING
        ready = _ready_proto()
        stub = MagicMock()
        stub.GetVolume = AsyncMock(side_effect=[validating, ready])
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data")
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("cwsandbox._volume._sleep", new_callable=AsyncMock),
        ):
            result = volume.wait_until_ready(timeout=5).result()
        assert result.state == VolumeState.READY
        assert stub.GetVolume.call_count == 2

    def test_wait_until_ready_times_out(self) -> None:
        validating = _ready_proto()
        validating.status.state = volume_pb2.VOLUME_STATE_VALIDATING
        stub = MagicMock()
        stub.GetVolume = AsyncMock(return_value=validating)
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data")
        now = 0.0

        def fake_monotonic() -> float:
            return now

        async def fake_sleep(seconds: float) -> None:
            nonlocal now
            now += seconds

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("cwsandbox._volume._monotonic", fake_monotonic),
            patch("cwsandbox._volume._sleep", fake_sleep),
        ):
            with pytest.raises(VolumeWaitTimeoutError):
                volume.wait_until_ready(timeout=5).result()
