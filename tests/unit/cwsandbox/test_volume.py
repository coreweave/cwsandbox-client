# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for registered Volume CRUD."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.rpc import error_details_pb2, status_pb2

from cwsandbox import AuthHeaders, PvcVolumeSource, Volume, VolumeState
from cwsandbox._defaults import DEFAULT_POLL_RPC_TIMEOUT_SECONDS
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

    def test_create_forwards_auth_and_retains_it(self) -> None:
        stub = MagicMock()
        stub.CreateVolume = AsyncMock(return_value=_ready_proto())
        stub.DeleteVolume = AsyncMock(return_value=_ready_proto())
        resolved: list[tuple[object, str]] = []

        def fake_resolve(auth: object = None, *, base_url: str = "") -> tuple[tuple[str, str], ...]:
            resolved.append((auth, base_url))
            return (("authorization", "custom"),)

        headers = AuthHeaders(headers={"Authorization": "Bearer tok"}, strategy="api_key")
        patches = _patch_volume_channel(stub)
        with (
            patch("cwsandbox._volume.resolve_auth_metadata", side_effect=fake_resolve),
            patches[1],
            patches[2],
            patches[3],
        ):
            volume = Volume.create(
                "team-data",
                pvc=PvcVolumeSource(runner_id="runner-1", namespace="ml", claim_name="data"),
                auth=headers,
                base_url="https://gw.example.test",
            ).result()
            volume.delete().result()

        assert resolved[0] == (headers, "https://gw.example.test")
        assert resolved[1] == (headers, "https://gw.example.test")
        assert stub.CreateVolume.call_args.kwargs["metadata"] == (("authorization", "custom"),)
        assert stub.DeleteVolume.call_args.kwargs["metadata"] == (("authorization", "custom"),)

    def test_create_provider_receives_effective_base_url(self) -> None:
        stub = MagicMock()
        stub.CreateVolume = AsyncMock(return_value=_ready_proto())

        class Provider:
            def __init__(self) -> None:
                self.base_url: str | None = None

            def resolve_auth(self, *, base_url: str) -> AuthHeaders:
                self.base_url = base_url
                return AuthHeaders(headers={"X-Test-Auth": "value"}, strategy="test")

        provider = Provider()
        patches = _patch_volume_channel(stub)
        with patches[1], patches[2], patches[3]:
            Volume.create(
                "team-data",
                pvc=PvcVolumeSource(runner_id="runner-1", namespace="ml", claim_name="data"),
                auth=provider,
                base_url="https://gw.example.test/",
            ).result()

        assert provider.base_url == "https://gw.example.test"
        assert stub.CreateVolume.call_args.kwargs["metadata"] == (("x-test-auth", "value"),)

    def test_wait_until_ready_caps_get_timeout_and_rejects_late_ready(self) -> None:
        ready = _ready_proto()
        stub = MagicMock()

        async def slow_get(*args: object, **kwargs: object) -> volume_pb2.Volume:
            await asyncio.sleep(0.2)
            return ready

        stub.GetVolume = AsyncMock(side_effect=slow_get)
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data")
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(VolumeWaitTimeoutError):
                volume.wait_until_ready(timeout=0.01).result()
        assert stub.GetVolume.call_args.kwargs["timeout"] <= 0.01

    def test_wait_until_ready_defaults_poll_rpc_timeout(self) -> None:
        stub = MagicMock()
        stub.GetVolume = AsyncMock(return_value=_ready_proto())
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data")
        with patches[0], patches[1], patches[2], patches[3]:
            volume.wait_until_ready(timeout=60).result()
        assert stub.GetVolume.call_args.kwargs["timeout"] == DEFAULT_POLL_RPC_TIMEOUT_SECONDS

    def test_delete_not_found_on_retry_is_success(self) -> None:
        stub = MagicMock()
        stub.DeleteVolume = AsyncMock(
            side_effect=[
                _volume_rpc_error("transient", code=grpc.StatusCode.UNAVAILABLE),
                _volume_rpc_error(CWSANDBOX_VOLUME_NOT_FOUND, code=grpc.StatusCode.NOT_FOUND),
            ]
        )
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data")
        with patches[0], patches[1], patches[2], patches[3]:
            result = volume.delete().result()
        assert result is volume
        assert stub.DeleteVolume.call_count == 2

    @pytest.mark.parametrize(
        ("method", "kwargs", "stub_attr"),
        [
            ("update", {"description": "shared"}, "UpdateVolume"),
            ("delete", {}, "DeleteVolume"),
            ("validate", {}, "ValidateVolume"),
        ],
    )
    def test_instance_methods_zero_timeout_uses_handle_default(
        self, method: str, kwargs: dict[str, str], stub_attr: str
    ) -> None:
        stub = MagicMock()
        stub.UpdateVolume = AsyncMock(return_value=_ready_proto())
        stub.DeleteVolume = AsyncMock(return_value=_ready_proto())
        stub.ValidateVolume = AsyncMock(return_value=_ready_proto())
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data", timeout_seconds=60.0)
        with patches[0], patches[1], patches[2], patches[3]:
            getattr(volume, method)(timeout_seconds=0, **kwargs).result()
            assert getattr(stub, stub_attr).call_args.kwargs["timeout"] == 60.0
            getattr(volume, method)(timeout_seconds=12, **kwargs).result()
            assert getattr(stub, stub_attr).call_args.kwargs["timeout"] == 12

    def test_delete_first_not_found_raises_unless_allow_missing(self) -> None:
        stub = MagicMock()
        stub.DeleteVolume = AsyncMock(
            side_effect=_volume_rpc_error(
                CWSANDBOX_VOLUME_NOT_FOUND, code=grpc.StatusCode.NOT_FOUND
            )
        )
        patches = _patch_volume_channel(stub)
        volume = Volume(volume_id="team-data")
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(VolumeNotFoundError):
                volume.delete().result()
            result = volume.delete(allow_missing=True).result()
        assert result is volume
