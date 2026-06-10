# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for file-system snapshot (FSS) functionality."""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from google.protobuf import timestamp_pb2
from google.rpc import error_details_pb2, status_pb2

import cwsandbox._sandbox as sandbox_module
from cwsandbox import (
    FileSystemSnapshotBucketConfig,
    FileSystemSnapshotBucketMode,
    FileSystemSnapshotOptions,
    FileSystemSnapshotStatus,
    FileSystemSnapshotTrigger,
    Sandbox,
    SandboxDefaults,
    SandboxStatus,
)
from cwsandbox._defaults import (
    DEFAULT_FSS_STOP_CLIENT_SLACK_SECONDS,
    DEFAULT_FSS_STOP_GRACE_FALLBACK_SECONDS,
    DEFAULT_FSS_STOP_TIMEOUT_SECONDS,
)
from cwsandbox._error_info import (
    CWSANDBOX_FSS_BACKEND_THROTTLED,
    CWSANDBOX_FSS_BUCKET_MISMATCH,
    CWSANDBOX_FSS_INFLIGHT_LIMIT,
    CWSANDBOX_FSS_NOT_FOUND,
    CWSANDBOX_FSS_NOT_READY,
    CWSANDBOX_FSS_NOT_SUPPORTED,
    CWSANDBOX_FSS_QUOTA_EXCEEDED,
    CWSANDBOX_FSS_RESTORE_FAILED,
    CWSANDBOX_FSS_SIZE_EXCEEDED,
    CWSANDBOX_FSS_WAIT_TIMEOUT,
)
from cwsandbox._proto import gateway_pb2
from cwsandbox._sandbox import _NotStarted, _Running, _Starting, _Stopping, _Terminal
from cwsandbox.exceptions import (
    SandboxError,
    SandboxRequestTimeoutError,
    SandboxSnapshotError,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SnapshotBackendThrottledError,
    SnapshotBucketMismatchError,
    SnapshotNotFoundError,
    SnapshotNotReadyError,
    SnapshotNotSupportedError,
    SnapshotOnStopConflictError,
    SnapshotQuotaExceededError,
    SnapshotSizeExceededError,
    SnapshotWaitTimeoutError,
)


def _fss_rpc_error(
    reason: str, code: grpc.StatusCode = grpc.StatusCode.FAILED_PRECONDITION
) -> grpc.RpcError:
    """Build an RpcError carrying an AIP-193 ErrorInfo with the given FSS reason."""
    info = error_details_pb2.ErrorInfo(reason=reason, domain="cwsandbox.com")
    status = status_pb2.Status(message="boom")
    status.details.add().Pack(info)
    status_bytes = status.SerializeToString()

    class _Err(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return code

        def details(self) -> str:
            return "boom"

        def trailing_metadata(self) -> list[tuple[str, bytes]]:  # type: ignore[override]
            return [("grpc-status-details-bin", status_bytes)]

    return _Err()


def _bare_rpc_error(code: grpc.StatusCode, details: str = "boom") -> grpc.RpcError:
    """Build an RpcError with only a status code (no AIP-193 details)."""

    class _Err(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return code

        def details(self) -> str:
            return details

        def trailing_metadata(self) -> list[tuple[str, bytes]]:  # type: ignore[override]
            return []

    return _Err()


# ---------------------------------------------------------------------------
# FileSystemSnapshotOptions
# ---------------------------------------------------------------------------


class TestFileSystemSnapshotOptions:
    def test_minimal(self) -> None:
        opts = FileSystemSnapshotOptions(mount_path="/workspace")
        assert opts.mount_path == "/workspace"
        assert opts.size is None
        assert opts.file_system_snapshot_id is None

    def test_restore(self) -> None:
        opts = FileSystemSnapshotOptions(
            mount_path="/data", size="10Gi", file_system_snapshot_id="fss-1"
        )
        assert opts.size == "10Gi"
        assert opts.file_system_snapshot_id == "fss-1"

    def test_empty_optionals_normalized_to_none(self) -> None:
        opts = FileSystemSnapshotOptions(mount_path="/data", size="", file_system_snapshot_id="")
        assert opts.size is None
        assert opts.file_system_snapshot_id is None

    @pytest.mark.parametrize("bad", ["", "relative/path", "/"])
    def test_invalid_mount_path_raises(self, bad: str) -> None:
        with pytest.raises(ValueError):
            FileSystemSnapshotOptions(mount_path=bad)

    def test_frozen(self) -> None:
        opts = FileSystemSnapshotOptions(mount_path="/workspace")
        with pytest.raises(dataclasses.FrozenInstanceError):
            opts.mount_path = "/other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Proto conversions
# ---------------------------------------------------------------------------


class TestProtoConversions:
    @pytest.mark.parametrize(
        ("proto_value", "expected"),
        [
            (gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_CREATING, FileSystemSnapshotStatus.CREATING),
            (gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_READY, FileSystemSnapshotStatus.READY),
            (gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_FAILED, FileSystemSnapshotStatus.FAILED),
            (gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_DELETING, FileSystemSnapshotStatus.DELETING),
        ],
    )
    def test_status_from_proto(self, proto_value: int, expected: FileSystemSnapshotStatus) -> None:
        assert sandbox_module._fss_status_from_proto(proto_value) == expected

    def test_status_from_proto_unknown(self) -> None:
        assert sandbox_module._fss_status_from_proto(999) == FileSystemSnapshotStatus.UNSPECIFIED

    def test_trigger_from_proto(self) -> None:
        assert (
            sandbox_module._fss_trigger_from_proto(gateway_pb2.FILE_SYSTEM_SNAPSHOT_TRIGGER_STOP)
            == FileSystemSnapshotTrigger.STOP
        )
        assert (
            sandbox_module._fss_trigger_from_proto(gateway_pb2.FILE_SYSTEM_SNAPSHOT_TRIGGER_MANUAL)
            == FileSystemSnapshotTrigger.MANUAL
        )

    def test_bucket_mode_from_proto(self) -> None:
        assert (
            sandbox_module._fss_bucket_mode_from_proto(
                gateway_pb2.FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_BRING_YOUR_OWN
            )
            == FileSystemSnapshotBucketMode.BRING_YOUR_OWN
        )

    def test_snapshot_from_proto(self) -> None:
        ts = timestamp_pb2.Timestamp()
        ts.FromJsonString("2026-06-07T10:00:00Z")
        proto = gateway_pb2.FileSystemSnapshot(
            file_system_snapshot_id="fss-9",
            status=gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_READY,
            status_reason="",
            size_bytes=4096,
            source_sandbox_id="sb-1",
            trigger=gateway_pb2.FILE_SYSTEM_SNAPSHOT_TRIGGER_MANUAL,
            idempotency_key="k",
            object_bucket="bucket-x",
            created_at=ts,
        )
        snap = sandbox_module._snapshot_from_proto(proto)
        assert snap.file_system_snapshot_id == "fss-9"
        assert snap.status == FileSystemSnapshotStatus.READY
        assert snap.size_bytes == 4096
        assert snap.source_sandbox_id == "sb-1"
        assert snap.trigger == FileSystemSnapshotTrigger.MANUAL
        assert snap.object_bucket == "bucket-x"
        assert snap.created_at is not None
        # Unset timestamps stay None.
        assert snap.updated_at is None
        assert snap.completed_at is None

    def test_bucket_config_from_proto(self) -> None:
        proto = gateway_pb2.FileSystemSnapshotBucketConfig(
            bucket_name="my-bucket",
            region="us-east-1",
            mode=gateway_pb2.FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_BRING_YOUR_OWN,
            effective_bucket_name="my-bucket",
        )
        cfg = sandbox_module._bucket_config_from_proto(proto)
        assert cfg.bucket_name == "my-bucket"
        assert cfg.region == "us-east-1"
        assert cfg.mode == FileSystemSnapshotBucketMode.BRING_YOUR_OWN
        assert cfg.effective_bucket_name == "my-bucket"

    def test_mount_kwargs_fresh(self) -> None:
        mount = sandbox_module._file_system_mount_kwargs(
            FileSystemSnapshotOptions(mount_path="/data")
        )
        assert mount == {"mount_path": "/data"}

    def test_mount_kwargs_restore(self) -> None:
        mount = sandbox_module._file_system_mount_kwargs(
            FileSystemSnapshotOptions(
                mount_path="/data", size="5Gi", file_system_snapshot_id="fss-1"
            )
        )
        assert mount == {
            "mount_path": "/data",
            "size": "5Gi",
            "file_system_snapshot": {"file_system_snapshot_id": "fss-1"},
        }


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestSnapshotErrorMapping:
    @pytest.mark.parametrize(
        ("reason", "cls"),
        [
            (CWSANDBOX_FSS_NOT_FOUND, SnapshotNotFoundError),
            (CWSANDBOX_FSS_NOT_READY, SnapshotNotReadyError),
            (CWSANDBOX_FSS_NOT_SUPPORTED, SnapshotNotSupportedError),
            (CWSANDBOX_FSS_SIZE_EXCEEDED, SnapshotSizeExceededError),
            (CWSANDBOX_FSS_QUOTA_EXCEEDED, SnapshotQuotaExceededError),
            (CWSANDBOX_FSS_BUCKET_MISMATCH, SnapshotBucketMismatchError),
            (CWSANDBOX_FSS_RESTORE_FAILED, SandboxSnapshotError),
        ],
    )
    def test_terminal_reasons(self, reason: str, cls: type[SandboxSnapshotError]) -> None:
        exc = sandbox_module._translate_snapshot_reason(
            reason,
            details="d",
            operation="op",
            file_system_snapshot_id="fss-9",
            metadata={},
            retry_delay=None,
        )
        assert isinstance(exc, cls)
        assert exc.reason == reason
        assert exc.file_system_snapshot_id == "fss-9"

    def test_wait_timeout_is_non_retryable_timeout(self) -> None:
        exc = sandbox_module._translate_snapshot_reason(
            CWSANDBOX_FSS_WAIT_TIMEOUT,
            details="d",
            operation="op",
            file_system_snapshot_id="fss-9",
            metadata={},
            retry_delay=None,
        )
        assert isinstance(exc, SnapshotWaitTimeoutError)
        assert isinstance(exc, SandboxTimeoutError)
        assert not isinstance(exc, sandbox_module._RETRYABLE_POLL_EXCEPTIONS)

    @pytest.mark.parametrize(
        "reason", [CWSANDBOX_FSS_BACKEND_THROTTLED, CWSANDBOX_FSS_INFLIGHT_LIMIT]
    )
    def test_transient_reasons_are_retryable(self, reason: str) -> None:
        exc = sandbox_module._translate_snapshot_reason(
            reason,
            details="d",
            operation="op",
            file_system_snapshot_id=None,
            metadata={},
            retry_delay=None,
        )
        assert isinstance(exc, SnapshotBackendThrottledError)
        assert isinstance(exc, SandboxUnavailableError)
        assert isinstance(exc, sandbox_module._RETRYABLE_POLL_EXCEPTIONS)

    def test_unknown_reason_returns_none(self) -> None:
        assert (
            sandbox_module._translate_snapshot_reason(
                "CWSANDBOX_NOT_AN_FSS_REASON",
                details="d",
                operation="op",
                file_system_snapshot_id=None,
                metadata={},
                retry_delay=None,
            )
            is None
        )

    def test_end_to_end_get_snapshot_not_supported(self, mock_api_key: str) -> None:
        """An FSS reason on a real RPC maps through _translate_rpc_error."""
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.GetFileSystemSnapshot = AsyncMock(
            side_effect=_fss_rpc_error(CWSANDBOX_FSS_NOT_SUPPORTED)
        )
        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("t:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub",
                return_value=mock_stub,
            ),
        ):
            with pytest.raises(SnapshotNotSupportedError):
                Sandbox.get_snapshot("fss-1").result()


# ---------------------------------------------------------------------------
# Start / stop wiring
# ---------------------------------------------------------------------------


class TestStartWiring:
    def test_run_param_builds_file_system_mount(self) -> None:
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            file_system_snapshot=FileSystemSnapshotOptions(
                mount_path="/workspace", size="10Gi", file_system_snapshot_id="fss-7"
            ),
        )
        mock_response = MagicMock()
        mock_response.sandbox_id = "sb-1"
        with patch.object(sandbox, "_ensure_client", new_callable=AsyncMock):
            sandbox._channel = MagicMock()
            sandbox._stub = MagicMock()
            sandbox._stub.Start = AsyncMock(return_value=mock_response)
            sandbox.start().result()
            request = sandbox._stub.Start.call_args[0][0]
            assert request.file_system.mount_path == "/workspace"
            assert request.file_system.size == "10Gi"
            assert request.file_system.file_system_snapshot.file_system_snapshot_id == "fss-7"

    def test_dict_coercion(self) -> None:
        sandbox = Sandbox(
            command="sleep",
            args=["infinity"],
            file_system_snapshot={"mount_path": "/data"},
        )
        assert isinstance(sandbox._start_kwargs["file_system_snapshot"], FileSystemSnapshotOptions)

    def test_defaults_fallback(self) -> None:
        defaults = SandboxDefaults(
            file_system_snapshot=FileSystemSnapshotOptions(mount_path="/workspace")
        )
        sandbox = Sandbox(command="sleep", args=["infinity"], defaults=defaults)
        assert sandbox._start_kwargs["file_system_snapshot"].mount_path == "/workspace"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            Sandbox(command="x", file_system_snapshot=123)  # type: ignore[arg-type]


class TestStopWiring:
    def test_snapshot_on_stop_sends_renamed_field_and_captures_id(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Starting(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = MagicMock()
        stub = sandbox._stub  # stop() tears down _stub on completion; keep a ref.
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.file_system_snapshot_id = "fss-new"
        sandbox._stub.Stop = AsyncMock(return_value=mock_response)

        with patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock):
            sandbox.stop(snapshot_on_stop=True, idempotency_key="k").result()

        request = stub.Stop.call_args[0][0]
        assert request.file_system_snapshot_on_stop is True
        assert request.wait_for_ready is True
        assert request.idempotency_key == "k"
        # Snapshot-on-stop uses the generous FSS ceiling, not graceful shutdown.
        assert request.max_timeout_seconds == int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS)
        assert sandbox.file_system_snapshot_id == "fss-new"

    def test_plain_stop_does_not_request_snapshot(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Starting(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = MagicMock()
        stub = sandbox._stub  # stop() tears down _stub on completion; keep a ref.
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.file_system_snapshot_id = ""
        sandbox._stub.Stop = AsyncMock(return_value=mock_response)

        with patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock):
            sandbox.stop().result()

        request = stub.Stop.call_args[0][0]
        assert request.file_system_snapshot_on_stop is False
        # wait_for_ready is only set when snapshotting.
        assert request.HasField("wait_for_ready") is False
        assert sandbox.file_system_snapshot_id is None

    def test_snapshot_on_stop_deadline_covers_archive_plus_grace(self) -> None:
        # The backend archives (max_timeout_seconds) THEN deletes the pod
        # (graceful_shutdown_seconds). The client deadline must cover both, not
        # just the archive — the proto archive budget stays the FSS default.
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Starting(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = MagicMock()
        stub = sandbox._stub
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.file_system_snapshot_id = "fss-new"
        sandbox._stub.Stop = AsyncMock(return_value=mock_response)

        with patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock):
            sandbox.stop(snapshot_on_stop=True, graceful_shutdown_seconds=30).result()

        request = stub.Stop.call_args[0][0]
        assert request.max_timeout_seconds == int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS)
        assert request.graceful_shutdown_seconds == 30
        timeout = stub.Stop.call_args.kwargs["timeout"]
        assert timeout == (
            int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS) + 30 + int(DEFAULT_FSS_STOP_CLIENT_SLACK_SECONDS)
        )

    def test_snapshot_on_stop_deadline_budgets_backend_grace_default_when_zero(self) -> None:
        # Sending graceful_shutdown_seconds=0 makes the backend substitute its
        # own grace default, so the client deadline budgets that default — not 0.
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Starting(sandbox_id="sb-1")
        sandbox._channel = MagicMock()
        sandbox._channel.close = AsyncMock()
        sandbox._stub = MagicMock()
        stub = sandbox._stub
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.file_system_snapshot_id = "fss-new"
        sandbox._stub.Stop = AsyncMock(return_value=mock_response)

        with patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock):
            sandbox.stop(snapshot_on_stop=True, graceful_shutdown_seconds=0).result()

        assert stub.Stop.call_args[0][0].graceful_shutdown_seconds == 0
        timeout = stub.Stop.call_args.kwargs["timeout"]
        assert timeout == (
            int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS)
            + int(DEFAULT_FSS_STOP_GRACE_FALLBACK_SECONDS)
            + int(DEFAULT_FSS_STOP_CLIENT_SLACK_SECONDS)
        )


class TestSnapshotOnStopConflict:
    """stop(snapshot_on_stop=True) must not silently coalesce into a stop that
    will not archive the mount. It raises SnapshotOnStopConflictError instead.
    """

    def test_raises_when_already_terminating(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        # TERMINATING (e.g. an external stop / TTL the poller observed): no
        # snapshot RPC will be sent for this drain.
        sandbox._state = _Stopping(sandbox_id="sb-1")
        with pytest.raises(SnapshotOnStopConflictError, match="already terminating"):
            sandbox.stop(snapshot_on_stop=True).result()

    def test_raises_when_already_stopped_without_snapshot(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)
        sandbox._file_system_snapshot_id = None
        with pytest.raises(SnapshotOnStopConflictError, match="already stopped"):
            sandbox.stop(snapshot_on_stop=True).result()

    def test_idempotent_when_snapshot_already_captured(self) -> None:
        # A prior snapshot-on-stop already produced an ID; a repeat request is
        # convergent (the snapshot the caller asked for exists), so it returns
        # rather than raising.
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Terminal(sandbox_id="sb-1", status=SandboxStatus.COMPLETED)
        sandbox._file_system_snapshot_id = "fss-prior"
        sandbox.stop(snapshot_on_stop=True).result()
        assert sandbox.file_system_snapshot_id == "fss-prior"

    def test_does_not_raise_when_cancelled_before_start(self) -> None:
        # Never started: no mount to archive, so this is the normal no-op path,
        # not a conflict.
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._state = _NotStarted(cancelled=True)
        sandbox._reject_unsatisfiable_snapshot_on_stop()  # does not raise

    def test_helper_raises_on_in_flight_plain_stop(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Running(sandbox_id="sb-1")
        # Simulate a plain stop already in flight (no snapshot).
        sandbox._stop_task = MagicMock()
        sandbox._stop_task.done.return_value = False
        sandbox._stop_snapshot_requested = False
        with pytest.raises(SnapshotOnStopConflictError, match="plain stop is already in progress"):
            sandbox._reject_unsatisfiable_snapshot_on_stop()

    def test_helper_allows_join_of_in_flight_snapshot_stop(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Running(sandbox_id="sb-1")
        # The in-flight stop is itself a snapshot-on-stop: joining is safe.
        sandbox._stop_task = MagicMock()
        sandbox._stop_task.done.return_value = False
        sandbox._stop_snapshot_requested = True
        sandbox._reject_unsatisfiable_snapshot_on_stop()  # does not raise

    async def test_in_flight_plain_stop_is_left_untouched_on_conflict(self) -> None:
        # End-to-end through _stop_async: the conflict is raised before the
        # join, so the in-flight plain stop is neither cancelled nor awaited.
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Running(sandbox_id="sb-1")

        async def _never() -> None:
            await asyncio.sleep(3600)

        sandbox._stop_task = asyncio.create_task(_never())
        sandbox._stop_snapshot_requested = False
        try:
            with pytest.raises(SnapshotOnStopConflictError):
                await sandbox._stop_async(snapshot_on_stop=True)
            assert not sandbox._stop_task.done()  # the plain stop kept running
        finally:
            sandbox._stop_task.cancel()

    def test_plain_stop_still_coalesces_when_terminating(self) -> None:
        # Regression: a plain stop joining a TERMINATING sandbox must NOT raise
        # (it observes the drain to terminal). Only snapshot requests conflict.
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._state = _Stopping(sandbox_id="sb-1")
        with patch.object(sandbox, "_await_terminal_after_stop", new_callable=AsyncMock):
            sandbox.stop().result()  # no SnapshotOnStopConflictError


# ---------------------------------------------------------------------------
# snapshot() and management methods
# ---------------------------------------------------------------------------


def _ready_snapshot_proto(file_system_snapshot_id: str = "fss-1") -> gateway_pb2.FileSystemSnapshot:
    return gateway_pb2.FileSystemSnapshot(
        file_system_snapshot_id=file_system_snapshot_id,
        status=gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_READY,
        source_sandbox_id="sb-1",
        trigger=gateway_pb2.FILE_SYSTEM_SNAPSHOT_TRIGGER_MANUAL,
    )


class TestSnapshotMethod:
    def test_snapshot_returns_id(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._stub = MagicMock()
        create_resp = MagicMock()
        create_resp.success = True
        create_resp.file_system_snapshot_id = "fss-1"
        sandbox._stub.CreateFileSystemSnapshot = AsyncMock(return_value=create_resp)
        sandbox._stub.GetFileSystemSnapshot = AsyncMock()
        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(
                sandbox, "_wait_until_running_async", new_callable=AsyncMock
            ) as wait_running,
        ):
            snapshot_id = sandbox.snapshot(idempotency_key="k").result()
        # snapshot() waits for RUNNING before archiving the mount.
        wait_running.assert_awaited()
        # snapshot() returns just the ID; it does NOT fetch the full record.
        assert snapshot_id == "fss-1"
        sandbox._stub.GetFileSystemSnapshot.assert_not_called()
        create_req = sandbox._stub.CreateFileSystemSnapshot.call_args[0][0]
        assert create_req.sandbox_id == "sb-1"
        assert create_req.wait_for_ready is True
        assert create_req.idempotency_key == "k"
        # wait_for_ready blocks on the archive, so the server-side ceiling is set.
        assert create_req.max_timeout_seconds == int(DEFAULT_FSS_STOP_TIMEOUT_SECONDS)

    def test_snapshot_failure_raises(self) -> None:
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._stub = MagicMock()
        create_resp = MagicMock()
        create_resp.success = False
        create_resp.error_message = "nope"
        sandbox._stub.CreateFileSystemSnapshot = AsyncMock(return_value=create_resp)
        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
        ):
            with pytest.raises(SandboxSnapshotError, match="nope"):
                sandbox.snapshot().result()


class TestManagementClassmethods:
    def _patch_channel(self, mock_stub: MagicMock):  # type: ignore[no-untyped-def]
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        return (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("t:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub",
                return_value=mock_stub,
            ),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
        )

    def test_get_snapshot(self) -> None:
        mock_stub = MagicMock()
        mock_stub.GetFileSystemSnapshot = AsyncMock(return_value=_ready_snapshot_proto("fss-2"))
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            snap = Sandbox.get_snapshot("fss-2").result()
        assert snap.file_system_snapshot_id == "fss-2"
        req = mock_stub.GetFileSystemSnapshot.call_args[0][0]
        assert req.file_system_snapshot_id == "fss-2"

    def test_list_snapshots_with_client_side_filters(self) -> None:
        protos = [
            gateway_pb2.FileSystemSnapshot(
                file_system_snapshot_id="a",
                status=gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_READY,
                source_sandbox_id="sb-1",
            ),
            gateway_pb2.FileSystemSnapshot(
                file_system_snapshot_id="b",
                status=gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_CREATING,
                source_sandbox_id="sb-1",
            ),
            gateway_pb2.FileSystemSnapshot(
                file_system_snapshot_id="c",
                status=gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_READY,
                source_sandbox_id="sb-2",
            ),
        ]
        mock_stub = MagicMock()
        mock_stub.ListFileSystemSnapshots = AsyncMock(
            return_value=gateway_pb2.ListFileSystemSnapshotsResponse(
                file_system_snapshots=protos, next_page_token=""
            )
        )
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            # No filter -> all three.
            all_snaps = Sandbox.list_snapshots().result()
            assert {s.file_system_snapshot_id for s in all_snaps} == {"a", "b", "c"}
            # Filter by source sandbox.
            sb1 = Sandbox.list_snapshots(source_sandbox_id="sb-1").result()
            assert {s.file_system_snapshot_id for s in sb1} == {"a", "b"}
            # Filter by status (string and enum both accepted).
            ready = Sandbox.list_snapshots(status=FileSystemSnapshotStatus.READY).result()
            assert {s.file_system_snapshot_id for s in ready} == {"a", "c"}
            # Combined filter.
            both = Sandbox.list_snapshots(source_sandbox_id="sb-1", status="ready").result()
            assert {s.file_system_snapshot_id for s in both} == {"a"}

    def test_list_snapshots_retry_restarts_pagination(self) -> None:
        """A mid-pagination transient retry restarts from page 1, no dropped pages.

        ``paginate_async`` mutates ``page_token`` in place, so the retried
        attempt must build a fresh request (page 1) rather than resuming at the
        last token; otherwise the already-collected first page is silently lost.
        """
        page1 = gateway_pb2.ListFileSystemSnapshotsResponse(
            file_system_snapshots=[
                gateway_pb2.FileSystemSnapshot(
                    file_system_snapshot_id="a",
                    status=gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_READY,
                )
            ],
            next_page_token="tok1",
        )
        page2 = gateway_pb2.ListFileSystemSnapshotsResponse(
            file_system_snapshots=[
                gateway_pb2.FileSystemSnapshot(
                    file_system_snapshot_id="b",
                    status=gateway_pb2.FILE_SYSTEM_SNAPSHOT_STATUS_READY,
                )
            ],
            next_page_token="",
        )
        failed_once = {"v": False}

        async def fake_list(request, *, metadata, timeout):  # type: ignore[no-untyped-def]
            if request.page_token == "":
                return page1
            # Reached page 2 (token "tok1"): fail transiently the first time.
            if not failed_once["v"]:
                failed_once["v"] = True
                raise _bare_rpc_error(grpc.StatusCode.UNAVAILABLE)
            return page2

        mock_stub = MagicMock()
        mock_stub.ListFileSystemSnapshots = fake_list
        patches = self._patch_channel(mock_stub)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock),
        ):
            listed = Sandbox.list_snapshots().result()
        # Both pages present: the retry restarted pagination from page 1 rather
        # than resuming at "tok1" and dropping page 1.
        assert {s.file_system_snapshot_id for s in listed} == {"a", "b"}

    def test_delete_snapshot_success(self) -> None:
        mock_stub = MagicMock()
        resp = MagicMock()
        resp.success = True
        mock_stub.DeleteFileSystemSnapshot = AsyncMock(return_value=resp)
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            assert Sandbox.delete_snapshot("fss-3").result() is None
        req = mock_stub.DeleteFileSystemSnapshot.call_args[0][0]
        assert req.file_system_snapshot_id == "fss-3"

    def test_delete_snapshot_missing_ok_suppresses_not_found(self) -> None:
        mock_stub = MagicMock()
        mock_stub.DeleteFileSystemSnapshot = AsyncMock(
            side_effect=_fss_rpc_error(CWSANDBOX_FSS_NOT_FOUND, code=grpc.StatusCode.NOT_FOUND)
        )
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            assert Sandbox.delete_snapshot("fss-3", missing_ok=True).result() is None

    def test_delete_snapshot_not_found_raises(self) -> None:
        mock_stub = MagicMock()
        mock_stub.DeleteFileSystemSnapshot = AsyncMock(
            side_effect=_fss_rpc_error(CWSANDBOX_FSS_NOT_FOUND, code=grpc.StatusCode.NOT_FOUND)
        )
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(SnapshotNotFoundError):
                Sandbox.delete_snapshot("fss-3").result()

    def test_get_snapshot_bare_not_found_maps_to_snapshot_not_found(self) -> None:
        """A bare gRPC NOT_FOUND (no FSS reason) still raises SnapshotNotFoundError."""
        mock_stub = MagicMock()
        mock_stub.GetFileSystemSnapshot = AsyncMock(
            side_effect=_bare_rpc_error(grpc.StatusCode.NOT_FOUND)
        )
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            with pytest.raises(SnapshotNotFoundError):
                Sandbox.get_snapshot("fss-x").result()

    def test_delete_snapshot_not_found_on_retry_is_success(self) -> None:
        """A committed delete whose response was lost: retry hits NOT_FOUND -> success.

        With missing_ok=False, a transient failure followed by NOT_FOUND on the
        retry is treated as success, because the delete's postcondition (the
        snapshot is gone) is satisfied.
        """
        mock_stub = MagicMock()
        mock_stub.DeleteFileSystemSnapshot = AsyncMock(
            side_effect=[
                _bare_rpc_error(grpc.StatusCode.UNAVAILABLE),
                _fss_rpc_error(CWSANDBOX_FSS_NOT_FOUND, code=grpc.StatusCode.NOT_FOUND),
            ]
        )
        patches = self._patch_channel(mock_stub)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock),
        ):
            assert Sandbox.delete_snapshot("fss-3").result() is None
        assert mock_stub.DeleteFileSystemSnapshot.call_count == 2


class TestBucketConfig:
    def _patch_channel(self, mock_stub: MagicMock):  # type: ignore[no-untyped-def]
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        return (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("t:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub",
                return_value=mock_stub,
            ),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
        )

    def test_get_bucket_config(self) -> None:
        mock_stub = MagicMock()
        mock_stub.GetFileSystemSnapshotBucketConfig = AsyncMock(
            return_value=gateway_pb2.FileSystemSnapshotBucketConfig(
                mode=gateway_pb2.FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_CW_MANAGED,
                effective_bucket_name="cw-bucket",
            )
        )
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            cfg = Sandbox.get_snapshot_bucket_config().result()
        assert isinstance(cfg, FileSystemSnapshotBucketConfig)
        assert cfg.mode == FileSystemSnapshotBucketMode.CW_MANAGED
        assert cfg.effective_bucket_name == "cw-bucket"

    def test_set_bucket_config(self) -> None:
        mock_stub = MagicMock()
        mock_stub.SetFileSystemSnapshotBucketConfig = AsyncMock(
            return_value=gateway_pb2.FileSystemSnapshotBucketConfig(
                bucket_name="byo",
                region="us-east-1",
                mode=gateway_pb2.FILE_SYSTEM_SNAPSHOT_BUCKET_MODE_BRING_YOUR_OWN,
                effective_bucket_name="byo",
            )
        )
        patches = self._patch_channel(mock_stub)
        with patches[0], patches[1], patches[2], patches[3]:
            cfg = Sandbox.set_snapshot_bucket_config(bucket_name="byo", region="us-east-1").result()
        assert cfg.mode == FileSystemSnapshotBucketMode.BRING_YOUR_OWN
        req = mock_stub.SetFileSystemSnapshotBucketConfig.call_args[0][0]
        assert req.bucket_name == "byo"
        assert req.region == "us-east-1"


# ---------------------------------------------------------------------------
# Client-side transient retry on FSS RPCs
# ---------------------------------------------------------------------------


class TestTransientRetry:
    """The FSS RPCs retry transient backend errors with a bounded budget."""

    def test_snapshot_retries_transient_then_succeeds(self) -> None:
        """A transient UNAVAILABLE on create is retried; the same auto-key reused."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._stub = MagicMock()
        ok = MagicMock()
        ok.success = True
        ok.file_system_snapshot_id = "fss-1"
        sandbox._stub.CreateFileSystemSnapshot = AsyncMock(
            side_effect=[_bare_rpc_error(grpc.StatusCode.UNAVAILABLE), ok]
        )
        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock),
        ):
            snapshot_id = sandbox.snapshot().result()

        assert snapshot_id == "fss-1"
        assert sandbox._stub.CreateFileSystemSnapshot.call_count == 2
        # The caller passed no idempotency_key, so one was generated and the
        # retried attempt reuses it (so a committed-but-lost create dedups).
        calls = sandbox._stub.CreateFileSystemSnapshot.call_args_list
        first_key = calls[0][0][0].idempotency_key
        second_key = calls[1][0][0].idempotency_key
        assert first_key
        assert first_key == second_key

    def test_snapshot_does_not_retry_failed_precondition(self) -> None:
        """A FAILED_PRECONDITION is fatal: one attempt, no retry."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._stub = MagicMock()
        sandbox._stub.CreateFileSystemSnapshot = AsyncMock(
            side_effect=_bare_rpc_error(
                grpc.StatusCode.FAILED_PRECONDITION, "requires a running sandbox"
            )
        )
        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            with pytest.raises(SandboxError, match="requires a running sandbox"):
                sandbox.snapshot().result()

        assert sandbox._stub.CreateFileSystemSnapshot.call_count == 1
        sleep.assert_not_called()

    def test_snapshot_does_not_retry_not_supported(self) -> None:
        """An org-not-enabled (NOT_SUPPORTED) reason is fatal: no retry."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._stub = MagicMock()
        sandbox._stub.CreateFileSystemSnapshot = AsyncMock(
            side_effect=_fss_rpc_error(CWSANDBOX_FSS_NOT_SUPPORTED)
        )
        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(SnapshotNotSupportedError):
                sandbox.snapshot().result()
        assert sandbox._stub.CreateFileSystemSnapshot.call_count == 1

    def test_snapshot_budget_zero_disables_retry(self) -> None:
        """With the budget at 0, the first transient error re-raises immediately."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._stub = MagicMock()
        sandbox._stub.CreateFileSystemSnapshot = AsyncMock(
            side_effect=_bare_rpc_error(grpc.StatusCode.UNAVAILABLE)
        )
        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.DEFAULT_FSS_RETRY_BUDGET_SECONDS", 0.0),
        ):
            with pytest.raises(SandboxUnavailableError):
                sandbox.snapshot().result()
        assert sandbox._stub.CreateFileSystemSnapshot.call_count == 1

    def test_snapshot_does_not_retry_deadline_exceeded(self) -> None:
        """A client DEADLINE_EXCEEDED on a wait-for-ready create is the ceiling
        being hit, not a transient blip: one attempt, no retry, so a wedged
        backend cannot trigger a second full-length attempt (~2x overrun)."""
        sandbox = Sandbox(command="sleep", args=["infinity"])
        sandbox._sandbox_id = "sb-1"
        sandbox._stub = MagicMock()
        sandbox._stub.CreateFileSystemSnapshot = AsyncMock(
            side_effect=_bare_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
        )
        with (
            patch.object(sandbox, "_ensure_client", new_callable=AsyncMock),
            patch.object(sandbox, "_wait_until_running_async", new_callable=AsyncMock),
            patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            with pytest.raises(SandboxRequestTimeoutError):
                sandbox.snapshot().result()
        assert sandbox._stub.CreateFileSystemSnapshot.call_count == 1
        sleep.assert_not_called()

    def test_get_snapshot_retries_transient_then_succeeds(self) -> None:
        """The management classmethods retry transient errors too."""
        mock_channel = MagicMock()
        mock_channel.close = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.GetFileSystemSnapshot = AsyncMock(
            side_effect=[
                _bare_rpc_error(grpc.StatusCode.UNAVAILABLE),
                _ready_snapshot_proto("fss-2"),
            ]
        )
        with (
            patch("cwsandbox._sandbox.parse_grpc_target", return_value=("t:443", True)),
            patch("cwsandbox._sandbox.create_channel", return_value=mock_channel),
            patch(
                "cwsandbox._sandbox.gateway_pb2_grpc.GatewayServiceStub",
                return_value=mock_stub,
            ),
            patch("cwsandbox._sandbox.resolve_auth_metadata", return_value=()),
            patch("cwsandbox._sandbox.asyncio.sleep", new_callable=AsyncMock),
        ):
            snap = Sandbox.get_snapshot("fss-2").result()
        assert snap.file_system_snapshot_id == "fss-2"
        assert mock_stub.GetFileSystemSnapshot.call_count == 2
