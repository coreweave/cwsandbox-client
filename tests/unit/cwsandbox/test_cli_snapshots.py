# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox snapshots CLI commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cwsandbox import FileSystemSnapshot, FileSystemSnapshotStatus, FileSystemSnapshotTrigger
from cwsandbox.cli import cli
from cwsandbox.exceptions import CWSandboxError, SnapshotNotFoundError
from tests.unit.cwsandbox.conftest import make_operation_ref


def _snapshot(snapshot_id: str = "fss-1") -> FileSystemSnapshot:
    return FileSystemSnapshot(
        file_system_snapshot_id=snapshot_id,
        status=FileSystemSnapshotStatus.READY,
        size_bytes=42,
        source_sandbox_id="sb-1",
        trigger=FileSystemSnapshotTrigger.MANUAL,
        request_id="create-1",
        object_bucket="bucket-1",
        source_volume_name="workspace",
        created_at=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
        updated_at=datetime(2026, 1, 15, 10, 31, 0, tzinfo=UTC),
        completed_at=datetime(2026, 1, 15, 10, 31, 0, tzinfo=UTC),
    )


class TestSnapshotsCommand:
    """Tests for the cwsandbox snapshots command group."""

    def test_snapshots_registered(self) -> None:
        """Snapshots command group is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["snapshots", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output
        assert "get" in result.output
        assert "list" in result.output
        assert "delete" in result.output

    def test_create_snapshot_prints_id(self) -> None:
        """cwsandbox snapshots create prints the created snapshot ID."""
        mock_sandbox = MagicMock()
        mock_sandbox.snapshot.return_value = make_operation_ref("fss-1")

        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = make_operation_ref(mock_sandbox)

            runner = CliRunner()
            result = runner.invoke(cli, ["snapshots", "create", "sb-1"])

        assert result.exit_code == 0
        assert result.output == "fss-1\n"
        mock_sandbox_cls.from_id.assert_called_once_with("sb-1", timeout_seconds=None)
        mock_sandbox.snapshot.assert_called_once_with(
            wait_for_ready=True,
            request_id=None,
        )

    def test_create_snapshot_options_and_json(self) -> None:
        """cwsandbox snapshots create passes options and can emit JSON."""
        mock_sandbox = MagicMock()
        mock_sandbox.snapshot.return_value = make_operation_ref("fss-2")

        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = make_operation_ref(mock_sandbox)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "snapshots",
                    "create",
                    "sb-1",
                    "--no-wait",
                    "--request-id",
                    "create-1",
                    "--timeout",
                    "7",
                    "--output",
                    "json",
                ],
            )

        assert result.exit_code == 0
        assert json.loads(result.output) == {"file_system_snapshot_id": "fss-2"}
        mock_sandbox_cls.from_id.assert_called_once_with("sb-1", timeout_seconds=7.0)
        mock_sandbox.snapshot.assert_called_once_with(
            wait_for_ready=False,
            request_id="create-1",
        )

    def test_get_snapshot_table(self) -> None:
        """cwsandbox snapshots get displays a snapshot table."""
        mock_op_ref = make_operation_ref(_snapshot("fss-3"))

        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.get_snapshot.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["snapshots", "get", "fss-3"])

        assert result.exit_code == 0
        assert "SNAPSHOT ID" in result.output
        assert "fss-3" in result.output
        assert "ready" in result.output
        assert "sb-1" in result.output
        assert "manual" in result.output
        mock_sandbox_cls.get_snapshot.assert_called_once_with("fss-3", timeout_seconds=None)

    def test_get_snapshot_json(self) -> None:
        """cwsandbox snapshots get --output json emits snapshot metadata."""
        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.get_snapshot.return_value = make_operation_ref(_snapshot("fss-4"))

            runner = CliRunner()
            result = runner.invoke(cli, ["snapshots", "get", "fss-4", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["file_system_snapshot_id"] == "fss-4"
        assert data["status"] == "ready"
        assert data["source_sandbox_id"] == "sb-1"
        assert data["trigger"] == "manual"
        assert data["request_id"] == "create-1"
        assert data["source_volume_name"] == "workspace"
        assert data["created_at"] == "2026-01-15T10:30:00+00:00"

    def test_list_snapshots_with_filters(self) -> None:
        """cwsandbox snapshots list passes filters to Sandbox.list_snapshots()."""
        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list_snapshots.return_value = make_operation_ref([])

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "snapshots",
                    "list",
                    "--source-sandbox-id",
                    "sb-1",
                    "--status",
                    "ready",
                    "--timeout",
                    "5",
                ],
            )

        assert result.exit_code == 0
        assert "No snapshots found." in result.output
        mock_sandbox_cls.list_snapshots.assert_called_once_with(
            source_sandbox_id="sb-1",
            status="ready",
            timeout_seconds=5.0,
        )

    def test_list_snapshots_json(self) -> None:
        """cwsandbox snapshots list --output json emits a snapshot list."""
        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list_snapshots.return_value = make_operation_ref([_snapshot("fss-5")])

            runner = CliRunner()
            result = runner.invoke(cli, ["snapshots", "list", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["file_system_snapshot_id"] == "fss-5"
        assert data[0]["status"] == "ready"

    def test_delete_snapshot_success(self) -> None:
        """cwsandbox snapshots delete deletes the selected snapshot."""
        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.delete_snapshot.return_value = make_operation_ref(None)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["snapshots", "delete", "fss-6", "--missing-ok", "--timeout", "9"],
            )

        assert result.exit_code == 0
        assert "Deleted fss-6." in result.output
        mock_sandbox_cls.delete_snapshot.assert_called_once_with(
            "fss-6",
            timeout_seconds=9.0,
            missing_ok=True,
        )

    def test_delete_snapshot_quiet(self) -> None:
        """cwsandbox snapshots delete --quiet suppresses success output."""
        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.delete_snapshot.return_value = make_operation_ref(None)

            runner = CliRunner()
            result = runner.invoke(cli, ["snapshots", "delete", "fss-6", "--quiet"])

        assert result.exit_code == 0
        assert result.output == ""

    def test_get_snapshot_api_error(self) -> None:
        """cwsandbox snapshots get shows clean errors for API failures."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SnapshotNotFoundError(
            "not found", file_system_snapshot_id="fss-missing"
        )

        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.get_snapshot.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["snapshots", "get", "fss-missing"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_create_snapshot_api_error(self) -> None:
        """cwsandbox snapshots create shows clean errors from snapshot()."""
        mock_sandbox = MagicMock()
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = CWSandboxError("snapshot failed")
        mock_sandbox.snapshot.return_value = mock_op_ref

        with patch("cwsandbox.cli.snapshots.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = make_operation_ref(mock_sandbox)

            runner = CliRunner()
            result = runner.invoke(cli, ["snapshots", "create", "sb-1"])

        assert result.exit_code == 1
        assert "snapshot failed" in result.output
