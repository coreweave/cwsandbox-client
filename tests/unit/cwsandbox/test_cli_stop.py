# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox stop CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cwsandbox._defaults import DEFAULT_GRACEFUL_SHUTDOWN_SECONDS
from cwsandbox.cli import cli
from cwsandbox.exceptions import SandboxNotFoundError
from tests.unit.cwsandbox.conftest import make_operation_ref


def _patch_sandbox(sandbox: MagicMock):
    """Patch cwsandbox.cli.stop.Sandbox.from_id to return *sandbox*."""
    return patch(
        "cwsandbox.cli.stop.Sandbox",
        **{"from_id.return_value": make_operation_ref(sandbox)},
    )


class TestStopCommand:
    """Tests for the cwsandbox stop CLI command."""

    def test_stop_registered(self) -> None:
        """Stop command is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["stop", "--help"])
        assert result.exit_code == 0
        assert "SANDBOX_ID" in result.output

    def test_stop_sandbox(self) -> None:
        """cwsandbox stop stops a sandbox and prints confirmation."""
        mock_sandbox = MagicMock()
        mock_sandbox.stop.return_value = make_operation_ref(None)

        with _patch_sandbox(mock_sandbox) as mock_sandbox_cls:
            runner = CliRunner()
            result = runner.invoke(cli, ["stop", "abc-123"])

        assert result.exit_code == 0
        assert "Stopped sandbox abc-123." in result.output
        mock_sandbox_cls.from_id.assert_called_once_with("abc-123")
        mock_sandbox.stop.assert_called_once_with(
            snapshot_on_stop=False,
            graceful_shutdown_seconds=DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
            missing_ok=False,
            wait_for_ready=True,
            request_id=None,
        )

    def test_stop_with_options(self) -> None:
        """cwsandbox stop passes stop options correctly."""
        mock_sandbox = MagicMock()
        mock_sandbox.stop.return_value = make_operation_ref(None)

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "stop",
                    "abc-123",
                    "--missing-ok",
                    "--snapshot-on-stop",
                    "--no-wait-for-snapshot",
                    "--request-id",
                    "idem-1",
                    "--graceful-shutdown-seconds",
                    "30",
                    "--quiet",
                ],
            )

        assert result.exit_code == 0
        assert result.output == ""
        mock_sandbox.stop.assert_called_once_with(
            snapshot_on_stop=True,
            graceful_shutdown_seconds=30.0,
            missing_ok=True,
            wait_for_ready=False,
            request_id="idem-1",
        )

    def test_stop_snapshot_on_stop_prints_snapshot_id(self) -> None:
        """cwsandbox stop --snapshot-on-stop prints the resulting snapshot ID."""
        mock_sandbox = MagicMock()
        mock_sandbox.stop.return_value = make_operation_ref(None)
        mock_sandbox.file_system_snapshot_id = "fss-123"

        with _patch_sandbox(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["stop", "abc-123", "--snapshot-on-stop"])

        assert result.exit_code == 0
        assert "Stopped sandbox abc-123. Snapshot fss-123." in result.output

    def test_stop_missing_ok_suppresses_from_id_not_found(self) -> None:
        """cwsandbox stop --missing-ok succeeds when the sandbox is already missing."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with patch("cwsandbox.cli.stop.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["stop", "bad-id", "--missing-ok"])

        assert result.exit_code == 0
        assert "Sandbox bad-id is already missing." in result.output

    def test_stop_sandbox_not_found(self) -> None:
        """cwsandbox stop shows clean error for SandboxNotFoundError."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with patch("cwsandbox.cli.stop.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["stop", "bad-id"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_stop_no_wait_for_snapshot_requires_snapshot_on_stop(self) -> None:
        """--no-wait-for-snapshot is only valid with --snapshot-on-stop."""
        runner = CliRunner()
        result = runner.invoke(cli, ["stop", "abc-123", "--no-wait-for-snapshot"])

        assert result.exit_code == 2
        assert "--no-wait-for-snapshot requires --snapshot-on-stop" in result.output

    def test_stop_request_id_requires_snapshot_on_stop(self) -> None:
        """--request-id is only valid with --snapshot-on-stop."""
        runner = CliRunner()
        result = runner.invoke(cli, ["stop", "abc-123", "--request-id", "idem-1"])

        assert result.exit_code == 2
        assert "--request-id requires --snapshot-on-stop" in result.output
