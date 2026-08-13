# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox delete CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cwsandbox.cli import cli
from cwsandbox.exceptions import SandboxNotFoundError
from tests.unit.cwsandbox.conftest import make_operation_ref


class TestDeleteCommand:
    """Tests for the cwsandbox delete CLI command."""

    def test_delete_registered(self) -> None:
        """Delete command is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "--help"])
        assert result.exit_code == 0
        assert "SANDBOX_ID" in result.output

    def test_delete_sandbox(self) -> None:
        """cwsandbox delete deletes a sandbox and prints confirmation."""
        with patch("cwsandbox.cli.delete.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.delete.return_value = make_operation_ref(None)

            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "abc-123"])

        assert result.exit_code == 0
        assert "Deleted sandbox abc-123." in result.output
        mock_sandbox_cls.delete.assert_called_once_with(
            "abc-123",
            missing_ok=False,
            timeout_seconds=None,
        )

    def test_delete_with_options(self) -> None:
        """cwsandbox delete passes --missing-ok, --timeout, and --quiet correctly."""
        with patch("cwsandbox.cli.delete.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.delete.return_value = make_operation_ref(None)

            runner = CliRunner()
            result = runner.invoke(
                cli, ["delete", "abc-123", "--missing-ok", "--timeout", "5", "--quiet"]
            )

        assert result.exit_code == 0
        assert result.output == ""
        mock_sandbox_cls.delete.assert_called_once_with(
            "abc-123",
            missing_ok=True,
            timeout_seconds=5.0,
        )

    def test_delete_sandbox_not_found(self) -> None:
        """cwsandbox delete shows clean error for SandboxNotFoundError."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with patch("cwsandbox.cli.delete.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.delete.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["delete", "bad-id"])

        assert result.exit_code == 1
        assert "not found" in result.output
