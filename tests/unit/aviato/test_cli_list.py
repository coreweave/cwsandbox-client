# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Tests for aviato list CLI command."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from aviato.cli import cli


class TestListCommand:
    """Tests for the aviato list CLI command."""

    def test_list_registered(self) -> None:
        """List command is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output

    def test_list_displays_sandboxes(self) -> None:
        """aviato list displays sandbox table."""
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "abc-123"
        mock_sb.status.value = "running"
        mock_sb.tower_id = "tower-1"
        mock_sb.runway_id = "runway-1"
        mock_sb.started_at = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)

        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = [mock_sb]

        with patch("aviato.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "abc-123" in result.output
        assert "running" in result.output
        assert "tower-1" in result.output
        assert "runway-1" in result.output
        assert "2026-01-15" in result.output

    def test_list_empty(self) -> None:
        """aviato list shows message when no sandboxes found."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = []

        with patch("aviato.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No sandboxes found" in result.output

    def test_list_with_filters(self) -> None:
        """aviato list passes filters to Sandbox.list()."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = []

        with patch("aviato.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(
                cli, ["list", "--status", "running", "--tag", "test", "--tag", "dev"]
            )

        assert result.exit_code == 0
        mock_sandbox_cls.list.assert_called_once_with(
            tags=["test", "dev"],
            status="running",
            runway_ids=None,
            tower_ids=None,
        )

    def test_list_invalid_status(self) -> None:
        """aviato list rejects invalid status values."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--status", "bogus"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_list_api_error(self) -> None:
        """aviato list shows error on API failure."""
        from aviato.exceptions import AviatoError

        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = AviatoError("connection failed")

        with patch("aviato.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["list"])

        assert result.exit_code != 0
        assert "connection failed" in result.output
