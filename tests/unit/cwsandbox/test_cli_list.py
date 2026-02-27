# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox ls CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cwsandbox.cli import cli
from cwsandbox.exceptions import CWSandboxError


class TestListCommand:
    """Tests for the cwsandbox ls CLI command."""

    def test_list_registered(self) -> None:
        """List command is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ls", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output

    def test_list_displays_sandboxes(self) -> None:
        """cwsandbox ls displays sandbox table."""
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "abc-123"
        mock_sb.status.value = "running"
        mock_sb.tower_id = "tower-1"
        mock_sb.runway_id = "runway-1"
        mock_sb.started_at = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)

        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = [mock_sb]

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["ls"])

        assert result.exit_code == 0
        assert "abc-123" in result.output
        assert "running" in result.output
        assert "tower-1" in result.output
        assert "runway-1" in result.output
        assert "2026-01-15" in result.output

    def test_list_empty(self) -> None:
        """cwsandbox ls shows message when no sandboxes found."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = []

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["ls"])

        assert result.exit_code == 0
        assert "No sandboxes found" in result.output

    def test_list_with_filters(self) -> None:
        """cwsandbox ls passes filters to Sandbox.list()."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = []

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(
                cli, ["ls", "--status", "running", "--tag", "test", "--tag", "dev"]
            )

        assert result.exit_code == 0
        mock_sandbox_cls.list.assert_called_once_with(
            tags=["test", "dev"],
            status="running",
            runway_ids=None,
            tower_ids=None,
        )

    def test_list_invalid_status(self) -> None:
        """cwsandbox ls rejects invalid status values."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ls", "--status", "bogus"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_list_output_json(self) -> None:
        """cwsandbox ls --output json emits valid JSON with expected fields."""
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "abc-123"
        mock_sb.status.value = "running"
        mock_sb.tower_id = "tower-1"
        mock_sb.runway_id = "runway-1"
        mock_sb.tower_group_id = "tg-1"
        mock_sb.started_at = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)

        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = [mock_sb]

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["ls", "--output", "json"])

        assert result.exit_code == 0
        expected = json.dumps(
            [
                {
                    "sandbox_id": "abc-123",
                    "status": "running",
                    "tower_id": "tower-1",
                    "runway_id": "runway-1",
                    "tower_group_id": "tg-1",
                    "started_at": "2026-01-15T10:30:00+00:00",
                }
            ],
            indent=2,
        )
        assert result.output.strip() == expected

    def test_list_output_json_empty(self) -> None:
        """cwsandbox ls --output json with no sandboxes emits []."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = []

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["ls", "--output", "json"])

        assert result.exit_code == 0
        assert result.output.strip() == "[]"

    def test_list_api_error(self) -> None:
        """cwsandbox ls shows clean error for CWSandboxError from API failure."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = CWSandboxError("connection failed")

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["ls"])

        assert result.exit_code == 1
        assert "connection failed" in result.output
