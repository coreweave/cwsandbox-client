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
        mock_sb.runner_id = "tower-1"
        mock_sb.profile_id = "runway-1"
        mock_sb.container_image = "python:3.11"
        mock_sb.tags = ("dev", "test")
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
        assert "python:3.11" in result.output
        assert "IMAGE" in result.output
        assert "2026-01-15" in result.output
        # Basic table does not show tower, runway, or tags
        assert "TOWER" not in result.output
        assert "TAGS" not in result.output

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
            profile_ids=None,
            runner_ids=None,
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
        mock_sb.runner_id = "tower-1"
        mock_sb.profile_id = "runway-1"
        mock_sb.runner_group_id = "tg-1"
        mock_sb.tags = ("dev", "test")
        mock_sb.container_image = "python:3.11"
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
                    "runner_id": "tower-1",
                    "profile_id": "runway-1",
                    "runner_group_id": "tg-1",
                    "started_at": "2026-01-15T10:30:00+00:00",
                    "container_image": "python:3.11",
                    "tags": ["dev", "test"],
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

    def test_list_json_no_tags(self) -> None:
        """cwsandbox ls --output json emits empty list for sandbox with no tags."""
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "abc-123"
        mock_sb.status.value = "running"
        mock_sb.runner_id = "tower-1"
        mock_sb.profile_id = "runway-1"
        mock_sb.runner_group_id = "tg-1"
        mock_sb.tags = None
        mock_sb.container_image = None
        mock_sb.started_at = None

        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = [mock_sb]

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["ls", "--output", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["tags"] == []
        assert data[0]["container_image"] is None

    def test_list_wide_output(self) -> None:
        """cwsandbox ls -o wide adds RUNNER, RUNNER GROUP, PROFILE, TAGS columns."""
        sb_with_tags = MagicMock()
        sb_with_tags.sandbox_id = "abc-123"
        sb_with_tags.status.value = "running"
        sb_with_tags.runner_id = "runner-1"
        sb_with_tags.runner_group_id = "rg-1"
        sb_with_tags.profile_id = "profile-1"
        sb_with_tags.container_image = "python:3.11"
        sb_with_tags.tags = ("gpu", "dev")
        sb_with_tags.started_at = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)

        sb_no_tags = MagicMock()
        sb_no_tags.sandbox_id = "def-456"
        sb_no_tags.status.value = "running"
        sb_no_tags.runner_id = "runner-1"
        sb_no_tags.runner_group_id = None
        sb_no_tags.profile_id = "profile-1"
        sb_no_tags.container_image = "python:3.11"
        sb_no_tags.tags = ()
        sb_no_tags.started_at = None

        mock_op_ref = MagicMock()
        mock_op_ref.result.return_value = [sb_with_tags, sb_no_tags]

        with patch("cwsandbox.cli.list.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.list.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["ls", "-o", "wide"])

        assert result.exit_code == 0
        assert "RUNNER" in result.output
        assert "RUNNER GROUP" in result.output
        assert "PROFILE" in result.output
        assert "TAGS" in result.output
        assert "gpu,dev" in result.output
        # Sandbox with no tags shows "-"
        lines = result.output.strip().split("\n")
        no_tag_line = [l for l in lines if "def-456" in l][0]
        assert no_tag_line.rstrip().endswith("-")

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
