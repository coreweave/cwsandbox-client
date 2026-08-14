# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Tests for cwsandbox get CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from cwsandbox.cli import cli
from cwsandbox.exceptions import SandboxNotFoundError
from tests.unit.cwsandbox.conftest import make_operation_ref


def _mock_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.sandbox_id = "abc-123"
    sandbox.status.value = "running"
    sandbox.runner_id = "runner-1"
    sandbox.runner_group_id = "group-1"
    sandbox.started_at = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)
    sandbox.returncode = None
    return sandbox


class TestGetCommand:
    """Tests for the cwsandbox get CLI command."""

    def test_get_registered(self) -> None:
        """Get command is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get", "--help"])
        assert result.exit_code == 0
        assert "SANDBOX_ID" in result.output

    def test_get_displays_sandbox_details(self) -> None:
        """cwsandbox get displays sandbox details."""
        mock_sandbox = _mock_sandbox()

        with patch("cwsandbox.cli.get.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = make_operation_ref(mock_sandbox)

            runner = CliRunner()
            result = runner.invoke(cli, ["get", "abc-123"])

        assert result.exit_code == 0
        assert "SANDBOX ID" in result.output
        assert "abc-123" in result.output
        assert "running" in result.output
        assert "runner-1" in result.output
        assert "2026-01-15 10:30:00 UTC" in result.output
        mock_sandbox_cls.from_id.assert_called_once_with("abc-123")

    def test_get_output_json(self) -> None:
        """cwsandbox get --output json emits valid JSON with expected fields."""
        mock_sandbox = _mock_sandbox()
        mock_sandbox.status.value = "completed"
        mock_sandbox.returncode = 0

        with patch("cwsandbox.cli.get.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = make_operation_ref(mock_sandbox)

            runner = CliRunner()
            result = runner.invoke(cli, ["get", "abc-123", "--output", "json"])

        assert result.exit_code == 0
        expected = json.dumps(
            {
                "sandbox_id": "abc-123",
                "status": "completed",
                "runner_id": "runner-1",
                "runner_group_id": "group-1",
                "started_at": "2026-01-15T10:30:00+00:00",
                "returncode": 0,
            },
            indent=2,
        )
        assert result.output.strip() == expected

    def test_get_sandbox_not_found(self) -> None:
        """cwsandbox get shows clean error for SandboxNotFoundError."""
        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with patch("cwsandbox.cli.get.Sandbox") as mock_sandbox_cls:
            mock_sandbox_cls.from_id.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["get", "bad-id"])

        assert result.exit_code == 1
        assert "not found" in result.output
