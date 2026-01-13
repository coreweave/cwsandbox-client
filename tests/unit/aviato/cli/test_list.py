"""Unit tests for aviato list command."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from aviato import SandboxStatus
from aviato.cli.main import cli


def _mock_operation_ref(
    return_value: object = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock OperationRef that returns value from .result()."""
    mock_ref = MagicMock()
    if side_effect is not None:
        mock_ref.result.side_effect = side_effect
    else:
        mock_ref.result.return_value = return_value
    return mock_ref


class TestListCommand:
    """Tests for aviato list command."""

    def test_list_no_sandboxes(self) -> None:
        """Test list command with no sandboxes."""
        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([])
            runner = CliRunner()
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0
            assert "No sandboxes found" in result.output

    def test_list_with_sandboxes(self) -> None:
        """Test list command with sandboxes."""
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "sb-test123"
        mock_sb.status = SandboxStatus.RUNNING
        mock_sb.started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)

        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([mock_sb])
            runner = CliRunner()
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0
            assert "sb-test123" in result.output
            assert SandboxStatus.RUNNING.value in result.output

    def test_list_status_filter(self) -> None:
        """Test list command with status filter."""
        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([])
            runner = CliRunner()
            result = runner.invoke(cli, ["list", "--status", SandboxStatus.RUNNING.value])
            assert result.exit_code == 0
            mock_list.assert_called_once_with(tags=None, status=SandboxStatus.RUNNING.value)

    def test_list_tag_filter(self) -> None:
        """Test list command with tag filter."""
        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([])
            runner = CliRunner()
            result = runner.invoke(cli, ["list", "--tag", "my-tag"])
            assert result.exit_code == 0
            mock_list.assert_called_once_with(tags=["my-tag"], status=None)

    def test_list_multiple_tags(self) -> None:
        """Test list command with multiple tag filters."""
        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([])
            runner = CliRunner()
            result = runner.invoke(cli, ["list", "--tag", "tag1", "--tag", "tag2"])
            assert result.exit_code == 0
            mock_list.assert_called_once_with(tags=["tag1", "tag2"], status=None)

    def test_list_json_output(self) -> None:
        """Test list command with JSON output."""
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "sb-json123"
        mock_sb.status = SandboxStatus.COMPLETED
        mock_sb.started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)

        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([mock_sb])
            runner = CliRunner()
            result = runner.invoke(cli, ["list", "-o", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) == 1
            assert data[0]["id"] == "sb-json123"
            assert data[0]["status"] == SandboxStatus.COMPLETED

    def test_list_quiet_output(self) -> None:
        """Test list command with quiet output."""
        mock_sb1 = MagicMock()
        mock_sb1.sandbox_id = "sb-quiet1"
        mock_sb2 = MagicMock()
        mock_sb2.sandbox_id = "sb-quiet2"

        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([mock_sb1, mock_sb2])
            runner = CliRunner()
            result = runner.invoke(cli, ["list", "-o", "quiet"])
            assert result.exit_code == 0
            lines = result.output.strip().split("\n")
            assert "sb-quiet1" in lines
            assert "sb-quiet2" in lines

    def test_list_error_handling(self) -> None:
        """Test list command handles errors gracefully."""
        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref(side_effect=Exception("Connection failed"))
            runner = CliRunner()
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 1
            assert "Error" in result.output
            assert "Connection failed" in result.output

    def test_list_verbose_output(self) -> None:
        """Test list command with verbose flag."""
        mock_sb = MagicMock()
        mock_sb.sandbox_id = "sb-verbose"
        mock_sb.status = SandboxStatus.RUNNING
        mock_sb.started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
        mock_sb.tower_id = "tower-xyz"
        mock_sb.runway_id = "runway-789"

        with patch("aviato.cli.sandbox.Sandbox.list") as mock_list:
            mock_list.return_value = _mock_operation_ref([mock_sb])
            runner = CliRunner()
            result = runner.invoke(cli, ["list", "-v"])
            assert result.exit_code == 0
            assert "TOWER" in result.output
            assert "RUNWAY" in result.output
