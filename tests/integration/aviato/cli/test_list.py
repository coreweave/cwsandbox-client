"""Integration tests for aviato list command.

These tests require a running Aviato backend.
Authentication is read from ~/.netrc or environment variables.
"""

import json

from click.testing import CliRunner

from aviato import Sandbox
from aviato.cli.main import cli


class TestListCommandIntegration:
    """Integration tests for aviato list command."""

    def test_list_returns_success(self) -> None:
        """Test list command runs successfully against real API."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0

    def test_list_shows_created_sandbox(self, test_sandbox: Sandbox, cli_test_tag: str) -> None:
        """Test list command shows a sandbox we just created."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tag", cli_test_tag])
        assert result.exit_code == 0
        assert test_sandbox.sandbox_id in result.output

    def test_list_json_output_is_valid(self, test_sandbox: Sandbox, cli_test_tag: str) -> None:
        """Test list --output json returns valid JSON."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tag", cli_test_tag, "-o", "json"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) >= 1

        # Find our test sandbox
        sandbox_ids = [sb["id"] for sb in data]
        assert test_sandbox.sandbox_id in sandbox_ids

    def test_list_quiet_output_contains_id(self, test_sandbox: Sandbox, cli_test_tag: str) -> None:
        """Test list --output quiet returns only IDs."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tag", cli_test_tag, "-o", "quiet"])
        assert result.exit_code == 0
        assert test_sandbox.sandbox_id in result.output

        # Should be just IDs, one per line (no headers or extra text)
        lines = [line for line in result.output.strip().split("\n") if line]
        assert len(lines) >= 1
        # Each line should be a valid sandbox ID (UUID or prefixed format)
        for line in lines:
            # Should not contain spaces (which would indicate table format)
            assert " " not in line.strip()

    def test_list_status_filter_running(self, test_sandbox: Sandbox, cli_test_tag: str) -> None:
        """Test list --status running returns running sandboxes."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["list", "--tag", cli_test_tag, "--status", "running"]
        )
        assert result.exit_code == 0
        assert test_sandbox.sandbox_id in result.output

    def test_list_status_filter_completed_excludes_running(
        self, test_sandbox: Sandbox, cli_test_tag: str
    ) -> None:
        """Test list --status completed excludes our running sandbox."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["list", "--tag", cli_test_tag, "--status", "completed", "-o", "quiet"]
        )
        assert result.exit_code == 0
        # Our test sandbox is running, not completed
        assert test_sandbox.sandbox_id not in result.output

    def test_list_verbose_shows_extra_columns(
        self, test_sandbox: Sandbox, cli_test_tag: str
    ) -> None:
        """Test list --verbose shows additional columns."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--tag", cli_test_tag, "-v"])
        assert result.exit_code == 0
        assert "TOWER" in result.output
        assert "RUNWAY" in result.output

    def test_list_nonexistent_tag_returns_empty(self) -> None:
        """Test list with nonexistent tag returns no results."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["list", "--tag", "nonexistent-tag-12345", "-o", "quiet"]
        )
        assert result.exit_code == 0
        # Should be empty or "No sandboxes found"
        assert result.output.strip() == "" or "No sandboxes found" in result.output
