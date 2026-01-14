"""Unit tests for aviato.cli.formatters module."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

from aviato.cli.formatters import (
    _format_age,
    format_sandbox_json,
    format_sandbox_quiet,
    format_sandbox_table,
)


class TestFormatAge:
    """Tests for _format_age helper."""

    def test_format_age_seconds(self) -> None:
        """Test formatting age in seconds."""
        from unittest.mock import patch

        with patch("aviato.cli.formatters.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 8, 12, 0, 30, tzinfo=UTC)
            started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
            result = _format_age(started_at)
            assert result == "30s"

    def test_format_age_minutes(self) -> None:
        """Test formatting age in minutes."""
        from unittest.mock import patch

        with patch("aviato.cli.formatters.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 8, 12, 5, 0, tzinfo=UTC)
            mock_dt.side_effect = datetime
            started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
            result = _format_age(started_at)
            assert result == "5m"

    def test_format_age_hours(self) -> None:
        """Test formatting age in hours."""
        from unittest.mock import patch

        with patch("aviato.cli.formatters.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 8, 14, 0, 0, tzinfo=UTC)
            started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
            result = _format_age(started_at)
            assert result == "2h"

    def test_format_age_days(self) -> None:
        """Test formatting age in days."""
        from unittest.mock import patch

        with patch("aviato.cli.formatters.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 10, 12, 0, 0, tzinfo=UTC)
            started_at = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
            result = _format_age(started_at)
            assert result == "2d"

    def test_format_age_future_timestamp(self) -> None:
        """Test formatting age when timestamp is in the future returns '-'."""
        from unittest.mock import patch

        with patch("aviato.cli.formatters.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)
            started_at = datetime(2025, 1, 8, 13, 0, 0, tzinfo=UTC)  # 1 hour in future
            result = _format_age(started_at)
            assert result == "-"


class TestFormatSandboxTable:
    """Tests for format_sandbox_table."""

    def test_empty_list(self) -> None:
        """Test formatting empty sandbox list."""
        result = format_sandbox_table([])
        assert result == "No sandboxes found."

    def test_single_sandbox(self, mock_sandbox: MagicMock) -> None:
        """Test formatting a single sandbox."""
        result = format_sandbox_table([mock_sandbox])
        assert "sb-abc123" in result
        assert "running" in result
        assert "ID" in result
        assert "STATUS" in result
        assert "AGE" in result

    def test_multiple_sandboxes(self, mock_sandboxes: list[MagicMock]) -> None:
        """Test formatting multiple sandboxes."""
        result = format_sandbox_table(mock_sandboxes)
        assert "sb-abc123" in result
        assert "sb-def456" in result
        assert "running" in result
        assert "completed" in result

    def test_verbose_mode(self, mock_sandbox: MagicMock) -> None:
        """Test verbose mode shows additional columns."""
        result = format_sandbox_table([mock_sandbox], verbose=True)
        assert "TOWER" in result
        assert "RUNWAY" in result
        assert "tower-abc" in result
        assert "runway-123" in result


class TestFormatSandboxJson:
    """Tests for format_sandbox_json."""

    def test_empty_list(self) -> None:
        """Test formatting empty sandbox list as JSON."""
        result = format_sandbox_json([])
        data = json.loads(result)
        assert data == []

    def test_single_sandbox(self, mock_sandbox: MagicMock) -> None:
        """Test formatting a single sandbox as JSON."""
        result = format_sandbox_json([mock_sandbox])
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["id"] == "sb-abc123"
        assert data[0]["status"] == "running"
        assert data[0]["started_at"] is not None

    def test_multiple_sandboxes(self, mock_sandboxes: list[MagicMock]) -> None:
        """Test formatting multiple sandboxes as JSON."""
        result = format_sandbox_json(mock_sandboxes)
        data = json.loads(result)
        assert len(data) == 2
        ids = [sb["id"] for sb in data]
        assert "sb-abc123" in ids
        assert "sb-def456" in ids


class TestFormatSandboxQuiet:
    """Tests for format_sandbox_quiet."""

    def test_empty_list(self) -> None:
        """Test formatting empty sandbox list as quiet."""
        result = format_sandbox_quiet([])
        assert result == ""

    def test_single_sandbox(self, mock_sandbox: MagicMock) -> None:
        """Test formatting a single sandbox as quiet."""
        result = format_sandbox_quiet([mock_sandbox])
        assert result == "sb-abc123"

    def test_multiple_sandboxes(self, mock_sandboxes: list[MagicMock]) -> None:
        """Test formatting multiple sandboxes as quiet."""
        result = format_sandbox_quiet(mock_sandboxes)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "sb-abc123" in lines
        assert "sb-def456" in lines
