"""Unit tests for aviato._integrations module."""

from unittest.mock import MagicMock, patch

import pytest


class TestIsWandbAvailable:
    """Tests for is_wandb_available function."""

    def test_returns_true_when_wandb_installed(self) -> None:
        """Test returns True when wandb can be imported."""
        from aviato import _integrations

        _integrations._wandb_available = None

        with patch.dict("sys.modules", {"wandb": MagicMock()}):
            result = _integrations.is_wandb_available()

        assert result is True

    def test_returns_false_when_wandb_not_installed(self) -> None:
        """Test returns False when wandb import fails."""
        from aviato import _integrations

        _integrations._wandb_available = None

        with patch.dict("sys.modules", {"wandb": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = _integrations.is_wandb_available()

        assert result is False

    def test_caches_result(self) -> None:
        """Test result is cached after first check."""
        from aviato import _integrations

        _integrations._wandb_available = True
        result = _integrations.is_wandb_available()
        assert result is True

        _integrations._wandb_available = None


class TestHasWandbCredentials:
    """Tests for has_wandb_credentials function."""

    def test_returns_true_when_api_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test returns True when WANDB_API_KEY is set."""
        from aviato._integrations import has_wandb_credentials

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        assert has_wandb_credentials() is True

    def test_returns_false_when_api_key_not_set(self) -> None:
        """Test returns False when WANDB_API_KEY is not set."""
        from aviato._integrations import has_wandb_credentials

        assert has_wandb_credentials() is False


class TestHasActiveWandbRun:
    """Tests for has_active_wandb_run function."""

    def test_returns_false_when_wandb_not_available(self) -> None:
        """Test returns False when wandb is not installed."""
        from aviato import _integrations

        _integrations._wandb_available = False
        result = _integrations.has_active_wandb_run()
        assert result is False
        _integrations._wandb_available = None

    def test_returns_true_when_run_active(self) -> None:
        """Test returns True when wandb.run is not None."""
        from aviato import _integrations

        _integrations._wandb_available = True

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = _integrations.has_active_wandb_run()

        assert result is True
        _integrations._wandb_available = None

    def test_returns_false_when_no_active_run(self) -> None:
        """Test returns False when wandb.run is None."""
        from aviato import _integrations

        _integrations._wandb_available = True

        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = _integrations.has_active_wandb_run()

        assert result is False
        _integrations._wandb_available = None


class TestShouldAutoReportWandb:
    """Tests for should_auto_report_wandb function."""

    def test_returns_false_when_no_credentials(self) -> None:
        """Test returns False when WANDB_API_KEY not set."""
        from aviato._integrations import should_auto_report_wandb

        assert should_auto_report_wandb() is False

    def test_returns_false_when_no_active_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test returns False when no active wandb run."""
        from aviato import _integrations

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        _integrations._wandb_available = True

        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = _integrations.should_auto_report_wandb()

        assert result is False
        _integrations._wandb_available = None

    def test_returns_true_when_credentials_and_active_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test returns True when both credentials and active run exist."""
        from aviato import _integrations

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        _integrations._wandb_available = True

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = _integrations.should_auto_report_wandb()

        assert result is True
        _integrations._wandb_available = None
