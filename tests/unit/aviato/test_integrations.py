"""Unit tests for aviato._integrations module."""

from unittest.mock import MagicMock, patch

from aviato._integrations import (
    has_active_wandb_run,
    has_wandb_credentials,
    is_wandb_available,
    should_auto_report_wandb,
)


class TestIsWandbAvailable:
    """Tests for is_wandb_available function."""

    def test_returns_true_when_importable(self) -> None:
        """Test returns True when wandb can be imported."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert is_wandb_available()

    def test_returns_false_when_not_importable(self) -> None:
        """Test returns False when wandb cannot be imported."""
        with patch.dict("sys.modules", {"wandb": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert not is_wandb_available()


class TestHasWandbCredentials:
    """Tests for has_wandb_credentials function."""

    def test_returns_true_when_api_key_set(self) -> None:
        """Test returns True when WANDB_API_KEY is set."""
        with patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
            assert has_wandb_credentials()

    def test_returns_false_when_api_key_not_set(self) -> None:
        """Test returns False when WANDB_API_KEY is not set and no netrc."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("aviato._integrations._read_api_key_from_netrc", return_value=None):
                assert not has_wandb_credentials()

    def test_returns_false_when_api_key_empty(self) -> None:
        """Test returns False when WANDB_API_KEY is empty and no netrc."""
        with patch.dict("os.environ", {"WANDB_API_KEY": ""}):
            with patch("aviato._integrations._read_api_key_from_netrc", return_value=None):
                assert not has_wandb_credentials()

    def test_returns_true_when_netrc_credentials_found(self) -> None:
        """Test returns True when netrc has W&B credentials."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("aviato._integrations._read_api_key_from_netrc", return_value="netrc-key"):
                assert has_wandb_credentials()


class TestHasActiveWandbRun:
    """Tests for has_active_wandb_run function."""

    def test_returns_true_when_run_active(self) -> None:
        """Test returns True when wandb.run is not None."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert has_active_wandb_run()

    def test_returns_false_when_no_run(self) -> None:
        """Test returns False when wandb.run is None."""
        mock_wandb = MagicMock()
        mock_wandb.run = None
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert not has_active_wandb_run()

    def test_returns_false_when_wandb_not_available(self) -> None:
        """Test returns False when wandb cannot be imported."""
        with patch.dict("sys.modules", {"wandb": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert not has_active_wandb_run()

    def test_returns_false_on_exception(self) -> None:
        """Test returns False when checking run raises exception."""
        mock_wandb = MagicMock()
        type(mock_wandb).run = property(lambda self: (_ for _ in ()).throw(RuntimeError))
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            assert not has_active_wandb_run()


class TestShouldAutoReportWandb:
    """Tests for should_auto_report_wandb function."""

    def test_returns_true_when_all_conditions_met(self) -> None:
        """Test returns True when all conditions are met."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
                assert should_auto_report_wandb()

    def test_returns_false_when_wandb_not_available(self) -> None:
        """Test returns False when wandb not importable."""
        with patch.dict("sys.modules", {"wandb": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
                    assert not should_auto_report_wandb()

    def test_returns_false_when_no_credentials(self) -> None:
        """Test returns False when no API key and no netrc."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch.dict("os.environ", {}, clear=True):
                with patch("aviato._integrations._read_api_key_from_netrc", return_value=None):
                    assert not should_auto_report_wandb()

    def test_returns_false_when_no_active_run(self) -> None:
        """Test returns False when no active run."""
        mock_wandb = MagicMock()
        mock_wandb.run = None
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
                assert not should_auto_report_wandb()
