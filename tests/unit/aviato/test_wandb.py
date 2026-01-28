"""Unit tests for aviato._wandb module."""

from unittest.mock import MagicMock, patch

import pytest


class TestWandbReporter:
    """Tests for WandbReporter class."""

    def test_init_creates_empty_metrics(self) -> None:
        """Test WandbReporter initializes with empty metrics."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        assert reporter._sandboxes_created == 0
        assert reporter._executions == 0
        assert reporter._successful_executions == 0

    def test_record_sandbox_created_increments_counter(self) -> None:
        """Test record_sandbox_created increments the counter."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        reporter.record_sandbox_created()
        assert reporter._sandboxes_created == 1

        reporter.record_sandbox_created()
        assert reporter._sandboxes_created == 2

    def test_record_execution_success(self) -> None:
        """Test record_execution with success=True."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        reporter.record_execution(success=True)

        assert reporter._executions == 1
        assert reporter._successful_executions == 1

    def test_record_execution_failure(self) -> None:
        """Test record_execution with success=False."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        reporter.record_execution(success=False)

        assert reporter._executions == 1
        assert reporter._successful_executions == 0

    def test_get_metrics_returns_correct_values(self) -> None:
        """Test get_metrics returns accumulated values."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_sandbox_created()
        reporter.record_execution(success=True)
        reporter.record_execution(success=True)
        reporter.record_execution(success=False)

        metrics = reporter.get_metrics()

        assert metrics["aviato/sandboxes_created"] == 2
        assert metrics["aviato/executions"] == 3
        assert metrics["aviato/success_rate"] == pytest.approx(2 / 3)

    def test_get_metrics_omits_success_rate_with_no_executions(self) -> None:
        """Test get_metrics omits success_rate when no executions."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()

        metrics = reporter.get_metrics()

        assert "aviato/sandboxes_created" in metrics
        assert "aviato/success_rate" not in metrics

    def test_reset_clears_metrics(self) -> None:
        """Test reset clears all accumulated metrics."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_execution(success=True)

        reporter.reset()

        assert reporter._sandboxes_created == 0
        assert reporter._executions == 0
        assert reporter._successful_executions == 0

    def test_has_metrics_false_when_empty(self) -> None:
        """Test has_metrics returns False with no metrics."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        assert reporter.has_metrics is False

    def test_has_metrics_true_with_sandboxes(self) -> None:
        """Test has_metrics returns True with sandbox count."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()
        assert reporter.has_metrics is True

    def test_has_metrics_true_with_executions(self) -> None:
        """Test has_metrics returns True with execution count."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_execution(success=True)
        assert reporter.has_metrics is True


class TestWandbReporterGetRun:
    """Tests for WandbReporter._get_run method."""

    def test_get_run_returns_none_when_wandb_not_installed(self) -> None:
        """Test _get_run returns None when wandb import fails."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        with patch("builtins.__import__", side_effect=ImportError):
            result = reporter._get_run()

        assert result is None

    def test_get_run_returns_none_when_no_active_run(self) -> None:
        """Test _get_run returns None when wandb.run is None."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = reporter._get_run()

        assert result is None

    def test_get_run_returns_run_when_active(self) -> None:
        """Test _get_run returns wandb.run when active."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.run = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = reporter._get_run()

        assert result is mock_run

    def test_get_run_caches_result(self) -> None:
        """Test _get_run caches the run after first check."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_wandb.run = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result1 = reporter._get_run()
            mock_wandb.run = MagicMock()
            result2 = reporter._get_run()

        assert result1 is result2


class TestWandbReporterLog:
    """Tests for WandbReporter.log method."""

    def test_log_returns_false_when_no_run(self) -> None:
        """Test log returns False when no wandb run available."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter._run_checked = True
        reporter._run = None

        result = reporter.log()

        assert result is False

    def test_log_returns_false_when_no_metrics(self) -> None:
        """Test log returns False when no metrics to log."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        reporter._run_checked = True
        reporter._run = mock_run

        result = reporter.log()

        assert result is False

    def test_log_calls_wandb_log_without_step(self) -> None:
        """Test log calls wandb.run.log without step."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        reporter._run_checked = True
        reporter._run = mock_run
        reporter.record_sandbox_created()

        result = reporter.log()

        assert result is True
        mock_run.log.assert_called_once()
        call_args = mock_run.log.call_args
        assert "aviato/sandboxes_created" in call_args[0][0]
        assert "step" not in call_args[1] or call_args[1]["step"] is None

    def test_log_calls_wandb_log_with_step(self) -> None:
        """Test log calls wandb.run.log with step."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        reporter._run_checked = True
        reporter._run = mock_run
        reporter.record_sandbox_created()

        result = reporter.log(step=100)

        assert result is True
        mock_run.log.assert_called_once()
        call_args = mock_run.log.call_args
        assert call_args[1]["step"] == 100

    def test_log_returns_false_on_exception(self) -> None:
        """Test log returns False when wandb.log raises exception."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        mock_run.log.side_effect = Exception("Network error")
        reporter._run_checked = True
        reporter._run = mock_run
        reporter.record_sandbox_created()

        result = reporter.log()

        assert result is False
