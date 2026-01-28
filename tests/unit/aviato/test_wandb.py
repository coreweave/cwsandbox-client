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
        assert reporter._exec_successes == 0
        assert reporter._exec_failures == 0
        assert reporter._exec_errors == 0
        assert reporter._startup_count == 0
        assert reporter._startup_total_seconds == 0.0
        assert reporter._startup_min_seconds is None
        assert reporter._startup_max_seconds is None
        assert reporter._exec_per_sandbox == {}

    def test_record_sandbox_created_increments_counter(self) -> None:
        """Test record_sandbox_created increments the counter."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        reporter.record_sandbox_created()
        assert reporter._sandboxes_created == 1

        reporter.record_sandbox_created()
        assert reporter._sandboxes_created == 2

    def test_record_exec_outcome_success(self) -> None:
        """Test record_exec_outcome with SUCCESS."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.SUCCESS)

        assert reporter._executions == 1
        assert reporter._exec_successes == 1
        assert reporter._exec_failures == 0
        assert reporter._exec_errors == 0

    def test_record_exec_outcome_failure(self) -> None:
        """Test record_exec_outcome with FAILURE."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.FAILURE)

        assert reporter._executions == 1
        assert reporter._exec_successes == 0
        assert reporter._exec_failures == 1
        assert reporter._exec_errors == 0

    def test_record_exec_outcome_error(self) -> None:
        """Test record_exec_outcome with ERROR."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.ERROR)

        assert reporter._executions == 1
        assert reporter._exec_successes == 0
        assert reporter._exec_failures == 0
        assert reporter._exec_errors == 1

    def test_record_exec_outcome_with_sandbox_id(self) -> None:
        """Test record_exec_outcome tracks per-sandbox counts."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.FAILURE, sandbox_id="sb-2")

        assert reporter._exec_per_sandbox == {"sb-1": 2, "sb-2": 1}
        assert reporter._executions == 3

    def test_record_exec_outcome_without_sandbox_id_does_not_track(self) -> None:
        """Test record_exec_outcome without sandbox_id doesn't update per-sandbox."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.SUCCESS)
        reporter.record_exec_outcome(ExecOutcome.FAILURE)

        assert reporter._exec_per_sandbox == {}
        assert reporter._executions == 2

    def test_record_startup_time_updates_all_stats(self) -> None:
        """Test record_startup_time updates count, total, min, and max."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        reporter.record_startup_time(5.0)
        assert reporter._startup_count == 1
        assert reporter._startup_total_seconds == 5.0
        assert reporter._startup_min_seconds == 5.0
        assert reporter._startup_max_seconds == 5.0

        reporter.record_startup_time(3.0)
        assert reporter._startup_count == 2
        assert reporter._startup_total_seconds == 8.0
        assert reporter._startup_min_seconds == 3.0
        assert reporter._startup_max_seconds == 5.0

        reporter.record_startup_time(7.0)
        assert reporter._startup_count == 3
        assert reporter._startup_total_seconds == 15.0
        assert reporter._startup_min_seconds == 3.0
        assert reporter._startup_max_seconds == 7.0

    def test_get_metrics_returns_correct_values(self) -> None:
        """Test get_metrics returns accumulated values."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.SUCCESS)
        reporter.record_exec_outcome(ExecOutcome.SUCCESS)
        reporter.record_exec_outcome(ExecOutcome.FAILURE)
        reporter.record_exec_outcome(ExecOutcome.ERROR)

        metrics = reporter.get_metrics()

        assert metrics["aviato/sandboxes_created"] == 2
        assert metrics["aviato/executions"] == 4
        assert metrics["aviato/exec_successes"] == 2
        assert metrics["aviato/exec_failures"] == 1
        assert metrics["aviato/exec_errors"] == 1
        assert metrics["aviato/success_rate"] == pytest.approx(2 / 4)
        assert metrics["aviato/error_rate"] == pytest.approx(1 / 4)

    def test_get_metrics_omits_rates_with_no_executions(self) -> None:
        """Test get_metrics omits rates when no executions."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()

        metrics = reporter.get_metrics()

        assert "aviato/sandboxes_created" in metrics
        assert "aviato/success_rate" not in metrics
        assert "aviato/error_rate" not in metrics

    def test_get_metrics_includes_startup_stats(self) -> None:
        """Test get_metrics includes startup stats when count > 0."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_startup_time(2.0)
        reporter.record_startup_time(4.0)
        reporter.record_startup_time(6.0)

        metrics = reporter.get_metrics()

        assert metrics["aviato/startup_count"] == 3
        assert metrics["aviato/avg_startup_seconds"] == pytest.approx(4.0)
        assert metrics["aviato/min_startup_seconds"] == 2.0
        assert metrics["aviato/max_startup_seconds"] == 6.0

    def test_get_metrics_omits_startup_stats_when_no_startup(self) -> None:
        """Test get_metrics omits startup stats when count == 0."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()

        metrics = reporter.get_metrics()

        assert "aviato/startup_count" not in metrics
        assert "aviato/avg_startup_seconds" not in metrics
        assert "aviato/min_startup_seconds" not in metrics
        assert "aviato/max_startup_seconds" not in metrics

    def test_get_metrics_includes_per_sandbox_stats(self) -> None:
        """Test get_metrics includes per-sandbox exec stats when tracking enabled."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-2")

        metrics = reporter.get_metrics()

        assert metrics["aviato/avg_execs_per_sandbox"] == pytest.approx(2.0)
        assert metrics["aviato/min_execs_per_sandbox"] == 1
        assert metrics["aviato/max_execs_per_sandbox"] == 3

    def test_get_metrics_omits_per_sandbox_stats_when_empty(self) -> None:
        """Test get_metrics omits per-sandbox stats when no sandbox_id provided."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()
        reporter.record_exec_outcome(ExecOutcome.SUCCESS)
        reporter.record_exec_outcome(ExecOutcome.FAILURE)

        metrics = reporter.get_metrics()

        assert "aviato/avg_execs_per_sandbox" not in metrics
        assert "aviato/min_execs_per_sandbox" not in metrics
        assert "aviato/max_execs_per_sandbox" not in metrics

    def test_reset_clears_metrics(self) -> None:
        """Test reset clears all accumulated metrics."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.FAILURE, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.ERROR)
        reporter.record_startup_time(5.0)

        reporter.reset()

        assert reporter._sandboxes_created == 0
        assert reporter._executions == 0
        assert reporter._exec_successes == 0
        assert reporter._exec_failures == 0
        assert reporter._exec_errors == 0
        assert reporter._startup_count == 0
        assert reporter._startup_total_seconds == 0.0
        assert reporter._startup_min_seconds is None
        assert reporter._startup_max_seconds is None
        assert reporter._exec_per_sandbox == {}

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
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()
        reporter.record_exec_outcome(ExecOutcome.SUCCESS)
        assert reporter.has_metrics is True

    def test_has_metrics_true_with_startup_times(self) -> None:
        """Test has_metrics returns True when only startup times recorded."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_startup_time(5.0)
        assert reporter.has_metrics is True

    def test_has_metrics_false_after_reset(self) -> None:
        """Test has_metrics returns False after reset clears all metrics."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.SUCCESS)
        reporter.record_startup_time(5.0)
        assert reporter.has_metrics is True

        reporter.reset()
        assert reporter.has_metrics is False


class TestWandbReporterPerSandbox:
    """Tests for per-sandbox exec tracking in WandbReporter."""

    def test_init_creates_empty_per_sandbox_dict(self) -> None:
        """Test WandbReporter initializes with empty per-sandbox dict."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()

        assert reporter._exec_per_sandbox == {}

    def test_record_exec_outcome_with_sandbox_id(self) -> None:
        """Test record_exec_outcome tracks per-sandbox count."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-123")

        assert reporter._exec_per_sandbox["sb-123"] == 1

    def test_record_exec_outcome_multiple_sandboxes(self) -> None:
        """Test record_exec_outcome tracks multiple sandboxes independently."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        # 3 execs for sandbox A
        for _ in range(3):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-a")
        # 5 execs for sandbox B
        for _ in range(5):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-b")
        # 8 execs for sandbox C
        for _ in range(8):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-c")

        assert reporter._exec_per_sandbox["sb-a"] == 3
        assert reporter._exec_per_sandbox["sb-b"] == 5
        assert reporter._exec_per_sandbox["sb-c"] == 8

    def test_record_exec_outcome_none_sandbox_id_updates_global_only(self) -> None:
        """Test record_exec_outcome without sandbox_id updates global but not per-sandbox."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id=None)

        assert reporter._executions == 1
        assert reporter._exec_per_sandbox == {}

    def test_get_metrics_per_sandbox_avg_min_max(self) -> None:
        """Test get_metrics returns correct avg/min/max for per-sandbox counts."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        # Track 3 sandboxes with exec counts [3, 5, 8]
        for _ in range(3):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-a")
        for _ in range(5):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-b")
        for _ in range(8):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-c")

        metrics = reporter.get_metrics()

        # avg = (3 + 5 + 8) / 3 = 5.33...
        assert metrics["aviato/avg_execs_per_sandbox"] == pytest.approx(5.33, rel=0.01)
        assert metrics["aviato/min_execs_per_sandbox"] == 3
        assert metrics["aviato/max_execs_per_sandbox"] == 8

    def test_get_metrics_single_sandbox(self) -> None:
        """Test get_metrics with single sandbox has avg=min=max."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        # Track 1 sandbox with 5 execs
        for _ in range(5):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-only")

        metrics = reporter.get_metrics()

        assert metrics["aviato/avg_execs_per_sandbox"] == 5
        assert metrics["aviato/min_execs_per_sandbox"] == 5
        assert metrics["aviato/max_execs_per_sandbox"] == 5

    def test_get_metrics_omits_per_sandbox_when_empty(self) -> None:
        """Test get_metrics omits per-sandbox keys when no sandbox_id provided."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.SUCCESS)

        metrics = reporter.get_metrics()

        assert "aviato/avg_execs_per_sandbox" not in metrics
        assert "aviato/min_execs_per_sandbox" not in metrics
        assert "aviato/max_execs_per_sandbox" not in metrics

    def test_reset_clears_per_sandbox_data(self) -> None:
        """Test reset clears per-sandbox tracking data."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-1")
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-2")

        reporter.reset()

        assert reporter._exec_per_sandbox == {}

    def test_per_sandbox_with_different_exec_counts(self) -> None:
        """Test per-sandbox tracking with varying exec counts."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        # Sandbox A: 1 exec, Sandbox B: 10 execs, Sandbox C: 3 execs
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-a")
        for _ in range(10):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-b")
        for _ in range(3):
            reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-c")

        metrics = reporter.get_metrics()

        assert metrics["aviato/min_execs_per_sandbox"] == 1
        assert metrics["aviato/max_execs_per_sandbox"] == 10
        # avg = (1 + 10 + 3) / 3 = 4.67
        assert metrics["aviato/avg_execs_per_sandbox"] == pytest.approx(4.67, rel=0.01)

    def test_per_sandbox_independent_of_outcome_types(self) -> None:
        """Test per-sandbox count tracks total execs regardless of outcome type."""
        from aviato._wandb import ExecOutcome, WandbReporter

        reporter = WandbReporter()

        # Record different outcome types for same sandbox
        reporter.record_exec_outcome(ExecOutcome.SUCCESS, sandbox_id="sb-mixed")
        reporter.record_exec_outcome(ExecOutcome.FAILURE, sandbox_id="sb-mixed")
        reporter.record_exec_outcome(ExecOutcome.ERROR, sandbox_id="sb-mixed")

        # Per-sandbox count should be 3 (total, not by outcome)
        assert reporter._exec_per_sandbox["sb-mixed"] == 3


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

    def test_get_run_rechecks_when_no_run_found(self) -> None:
        """Test _get_run re-checks for wandb.run until one is found.

        This tests the late wandb.init scenario: Session is created before
        wandb.init() is called, so the first _get_run() finds no active run.
        Later, after wandb.init() is called, _get_run() should find the run.
        """
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        mock_wandb = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # First call: no run yet (simulates before wandb.init)
            mock_wandb.run = None
            result1 = reporter._get_run()
            assert result1 is None

            # Second call: run now exists (simulates after wandb.init)
            mock_wandb.run = mock_run
            result2 = reporter._get_run()
            assert result2 is mock_run

    def test_log_finds_run_after_late_wandb_init(self) -> None:
        """Test log() works after late wandb.init.

        This tests the full scenario: create reporter, call log() (returns False
        because no run), then wandb.init() is called, then log() (returns True).
        """
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        reporter.record_sandbox_created()

        mock_run = MagicMock()
        mock_wandb = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # First call: no run (returns False)
            mock_wandb.run = None
            result1 = reporter.log()
            assert result1 is False

            # Second call: run exists (returns True)
            mock_wandb.run = mock_run
            result2 = reporter.log()
            assert result2 is True
            mock_run.log.assert_called_once()


class TestWandbReporterLog:
    """Tests for WandbReporter.log method."""

    def test_log_returns_false_when_no_run(self) -> None:
        """Test log returns False when no wandb run available."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = reporter.log()

        assert result is False

    def test_log_returns_false_when_no_metrics(self) -> None:
        """Test log returns False when no metrics to log."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
        reporter._run = mock_run

        result = reporter.log()

        assert result is False

    def test_log_calls_wandb_log_without_step(self) -> None:
        """Test log calls wandb.run.log without step."""
        from aviato._wandb import WandbReporter

        reporter = WandbReporter()
        mock_run = MagicMock()
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
        reporter._run = mock_run
        reporter.record_sandbox_created()

        result = reporter.log()

        assert result is False
