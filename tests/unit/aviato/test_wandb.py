# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Unit tests for aviato._wandb module."""

from unittest.mock import MagicMock, patch

from aviato._wandb import ExecOutcome, WandbReporter


class TestExecOutcome:
    """Tests for ExecOutcome enum."""

    def test_values(self) -> None:
        """Test ExecOutcome enum values."""
        assert ExecOutcome.COMPLETED_OK == "completed_ok"
        assert ExecOutcome.COMPLETED_NONZERO == "completed_nonzero"
        assert ExecOutcome.FAILURE == "failure"


class TestWandbReporter:
    """Tests for WandbReporter class."""

    def test_initial_state(self) -> None:
        """Test reporter starts with zeroed counters."""
        reporter = WandbReporter()
        assert not reporter.has_metrics
        assert reporter.get_metrics() == {
            "aviato/sandboxes_created": 0,
            "aviato/exec_completed_ok": 0,
            "aviato/exec_completed_nonzero": 0,
            "aviato/exec_failures": 0,
        }

    def test_record_sandbox_created(self) -> None:
        """Test recording sandbox creation."""
        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_sandbox_created()

        assert reporter.has_metrics
        metrics = reporter.get_metrics()
        assert metrics["aviato/sandboxes_created"] == 2

    def test_record_exec_outcome_completed_ok(self) -> None:
        """Test recording successful exec completion."""
        reporter = WandbReporter()
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_OK)
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_OK, "sandbox-123")

        assert reporter.has_metrics
        metrics = reporter.get_metrics()
        assert metrics["aviato/exec_completed_ok"] == 2
        assert metrics["aviato/exec_completed_nonzero"] == 0
        assert metrics["aviato/exec_failures"] == 0

    def test_record_exec_outcome_completed_nonzero(self) -> None:
        """Test recording non-zero exit code completion."""
        reporter = WandbReporter()
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_NONZERO)

        metrics = reporter.get_metrics()
        assert metrics["aviato/exec_completed_nonzero"] == 1
        assert metrics["aviato/exec_completed_ok"] == 0

    def test_record_exec_outcome_failure(self) -> None:
        """Test recording exec failure."""
        reporter = WandbReporter()
        reporter.record_exec_outcome(ExecOutcome.FAILURE)

        metrics = reporter.get_metrics()
        assert metrics["aviato/exec_failures"] == 1

    def test_record_startup_time(self) -> None:
        """Test recording startup times."""
        reporter = WandbReporter()
        reporter.record_startup_time(2.5)
        reporter.record_startup_time(3.5)

        assert reporter.has_metrics
        metrics = reporter.get_metrics()
        assert metrics["aviato/avg_startup_time"] == 3.0

    def test_get_metrics_no_startup_times(self) -> None:
        """Test get_metrics excludes avg_startup_time when no times recorded."""
        reporter = WandbReporter()
        reporter.record_sandbox_created()

        metrics = reporter.get_metrics()
        assert "aviato/avg_startup_time" not in metrics

    def test_reset(self) -> None:
        """Test resetting all counters."""
        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_OK)
        reporter.record_startup_time(1.0)

        assert reporter.has_metrics

        reporter.reset()

        assert not reporter.has_metrics
        assert reporter.get_metrics() == {
            "aviato/sandboxes_created": 0,
            "aviato/exec_completed_ok": 0,
            "aviato/exec_completed_nonzero": 0,
            "aviato/exec_failures": 0,
        }

    def test_has_metrics_false_when_empty(self) -> None:
        """Test has_metrics returns False when no metrics recorded."""
        reporter = WandbReporter()
        assert not reporter.has_metrics

    def test_has_metrics_true_with_any_counter(self) -> None:
        """Test has_metrics returns True when any counter is non-zero."""
        reporter = WandbReporter()

        reporter.record_sandbox_created()
        assert reporter.has_metrics

        reporter.reset()
        reporter.record_exec_outcome(ExecOutcome.FAILURE)
        assert reporter.has_metrics

        reporter.reset()
        reporter.record_startup_time(1.0)
        assert reporter.has_metrics


class TestWandbReporterLog:
    """Tests for WandbReporter.log method."""

    def test_log_returns_false_when_no_metrics(self) -> None:
        """Test log returns False when no metrics to log."""
        reporter = WandbReporter()
        assert not reporter.log()

    def test_log_returns_false_when_wandb_not_available(self) -> None:
        """Test log returns False when wandb is not importable."""
        reporter = WandbReporter()
        reporter.record_sandbox_created()

        with patch.dict("sys.modules", {"wandb": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = reporter.log()
                assert not result

    def test_log_returns_false_when_no_active_run(self) -> None:
        """Test log returns False when wandb.run is None."""
        reporter = WandbReporter()
        reporter.record_sandbox_created()

        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = reporter.log()
            assert not result

    def test_log_calls_wandb_log_and_resets(self) -> None:
        """Test log calls wandb.log with metrics and resets counters."""
        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_OK)

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = reporter.log(step=42)

            assert result
            mock_wandb.log.assert_called_once()
            call_args = mock_wandb.log.call_args
            assert call_args[0][0]["aviato/sandboxes_created"] == 1
            assert call_args[0][0]["aviato/exec_completed_ok"] == 1
            assert call_args[1]["step"] == 42

        # Counters should be reset after logging
        assert not reporter.has_metrics

    def test_log_without_step(self) -> None:
        """Test log works without step parameter."""
        reporter = WandbReporter()
        reporter.record_sandbox_created()

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = reporter.log()

            assert result
            call_args = mock_wandb.log.call_args
            assert call_args[1]["step"] is None
