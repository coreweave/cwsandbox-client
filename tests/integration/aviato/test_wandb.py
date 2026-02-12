# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Integration tests for W&B metrics collection.

These tests run real sandboxes but mock the wandb SDK to verify
metric collection and guardrails work correctly without requiring
W&B credentials.

The TestLiveWandbIntegration class requires WANDB_API_KEY to be set
and performs actual logging to W&B with API verification.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aviato import SandboxDefaults, Session
from aviato._wandb import ExecOutcome

# Project for live W&B integration tests
WANDB_TEST_PROJECT = "aviato-integration-tests"


def _wandb_available() -> bool:
    """Check if wandb is installed."""
    try:
        import wandb  # noqa: F401

        return True
    except ImportError:
        return False


def _wandb_api_key_set() -> bool:
    """Check if WANDB_API_KEY is set."""
    return bool(os.environ.get("WANDB_API_KEY"))


@pytest.fixture
def require_wandb_live() -> None:
    """Skip test if wandb package or WANDB_API_KEY is not available.

    Uses fixture-based skip (evaluated at runtime) rather than skipif decorator
    (evaluated at import time) to ensure .env file has been loaded by pytest-dotenv.
    """
    if not _wandb_available():
        pytest.skip("wandb package not installed")
    if not _wandb_api_key_set():
        pytest.skip("WANDB_API_KEY environment variable not set")


@contextmanager
def wandb_test_run(
    run_name: str,
) -> Generator[Any, None, None]:
    """Create a W&B run for testing and ensure cleanup on exit.

    This context manager handles:
    - Creating a test run in the aviato-integration-tests project
    - Yielding the run for test use
    - Finishing the run and deleting it from W&B
    """
    import wandb

    run = wandb.init(  # type: ignore[attr-defined]
        project=WANDB_TEST_PROJECT,
        name=run_name,
        tags=["integration-test", "auto-cleanup"],
    )

    try:
        yield run
    finally:
        run_id = run.id
        run.finish()

        # Delete the run from W&B to clean up
        try:
            api = wandb.Api()  # type: ignore[attr-defined]
            entity = run.entity
            api_run = api.run(f"{entity}/{WANDB_TEST_PROJECT}/{run_id}")
            api_run.delete()
        except Exception:
            # Cleanup failure should not fail the test
            pass


class TestMetricCollection:
    """Tests that verify metric collection with mocked wandb."""

    def test_session_tracks_sandbox_creation_count(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test that Session tracks sandbox creation count across multiple sandboxes."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                # Create multiple sandboxes
                session.sandbox()
                session.sandbox()
                session.sandbox()

                # Verify count before close
                metrics = session._reporter.get_metrics()
                assert metrics["aviato/sandboxes_created"] == 3

    def test_exec_success_increments_completed_ok(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test that successful exec (returncode=0) increments completed_ok."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                sandbox = session.sandbox()

                # Verify initial state
                metrics_before = session._reporter.get_metrics()
                ok_before = metrics_before["aviato/exec_completed_ok"]

                # Execute successful command
                result = sandbox.exec(["echo", "hello"]).result()
                assert result.returncode == 0

                # Verify exec outcome was automatically tracked
                metrics = session._reporter.get_metrics()
                assert metrics["aviato/exec_completed_ok"] == ok_before + 1
                assert metrics["aviato/exec_completed_nonzero"] == 0
                assert metrics["aviato/exec_failures"] == 0

    def test_exec_failure_increments_completed_nonzero(
        self, sandbox_defaults: SandboxDefaults
    ) -> None:
        """Test that failed exec (non-zero exit) increments completed_nonzero."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                sandbox = session.sandbox()

                # Verify initial state
                metrics_before = session._reporter.get_metrics()
                nonzero_before = metrics_before["aviato/exec_completed_nonzero"]

                # Execute command that exits non-zero
                result = sandbox.exec(["sh", "-c", "exit 1"]).result()
                assert result.returncode != 0

                # Verify exec outcome was automatically tracked
                metrics = session._reporter.get_metrics()
                assert metrics["aviato/exec_completed_ok"] == 0
                assert metrics["aviato/exec_completed_nonzero"] == nonzero_before + 1
                assert metrics["aviato/exec_failures"] == 0

    def test_exec_error_increments_failures(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test that exec error (exception during exec) increments failures."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                session.sandbox()

                # Record a failure outcome (simulating exception during exec)
                session._reporter.record_exec_outcome(ExecOutcome.FAILURE)

                metrics = session._reporter.get_metrics()
                assert metrics["aviato/exec_completed_ok"] == 0
                assert metrics["aviato/exec_completed_nonzero"] == 0
                assert metrics["aviato/exec_failures"] == 1

    def test_startup_times_recorded_and_averaged(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test that startup times are recorded and averaged correctly."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                # Record multiple startup times
                session._reporter.record_startup_time(2.0)
                session._reporter.record_startup_time(4.0)
                session._reporter.record_startup_time(6.0)

                metrics = session._reporter.get_metrics()
                # Average of 2.0, 4.0, 6.0 = 4.0
                assert metrics["aviato/avg_startup_seconds"] == 4.0


class TestGuardrails:
    """Tests for guardrails that prevent unwanted W&B logging."""

    def test_no_logging_when_no_active_wandb_run(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test no logging when wandb.run is None."""
        mock_wandb = MagicMock()
        # wandb.run is None (no active run)
        mock_wandb.run = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                # Create sandbox to generate metrics
                session.sandbox()

                # Verify metrics were collected
                assert session._reporter.has_metrics
                metrics = session._reporter.get_metrics()
                assert metrics["aviato/sandboxes_created"] == 1

                # Try to log - should return False (no active run)
                logged = session._reporter.log()
                assert logged is False

    def test_no_logging_when_report_to_empty(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test no logging when report_to=[] (explicit opt-out)."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # Explicit opt-out with empty list
            with Session(sandbox_defaults, report_to=[]) as session:
                # Create sandbox to generate metrics
                session.sandbox()

                # log_metrics should return False (reporting disabled)
                logged = session.log_metrics()
                assert logged is False

                # wandb.run.log should NOT have been called
                mock_wandb.run.log.assert_not_called()

    def test_metrics_collected_even_when_not_logging(
        self, sandbox_defaults: SandboxDefaults
    ) -> None:
        """Test that metrics ARE collected even when no wandb run is active."""
        mock_wandb = MagicMock()
        mock_wandb.run = None  # No active run - log() returns False

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                # Create sandboxes
                session.sandbox()
                session.sandbox()

                # Manually record exec outcomes and startup times
                session._reporter.record_exec_outcome(ExecOutcome.COMPLETED_OK)
                session._reporter.record_startup_time(1.5)

                # Verify metrics were collected for later retrieval
                assert session._reporter.has_metrics
                metrics = session._reporter.get_metrics()

                assert metrics["aviato/sandboxes_created"] == 2
                assert metrics["aviato/exec_completed_ok"] == 1
                assert metrics["aviato/avg_startup_seconds"] == 1.5

                # But logging returns False since no active run
                assert session._reporter.log() is False

    def test_log_metrics_disabled_when_report_to_not_wandb(
        self, sandbox_defaults: SandboxDefaults
    ) -> None:
        """Test that log_metrics returns False when report_to excludes wandb."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # report_to with some other value (not wandb) - reporter is None
            with Session(sandbox_defaults, report_to=["other"]) as session:
                session.sandbox()

                # No reporter means no metrics collection
                assert session.get_metrics() == {}

                # log_metrics should return False (no reporter)
                logged = session.log_metrics()
                assert logged is False

                # wandb.run.log should NOT have been called
                mock_wandb.run.log.assert_not_called()


class TestMetricLogging:
    """Tests for the actual W&B logging behavior when enabled."""

    def test_log_metrics_called_on_session_close(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test that metrics are logged automatically on session close."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                session.sandbox()

                # Don't call log_metrics manually - let session close do it
                pass

            # Reporter calls run.log(), not wandb.log()
            mock_wandb.run.log.assert_called_once()
            call_args = mock_wandb.run.log.call_args
            logged_metrics = call_args[0][0]

            assert "aviato/sandboxes_created" in logged_metrics
            assert logged_metrics["aviato/sandboxes_created"] == 1

    def test_log_metrics_with_step(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test that step parameter is passed to wandb.log."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                session.sandbox()

                # Log with step number
                logged = session.log_metrics(step=42)
                assert logged is True

                # Reporter calls run.log(), not wandb.log()
                mock_wandb.run.log.assert_called()
                call_kwargs = mock_wandb.run.log.call_args[1]
                assert call_kwargs["step"] == 42

    def test_log_metrics_resets_counters_by_default(
        self, sandbox_defaults: SandboxDefaults
    ) -> None:
        """Test that log_metrics resets counters by default."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                session.sandbox()

                # First log should succeed
                logged = session.log_metrics()
                assert logged is True

                # After reset, no metrics to log
                logged = session.log_metrics()
                assert logged is False

    def test_log_metrics_without_reset(self, sandbox_defaults: SandboxDefaults) -> None:
        """Test that log_metrics preserves counters when reset=False."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                session.sandbox()

                # First log without reset
                logged = session.log_metrics(reset=False)
                assert logged is True

                # Second log should still have metrics
                logged = session.log_metrics(reset=False)
                assert logged is True

                # Both calls should have the same count
                assert mock_wandb.run.log.call_count == 2


class TestLiveWandbIntegration:
    """Live integration tests that log to real W&B and verify via API.

    These tests require:
    - wandb package installed
    - WANDB_API_KEY environment variable set

    Each test creates a temporary W&B run, logs metrics, verifies them
    via the W&B API, and cleans up the run afterward.
    """

    def test_training_loop_with_api_verification(
        self, sandbox_defaults: SandboxDefaults, require_wandb_live: None
    ) -> None:
        """Simulate a training loop and verify metrics via W&B API.

        This test:
        1. Creates a W&B run
        2. Creates a Session with report_to=["wandb"]
        3. Runs 2 epochs with sandboxes executing various commands
        4. Logs metrics at each epoch via session.log_metrics()
        5. Finishes the run and verifies metrics via the W&B API
        6. Cleans up the test run
        """
        import wandb

        run_name = f"integration-test-{int(time.time())}"

        # Create run manually so we can control when it finishes
        run = wandb.init(
            project=WANDB_TEST_PROJECT,
            name=run_name,
            tags=["integration-test", "auto-cleanup"],
        )
        run_id = run.id
        entity = run.entity

        try:
            with Session(sandbox_defaults, report_to=["wandb"]) as session:
                # Run 2 epochs, each with one sandbox
                for epoch in range(2):
                    sandbox = session.sandbox()
                    sandbox.wait()

                    # Success: echo command (returncode 0)
                    result = sandbox.exec(["echo", f"epoch {epoch} complete"]).result()
                    assert result.returncode == 0

                    # Failure: non-zero exit
                    result = sandbox.exec(["sh", "-c", "exit 1"]).result()
                    assert result.returncode != 0

                    # Another success for variety
                    result = sandbox.exec(["python", "-c", "print('hello')"]).result()
                    assert result.returncode == 0

                    # Log metrics at end of epoch
                    logged = session.log_metrics(step=epoch)
                    assert logged is True

                # Session close will log final metrics

            # Finish the run BEFORE querying the API
            # The API can only see data from finished runs
            run.finish()

            # Verify metrics via W&B API
            api = wandb.Api()
            api_run = api.run(f"{entity}/{WANDB_TEST_PROJECT}/{run_id}")

            # Poll until data appears or timeout (10s max)
            history: list[dict[str, Any]] = []
            for _ in range(10):
                history = list(api_run.scan_history())
                if len(history) >= 2:
                    break
                time.sleep(1)

            # We should have at least 2 log calls (one per epoch)
            assert len(history) >= 2, f"Expected at least 2 log entries, got {len(history)}"

            # Verify the metrics keys exist in at least one entry
            all_keys: set[str] = set()
            for row in history:
                all_keys.update(row.keys())

            assert "aviato/sandboxes_created" in all_keys
            assert "aviato/exec_completed_ok" in all_keys
            assert "aviato/exec_completed_nonzero" in all_keys

            # Verify metric values in at least one row
            # (values reset between log calls, so we check individual rows)
            found_sandbox_metric = False
            found_exec_ok_metric = False
            for row in history:
                if "aviato/sandboxes_created" in row and row["aviato/sandboxes_created"] > 0:
                    found_sandbox_metric = True
                if "aviato/exec_completed_ok" in row and row["aviato/exec_completed_ok"] > 0:
                    found_exec_ok_metric = True

            assert found_sandbox_metric, "Should have logged sandboxes_created > 0"
            assert found_exec_ok_metric, "Should have logged exec_completed_ok > 0"

        finally:
            # Clean up: delete the test run from W&B
            try:
                if not run.finished:
                    run.finish()
                api = wandb.Api()
                api_run = api.run(f"{entity}/{WANDB_TEST_PROJECT}/{run_id}")
                api_run.delete()
            except Exception:
                pass  # Cleanup failure should not fail the test
