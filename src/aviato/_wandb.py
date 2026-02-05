# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Weights & Biases metrics reporting for sandbox usage tracking.

This module provides metrics collection for sandbox operations, enabling
tracking of sandbox creation rates, execution outcomes, and startup times.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ExecOutcome(StrEnum):
    """Outcome of a sandbox exec operation."""

    COMPLETED_OK = "completed_ok"
    COMPLETED_NONZERO = "completed_nonzero"
    FAILURE = "failure"


class WandbReporter:
    """Collects metrics for W&B logging during sandbox operations.

    WandbReporter accumulates metrics across sandbox operations within a session.
    Metrics can be logged to W&B at any step, or automatically on session close.

    The reporter prepares metrics for logging. The log() convenience method
    handles the wandb import and logging, or callers can use get_metrics()
    to log directly.

    Example:
        ```python
        reporter = WandbReporter()

        # Record operations
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_OK, "sandbox-123")
        reporter.record_startup_time(2.5)

        # Get metrics for logging
        if reporter.has_metrics:
            metrics = reporter.get_metrics()
            wandb.log(metrics, step=current_step)
            reporter.reset()
        ```

    Metrics collected:
        - aviato/sandboxes_created: Number of sandboxes created
        - aviato/exec_completed_ok: Execs that completed with returncode 0
        - aviato/exec_completed_nonzero: Execs that completed with non-zero returncode
          (when check=False; with check=True, non-zero exits count as failures)
        - aviato/exec_failures: Execs that failed (exception raised, including
          SandboxExecutionError from check=True with non-zero exit)
        - aviato/avg_startup_time: Average sandbox startup time in seconds
    """

    def __init__(self) -> None:
        """Initialize a new WandbReporter with zeroed counters."""
        self._sandboxes_created = 0
        self._exec_completed_ok = 0
        self._exec_completed_nonzero = 0
        self._exec_failures = 0
        self._startup_times: list[float] = []

    def record_sandbox_created(self) -> None:
        """Record that a sandbox was created."""
        self._sandboxes_created += 1

    def record_exec_outcome(self, outcome: ExecOutcome, sandbox_id: str | None = None) -> None:
        """Record the outcome of an exec operation.

        Args:
            outcome: The outcome of the exec operation
            sandbox_id: Optional sandbox ID for logging context
        """
        match outcome:
            case ExecOutcome.COMPLETED_OK:
                self._exec_completed_ok += 1
            case ExecOutcome.COMPLETED_NONZERO:
                self._exec_completed_nonzero += 1
            case ExecOutcome.FAILURE:
                self._exec_failures += 1

        logger.debug(
            "Recorded exec outcome %s for sandbox %s",
            outcome.value,
            sandbox_id or "unknown",
        )

    def record_startup_time(self, seconds: float) -> None:
        """Record a sandbox startup time measurement.

        Args:
            seconds: Time in seconds from start request to RUNNING status
        """
        self._startup_times.append(seconds)
        logger.debug("Recorded startup time: %.2fs", seconds)

    def get_metrics(self) -> dict[str, Any]:
        """Get the current metrics as a dictionary suitable for wandb.log().

        Returns:
            Dictionary with aviato/ prefixed metric names and values
        """
        metrics: dict[str, Any] = {
            "aviato/sandboxes_created": self._sandboxes_created,
            "aviato/exec_completed_ok": self._exec_completed_ok,
            "aviato/exec_completed_nonzero": self._exec_completed_nonzero,
            "aviato/exec_failures": self._exec_failures,
        }

        if self._startup_times:
            metrics["aviato/avg_startup_time"] = sum(self._startup_times) / len(self._startup_times)

        return metrics

    @property
    def has_metrics(self) -> bool:
        """Return True if any metrics have been recorded."""
        return (
            self._sandboxes_created > 0
            or self._exec_completed_ok > 0
            or self._exec_completed_nonzero > 0
            or self._exec_failures > 0
            or len(self._startup_times) > 0
        )

    def reset(self) -> None:
        """Reset all counters to zero."""
        self._sandboxes_created = 0
        self._exec_completed_ok = 0
        self._exec_completed_nonzero = 0
        self._exec_failures = 0
        self._startup_times.clear()

    def log(self, step: int | None = None, *, reset: bool = True) -> bool:
        """Log metrics to W&B if available and there are metrics to log.

        This is a convenience method that checks for wandb availability,
        logs metrics, and optionally resets the counters.

        Note: This method imports wandb when called.

        Args:
            step: Optional step number for wandb.log()
            reset: If True (default), reset counters after logging

        Returns:
            True if metrics were logged, False otherwise
        """
        if not self.has_metrics:
            return False

        try:
            import wandb
        except (ImportError, ModuleNotFoundError):
            logger.warning("wandb not installed, skipping log")
            return False

        if getattr(wandb, "run", None) is None:
            logger.debug("No active wandb run, skipping log")
            return False

        metrics = self.get_metrics()
        wandb.log(metrics, step=step)
        if reset:
            self.reset()
        return True
