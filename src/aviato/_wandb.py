# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""W&B (Weights & Biases) integration for sandbox metrics logging.

This module provides automatic logging of sandbox usage metrics to wandb
when an active wandb run exists. Metrics include sandbox creation counts,
execution counts, and success rates.

The integration follows the ART pattern: auto-detect via WANDB_API_KEY and
active run, never create new runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aviato._types import ExecOutcome

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = ["ExecOutcome", "WandbReporter"]


class WandbReporter:
    """Collects and logs sandbox metrics to an active wandb run.

    This reporter accumulates metrics during a session and logs them
    either on-demand via log() or automatically on session close.

    Metrics tracked:
    - aviato/sandboxes_created: Total sandboxes requested via session (includes unstarted)
    - aviato/executions: Total exec() calls
    - aviato/exec_completed_ok: Completed executions (returncode=0)
    - aviato/exec_completed_nonzero: Completed executions (returncode!=0)
    - aviato/exec_failures: Failed executions (timeouts, cancellations, transport)
    - aviato/exec_completion_rate: Fraction of exec() that completed (returncode=0)
    - aviato/exec_failure_rate: Fraction of exec() that failed to complete
    - aviato/startup_count: Number of startup times recorded
    - aviato/avg_startup_seconds: Average sandbox startup time
    - aviato/min_startup_seconds: Minimum sandbox startup time
    - aviato/max_startup_seconds: Maximum sandbox startup time
    - aviato/avg_execs_per_sandbox: Average exec() calls per sandbox
    - aviato/min_execs_per_sandbox: Minimum exec() calls in any sandbox
    - aviato/max_execs_per_sandbox: Maximum exec() calls in any sandbox
    Example:
        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_OK, sandbox_id="sb-123")
        reporter.record_exec_outcome(ExecOutcome.COMPLETED_NONZERO, sandbox_id="sb-123")
        reporter.record_startup_time(2.5)
        reporter.log(step=100)  # Logs metrics to wandb at step 100
    """

    def __init__(self) -> None:
        self._run: Run | None = None
        self._wandb_unavailable: bool = False
        self._sandboxes_created = 0
        self._executions = 0
        self._exec_completed_ok = 0
        self._exec_completed_nonzero = 0
        self._exec_failures = 0
        self._startup_count: int = 0
        self._startup_total_seconds: float = 0.0
        self._startup_min_seconds: float | None = None
        self._startup_max_seconds: float | None = None
        self._per_sandbox_exec_counts: dict[str, int] = {}

    def _get_run(self) -> Run | None:
        """Lazily get the active wandb run.

        Re-checks for wandb.run on each call until a run is found.
        Cached for session lifetime - one run per session is the expected
        usage pattern. If wandb.finish() is called and a new run starts,
        the reporter keeps the original reference.

        Returns:
            The active wandb.run if one exists, None otherwise.
        """
        if self._run is not None:
            return self._run

        if self._wandb_unavailable:
            return None

        try:
            import wandb

            if wandb.run is not None:
                self._run = wandb.run
                logger.debug("Attached to wandb run: %s", wandb.run.name)
        except ImportError:
            self._wandb_unavailable = True
            logger.debug("wandb not installed, metrics will not be logged")
        except Exception as e:
            logger.warning("Failed to attach to wandb run: %s", e)

        return self._run

    def record_sandbox_created(self) -> None:
        """Record that a sandbox was created."""
        self._sandboxes_created += 1

    def record_exec_outcome(self, outcome: ExecOutcome, sandbox_id: str | None = None) -> None:
        """Record an exec() call outcome.

        Args:
            outcome: The outcome classification (COMPLETED_OK, COMPLETED_NONZERO, or FAILURE).
            sandbox_id: Optional sandbox ID for per-sandbox exec count tracking.
        """
        self._executions += 1
        if outcome == ExecOutcome.COMPLETED_OK:
            self._exec_completed_ok += 1
        elif outcome == ExecOutcome.COMPLETED_NONZERO:
            self._exec_completed_nonzero += 1
        elif outcome == ExecOutcome.FAILURE:
            self._exec_failures += 1

        if sandbox_id is not None and self._get_run() is not None:
            self._per_sandbox_exec_counts[sandbox_id] = (
                self._per_sandbox_exec_counts.get(sandbox_id, 0) + 1
            )

    def record_startup_time(self, startup_seconds: float) -> None:
        """Record sandbox startup time.

        Args:
            startup_seconds: Time in seconds for sandbox to reach RUNNING state.
        """
        self._startup_count += 1
        self._startup_total_seconds += startup_seconds
        if self._startup_min_seconds is None or startup_seconds < self._startup_min_seconds:
            self._startup_min_seconds = startup_seconds
        if self._startup_max_seconds is None or startup_seconds > self._startup_max_seconds:
            self._startup_max_seconds = startup_seconds

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics as a dictionary.

        Returns:
            Dictionary with aviato/* prefixed metric names and values.
        """
        metrics: dict[str, Any] = {
            "aviato/sandboxes_created": self._sandboxes_created,
            "aviato/executions": self._executions,
            "aviato/exec_completed_ok": self._exec_completed_ok,
            "aviato/exec_completed_nonzero": self._exec_completed_nonzero,
            "aviato/exec_failures": self._exec_failures,
        }

        if self._executions > 0:
            metrics["aviato/exec_completion_rate"] = self._exec_completed_ok / self._executions
            metrics["aviato/exec_failure_rate"] = self._exec_failures / self._executions

        if self._startup_count > 0:
            metrics["aviato/startup_count"] = self._startup_count
            metrics["aviato/avg_startup_seconds"] = (
                self._startup_total_seconds / self._startup_count
            )
            metrics["aviato/min_startup_seconds"] = self._startup_min_seconds
            metrics["aviato/max_startup_seconds"] = self._startup_max_seconds

        if self._per_sandbox_exec_counts:
            counts = self._per_sandbox_exec_counts.values()
            metrics["aviato/avg_execs_per_sandbox"] = sum(counts) / len(counts)
            metrics["aviato/min_execs_per_sandbox"] = min(counts)
            metrics["aviato/max_execs_per_sandbox"] = max(counts)

        return metrics

    def log(self, step: int | None = None) -> bool:
        """Log accumulated metrics to wandb.

        If no wandb run is active, this is a no-op.

        Args:
            step: Optional step number to associate with metrics.
                If provided, metrics are logged at this training step.

        Returns:
            True if metrics were logged, False if no run available
            or no metrics to log.
        """
        run = self._get_run()
        if run is None:
            return False

        if not self.has_metrics:
            return False

        metrics = self.get_metrics()

        try:
            if step is not None:
                run.log(metrics, step=step)
            else:
                run.log(metrics)
            logger.debug("Logged metrics to wandb: %s", metrics)
            return True
        except Exception as e:
            logger.warning("Failed to log metrics to wandb: %s", e)
            return False

    def reset(self) -> None:
        """Reset accumulated metrics to zero.

        Call this after logging if you want to track metrics
        per-step rather than cumulative.
        """
        self._sandboxes_created = 0
        self._executions = 0
        self._exec_completed_ok = 0
        self._exec_completed_nonzero = 0
        self._exec_failures = 0
        self._startup_count = 0
        self._startup_total_seconds = 0.0
        self._startup_min_seconds = None
        self._startup_max_seconds = None
        self._per_sandbox_exec_counts = {}

    @property
    def has_metrics(self) -> bool:
        """Check if any metrics have been recorded."""
        return self._sandboxes_created > 0 or self._executions > 0 or self._startup_count > 0
