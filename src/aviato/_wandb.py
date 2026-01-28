"""W&B (Weights & Biases) integration for sandbox metrics logging.

This module provides automatic logging of sandbox usage metrics to wandb
when an active wandb run exists. Metrics include sandbox creation counts,
execution counts, and success rates.

The integration follows the ART pattern: auto-detect via WANDB_API_KEY and
active run, never create new runs.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)


class ExecOutcome(Enum):
    """Outcome classification for exec() calls.

    Taxonomy:
    - SUCCESS: returncode == 0
    - FAILURE: returncode != 0 (process completed but failed)
    - ERROR: SandboxTimeoutError, cancellation, transport failures
    """

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class WandbReporter:
    """Collects and logs sandbox metrics to an active wandb run.

    This reporter accumulates metrics during a session and logs them
    either on-demand via log() or automatically on session close.

    Metrics tracked:
    - aviato/sandboxes_created: Total sandboxes created via session
    - aviato/executions: Total exec() calls
    - aviato/exec_successes: Successful executions (returncode=0)
    - aviato/exec_failures: Failed executions (returncode!=0)
    - aviato/exec_errors: Errors (timeouts, cancellations, transport)
    - aviato/success_rate: Fraction of exec() with returncode=0
    - aviato/error_rate: Fraction of exec() that errored

    Example:
        reporter = WandbReporter()
        reporter.record_sandbox_created()
        reporter.record_exec_outcome(ExecOutcome.SUCCESS)
        reporter.record_exec_outcome(ExecOutcome.FAILURE)
        reporter.log(step=100)  # Logs metrics to wandb at step 100
    """

    def __init__(self) -> None:
        self._run: Run | None = None
        self._run_checked = False
        self._sandboxes_created = 0
        self._executions = 0
        self._exec_successes = 0
        self._exec_failures = 0
        self._exec_errors = 0

    def _get_run(self) -> Run | None:
        """Lazily get the active wandb run.

        Only checks for a run once; subsequent calls return the cached result.
        This avoids repeated wandb imports and run lookups.

        Returns:
            The active wandb.run if one exists, None otherwise.
        """
        if self._run_checked:
            return self._run

        self._run_checked = True
        try:
            import wandb

            if wandb.run is not None:
                self._run = wandb.run
                logger.debug("Attached to wandb run: %s", wandb.run.name)
        except ImportError:
            logger.debug("wandb not installed, metrics will not be logged")
        except Exception as e:
            logger.warning("Failed to attach to wandb run: %s", e)

        return self._run

    def record_sandbox_created(self) -> None:
        """Record that a sandbox was created."""
        self._sandboxes_created += 1

    def record_exec_outcome(self, outcome: ExecOutcome) -> None:
        """Record an exec() call outcome.

        Args:
            outcome: The outcome classification (SUCCESS, FAILURE, or ERROR).
        """
        self._executions += 1
        if outcome == ExecOutcome.SUCCESS:
            self._exec_successes += 1
        elif outcome == ExecOutcome.FAILURE:
            self._exec_failures += 1
        elif outcome == ExecOutcome.ERROR:
            self._exec_errors += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics as a dictionary.

        Returns:
            Dictionary with aviato/* prefixed metric names and values.
        """
        metrics: dict[str, Any] = {
            "aviato/sandboxes_created": self._sandboxes_created,
            "aviato/executions": self._executions,
            "aviato/exec_successes": self._exec_successes,
            "aviato/exec_failures": self._exec_failures,
            "aviato/exec_errors": self._exec_errors,
        }

        if self._executions > 0:
            metrics["aviato/success_rate"] = self._exec_successes / self._executions
            metrics["aviato/error_rate"] = self._exec_errors / self._executions

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
        self._exec_successes = 0
        self._exec_failures = 0
        self._exec_errors = 0

    @property
    def has_metrics(self) -> bool:
        """Check if any metrics have been recorded."""
        return self._sandboxes_created > 0 or self._executions > 0
