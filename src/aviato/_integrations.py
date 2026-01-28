"""Optional integration detection utilities.

These functions provide lazy detection of optional dependencies
without creating hard requirements on those packages.
"""

from __future__ import annotations

import os

_wandb_available: bool | None = None


def is_wandb_available() -> bool:
    """Check if wandb is installed and importable.

    Uses caching to avoid repeated import attempts.

    Returns:
        True if wandb can be imported, False otherwise.
    """
    global _wandb_available
    if _wandb_available is None:
        try:
            import wandb  # noqa: F401

            _wandb_available = True
        except ImportError:
            _wandb_available = False
    return _wandb_available


def has_wandb_credentials() -> bool:
    """Check if wandb credentials are available.

    Looks for WANDB_API_KEY environment variable.

    Returns:
        True if WANDB_API_KEY is set, False otherwise.
    """
    return "WANDB_API_KEY" in os.environ


def has_active_wandb_run() -> bool:
    """Check if there is an active wandb run.

    Only returns True if wandb is installed AND a run is active.

    Returns:
        True if wandb.run is not None, False otherwise.
    """
    if not is_wandb_available():
        return False
    try:
        import wandb

        return wandb.run is not None
    except Exception:
        return False


def should_auto_report_wandb() -> bool:
    """Determine if wandb reporting should be auto-enabled.

    Returns True when all conditions are met:
    - wandb is installed
    - WANDB_API_KEY environment variable is set
    - An active wandb run exists (wandb.run is not None)

    We only attach to existing runs - we never create new ones.

    Returns:
        True if all conditions are met, False otherwise.
    """
    return has_wandb_credentials() and has_active_wandb_run()
