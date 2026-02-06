# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Integration detection utilities for external services.

This module provides functions to check for the availability and
configuration of external integrations like Weights & Biases.
"""

from __future__ import annotations

import os

from aviato._auth import _read_api_key_from_netrc


def is_wandb_available() -> bool:
    """Check if wandb is installed and importable.

    Returns:
        True if wandb can be imported, False otherwise
    """
    try:
        import wandb  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError, AttributeError, OSError):
        # ImportError/ModuleNotFoundError: package not installed
        # AttributeError: package installed but broken (missing attributes during init)
        # OSError: file system issues loading native extensions
        return False


def has_wandb_credentials() -> bool:
    """Check if W&B API credentials are configured.

    Checks for WANDB_API_KEY environment variable or ~/.netrc credentials.

    Returns:
        True if credentials are available, False otherwise
    """
    return bool(os.environ.get("WANDB_API_KEY") or _read_api_key_from_netrc())


def has_active_wandb_run() -> bool:
    """Check if there is an active W&B run.

    Returns:
        True if wandb.run is not None, False otherwise
    """
    if not is_wandb_available():
        return False

    try:
        import wandb

        return getattr(wandb, "run", None) is not None
    except Exception:
        return False


def should_auto_report_wandb() -> bool:
    """Check if automatic W&B reporting should be enabled.

    Auto-reporting is enabled when:
    1. wandb is installed
    2. WANDB_API_KEY is set
    3. There is an active wandb run

    Returns:
        True if all conditions are met, False otherwise
    """
    return is_wandb_available() and has_wandb_credentials() and has_active_wandb_run()
