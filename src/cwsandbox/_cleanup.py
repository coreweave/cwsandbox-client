# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Cleanup handlers for graceful shutdown of sandboxes.

This module installs atexit and signal handlers to ensure all sandboxes are
properly stopped when the process exits. This prevents orphaned sandboxes
from consuming resources after the client process terminates.

The handlers are installed automatically when this module is imported.
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from collections.abc import Callable
from types import FrameType

logger = logging.getLogger(__name__)

# Type alias for signal handlers
_SignalHandler = Callable[[int, FrameType | None], None] | int | None

# Module-level state for cleanup coordination
_cleanup_in_progress: bool = False
_original_sigint: _SignalHandler = None
_original_sigterm: _SignalHandler = None
_handlers_installed: bool = False


def _cleanup() -> None:
    """Clean up all sandboxes.

    This function is called during process shutdown to stop all sandboxes.
    It guards against re-entrancy using a module-level flag.
    """
    global _cleanup_in_progress
    if _cleanup_in_progress:
        return
    _cleanup_in_progress = True

    try:
        # Import here to avoid circular imports
        from cwsandbox._loop_manager import _LoopManager

        manager = _LoopManager.get()
        manager.cleanup_all()
    except Exception:
        logger.exception("Error during cleanup")


def _signal_handler(signum: int, frame: FrameType | None) -> None:
    """Handle SIGINT and SIGTERM signals.

    On first signal, performs cleanup and chains to original handler.
    On second signal during cleanup, forces immediate exit.

    Args:
        signum: The signal number received.
        frame: The current stack frame (unused).
    """
    global _cleanup_in_progress

    if _cleanup_in_progress:
        # Second signal during cleanup - force exit
        sys.exit(128 + signum)

    _cleanup()

    # Chain to original handler
    original = _original_sigint if signum == signal.SIGINT else _original_sigterm

    if original == signal.SIG_DFL:
        # Default handler - exit with signal code
        sys.exit(128 + signum)
    elif original == signal.SIG_IGN:
        # Ignore - do nothing
        pass
    elif callable(original):
        # User-defined handler - chain to it
        original(signum, frame)


def _install_handlers() -> None:
    """Install atexit and signal handlers.

    This function is called automatically on module import. It registers:
    - An atexit handler for normal process exit
    - Signal handlers for SIGINT (Ctrl+C) and SIGTERM

    The original signal handlers are preserved and chained after cleanup.
    """
    global _original_sigint, _original_sigterm, _handlers_installed

    if _handlers_installed:
        return

    # Register atexit handler
    atexit.register(_cleanup)

    # Install signal handlers, preserving originals for chaining
    _original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    _original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    _handlers_installed = True
    logger.debug("Installed cleanup handlers")


def _reset_for_testing() -> None:
    """Reset cleanup state for testing.

    This function is intended for use in tests only. It resets the
    module-level state to allow handlers to be reinstalled.
    """
    global _cleanup_in_progress, _handlers_installed
    global _original_sigint, _original_sigterm

    _cleanup_in_progress = False
    _handlers_installed = False

    # Restore original handlers if they were saved
    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)
        _original_sigint = None
    if _original_sigterm is not None:
        signal.signal(signal.SIGTERM, _original_sigterm)
        _original_sigterm = None


# Install handlers on module import
_install_handlers()
