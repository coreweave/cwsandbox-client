# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Cleanup handlers for graceful shutdown of sandboxes.

This module registers atexit and signal handlers so owned sandboxes can be
stopped when the process exits. That prevents orphaned sandboxes from
consuming resources after the client process terminates.

Handlers are not installed on import. Activation happens when the caller
first owns a sandbox:

- constructing a standalone ``Sandbox`` (including ``run()``,
  ``run_from_template()``, and ``run_from_file()``)
- a ``Session`` first owning or adopting a sandbox

``atexit`` may be registered from any thread. SIGINT/SIGTERM installation
requires Python's main thread. An off-main activation registers ``atexit``,
skips signals without raising, and leaves signal installation pending so a
later main-thread activation can recover it.

Embedded hosts that own process lifecycle can call
:func:`disable_signal_handlers` or set
``CWSANDBOX_DISABLE_SIGNAL_HANDLERS`` to a truthy value (``1``, ``true``,
``yes``, ``on``) before the first owned sandbox. That skips SIGINT/SIGTERM
installation. Changing the environment variable after installation has no
effect. A late :func:`disable_signal_handlers` call raises
``RuntimeError``. ``atexit`` remains registered. There is no public
re-enable path.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from types import FrameType

logger = logging.getLogger(__name__)

# Type alias for signal handlers
_SignalHandler = Callable[[int, FrameType | None], None] | int | None

_DISABLE_SIGNAL_HANDLERS_ENV = "CWSANDBOX_DISABLE_SIGNAL_HANDLERS"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

# Module-level state for cleanup coordination
_cleanup_in_progress: bool = False
_original_sigint: _SignalHandler = None
_original_sigterm: _SignalHandler = None
_atexit_registered: bool = False
_signals_installed: bool = False
_signals_disabled: bool = False
_activation_lock = threading.Lock()


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


def _running_on_main_thread() -> bool:
    """Return True when the current thread can install signal handlers."""
    return threading.current_thread() is threading.main_thread()


def _env_disables_signal_handlers() -> bool:
    """Return True when the startup environment requests a signal opt-out."""
    raw = os.environ.get(_DISABLE_SIGNAL_HANDLERS_ENV, "")
    return raw.strip().lower() in _TRUTHY_ENV_VALUES


def _is_restorable_handler(handler: _SignalHandler) -> bool:
    """Return True when *handler* is safe to pass back to ``signal.signal``."""
    if handler is None:
        return False
    if handler in (signal.SIG_DFL, signal.SIG_IGN):
        return True
    module = getattr(type(handler), "__module__", "")
    return callable(handler) and not module.startswith("unittest.mock")


def _restore_original_handler(signum: int, original: _SignalHandler) -> None:
    """Restore a captured handler, ignoring test doubles and off-main errors."""
    if not _is_restorable_handler(original):
        return
    try:
        signal.signal(signum, original)
    except (TypeError, ValueError, OSError):
        pass


def _activate_cleanup_handlers() -> None:
    """Register atexit and, on the main thread, SIGINT/SIGTERM handlers.

    Safe to call from any thread and any number of times. ``atexit`` is
    registered once from the first activation. Signal handlers install
    once, and only on the main thread. An earlier off-main skip leaves
    signal installation pending for a later main-thread call. A process
    opt-out via :func:`disable_signal_handlers` or
    ``CWSANDBOX_DISABLE_SIGNAL_HANDLERS`` must happen before installation
    and skips signals while still registering ``atexit``.
    """
    global _original_sigint, _original_sigterm, _atexit_registered
    global _signals_installed, _signals_disabled

    with _activation_lock:
        if not _atexit_registered:
            atexit.register(_cleanup)
            _atexit_registered = True
            logger.debug("Registered atexit cleanup handler")

        if _signals_installed:
            return

        if _signals_disabled or _env_disables_signal_handlers():
            _signals_disabled = True
            logger.debug("Skipping signal handler install; host owns SIGINT/SIGTERM")
            return

        if not _running_on_main_thread():
            logger.debug("Skipping signal handler install off the main thread")
            return

        try:
            _original_sigint = signal.signal(signal.SIGINT, _signal_handler)
            _original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
        except ValueError:
            # Python rejects signal.signal() off the main thread. Treat this
            # as a pending install so a later main-thread call can recover.
            logger.debug("Signal handler install rejected; will retry on main thread")
            return

        _signals_installed = True
        logger.debug("Installed SIGINT/SIGTERM cleanup handlers")


def disable_signal_handlers() -> None:
    """Stop installing cwsandbox SIGINT/SIGTERM handlers for this process.

    Call this from an embedded host before creating the first owned sandbox
    so the host keeps process-signal ownership. The call is process-wide and
    idempotent. ``atexit`` cleanup still registers lazily when a sandbox
    becomes owned.

    If handlers are already installed, this raises ``RuntimeError`` without
    changing disable or install state.

    There is no public re-enable path. The host is responsible for graceful
    shutdown and for explicitly stopping resources it owns.

    Examples:
        ```python
        import cwsandbox

        cwsandbox.disable_signal_handlers()
        asyncio.run(run_worker())
        ```
    """
    global _signals_disabled

    with _activation_lock:
        if _signals_installed:
            raise RuntimeError(
                "disable_signal_handlers() must be called before cwsandbox installs signal handlers"
            )

        _signals_disabled = True
        logger.debug("Disabled cwsandbox SIGINT/SIGTERM handlers")


def _install_handlers() -> None:
    """Backward-compatible alias for :func:`_activate_cleanup_handlers`."""
    _activate_cleanup_handlers()


def _reset_for_testing() -> None:
    """Reset cleanup state for testing.

    This function is intended for use in tests only. It restores original
    signal handlers when they were captured, unregisters the atexit
    callback, and clears activation and disable state so handlers can be
    reinstalled.
    """
    global _cleanup_in_progress, _atexit_registered, _signals_installed
    global _signals_disabled, _original_sigint, _original_sigterm

    with _activation_lock:
        if _signals_installed:
            _restore_original_handler(signal.SIGINT, _original_sigint)
            _restore_original_handler(signal.SIGTERM, _original_sigterm)

        if _atexit_registered:
            atexit.unregister(_cleanup)

        _cleanup_in_progress = False
        _atexit_registered = False
        _signals_installed = False
        _signals_disabled = False
        _original_sigint = None
        _original_sigterm = None
