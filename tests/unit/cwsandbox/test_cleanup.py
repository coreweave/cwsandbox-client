# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""Unit tests for cwsandbox._cleanup module."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from cwsandbox._cleanup import (
    _activate_cleanup_handlers,
    _cleanup,
    _install_handlers,
    _reset_for_testing,
    _signal_handler,
    disable_signal_handlers,
)


@pytest.fixture(autouse=True)
def reset_cleanup_state():
    """Reset cleanup state before and after each test."""
    _reset_for_testing()
    yield
    _reset_for_testing()


class TestCleanup:
    """Tests for _cleanup function."""

    def test_cleanup_calls_loop_manager_cleanup_all(self) -> None:
        """Test _cleanup calls _LoopManager.get().cleanup_all()."""
        with patch("cwsandbox._loop_manager._LoopManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.get.return_value = mock_manager

            _cleanup()

            mock_manager_class.get.assert_called_once()
            mock_manager.cleanup_all.assert_called_once()

    def test_cleanup_guards_against_reentrancy(self) -> None:
        """Test _cleanup only runs once even if called multiple times."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False

        with patch("cwsandbox._loop_manager._LoopManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.get.return_value = mock_manager

            # First call should run
            _cleanup()
            assert mock_manager.cleanup_all.call_count == 1

            # Second call should be skipped
            _cleanup()
            assert mock_manager.cleanup_all.call_count == 1

    def test_cleanup_handles_exception(self) -> None:
        """Test _cleanup handles exceptions gracefully."""
        with patch("cwsandbox._loop_manager._LoopManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.cleanup_all.side_effect = RuntimeError("test error")
            mock_manager_class.get.return_value = mock_manager

            # Should not raise
            _cleanup()


class TestSignalHandler:
    """Tests for _signal_handler function."""

    def test_signal_handler_calls_cleanup(self) -> None:
        """Test signal handler calls _cleanup."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigint = signal.SIG_IGN

        with patch("cwsandbox._loop_manager._LoopManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.get.return_value = mock_manager

            _signal_handler(signal.SIGINT, None)

            mock_manager.cleanup_all.assert_called_once()

    def test_signal_handler_chains_to_sig_dfl(self) -> None:
        """Test signal handler exits with signal code for SIG_DFL."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigint = signal.SIG_DFL

        with patch("cwsandbox._loop_manager._LoopManager"):
            with pytest.raises(SystemExit) as exc_info:
                _signal_handler(signal.SIGINT, None)

            # Exit code should be 128 + signal number
            assert exc_info.value.code == 128 + signal.SIGINT

    def test_signal_handler_chains_to_sig_ign(self) -> None:
        """Test signal handler does nothing for SIG_IGN."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigint = signal.SIG_IGN

        with patch("cwsandbox._loop_manager._LoopManager"):
            # Should not raise
            _signal_handler(signal.SIGINT, None)

    def test_signal_handler_chains_to_callable(self) -> None:
        """Test signal handler chains to callable original handler."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False

        original_handler = MagicMock()
        cleanup_module._original_sigint = original_handler

        with patch("cwsandbox._loop_manager._LoopManager"):
            _signal_handler(signal.SIGINT, None)

            original_handler.assert_called_once_with(signal.SIGINT, None)

    def test_second_signal_forces_exit(self) -> None:
        """Test second signal during cleanup forces immediate exit."""
        import cwsandbox._cleanup as cleanup_module

        # Simulate cleanup already in progress
        cleanup_module._cleanup_in_progress = True

        with pytest.raises(SystemExit) as exc_info:
            _signal_handler(signal.SIGINT, None)

        # Exit code should be 128 + signal number
        assert exc_info.value.code == 128 + signal.SIGINT

    def test_signal_handler_handles_sigterm(self) -> None:
        """Test signal handler works for SIGTERM."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigterm = signal.SIG_DFL

        with patch("cwsandbox._loop_manager._LoopManager"):
            with pytest.raises(SystemExit) as exc_info:
                _signal_handler(signal.SIGTERM, None)

            assert exc_info.value.code == 128 + signal.SIGTERM


class TestActivateCleanupHandlers:
    """Tests for lazy, lock-guarded handler activation."""

    def test_activate_registers_atexit_once(self) -> None:
        """Test activation registers atexit exactly once."""
        with patch("cwsandbox._cleanup.atexit.register") as mock_register:
            with patch("cwsandbox._cleanup.signal.signal"):
                _activate_cleanup_handlers()
                _activate_cleanup_handlers()
                _activate_cleanup_handlers()

                mock_register.assert_called_once_with(_cleanup)

    def test_main_thread_installs_sigint_and_sigterm_once(self) -> None:
        """Test main-thread activation installs SIGINT and SIGTERM exactly once."""
        with patch("cwsandbox._cleanup.atexit.register"):
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                _activate_cleanup_handlers()
                _activate_cleanup_handlers()

                calls = mock_signal.call_args_list
                sigint_calls = [c for c in calls if c[0][0] == signal.SIGINT]
                sigterm_calls = [c for c in calls if c[0][0] == signal.SIGTERM]
                assert len(sigint_calls) == 1
                assert len(sigterm_calls) == 1
                assert sigint_calls[0][0][1] == _signal_handler
                assert sigterm_calls[0][0][1] == _signal_handler

    def test_activate_preserves_original_handlers(self) -> None:
        """Test activation captures existing handlers for chaining."""
        import cwsandbox._cleanup as cleanup_module

        original_sigint = MagicMock()
        original_sigterm = MagicMock()

        with patch("cwsandbox._cleanup.atexit.register"):
            with patch(
                "cwsandbox._cleanup.signal.signal",
                side_effect=[original_sigint, original_sigterm],
            ):
                _activate_cleanup_handlers()

                assert cleanup_module._original_sigint == original_sigint
                assert cleanup_module._original_sigterm == original_sigterm

    def test_worker_thread_skips_signal_install(self) -> None:
        """Test worker-thread activation does not call signal.signal or raise."""
        import cwsandbox._cleanup as cleanup_module

        errors: list[BaseException] = []

        with patch("cwsandbox._cleanup.atexit.register") as mock_register:
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:

                def worker() -> None:
                    try:
                        _activate_cleanup_handlers()
                    except BaseException as exc:  # noqa: BLE001 - collect for assertion
                        errors.append(exc)

                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()

        assert errors == []
        mock_register.assert_called_once_with(_cleanup)
        mock_signal.assert_not_called()
        assert cleanup_module._atexit_registered is True
        assert cleanup_module._signals_installed is False

    def test_off_main_skip_leaves_signals_pending(self) -> None:
        """Test an off-main skip does not mark signals installed."""
        import cwsandbox._cleanup as cleanup_module

        with patch("cwsandbox._cleanup.atexit.register"):
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                with patch("cwsandbox._cleanup._running_on_main_thread", return_value=False):
                    _activate_cleanup_handlers()

                mock_signal.assert_not_called()
                assert cleanup_module._atexit_registered is True
                assert cleanup_module._signals_installed is False

    def test_later_main_thread_activation_recovers_signals(self) -> None:
        """Test a later main-thread call installs handlers after an off-main skip."""
        import cwsandbox._cleanup as cleanup_module

        with patch("cwsandbox._cleanup.atexit.register") as mock_register:
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                with patch("cwsandbox._cleanup._running_on_main_thread", return_value=False):
                    _activate_cleanup_handlers()

                assert cleanup_module._signals_installed is False
                mock_signal.assert_not_called()

                _activate_cleanup_handlers()

                mock_register.assert_called_once_with(_cleanup)
                assert mock_signal.call_count == 2
                assert cleanup_module._signals_installed is True

    def test_concurrent_activation_is_idempotent(self) -> None:
        """Test concurrent first use registers atexit once and recovers signals once."""
        import cwsandbox._cleanup as cleanup_module

        with patch("cwsandbox._cleanup.atexit.register") as mock_register:
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                errors: list[BaseException] = []

                def worker() -> None:
                    try:
                        _activate_cleanup_handlers()
                    except BaseException as exc:  # noqa: BLE001 - collect for assertion
                        errors.append(exc)

                threads = [threading.Thread(target=worker) for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                assert errors == []
                mock_register.assert_called_once_with(_cleanup)
                mock_signal.assert_not_called()
                assert cleanup_module._signals_installed is False

                _activate_cleanup_handlers()
                _activate_cleanup_handlers()

                assert mock_signal.call_count == 2
                assert cleanup_module._signals_installed is True

    def test_install_handlers_alias_activates(self) -> None:
        """Test the compatibility alias still activates handlers."""
        with patch("cwsandbox._cleanup.atexit.register") as mock_register:
            with patch("cwsandbox._cleanup.signal.signal"):
                _install_handlers()

                mock_register.assert_called_once_with(_cleanup)


class TestResetForTesting:
    """Tests for _reset_for_testing function."""

    def test_reset_clears_cleanup_in_progress(self) -> None:
        """Test _reset_for_testing clears cleanup_in_progress flag."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = True
        _reset_for_testing()
        assert cleanup_module._cleanup_in_progress is False

    def test_reset_clears_both_registration_states(self) -> None:
        """Test _reset_for_testing clears atexit and signal registration flags."""
        import cwsandbox._cleanup as cleanup_module

        cleanup_module._atexit_registered = True
        cleanup_module._signals_installed = True
        cleanup_module._signals_disabled = True
        _reset_for_testing()
        assert cleanup_module._atexit_registered is False
        assert cleanup_module._signals_installed is False
        assert cleanup_module._signals_disabled is False

    def test_reset_unregisters_atexit_and_restores_handlers(self) -> None:
        """Test reset restores captured handlers and unregisters atexit."""
        import cwsandbox._cleanup as cleanup_module

        original_sigint = signal.SIG_IGN
        original_sigterm = signal.SIG_IGN

        with patch("cwsandbox._cleanup.atexit.register"):
            with patch(
                "cwsandbox._cleanup.signal.signal",
                side_effect=[original_sigint, original_sigterm],
            ):
                _activate_cleanup_handlers()

        with (
            patch("cwsandbox._cleanup.atexit.unregister") as mock_unregister,
            patch("cwsandbox._cleanup.signal.signal") as mock_signal,
        ):
            _reset_for_testing()

        mock_unregister.assert_called_once_with(_cleanup)
        mock_signal.assert_any_call(signal.SIGINT, original_sigint)
        mock_signal.assert_any_call(signal.SIGTERM, original_sigterm)
        assert cleanup_module._original_sigint is None
        assert cleanup_module._original_sigterm is None


def _run_fresh(script: str, *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    """Run *script* in a fresh interpreter that can import this checkout."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _run_fresh_until_sigterm(
    script: str,
    *,
    ready_timeout: float = 10.0,
    exit_timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    """Wait boundedly for READY, send SIGTERM, and always reap the child."""
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], ready_timeout)
        if not ready:
            raise TimeoutError(f"child did not print READY within {ready_timeout}s")

        line = proc.stdout.readline()
        if line.strip() != "READY":
            raise RuntimeError(f"child printed {line.strip()!r} before READY")

        os.kill(proc.pid, signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=exit_timeout)
        return subprocess.CompletedProcess(
            proc.args,
            proc.returncode,
            stdout=f"{line}{stdout}",
            stderr=stderr,
        )
    except Exception as exc:
        if proc.poll() is None:
            proc.kill()
        stdout, stderr = proc.communicate(timeout=exit_timeout)
        raise AssertionError(f"{exc}\nchild stdout:\n{stdout}\nchild stderr:\n{stderr}") from exc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=exit_timeout)


class TestFreshProcessCleanup:
    """Fresh-process coverage for import, threads, and signal compatibility."""

    def test_main_thread_import_leaves_handlers_unchanged(self) -> None:
        """Importing cwsandbox on the main thread must not replace signals."""
        script = """
import signal
before_int = signal.getsignal(signal.SIGINT)
before_term = signal.getsignal(signal.SIGTERM)
import cwsandbox
after_int = signal.getsignal(signal.SIGINT)
after_term = signal.getsignal(signal.SIGTERM)
print(before_int is after_int and before_term is after_term)
"""
        result = _run_fresh(script)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"

    def test_worker_thread_import_succeeds(self) -> None:
        """Importing cwsandbox off the main thread must not raise."""
        script = """
import signal
import threading

before_int = signal.getsignal(signal.SIGINT)
before_term = signal.getsignal(signal.SIGTERM)
errors = []

def worker():
    try:
        import cwsandbox
        from cwsandbox import Sandbox
    except Exception as exc:
        errors.append(repr(exc))

thread = threading.Thread(target=worker)
thread.start()
thread.join()
after_int = signal.getsignal(signal.SIGINT)
after_term = signal.getsignal(signal.SIGTERM)
print("errors", errors)
print("unchanged", before_int is after_int and before_term is after_term)
"""
        result = _run_fresh(script)
        assert result.returncode == 0, result.stderr
        assert "errors []" in result.stdout
        assert "unchanged True" in result.stdout

    def test_main_thread_owned_sandbox_installs_legacy_handler(self) -> None:
        """First owned sandbox on the main thread installs the SDK handler."""
        script = """
import signal
from cwsandbox import Sandbox
from cwsandbox._cleanup import _signal_handler

Sandbox()
print(signal.getsignal(signal.SIGINT) is _signal_handler)
print(signal.getsignal(signal.SIGTERM) is _signal_handler)
"""
        result = _run_fresh(script)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["True", "True"]

    def test_session_from_id_installs_handler_on_calling_thread(self) -> None:
        """Session.from_id(adopt=True) activates before background registration."""
        script = """
import signal
from unittest.mock import AsyncMock, patch
from cwsandbox import Session
from cwsandbox._cleanup import _signal_handler

class SandboxStub:
    sandbox_id = "test-123"
    _session = None

    async def _stop_async(self):
        pass

session = Session(report_to=[])
with patch(
    "cwsandbox._sandbox.Sandbox._from_id_async",
    new=AsyncMock(return_value=SandboxStub()),
):
    sandbox = session.from_id("test-123").result()

print(sandbox.sandbox_id)
print(signal.getsignal(signal.SIGTERM) is _signal_handler)
session.close().result()
"""
        result = _run_fresh(script)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["test-123", "True"]

    def test_off_main_owned_sandbox_recovers_on_main_thread(self) -> None:
        """Worker-thread first use skips signals; later main-thread use installs them."""
        script = """
import signal
import threading
from cwsandbox._cleanup import _activate_cleanup_handlers, _signal_handler

def worker():
    from cwsandbox import Sandbox
    Sandbox()

thread = threading.Thread(target=worker)
thread.start()
thread.join()
print("after_worker", signal.getsignal(signal.SIGTERM) is _signal_handler)
_activate_cleanup_handlers()
print("after_main", signal.getsignal(signal.SIGTERM) is _signal_handler)
"""
        result = _run_fresh(script)
        assert result.returncode == 0, result.stderr
        assert "after_worker False" in result.stdout
        assert "after_main True" in result.stdout

    @pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="Unix SIGTERM")
    def test_custom_host_handler_is_chained(self) -> None:
        """A host handler installed before first sandbox use is chained."""
        script = """
import os
import signal
import time
from cwsandbox import Sandbox

def host(signum, frame):
    print("HOST", flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, host)
Sandbox()
print("READY", flush=True)
time.sleep(30)
"""
        result = _run_fresh_until_sigterm(script)
        assert result.returncode == 0, result.stderr
        assert "HOST" in result.stdout

    @pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="Unix SIGTERM")
    def test_activated_child_keeps_sigterm_exit_behavior(self) -> None:
        """An activated child still exits 143 on SIGTERM with the default handler."""
        script = """
import time
from cwsandbox import Sandbox

Sandbox()
print("READY", flush=True)
time.sleep(30)
"""
        result = _run_fresh_until_sigterm(script)
        assert result.returncode == 128 + signal.SIGTERM, result.stderr

    @pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="Unix SIGTERM")
    def test_api_opt_out_lets_host_handle_sigterm(self) -> None:
        """API disable lets a host SIGTERM handler run instead of SystemExit(143)."""
        script = """
import os
import signal
import time
import cwsandbox
from cwsandbox import Sandbox

def host(signum, frame):
    print("HOST", flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, host)
cwsandbox.disable_signal_handlers()
Sandbox()
print("READY", flush=True)
time.sleep(30)
"""
        result = _run_fresh_until_sigterm(script)
        assert result.returncode == 0, result.stderr
        assert "HOST" in result.stdout

    @pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="Unix SIGTERM")
    def test_env_opt_out_leaves_host_handlers_intact(self) -> None:
        """Environment disable leaves host handlers in place after first sandbox."""
        script = """
import os
import signal
from cwsandbox import Sandbox
from cwsandbox._cleanup import _signal_handler

def host(signum, frame):
    raise SystemExit(0)

os.environ["CWSANDBOX_DISABLE_SIGNAL_HANDLERS"] = "1"
signal.signal(signal.SIGINT, host)
signal.signal(signal.SIGTERM, host)
Sandbox()
print(signal.getsignal(signal.SIGINT) is host)
print(signal.getsignal(signal.SIGTERM) is host)
print(signal.getsignal(signal.SIGTERM) is _signal_handler)
"""
        result = _run_fresh(script)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["True", "True", "False"]


class TestDisableSignalHandlers:
    """Tests for the public host signal-handler opt-out."""

    def test_disable_before_activate_skips_signals_and_keeps_atexit(self) -> None:
        """API disable before first sandbox skips signals but still registers atexit."""
        import cwsandbox._cleanup as cleanup_module

        with patch("cwsandbox._cleanup.atexit.register") as mock_register:
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                disable_signal_handlers()
                _activate_cleanup_handlers()

                mock_register.assert_called_once_with(_cleanup)
                mock_signal.assert_not_called()
                assert cleanup_module._atexit_registered is True
                assert cleanup_module._signals_installed is False
                assert cleanup_module._signals_disabled is True

    def test_repeated_disable_is_idempotent(self) -> None:
        """Repeated disable calls are safe and remain disabled."""
        import cwsandbox._cleanup as cleanup_module

        disable_signal_handlers()
        disable_signal_handlers()
        disable_signal_handlers()
        assert cleanup_module._signals_disabled is True
        assert cleanup_module._signals_installed is False

    def test_late_disable_raises_without_mutating_state(self) -> None:
        """A late API call raises and leaves installed handlers unchanged."""
        import cwsandbox._cleanup as cleanup_module

        _activate_cleanup_handlers()
        original_int = cleanup_module._original_sigint
        original_term = cleanup_module._original_sigterm

        with pytest.raises(RuntimeError, match="before"):
            disable_signal_handlers()

        assert cleanup_module._signals_disabled is False
        assert cleanup_module._signals_installed is True
        assert cleanup_module._original_sigint is original_int
        assert cleanup_module._original_sigterm is original_term
        assert signal.getsignal(signal.SIGINT) is _signal_handler
        assert signal.getsignal(signal.SIGTERM) is _signal_handler

    def test_late_env_change_does_not_uninstall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting the environment variable after install does not restore handlers."""
        import cwsandbox._cleanup as cleanup_module

        with patch("cwsandbox._cleanup.atexit.register"):
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                _activate_cleanup_handlers()
                assert mock_signal.call_count == 2
                monkeypatch.setenv("CWSANDBOX_DISABLE_SIGNAL_HANDLERS", "1")
                _activate_cleanup_handlers()

        assert cleanup_module._signals_installed is True
        assert cleanup_module._signals_disabled is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " Yes "])
    def test_truthy_env_disables_signals(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Documented truthy environment values skip signal installation."""
        import cwsandbox._cleanup as cleanup_module

        monkeypatch.setenv("CWSANDBOX_DISABLE_SIGNAL_HANDLERS", value)
        with patch("cwsandbox._cleanup.atexit.register") as mock_register:
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                _activate_cleanup_handlers()

                mock_register.assert_called_once_with(_cleanup)
                mock_signal.assert_not_called()
                assert cleanup_module._signals_disabled is True
                assert cleanup_module._signals_installed is False

    @pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
    def test_falsy_env_preserves_default_install(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset-style and false environment values keep the default install."""
        import cwsandbox._cleanup as cleanup_module

        monkeypatch.setenv("CWSANDBOX_DISABLE_SIGNAL_HANDLERS", value)
        with patch("cwsandbox._cleanup.atexit.register"):
            with patch("cwsandbox._cleanup.signal.signal") as mock_signal:
                _activate_cleanup_handlers()

                assert mock_signal.call_count == 2
                assert cleanup_module._signals_disabled is False
                assert cleanup_module._signals_installed is True

    def test_reset_clears_api_disabled_state(self) -> None:
        """Test reset clears the public disable flag so later tests can reinstall."""
        import cwsandbox._cleanup as cleanup_module

        disable_signal_handlers()
        assert cleanup_module._signals_disabled is True
        _reset_for_testing()
        assert cleanup_module._signals_disabled is False
