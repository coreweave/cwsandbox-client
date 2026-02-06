# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Unit tests for aviato._cleanup module."""

import signal
from unittest.mock import MagicMock, patch

import pytest

from aviato._cleanup import (
    _cleanup,
    _install_handlers,
    _reset_for_testing,
    _signal_handler,
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
        with patch("aviato._loop_manager._LoopManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.get.return_value = mock_manager

            _cleanup()

            mock_manager_class.get.assert_called_once()
            mock_manager.cleanup_all.assert_called_once()

    def test_cleanup_guards_against_reentrancy(self) -> None:
        """Test _cleanup only runs once even if called multiple times."""
        import aviato._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False

        with patch("aviato._loop_manager._LoopManager") as mock_manager_class:
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
        with patch("aviato._loop_manager._LoopManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.cleanup_all.side_effect = RuntimeError("test error")
            mock_manager_class.get.return_value = mock_manager

            # Should not raise
            _cleanup()


class TestSignalHandler:
    """Tests for _signal_handler function."""

    def test_signal_handler_calls_cleanup(self) -> None:
        """Test signal handler calls _cleanup."""
        import aviato._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigint = signal.SIG_IGN

        with patch("aviato._loop_manager._LoopManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.get.return_value = mock_manager

            _signal_handler(signal.SIGINT, None)

            mock_manager.cleanup_all.assert_called_once()

    def test_signal_handler_chains_to_sig_dfl(self) -> None:
        """Test signal handler exits with signal code for SIG_DFL."""
        import aviato._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigint = signal.SIG_DFL

        with patch("aviato._loop_manager._LoopManager"):
            with pytest.raises(SystemExit) as exc_info:
                _signal_handler(signal.SIGINT, None)

            # Exit code should be 128 + signal number
            assert exc_info.value.code == 128 + signal.SIGINT

    def test_signal_handler_chains_to_sig_ign(self) -> None:
        """Test signal handler does nothing for SIG_IGN."""
        import aviato._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigint = signal.SIG_IGN

        with patch("aviato._loop_manager._LoopManager"):
            # Should not raise
            _signal_handler(signal.SIGINT, None)

    def test_signal_handler_chains_to_callable(self) -> None:
        """Test signal handler chains to callable original handler."""
        import aviato._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False

        original_handler = MagicMock()
        cleanup_module._original_sigint = original_handler

        with patch("aviato._loop_manager._LoopManager"):
            _signal_handler(signal.SIGINT, None)

            original_handler.assert_called_once_with(signal.SIGINT, None)

    def test_second_signal_forces_exit(self) -> None:
        """Test second signal during cleanup forces immediate exit."""
        import aviato._cleanup as cleanup_module

        # Simulate cleanup already in progress
        cleanup_module._cleanup_in_progress = True

        with pytest.raises(SystemExit) as exc_info:
            _signal_handler(signal.SIGINT, None)

        # Exit code should be 128 + signal number
        assert exc_info.value.code == 128 + signal.SIGINT

    def test_signal_handler_handles_sigterm(self) -> None:
        """Test signal handler works for SIGTERM."""
        import aviato._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = False
        cleanup_module._original_sigterm = signal.SIG_DFL

        with patch("aviato._loop_manager._LoopManager"):
            with pytest.raises(SystemExit) as exc_info:
                _signal_handler(signal.SIGTERM, None)

            assert exc_info.value.code == 128 + signal.SIGTERM


class TestInstallHandlers:
    """Tests for _install_handlers function."""

    def test_install_handlers_registers_atexit(self) -> None:
        """Test _install_handlers registers atexit handler."""
        with patch("aviato._cleanup.atexit.register") as mock_register:
            with patch("aviato._cleanup.signal.signal"):
                _install_handlers()

                mock_register.assert_called_once_with(_cleanup)

    def test_install_handlers_installs_sigint_handler(self) -> None:
        """Test _install_handlers installs SIGINT handler."""
        with patch("aviato._cleanup.atexit.register"):
            with patch("aviato._cleanup.signal.signal") as mock_signal:
                _install_handlers()

                # Should have been called for SIGINT
                calls = mock_signal.call_args_list
                sigint_calls = [c for c in calls if c[0][0] == signal.SIGINT]
                assert len(sigint_calls) == 1
                assert sigint_calls[0][0][1] == _signal_handler

    def test_install_handlers_installs_sigterm_handler(self) -> None:
        """Test _install_handlers installs SIGTERM handler."""
        with patch("aviato._cleanup.atexit.register"):
            with patch("aviato._cleanup.signal.signal") as mock_signal:
                _install_handlers()

                # Should have been called for SIGTERM
                calls = mock_signal.call_args_list
                sigterm_calls = [c for c in calls if c[0][0] == signal.SIGTERM]
                assert len(sigterm_calls) == 1
                assert sigterm_calls[0][0][1] == _signal_handler

    def test_install_handlers_only_installs_once(self) -> None:
        """Test _install_handlers only installs handlers once."""
        with patch("aviato._cleanup.atexit.register") as mock_register:
            with patch("aviato._cleanup.signal.signal"):
                _install_handlers()
                _install_handlers()
                _install_handlers()

                # Should only register once
                assert mock_register.call_count == 1

    def test_install_handlers_preserves_original_handlers(self) -> None:
        """Test _install_handlers preserves original signal handlers."""
        import aviato._cleanup as cleanup_module

        original_sigint = MagicMock()
        original_sigterm = MagicMock()

        with patch("aviato._cleanup.atexit.register"):
            with patch(
                "aviato._cleanup.signal.signal",
                side_effect=[original_sigint, original_sigterm],
            ):
                _install_handlers()

                assert cleanup_module._original_sigint == original_sigint
                assert cleanup_module._original_sigterm == original_sigterm


class TestResetForTesting:
    """Tests for _reset_for_testing function."""

    def test_reset_clears_cleanup_in_progress(self) -> None:
        """Test _reset_for_testing clears cleanup_in_progress flag."""
        import aviato._cleanup as cleanup_module

        cleanup_module._cleanup_in_progress = True
        _reset_for_testing()
        assert cleanup_module._cleanup_in_progress is False

    def test_reset_clears_handlers_installed(self) -> None:
        """Test _reset_for_testing clears handlers_installed flag."""
        import aviato._cleanup as cleanup_module

        cleanup_module._handlers_installed = True
        _reset_for_testing()
        assert cleanup_module._handlers_installed is False
