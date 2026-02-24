# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""Tests for aviato shell CLI command."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from aviato.cli import cli

from .conftest import make_operation_ref


@contextmanager
def _terminal_env(
    mock_sandbox: MagicMock,
    *,
    os_read_data: list[bytes] | None = None,
    exit_code: int = 0,
    output: Iterator[bytes] | None = None,
    terminal_size: tuple[int, int] = (80, 24),
) -> Iterator[dict[str, Any]]:
    """Set up the full terminal mock environment for shell command tests.

    Patches the 9 targets needed to simulate a Unix terminal session without
    touching real file descriptors or terminal state.

    Yields a dict with ``session``, ``sandbox``, and ``tcsetattr`` for
    assertions in individual tests.
    """
    import termios as _termios
    import tty as _tty

    mock_session = MagicMock()
    mock_session.output = output if output is not None else iter([])
    mock_session.stdin = MagicMock()
    mock_session.stdin.write.return_value.result.return_value = None
    mock_session.stdin.close.return_value.result.return_value = None
    mock_session.wait.return_value = exit_code
    mock_sandbox.shell.return_value = mock_session

    if os_read_data is None:
        os_read_data = [b""]

    def raise_system_exit(code: int = 0) -> None:
        raise SystemExit(code)

    op_ref = make_operation_ref(mock_sandbox)

    with (
        patch("aviato.cli.shell.platform") as mock_platform,
        patch("aviato.cli.shell.Sandbox") as mock_sandbox_cls,
        patch("aviato.cli.shell.shutil.get_terminal_size") as mock_size,
        patch("aviato.cli.shell.os.read") as mock_os_read,
        patch("aviato.cli.shell.sys") as mock_sys,
        patch("aviato.cli.shell.signal") as mock_signal,
        patch.object(_termios, "tcgetattr", return_value=[]),
        patch.object(_termios, "tcsetattr") as mock_tcsetattr,
        patch.object(_tty, "setraw"),
    ):
        mock_platform.system.return_value = "Linux"
        mock_sandbox_cls.from_id.return_value = op_ref
        mock_size.return_value = os.terminal_size(terminal_size)
        mock_os_read.side_effect = os_read_data
        mock_sys.stdin.fileno.return_value = 5
        mock_sys.stdin.isatty.return_value = True
        mock_sys.stdout.buffer = MagicMock()
        mock_sys.stdout.isatty.return_value = True
        mock_sys.exit = raise_system_exit
        mock_signal.SIGWINCH = 28

        yield {
            "session": mock_session,
            "sandbox": mock_sandbox,
            "tcsetattr": mock_tcsetattr,
        }


class TestShellCommand:
    """Tests for the aviato shell CLI command."""

    def test_shell_registered(self) -> None:
        """Shell command is registered on the CLI group."""
        runner = CliRunner()
        result = runner.invoke(cli, ["shell", "--help"])
        assert result.exit_code == 0
        assert "SANDBOX_ID" in result.output

    def test_shell_windows_error(self) -> None:
        """aviato shell exits with error on Windows."""
        with patch("aviato.cli.shell.platform") as mock_platform:
            mock_platform.system.return_value = "Windows"

            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "test-sandbox-id"])

        assert result.exit_code != 0
        assert "not supported on Windows" in result.output

    def test_shell_no_tty_error(self) -> None:
        """aviato shell exits with error when stdin is not a TTY."""
        with (
            patch("aviato.cli.shell.platform") as mock_platform,
            patch("aviato.cli.shell.sys") as mock_sys,
        ):
            mock_platform.system.return_value = "Linux"
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdout.isatty.return_value = True

            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "test-sandbox-id"])

        assert result.exit_code != 0
        assert "requires a TTY" in result.output

    def test_shell_happy_path(self) -> None:
        """aviato shell streams output and exits with the session exit code."""
        mock_sandbox = MagicMock()
        with _terminal_env(
            mock_sandbox,
            os_read_data=[b"some input", b""],
            output=iter([b"hello from shell\n"]),
        ) as env:
            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "test-sandbox-id"])

        assert result.exit_code == 0
        mock_sandbox.shell.assert_called_once_with(
            ["/bin/bash"],
            width=80,
            height=24,
        )
        # Verify terminal was restored
        env["tcsetattr"].assert_called_once()

    def test_shell_custom_cmd(self) -> None:
        """aviato shell --cmd passes custom command to shell()."""
        mock_sandbox = MagicMock()
        with _terminal_env(
            mock_sandbox,
            terminal_size=(120, 40),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "test-sandbox-id", "--cmd", "python main.py"])

        assert result.exit_code == 0
        mock_sandbox.shell.assert_called_once_with(
            ["python", "main.py"],
            width=120,
            height=40,
        )

    def test_shell_sandbox_not_found(self) -> None:
        """aviato shell shows error when sandbox is not found."""
        from aviato.exceptions import SandboxNotFoundError

        mock_op_ref = MagicMock()
        mock_op_ref.result.side_effect = SandboxNotFoundError("not found", sandbox_id="bad-id")

        with (
            patch("aviato.cli.shell.platform") as mock_platform,
            patch("aviato.cli.shell.Sandbox") as mock_sandbox_cls,
            patch("aviato.cli.shell.sys") as mock_sys,
        ):
            mock_platform.system.return_value = "Linux"
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            mock_sandbox_cls.from_id.return_value = mock_op_ref

            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "bad-id"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_shell_nonzero_exit_code(self) -> None:
        """aviato shell exits with the remote session's nonzero exit code."""
        mock_sandbox = MagicMock()
        with _terminal_env(mock_sandbox, exit_code=42):
            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "test-sandbox-id"])

        assert result.exit_code == 42

    def test_shell_keyboard_interrupt(self) -> None:
        """aviato shell exits 130 on KeyboardInterrupt."""

        def _interrupted_output() -> Iterator[bytes]:
            raise KeyboardInterrupt
            yield  # pragma: no cover -- makes this a generator

        mock_sandbox = MagicMock()
        with _terminal_env(
            mock_sandbox,
            output=_interrupted_output(),
        ) as env:
            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "test-sandbox-id"])

        assert result.exit_code == 130
        # Terminal must still be restored
        env["tcsetattr"].assert_called_once()

    def test_shell_aviato_error_from_shell_call(self) -> None:
        """aviato shell shows error when sandbox.shell() raises AviatoError."""
        from aviato.exceptions import AviatoError

        mock_sandbox = MagicMock()
        mock_sandbox.shell.side_effect = AviatoError("shell creation failed")

        with _terminal_env(mock_sandbox):
            runner = CliRunner()
            result = runner.invoke(cli, ["shell", "test-sandbox-id"])

        assert result.exit_code != 0
        assert "shell creation failed" in result.output
