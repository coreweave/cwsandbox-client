# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox shell — interactive shell in a sandbox."""

from __future__ import annotations

import logging
import os
import platform
import shlex
import shutil
import signal
import sys
import threading
from types import FrameType

import click

from cwsandbox import Sandbox

logger = logging.getLogger(__name__)


def _validate_cmd(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Validate --cmd is a parseable, non-empty shell command."""
    try:
        parts = shlex.split(value)
    except ValueError as e:
        raise click.BadParameter(str(e)) from None
    if not parts:
        raise click.BadParameter("must not be empty.")
    return value


@click.command()
@click.argument("sandbox_id")
@click.option(
    "--cmd",
    default="/bin/bash",
    callback=_validate_cmd,
    help="Command to run (default: /bin/bash). Accepts full command strings.",
)
def shell(sandbox_id: str, cmd: str) -> None:
    """Open an interactive shell in a sandbox.

    SANDBOX_ID is the ID of the sandbox to connect to.

    The terminal runs in raw mode, so Ctrl+C is forwarded to the remote
    process instead of exiting locally. To exit, type 'exit' or press Ctrl+D.

    Examples:

        cwsandbox sh <sandbox-id>

        cwsandbox sh <sandbox-id> --cmd /bin/zsh

        cwsandbox sh <sandbox-id> --cmd "python main.py"
    """
    if platform.system() == "Windows":
        raise click.ClickException("Interactive shell is not supported on Windows.")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise click.ClickException(
            "Interactive shell requires a TTY (stdin and stdout must be a terminal)."
        )

    # Late imports for Unix-only modules
    import termios
    import tty

    sandbox = Sandbox.from_id(sandbox_id).result()

    # Pass initial size so remote PTY matches local terminal
    size = shutil.get_terminal_size()

    command = shlex.split(cmd)

    session = sandbox.shell(
        command,
        width=size.columns,
        height=size.lines,
    )

    # Raw mode forwards all keystrokes (incl. Ctrl+C) to the remote session
    stdin_fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(stdin_fd)

    # Keep remote PTY dimensions in sync when user resizes window
    def on_sigwinch(signum: int, frame: FrameType | None) -> None:
        new_size = shutil.get_terminal_size()
        try:
            session.resize(new_size.columns, new_size.lines)  # fire-and-forget
        except Exception:
            logger.debug("Failed to resize remote PTY", exc_info=True)

    old_sigwinch = signal.getsignal(signal.SIGWINCH)
    signal.signal(signal.SIGWINCH, on_sigwinch)

    exit_code = 1
    try:
        tty.setraw(stdin_fd)

        # Background thread avoids blocking the output loop while waiting for user input
        def forward_stdin() -> None:
            try:
                while True:
                    data = os.read(stdin_fd, 1024)
                    if not data:
                        session.stdin.close().result(timeout=5.0)
                        break
                    session.stdin.write(data).result(timeout=5.0)
            except Exception:
                logger.debug("stdin forwarding stopped", exc_info=True)

        stdin_thread = threading.Thread(target=forward_stdin, daemon=True)
        stdin_thread.start()

        # Forward remote output to local terminal (raw bytes, no re-encode)
        for chunk in session.output:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

        # Collect exit code after output stream ends
        try:
            exit_code = session.wait(timeout=5.0)
        except Exception:
            logger.debug("Failed to collect exit code", exc_info=True)
            exit_code = 1

    except KeyboardInterrupt:
        exit_code = 130

    finally:
        # Must restore before printing anything, otherwise output is garbled
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
        signal.signal(signal.SIGWINCH, old_sigwinch)

    sys.exit(exit_code)
