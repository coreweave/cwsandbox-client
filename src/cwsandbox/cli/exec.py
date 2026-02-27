# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox exec — execute a command in a sandbox."""

from __future__ import annotations

import sys
import threading

import click

from cwsandbox import Sandbox


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("sandbox_id")
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option(
    "--cwd",
    "-w",
    default=None,
    help="Working directory for the command.",
)
@click.option(
    "--timeout",
    "-t",
    "timeout_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Timeout in seconds.",
)
def exec_command(
    sandbox_id: str,
    command: tuple[str, ...],
    cwd: str | None,
    timeout_seconds: float | None,
) -> None:
    """Execute a command in a sandbox.

    SANDBOX_ID is the ID of the sandbox to run the command in.

    Examples:

        cwsandbox exec <sandbox-id> echo hello

        cwsandbox exec <sandbox-id> python -c "print('hello')"

        cwsandbox exec <sandbox-id> --cwd /app python script.py
    """
    sandbox = Sandbox.from_id(sandbox_id).result()

    try:
        process = sandbox.exec(
            list(command),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )

        def _drain_stderr() -> None:
            try:
                for chunk in process.stderr:
                    click.echo(chunk, nl=False, err=True)
            except BrokenPipeError:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr)
        stderr_thread.start()

        try:
            for line in process.stdout:
                click.echo(line, nl=False)

            result = process.result()
        finally:
            stderr_thread.join()
    except KeyboardInterrupt:
        sys.exit(130)
    except BrokenPipeError:
        pass  # Piped to head/etc — exit cleanly
    else:
        sys.exit(result.returncode)
