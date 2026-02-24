# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""aviato exec — execute a command in a sandbox."""

from __future__ import annotations

import sys
import threading

import click

from aviato import Sandbox
from aviato.exceptions import AviatoError


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("sandbox_id")
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.option(
    "--cwd",
    default=None,
    help="Working directory for the command.",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=float,
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

        aviato exec <sandbox-id> echo hello

        aviato exec <sandbox-id> sh -c "echo hello > /tmp/file"

        aviato exec <sandbox-id> --cwd /app python script.py
    """
    try:
        sandbox = Sandbox.from_id(sandbox_id).result()
    except AviatoError as e:
        raise click.ClickException(str(e)) from None

    try:
        process = sandbox.exec(
            list(command),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )

        def _drain_stderr() -> None:
            for chunk in process.stderr:
                click.echo(chunk, nl=False, err=True)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        try:
            for line in process.stdout:
                click.echo(line, nl=False)
        finally:
            stderr_thread.join(timeout=5.0)

        result = process.result()
    except KeyboardInterrupt:
        sys.exit(130)
    except BrokenPipeError:
        pass  # Piped to head/etc — exit cleanly
    except AviatoError as e:
        raise click.ClickException(str(e)) from None
    else:
        sys.exit(result.returncode)
