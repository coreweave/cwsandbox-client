# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox delete - delete a sandbox."""

from __future__ import annotations

import click

from cwsandbox import Sandbox


@click.command("delete")
@click.argument("sandbox_id")
@click.option(
    "--missing-ok",
    is_flag=True,
    default=False,
    help="Do not fail if the sandbox is already missing.",
)
@click.option(
    "--timeout",
    "-t",
    "timeout_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Timeout in seconds.",
)
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress success output.")
def delete_sandbox(
    sandbox_id: str,
    missing_ok: bool,
    timeout_seconds: float | None,
    quiet: bool,
) -> None:
    """Delete a sandbox.

    SANDBOX_ID is the ID of the sandbox to delete.
    """
    Sandbox.delete(
        sandbox_id,
        missing_ok=missing_ok,
        timeout_seconds=timeout_seconds,
    ).result()
    if not quiet:
        click.echo(f"Deleted sandbox {sandbox_id}.")
