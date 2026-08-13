# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox stop - stop a sandbox."""

from __future__ import annotations

import click

from cwsandbox import Sandbox
from cwsandbox._defaults import DEFAULT_GRACEFUL_SHUTDOWN_SECONDS
from cwsandbox.exceptions import SandboxNotFoundError


@click.command("stop")
@click.argument("sandbox_id")
@click.option(
    "--missing-ok",
    is_flag=True,
    default=False,
    help="Do not fail if the sandbox is already missing.",
)
@click.option(
    "--snapshot-on-stop",
    is_flag=True,
    default=False,
    help="Capture a file-system snapshot before stopping.",
)
@click.option(
    "--wait-for-snapshot/--no-wait-for-snapshot",
    default=True,
    help="Wait for a snapshot-on-stop snapshot to reach ready or failed.",
)
@click.option(
    "--request-id",
    default=None,
    help="Client-supplied request ID for snapshot-on-stop.",
)
@click.option(
    "--graceful-shutdown-seconds",
    type=click.FloatRange(min=0),
    default=DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    show_default=True,
    help="Seconds to wait for graceful shutdown.",
)
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress success output.")
def stop_sandbox(
    sandbox_id: str,
    missing_ok: bool,
    snapshot_on_stop: bool,
    wait_for_snapshot: bool,
    request_id: str | None,
    graceful_shutdown_seconds: float,
    quiet: bool,
) -> None:
    """Stop a sandbox.

    SANDBOX_ID is the ID of the sandbox to stop.
    """
    if not snapshot_on_stop and not wait_for_snapshot:
        raise click.UsageError("--no-wait-for-snapshot requires --snapshot-on-stop.")
    if request_id is not None and not snapshot_on_stop:
        raise click.UsageError("--request-id requires --snapshot-on-stop.")

    try:
        sandbox = Sandbox.from_id(sandbox_id).result()
    except SandboxNotFoundError:
        if not missing_ok:
            raise
        if not quiet:
            click.echo(f"Sandbox {sandbox_id} is already missing.")
        return

    sandbox.stop(
        snapshot_on_stop=snapshot_on_stop,
        graceful_shutdown_seconds=graceful_shutdown_seconds,
        missing_ok=missing_ok,
        wait_for_ready=wait_for_snapshot,
        request_id=request_id,
    ).result()

    if quiet:
        return

    snapshot_id = sandbox.file_system_snapshot_id
    if snapshot_on_stop and snapshot_id:
        click.echo(f"Stopped sandbox {sandbox_id}. Snapshot {snapshot_id}.")
    else:
        click.echo(f"Stopped sandbox {sandbox_id}.")
