# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox logs — stream logs from a sandbox."""

from __future__ import annotations

from datetime import UTC, datetime

import click

from cwsandbox import Sandbox


@click.command()
@click.argument("sandbox_id")
@click.option(
    "--follow", "-f", is_flag=True, default=False, help="Follow log output (like tail -f)."
)
@click.option(
    "--tail",
    "tail_lines",
    type=click.IntRange(min=0),
    default=None,
    help="Number of recent lines to show.",
)
@click.option(
    "--since",
    "since_time",
    type=click.DateTime(),
    default=None,
    help="Show logs since timestamp (e.g. 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS').",
)
@click.option("--timestamps", "-t", is_flag=True, default=False, help="Show timestamps.")
def logs(
    sandbox_id: str,
    follow: bool,
    tail_lines: int | None,
    since_time: datetime | None,
    timestamps: bool,
) -> None:
    """Stream logs from a sandbox.

    SANDBOX_ID is the ID of the sandbox to stream logs from.
    """
    if since_time is not None and since_time.tzinfo is None:
        since_time = since_time.replace(tzinfo=UTC)

    sandbox = Sandbox.from_id(sandbox_id).result()

    reader = sandbox.stream_logs(
        follow=follow,
        tail_lines=tail_lines,
        since_time=since_time,
        timestamps=timestamps,
    )
    try:
        for line in reader:
            click.echo(line, nl=False)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
    finally:
        reader.close()
