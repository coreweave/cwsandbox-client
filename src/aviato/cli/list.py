# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: aviato-client

"""aviato list — list sandboxes."""

from __future__ import annotations

import click

from aviato import Sandbox, SandboxStatus
from aviato.exceptions import AviatoError

_STATUS_CHOICES = [s.value for s in SandboxStatus if s != SandboxStatus.UNSPECIFIED]


@click.command("list")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(_STATUS_CHOICES, case_sensitive=False),
    help="Filter by status.",
)
@click.option("--tag", "-t", "tags", multiple=True, help="Filter by tag (repeatable).")
@click.option("--runway-id", "runway_ids", multiple=True, help="Filter by runway ID (repeatable).")
@click.option("--tower-id", "tower_ids", multiple=True, help="Filter by tower ID (repeatable).")
def list_sandboxes(
    status: str | None,
    tags: tuple[str, ...],
    runway_ids: tuple[str, ...],
    tower_ids: tuple[str, ...],
) -> None:
    """List sandboxes.

    Displays sandbox ID, status, tower, runway, and started time for matching sandboxes.
    """
    try:
        sandboxes = Sandbox.list(
            tags=list(tags) if tags else None,
            status=status,
            runway_ids=list(runway_ids) if runway_ids else None,
            tower_ids=list(tower_ids) if tower_ids else None,
        ).result()
    except AviatoError as e:
        raise click.ClickException(str(e)) from None

    if not sandboxes:
        click.echo("No sandboxes found.")
        return

    click.echo(f"{'SANDBOX ID':<40} {'STATUS':<14} {'TOWER':<20} {'RUNWAY':<20} {'STARTED AT'}")
    click.echo(f"{'-' * 40} {'-' * 14} {'-' * 20} {'-' * 20} {'-' * 24}")

    for sb in sandboxes:
        sid = sb.sandbox_id or "-"
        st = sb.status.value if sb.status else "-"
        tower = sb.tower_id or "-"
        runway = sb.runway_id or "-"
        started = sb.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if sb.started_at else "-"
        click.echo(f"{sid:<40} {st:<14} {tower:<20} {runway:<20} {started}")
