# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: cwsandbox-client

"""cwsandbox ls — list sandboxes."""

from __future__ import annotations

import json

import click

from cwsandbox import Sandbox, SandboxStatus

_STATUS_CHOICES = [s.value for s in SandboxStatus if s != SandboxStatus.UNSPECIFIED]


@click.command("ls")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(_STATUS_CHOICES, case_sensitive=False),
    help="Filter by status.",
)
@click.option("--tag", "-t", "tags", multiple=True, help="Filter by tag (repeatable).")
@click.option(
    "--runway-id", "-r", "runway_ids", multiple=True, help="Filter by runway ID (repeatable)."
)
@click.option(
    "--tower-id", "-T", "tower_ids", multiple=True, help="Filter by tower ID (repeatable)."
)
@click.option(
    "--output",
    "-o",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format.",
)
@click.option("--show-tags", is_flag=True, default=False, help="Show tags column in table output.")
def list_sandboxes(
    status: str | None,
    tags: tuple[str, ...],
    runway_ids: tuple[str, ...],
    tower_ids: tuple[str, ...],
    output_format: str,
    show_tags: bool,
) -> None:
    """List sandboxes.

    Displays sandbox ID, status, tower, runway, and started time for matching sandboxes.
    """
    sandboxes = Sandbox.list(
        tags=list(tags) if tags else None,
        status=status,
        runway_ids=list(runway_ids) if runway_ids else None,
        tower_ids=list(tower_ids) if tower_ids else None,
    ).result()

    if output_format == "json":
        data = [
            {
                "sandbox_id": sb.sandbox_id,
                "status": sb.status.value if sb.status else None,
                "tower_id": sb.tower_id,
                "runway_id": sb.runway_id,
                "tower_group_id": sb.tower_group_id,
                "started_at": sb.started_at.isoformat() if sb.started_at else None,
                "tags": list(sb.tags) if sb.tags else [],
            }
            for sb in sandboxes
        ]
        click.echo(json.dumps(data, indent=2))
        return

    if not sandboxes:
        click.echo("No sandboxes found.")
        return

    header = f"{'SANDBOX ID':<40} {'STATUS':<14} {'TOWER':<20} {'RUNWAY':<20} {'STARTED AT':<24}"
    sep = f"{'-' * 40} {'-' * 14} {'-' * 20} {'-' * 20} {'-' * 24}"
    if show_tags:
        header += f" {'TAGS'}"
        sep += f" {'-' * 40}"
    click.echo(header)
    click.echo(sep)

    for sb in sandboxes:
        sid = sb.sandbox_id or "-"
        st = sb.status.value if sb.status else "-"
        tower = sb.tower_id or "-"
        runway = sb.runway_id or "-"
        started = sb.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if sb.started_at else "-"
        line = f"{sid:<40} {st:<14} {tower:<20} {runway:<20} {started:<24}"
        if show_tags:
            line += f" {','.join(sb.tags) if sb.tags else '<none>'}"
        click.echo(line)
