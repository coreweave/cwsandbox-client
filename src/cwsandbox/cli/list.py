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
    "--profile-id", "-r", "profile_ids", multiple=True, help="Filter by profile ID (repeatable)."
)
@click.option(
    "--profile-name",
    "profile_names",
    multiple=True,
    help="Filter by profile name (repeatable).",
)
@click.option(
    "--runner-id", "-T", "runner_ids", multiple=True, help="Filter by runner ID (repeatable)."
)
@click.option(
    "--output",
    "-o",
    "output_format",
    default="table",
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format.",
)
def list_sandboxes(
    status: str | None,
    tags: tuple[str, ...],
    profile_ids: tuple[str, ...],
    profile_names: tuple[str, ...],
    runner_ids: tuple[str, ...],
    output_format: str,
) -> None:
    """List sandboxes.

    Displays sandbox ID, status, runner, profile, and started time for matching sandboxes.
    """
    sandboxes = Sandbox.list(
        tags=list(tags) if tags else None,
        status=status,
        profile_ids=list(profile_ids) if profile_ids else None,
        profile_names=list(profile_names) if profile_names else None,
        runner_ids=list(runner_ids) if runner_ids else None,
    ).result()

    if output_format == "json":
        data = [
            {
                "sandbox_id": sb.sandbox_id,
                "status": sb.status.value if sb.status else None,
                "runner_id": sb.runner_id,
                "profile_id": sb.profile_id,
                "runner_group_id": sb.runner_group_id,
                "started_at": sb.started_at.isoformat() if sb.started_at else None,
            }
            for sb in sandboxes
        ]
        click.echo(json.dumps(data, indent=2))
        return

    if not sandboxes:
        click.echo("No sandboxes found.")
        return

    click.echo(f"{'SANDBOX ID':<40} {'STATUS':<14} {'TOWER':<20} {'RUNWAY':<20} {'STARTED AT'}")
    click.echo(f"{'-' * 40} {'-' * 14} {'-' * 20} {'-' * 20} {'-' * 24}")

    for sb in sandboxes:
        sid = sb.sandbox_id or "-"
        st = sb.status.value if sb.status else "-"
        runner = sb.runner_id or "-"
        profile = sb.profile_id or "-"
        started = sb.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if sb.started_at else "-"
        click.echo(f"{sid:<40} {st:<14} {runner:<20} {profile:<20} {started}")
